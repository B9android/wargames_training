"""GraphQL Gateway — permanent, GraphQL-first client for all GitHub operations.

Replaces the old projects_v2.py with a unified, testable gateway.
All write paths respect DRY_RUN=true.  REST is used only where GraphQL
has no equivalent (e.g. issue creation via REST which is then synced via GraphQL).
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from typing import Any

from agent_platform.errors import AgentError, DryRunViolation, GraphQLError, ResourceNotFound
from agent_platform.telemetry import log_event

# Default project board title — overridable at construction time.
DEFAULT_PROJECT_TITLE = "Wargames Training — Master Board"


class GraphQLGateway:
    """GraphQL-first client for GitHub Projects v2, issues, labels, milestones, iterations."""

    def __init__(
        self,
        github_token: str,
        repo_name: str,
        *,
        dry_run: bool = False,
        project_title: str = DEFAULT_PROJECT_TITLE,
    ) -> None:
        self.github_token = github_token
        self.repo_name = repo_name
        self.dry_run = dry_run
        self.project_title = project_title

        owner, _, _ = repo_name.partition("/")
        self._owner = owner

        # Caches (TTL invalidation done manually via invalidate_cache())
        self._project_id: str | None = None
        self._field_ids: dict[str, dict[str, Any]] | None = None  # name → {id, dataType, options}
        self._active_sprint_id: str | None = None
        self._rest_fallback_count: int = 0

    # ------------------------------------------------------------------
    # Core GraphQL execution
    # ------------------------------------------------------------------

    def _gql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        *,
        allow_partial_errors: bool = False,
        retries: int = 3,
    ) -> dict[str, Any]:
        """Execute a GraphQL query with exponential-backoff retry."""
        import time as _time

        payload = {"query": query.strip()}
        if variables:
            payload["variables"] = variables

        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                proc = subprocess.run(
                    ["gh", "api", "graphql", "--input", "-"],
                    input=json.dumps(payload),
                    capture_output=True,
                    text=True,
                    env=self._gh_env(),
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"gh CLI error: {proc.stderr.strip()}")
                response = json.loads(proc.stdout)
                if "errors" in response and not allow_partial_errors:
                    msgs = "; ".join(e.get("message", "unknown") for e in response["errors"])
                    raise GraphQLError(msgs)
                return response.get("data") or {}
            except GraphQLError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    delay = 2 ** (attempt - 1)
                    log_event("graphql_retry", attempt=attempt, delay=delay, error=str(exc))
                    _time.sleep(delay)

        raise AgentError(f"GraphQL call failed after {retries} attempts: {last_exc}")

    def _gh_env(self) -> dict[str, str]:
        import os
        env = os.environ.copy()
        env["GH_TOKEN"] = self.github_token
        return env

    def _guard_write(self, operation: str) -> None:
        """Raise if DRY_RUN is set and we are about to mutate."""
        if self.dry_run:
            raise DryRunViolation(f"Mutation {operation!r} blocked: DRY_RUN=true")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        self._project_id = None
        self._field_ids = None
        self._active_sprint_id = None

    # ------------------------------------------------------------------
    # Project identification
    # ------------------------------------------------------------------

    def get_project_id(self) -> str:
        if self._project_id:
            return self._project_id

        data = self._gql(
            """
            query($login: String!) {
              user(login: $login) {
                projectsV2(first: 30) {
                  nodes { id number title }
                }
              }
            }
            """,
            {"login": self._owner},
        )
        nodes = data.get("user", {}).get("projectsV2", {}).get("nodes", [])
        project = next((p for p in nodes if p["title"] == self.project_title), None)
        if not project:
            raise ResourceNotFound(f"Project '{self.project_title}' not found in {self._owner}'s projects")
        self._project_id = project["id"]
        log_event("project_id_resolved", project=self.project_title, id=self._project_id)
        return self._project_id

    # ------------------------------------------------------------------
    # Field introspection
    # ------------------------------------------------------------------

    def get_fields(self, *, invalidate: bool = False) -> dict[str, dict[str, Any]]:
        """Return {fieldName: {id, dataType, options?: [{id, name}]}}."""
        if self._field_ids is not None and not invalidate:
            return self._field_ids

        project_id = self.get_project_id()
        data = self._gql(
            """
            query($id: ID!) {
              node(id: $id) {
                ... on ProjectV2 {
                  fields(first: 50) {
                    nodes {
                      ... on ProjectV2Field         { id name dataType }
                      ... on ProjectV2IterationField { id name dataType
                        configuration {
                          iterations { id title startDate duration }
                          completedIterations { id title startDate duration }
                        }
                      }
                      ... on ProjectV2SingleSelectField {
                        id name dataType
                        options { id name }
                      }
                    }
                  }
                }
              }
            }
            """,
            {"id": project_id},
        )
        nodes = data.get("node", {}).get("fields", {}).get("nodes", [])
        self._field_ids = {n["name"]: n for n in nodes if n.get("id")}
        log_event("fields_cached", count=len(self._field_ids), fields=list(self._field_ids))
        return self._field_ids

    def get_field_id(self, field_name: str) -> str:
        fields = self.get_fields()
        if field_name not in fields:
            raise ResourceNotFound(f"Field '{field_name}' not found. Available: {list(fields)}")
        return fields[field_name]["id"]

    def get_single_select_option_id(self, field_name: str, option_name: str) -> str:
        """Resolve a single-select option name to its internal option ID."""
        fields = self.get_fields()
        field = fields.get(field_name)
        if not field:
            raise ResourceNotFound(f"Field '{field_name}' not found")
        options = field.get("options", [])
        opt = next((o for o in options if o["name"].lower() == option_name.lower()), None)
        if not opt:
            available = [o["name"] for o in options]
            raise ResourceNotFound(f"Option '{option_name}' not found in '{field_name}'. Available: {available}")
        return opt["id"]

    # ------------------------------------------------------------------
    # Sprint / iteration
    # ------------------------------------------------------------------

    def get_active_sprint_id(self) -> str | None:
        if self._active_sprint_id:
            return self._active_sprint_id

        fields = self.get_fields()
        sprint_field = fields.get("Sprint")
        if not sprint_field:
            return None

        iters = sprint_field.get("configuration", {}).get("iterations", [])
        if not iters:
            return None

        # Most recent non-completed iteration
        sprint_id = iters[-1]["id"]
        self._active_sprint_id = sprint_id
        log_event("active_sprint_resolved", sprint_id=sprint_id, title=iters[-1].get("title"))
        return sprint_id

    # ------------------------------------------------------------------
    # Issue node resolution
    # ------------------------------------------------------------------

    def get_issue_node_id(self, issue_number: int) -> str:
        owner, _, repo = self.repo_name.partition("/")
        data = self._gql(
            """
            query($owner: String!, $repo: String!, $number: Int!) {
              repository(owner: $owner, name: $repo) {
                issue(number: $number) { id }
              }
            }
            """,
            {"owner": owner, "repo": repo, "number": issue_number},
        )
        node_id = data.get("repository", {}).get("issue", {}).get("id")
        if not node_id:
            raise ResourceNotFound(f"Issue #{issue_number} not found in {self.repo_name}")
        return node_id

    def get_pr_node_id(self, pr_number: int) -> str:
        owner, _, repo = self.repo_name.partition("/")
        data = self._gql(
            """
            query($owner: String!, $repo: String!, $number: Int!) {
              repository(owner: $owner, name: $repo) {
                pullRequest(number: $number) { id }
              }
            }
            """,
            {"owner": owner, "repo": repo, "number": pr_number},
        )
        node_id = data.get("repository", {}).get("pullRequest", {}).get("id")
        if not node_id:
            raise ResourceNotFound(f"PR #{pr_number} not found in {self.repo_name}")
        return node_id

    # ------------------------------------------------------------------
    # Bulk issue reads (GraphQL-first)
    # ------------------------------------------------------------------

    def get_issues_bulk(
        self,
        *,
        states: list[str] | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Paginated issue fetch using GraphQL — replaces all REST get_issues() calls."""
        owner, _, repo = self.repo_name.partition("/")
        state_filter = states or ["OPEN"]

        results: list[dict[str, Any]] = []
        cursor: str | None = None

        while len(results) < limit:
            page_size = min(50, limit - len(results))
            data = self._gql(
                """
                query($owner: String!, $repo: String!, $states: [IssueState!],
                      $first: Int!, $after: String) {
                  repository(owner: $owner, name: $repo) {
                    issues(states: $states, first: $first, after: $after,
                           orderBy: {field: UPDATED_AT, direction: DESC}) {
                      pageInfo { hasNextPage endCursor }
                      nodes {
                        number title body state createdAt updatedAt closedAt
                        labels(first: 20) { nodes { name } }
                        milestone { number title dueOn }
                        assignees(first: 5) { nodes { login } }
                      }
                    }
                  }
                }
                """,
                {
                    "owner": owner,
                    "repo": repo,
                    "states": state_filter,
                    "first": page_size,
                    "after": cursor,
                },
            )
            page = data.get("repository", {}).get("issues", {})
            nodes = page.get("nodes", [])
            results.extend(nodes)

            page_info = page.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

        log_event("issues_bulk_fetched", count=len(results), states=state_filter)
        return results

    # ------------------------------------------------------------------
    # Project item management
    # ------------------------------------------------------------------

    def add_issue_to_project(self, issue_node_id: str) -> str | None:
        """Add an issue to the project board. Returns item_id or None."""
        self._guard_write("add_issue_to_project")
        project_id = self.get_project_id()
        try:
            data = self._gql(
                """
                mutation($projectId: ID!, $contentId: ID!) {
                  addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
                    item { id }
                  }
                }
                """,
                {"projectId": project_id, "contentId": issue_node_id},
            )
            item_id = data.get("addProjectV2ItemById", {}).get("item", {}).get("id")
            if item_id:
                log_event("issue_added_to_project", issue_node_id=issue_node_id, item_id=item_id)
            return item_id
        except Exception as exc:
            log_event("add_issue_to_project_failed", issue_node_id=issue_node_id, error=str(exc))
            return None

    def get_project_item_id(self, issue_node_id: str) -> str | None:
        project_id = self.get_project_id()
        try:
            data = self._gql(
                """
                query($projectId: ID!, $first: Int!, $after: String) {
                  node(id: $projectId) {
                    ... on ProjectV2 {
                      items(first: $first, after: $after) {
                        pageInfo { hasNextPage endCursor }
                        nodes { id content { ... on Issue { id } } }
                      }
                    }
                  }
                }
                """,
                {"projectId": project_id, "first": 100, "after": None},
                allow_partial_errors=True,
            )
            items = data.get("node", {}).get("items", {}).get("nodes", [])
            for item in items:
                if item.get("content", {}).get("id") == issue_node_id:
                    return item["id"]
        except Exception as exc:
            log_event("get_project_item_id_failed", issue_node_id=issue_node_id, error=str(exc))
        return None

    def ensure_item_in_project(self, issue_node_id: str) -> str | None:
        """Add issue to project if not already present. Returns item_id."""
        existing = self.get_project_item_id(issue_node_id)
        if existing:
            return existing
        return self.add_issue_to_project(issue_node_id)

    # ------------------------------------------------------------------
    # Field updates
    # ------------------------------------------------------------------

    def update_field_value(
        self,
        item_id: str,
        field_name: str,
        value: str,
        *,
        field_type: str | None = None,
    ) -> bool:
        """Update a single project field value."""
        self._guard_write(f"update_field_value:{field_name}")
        project_id = self.get_project_id()
        fields = self.get_fields()
        field = fields.get(field_name)
        if not field:
            log_event("field_not_found", field=field_name, available=list(fields))
            return False

        resolved_type = field_type or field.get("dataType", "TEXT")

        if resolved_type == "SINGLE_SELECT":
            # Resolve option name → option ID
            try:
                option_id = self.get_single_select_option_id(field_name, value)
                value_input = {"singleSelectOptionId": option_id}
            except ResourceNotFound:
                # value may already be an option ID; use as-is
                value_input = {"singleSelectOptionId": value}
        elif resolved_type in ("ITERATION", "SPRINT"):
            value_input = {"iterationId": value}
        elif resolved_type == "NUMBER":
            value_input = {"number": float(value)}
        else:
            value_input = {"text": str(value)}

        try:
            self._gql(
                """
                mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: ProjectV2FieldValue!) {
                  updateProjectV2ItemFieldValue(input: {
                    projectId: $projectId itemId: $itemId fieldId: $fieldId value: $value
                  }) { clientMutationId }
                }
                """,
                {
                    "projectId": project_id,
                    "itemId": item_id,
                    "fieldId": field["id"],
                    "value": value_input,
                },
            )
            log_event("field_updated", field=field_name, value=value)
            return True
        except Exception as exc:
            log_event("field_update_failed", field=field_name, error=str(exc))
            return False

    def batch_update_fields(
        self,
        item_id: str,
        field_updates: dict[str, tuple[str, str]],   # {field_name: (value, field_type)}
    ) -> dict[str, bool]:
        """Update multiple fields in sequence.  Returns {field_name: success}."""
        results: dict[str, bool] = {}
        for field_name, (value, field_type) in field_updates.items():
            results[field_name] = self.update_field_value(item_id, field_name, value, field_type=field_type)
        return results

    # ------------------------------------------------------------------
    # High-level composite: ensure issue is on board with fields set
    # ------------------------------------------------------------------

    def ensure_issue_in_project_with_fields(
        self,
        issue_number: int,
        *,
        field_updates: dict[str, tuple[str, str]] | None = None,
    ) -> bool:
        """Idempotent: add issue to project and set all specified fields."""
        self._guard_write("ensure_issue_in_project_with_fields")
        try:
            issue_node_id = self.get_issue_node_id(issue_number)
            item_id = self.ensure_item_in_project(issue_node_id)
            if not item_id:
                log_event("project_sync_no_item_id", issue=issue_number)
                return False
            if field_updates:
                self.batch_update_fields(item_id, field_updates)
            log_event("issue_project_synced", issue=issue_number)
            return True
        except DryRunViolation:
            raise
        except Exception as exc:
            log_event("ensure_issue_in_project_failed", issue=issue_number, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Label operations via GraphQL
    # ------------------------------------------------------------------

    def add_labels_to_issue(self, issue_node_id: str, label_node_ids: list[str]) -> bool:
        """Add labels to an issue via GraphQL mutation."""
        self._guard_write("add_labels_to_issue")
        if not label_node_ids:
            return True
        try:
            self._gql(
                """
                mutation($issueId: ID!, $labelIds: [ID!]!) {
                  addLabelsToLabelable(input: {labelableId: $issueId, labelIds: $labelIds}) {
                    clientMutationId
                  }
                }
                """,
                {"issueId": issue_node_id, "labelIds": label_node_ids},
            )
            log_event("labels_added", issue_node_id=issue_node_id, count=len(label_node_ids))
            return True
        except Exception as exc:
            log_event("add_labels_failed", error=str(exc))
            return False

    def remove_labels_from_issue(self, issue_node_id: str, label_node_ids: list[str]) -> bool:
        """Remove labels from an issue via GraphQL mutation."""
        self._guard_write("remove_labels_from_issue")
        if not label_node_ids:
            return True
        try:
            self._gql(
                """
                mutation($issueId: ID!, $labelIds: [ID!]!) {
                  removeLabelsFromLabelable(input: {labelableId: $issueId, labelIds: $labelIds}) {
                    clientMutationId
                  }
                }
                """,
                {"issueId": issue_node_id, "labelIds": label_node_ids},
            )
            log_event("labels_removed", issue_node_id=issue_node_id, count=len(label_node_ids))
            return True
        except Exception as exc:
            log_event("remove_labels_failed", error=str(exc))
            return False

    def get_repo_label_node_ids(self, label_names: list[str]) -> dict[str, str]:
        """Return {label_name: node_id} for the requested labels. Unknown labels are omitted."""
        owner, _, repo = self.repo_name.partition("/")
        if not label_names:
            return {}
        try:
            data = self._gql(
                """
                query($owner: String!, $repo: String!, $first: Int!) {
                  repository(owner: $owner, name: $repo) {
                    labels(first: $first) {
                      nodes { id name }
                    }
                  }
                }
                """,
                {"owner": owner, "repo": repo, "first": 100},
            )
            all_labels = data.get("repository", {}).get("labels", {}).get("nodes", [])
            name_set = {n.lower() for n in label_names}
            return {lbl["name"]: lbl["id"] for lbl in all_labels if lbl["name"].lower() in name_set}
        except Exception as exc:
            log_event("label_node_id_fetch_failed", error=str(exc))
            return {}

    # ------------------------------------------------------------------
    # PR relationship metadata
    # ------------------------------------------------------------------

    def get_pr_linked_issues(self, pr_number: int) -> list[int]:
        """Return issue numbers linked to a PR via closing references."""
        owner, _, repo = self.repo_name.partition("/")
        try:
            data = self._gql(
                """
                query($owner: String!, $repo: String!, $number: Int!) {
                  repository(owner: $owner, name: $repo) {
                    pullRequest(number: $number) {
                      closingIssuesReferences(first: 20) {
                        nodes { number }
                      }
                    }
                  }
                }
                """,
                {"owner": owner, "repo": repo, "number": pr_number},
            )
            refs = (
                data.get("repository", {})
                .get("pullRequest", {})
                .get("closingIssuesReferences", {})
                .get("nodes", [])
            )
            return [r["number"] for r in refs]
        except Exception as exc:
            log_event("pr_linked_issues_failed", pr=pr_number, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Milestone bulk fetch
    # ------------------------------------------------------------------

    def get_open_milestones(self) -> list[dict[str, Any]]:
        owner, _, repo = self.repo_name.partition("/")
        try:
            data = self._gql(
                """
                query($owner: String!, $repo: String!) {
                  repository(owner: $owner, name: $repo) {
                    milestones(first: 30, states: [OPEN], orderBy: {field: DUE_DATE, direction: ASC}) {
                      nodes {
                        number title dueOn
                        openIssues: issues(states: [OPEN]) { totalCount }
                        closedIssues: issues(states: [CLOSED]) { totalCount }
                      }
                    }
                  }
                }
                """,
                {"owner": owner, "repo": repo},
            )
            return data.get("repository", {}).get("milestones", {}).get("nodes", [])
        except Exception as exc:
            log_event("milestones_fetch_failed", error=str(exc))
            self._rest_fallback_count += 1
            return []

