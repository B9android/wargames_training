"""GitHub Projects v2 GraphQL abstraction layer for autonomous field management."""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from common import AgentError, log_event, retry


class ProjectsV2Client:
    """GraphQL client for GitHub Projects v2 operations with retry and caching."""

    def __init__(self, github_token: str, repo_name: str, project_title: str = "Wargames Training — Master Board"):
        """Initialize client with GitHub token and repository context.
        
        Args:
            github_token: GitHub personal access token
            repo_name: Repository name (owner/repo format)
            project_title: Project board title (defaults to main board)
        """
        self.github_token = github_token
        self.repo_name = repo_name
        self.project_title = project_title
        self._project_id_cache: str | None = None
        self._field_ids_cache: dict[str, str] | None = None
        self._active_sprint_cache: str | None = None

    def _gql(
        self,
        query: str,
        variables: dict | None = None,
        *,
        allow_errors: bool = False,
        retries: int = 3,
    ) -> dict:
        """Execute GraphQL query with retry logic.
        
        Args:
            query: GraphQL query string
            variables: Query variables dict
            allow_errors: If True, don't raise on GraphQL errors
            retries: Number of retries for transient failures
            
        Returns:
            Data portion of GraphQL response
        """
        payload = {"query": query.strip()}
        if variables:
            payload["variables"] = variables

        def execute():
            result = subprocess.run(
                ["gh", "api", "graphql", "--input", "-"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"gh CLI error: {result.stderr}")
            return json.loads(result.stdout)

        response = retry(execute, retries=retries)
        
        if "errors" in response and not allow_errors:
            messages = "\n".join(err.get("message", "unknown") for err in response["errors"])
            log_event("graphql_error", query_type=query.split()[0], error=messages)
            raise RuntimeError(f"GraphQL error(s): {messages}")

        return response.get("data") or {}

    def get_project_id(self) -> str:
        """Get project ID, with caching."""
        if self._project_id_cache:
            return self._project_id_cache

        owner, _ = self.repo_name.split("/")
        data = self._gql(
            """
            query($login: String!) {
                user(login: $login) {
                    projectsV2(first: 20) {
                        nodes { id number url title }
                    }
                }
            }
            """,
            {"login": owner},
        )

        projects = data["user"]["projectsV2"]["nodes"]
        project = next((p for p in projects if p["title"] == self.project_title), None)
        if not project:
            raise AgentError(f"Project '{self.project_title}' not found in {owner}'s projects")

        self._project_id_cache = project["id"]
        log_event("project_id_resolved", project=self.project_title, id=project["id"])
        return self._project_id_cache

    def get_field_ids(self, *, invalidate: bool = False) -> dict[str, str]:
        """Get mapping of field names to field IDs, with caching.
        
        Args:
            invalidate: Force refresh cache
            
        Returns:
            Dict mapping field name → field ID
        """
        if self._field_ids_cache and not invalidate:
            return self._field_ids_cache

        project_id = self.get_project_id()
        data = self._gql(
            """
            query($id: ID!) {
                node(id: $id) {
                    ... on ProjectV2 {
                        fields(first: 30) {
                            nodes {
                                ... on ProjectV2Field { id name dataType }
                                ... on ProjectV2IterationField { id name dataType }
                                ... on ProjectV2SingleSelectField { id name dataType }
                            }
                        }
                    }
                }
            }
            """,
            {"id": project_id},
        )

        fields = data["node"]["fields"]["nodes"]
        self._field_ids_cache = {field["name"]: field["id"] for field in fields}
        log_event("field_ids_cached", count=len(self._field_ids_cache), fields=list(self._field_ids_cache.keys()))
        return self._field_ids_cache

    def add_issue_to_project(self, issue_id: str) -> bool:
        """Add issue to project board.
        
        Args:
            issue_id: GitHub issue node ID (graphql ID, not number)
            
        Returns:
            True if successful
        """
        project_id = self.get_project_id()
        try:
            data = self._gql(
                """
                mutation($projectId: ID!, $issueId: ID!) {
                    addProjectV2ItemById(input: {
                        projectId: $projectId
                        contentId: $issueId
                    }) {
                        item { id }
                    }
                }
                """,
                {"projectId": project_id, "issueId": issue_id},
            )
            item_id = data.get("addProjectV2ItemById", {}).get("item", {}).get("id")
            if item_id:
                log_event("issue_added_to_project", issue_id=issue_id, item_id=item_id)
                return True
        except Exception as exc:
            log_event("add_issue_failed", issue_id=issue_id, error=str(exc))
        return False

    def update_field_value(
        self,
        item_id: str,
        field_name: str,
        field_value: str | int | None,
        field_type: str = "SINGLE_SELECT",
    ) -> bool:
        """Update custom field value on a project item.
        
        Args:
            item_id: Project item ID
            field_name: Custom field name (e.g., "Version", "Status")
            field_value: Field value (option ID for single-select, text for text, number for number)
            field_type: Field type (SINGLE_SELECT, TEXT, NUMBER, ITERATION)
            
        Returns:
            True if successful
        """
        if not field_value:
            log_event("field_update_skipped", field=field_name, reason="no value")
            return True

        project_id = self.get_project_id()
        field_ids = self.get_field_ids()
        field_id = field_ids.get(field_name)

        if not field_id:
            log_event("field_not_found", field=field_name, available=list(field_ids.keys()))
            return False

        # Build value input based on field type
        if field_type == "SINGLE_SELECT":
            value_input = {"singleSelectOptionId": field_value}
        elif field_type == "TEXT":
            value_input = {"text": str(field_value)}
        elif field_type == "NUMBER":
            value_input = {"number": int(field_value)}
        elif field_type == "ITERATION":
            value_input = {"iterationId": field_value}
        else:
            log_event("unknown_field_type", field=field_name, type=field_type)
            return False

        try:
            self._gql(
                """
                mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: ProjectV2FieldValue!) {
                    updateProjectV2ItemFieldValue(input: {
                        projectId: $projectId
                        itemId: $itemId
                        fieldId: $fieldId
                        value: $value
                    }) {
                        clientMutationId
                    }
                }
                """,
                {
                    "projectId": project_id,
                    "itemId": item_id,
                    "fieldId": field_id,
                    "value": value_input,
                },
                retries=2,
            )
            log_event("field_updated", field=field_name, value=field_value)
            return True
        except Exception as exc:
            log_event("field_update_failed", field=field_name, error=str(exc))
            return False

    def get_active_sprint_id(self) -> str | None:
        """Get the current/active sprint iteration ID.
        
        Returns:
            Iteration ID if found, None otherwise
        """
        if self._active_sprint_cache:
            return self._active_sprint_cache

        project_id = self.get_project_id()
        try:
            data = self._gql(
                """
                query($id: ID!) {
                    node(id: $id) {
                        ... on ProjectV2 {
                            fields(first: 20) {
                                nodes {
                                    ... on ProjectV2IterationField {
                                        id
                                        name
                                        configuration {
                                            iterations(last: 1) {
                                                nodes { id title startDate duration }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                """,
                {"id": project_id},
            )
            
            fields = data.get("node", {}).get("fields", {}).get("nodes", [])
            for field in fields:
                if field.get("name") == "Sprint":
                    iterations = field.get("configuration", {}).get("iterations", {}).get("nodes", [])
                    if iterations:
                        sprint_id = iterations[0].get("id")
                        self._active_sprint_cache = sprint_id
                        log_event("active_sprint_found", sprint_id=sprint_id, title=iterations[0].get("title"))
                        return sprint_id
        except Exception as exc:
            log_event("active_sprint_query_failed", error=str(exc))

        return None

    def get_issue_node_id(self, repo_name: str, issue_number: int) -> str | None:
        """Convert issue number to GraphQL node ID.
        
        Args:
            repo_name: Repository (owner/repo)
            issue_number: Issue number
            
        Returns:
            GraphQL node ID or None if not found
        """
        owner, repo = repo_name.split("/")
        try:
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
            if node_id:
                log_event("issue_node_id_resolved", issue=issue_number, node_id=node_id)
            return node_id
        except Exception as exc:
            log_event("get_issue_node_id_failed", issue=issue_number, error=str(exc))
            return None

    def get_project_item_id(self, issue_node_id: str) -> str | None:
        """Get project item ID for a given issue node ID.
        
        Args:
            issue_node_id: GraphQL issue node ID
            
        Returns:
            Project item ID or None
        """
        project_id = self.get_project_id()
        try:
            data = self._gql(
                """
                query($projectId: ID!, $issueId: ID!) {
                    node(id: $projectId) {
                        ... on ProjectV2 {
                            items(first: 1, filter: { contentId: $issueId }) {
                                nodes { id }
                            }
                        }
                    }
                }
                """,
                {"projectId": project_id, "issueId": issue_node_id},
                allow_errors=True,
            )
            items = data.get("node", {}).get("items", {}).get("nodes", [])
            if items:
                item_id = items[0].get("id")
                log_event("project_item_found", issue_node_id=issue_node_id, item_id=item_id)
                return item_id
        except Exception as exc:
            log_event("get_project_item_id_failed", issue_node_id=issue_node_id, error=str(exc))
        
        return None

    def ensure_issue_in_project_with_fields(
        self,
        issue_number: int,
        field_updates: dict[str, tuple[str, str]] | None = None,
    ) -> bool:
        """Add issue to project and update custom fields in one operation.
        
        Args:
            issue_number: Issue number to add
            field_updates: Dict of field_name → (value, type) pairs
                          E.g., {"Version": ("v1", "SINGLE_SELECT"), "Story Points": ("5", "NUMBER")}
                          
        Returns:
            True if all operations succeeded
        """
        # Get issue node ID
        issue_node_id = self.get_issue_node_id(self.repo_name, issue_number)
        if not issue_node_id:
            log_event("ensure_issue_failed", issue=issue_number, reason="node_id_not_found")
            return False

        # Add to project
        if not self.add_issue_to_project(issue_node_id):
            log_event("ensure_issue_failed", issue=issue_number, reason="add_failed")
            return False

        # Get item ID and update fields
        item_id = self.get_project_item_id(issue_node_id)
        if not item_id:
            log_event("ensure_issue_failed", issue=issue_number, reason="item_id_not_found")
            return False

        # Update custom fields
        if field_updates:
            for field_name, (value, field_type) in field_updates.items():
                if not self.update_field_value(item_id, field_name, value, field_type):
                    log_event("ensure_issue_partial_fail", issue=issue_number, field=field_name)
                    # Continue with other fields even if one fails

        log_event("ensure_issue_complete", issue=issue_number, fields=len(field_updates or {}))
        return True
