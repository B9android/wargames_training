#!/usr/bin/env python3
"""Idempotent GitHub Projects v2 board bootstrap for wargames_training.

Safe defaults:
- Reuses an existing board when found.
- Does not delete duplicates unless explicitly requested.
- Supports dry-run mode for audit-only execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

REPO_OWNER = "B9android"
REPO_NAME = "wargames_training"
PROJECT_TITLE = "Wargames Training — Master Board"

CUSTOM_FIELDS = [
    {"name": "Story Points", "dataType": "NUMBER"},
    {"name": "W&B Run", "dataType": "TEXT"},
    {"name": "Git Commit", "dataType": "TEXT"},
    {
        "name": "Experiment Status",
        "dataType": "SINGLE_SELECT",
        "options": [
            {"name": "⏳ Running", "color": "YELLOW", "description": ""},
            {"name": "✅ Success", "color": "GREEN", "description": ""},
            {"name": "❌ Failed", "color": "RED", "description": ""},
            {"name": "🤔 Mixed", "color": "ORANGE", "description": ""},
            {"name": "💥 Crashed", "color": "RED", "description": ""},
        ],
    },
    {
        "name": "Version",
        "dataType": "SINGLE_SELECT",
        "options": [
            {"name": "v1", "color": "BLUE", "description": "Foundation"},
            {"name": "v2", "color": "GREEN", "description": "Multi-agent"},
            {"name": "v3", "color": "YELLOW", "description": "Hierarchy"},
            {"name": "v4", "color": "ORANGE", "description": "League"},
            {"name": "vFuture", "color": "GRAY", "description": "Not scheduled"},
        ],
    },
    {"name": "Sprint", "dataType": "ITERATION"},
]

VIEWS = [
    ("🗓️ Current Sprint", "BOARD_LAYOUT"),
    ("📋 Full Backlog", "TABLE_LAYOUT"),
    ("🧪 Experiments", "TABLE_LAYOUT"),
    ("🚨 Blockers", "BOARD_LAYOUT"),
    ("🤖 Agent Activity", "TABLE_LAYOUT"),
    ("🗺️ Roadmap", "ROADMAP_LAYOUT"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up or sync the project board")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without mutating GitHub")
    parser.add_argument(
        "--delete-duplicates",
        action="store_true",
        help="Delete duplicate matching projects (off by default for safety)",
    )
    return parser.parse_args()


def gql(query: str, variables: dict | None = None, *, allow_errors: bool = False) -> dict:
    payload = {"query": query.strip()}
    if variables:
        payload["variables"] = variables

    result = subprocess.run(
        ["gh", "api", "graphql", "--input", "-"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh CLI error:\n{result.stderr}")

    data = json.loads(result.stdout)
    if "errors" in data and not allow_errors:
        messages = "\n".join(err.get("message", "unknown GraphQL error") for err in data["errors"])
        raise RuntimeError(f"GraphQL error(s):\n{messages}")

    return data.get("data") or {}


def maybe_run(dry_run: bool, description: str, fn):
    if dry_run:
        print(f"     [dry-run] {description}")
        return None
    return fn()


def create_field(project_id: str, field: dict, dry_run: bool) -> None:
    dtype = field["dataType"]
    name = field["name"]
    if dtype == "SINGLE_SELECT":
        query = """
        mutation($projectId: ID!, $name: String!, $options: [ProjectV2SingleSelectFieldOptionInput!]!) {
            createProjectV2Field(input: {
                projectId: $projectId
                dataType: SINGLE_SELECT
                name: $name
                singleSelectOptions: $options
            }) {
                projectV2Field { ... on ProjectV2SingleSelectField { id name } }
            }
        }
        """
        vars_ = {"projectId": project_id, "name": name, "options": field["options"]}
    elif dtype == "NUMBER":
        query = """
        mutation($projectId: ID!, $name: String!) {
            createProjectV2Field(input: { projectId: $projectId, dataType: NUMBER, name: $name }) {
                projectV2Field { ... on ProjectV2Field { id name } }
            }
        }
        """
        vars_ = {"projectId": project_id, "name": name}
    elif dtype == "TEXT":
        query = """
        mutation($projectId: ID!, $name: String!) {
            createProjectV2Field(input: { projectId: $projectId, dataType: TEXT, name: $name }) {
                projectV2Field { ... on ProjectV2Field { id name } }
            }
        }
        """
        vars_ = {"projectId": project_id, "name": name}
    elif dtype == "ITERATION":
        query = """
        mutation($projectId: ID!, $name: String!) {
            createProjectV2Field(input: { projectId: $projectId, dataType: ITERATION, name: $name }) {
                projectV2Field { ... on ProjectV2IterationField { id name } }
            }
        }
        """
        vars_ = {"projectId": project_id, "name": name}
    else:
        print(f"     ! Unknown field type for {name}: {dtype}")
        return

    maybe_run(dry_run, f"create field {name} ({dtype})", lambda: gql(query, vars_))
    print(f"     + {name} ({dtype})")


def create_view(project_id: str, name: str, layout: str, dry_run: bool) -> bool:
    query = """
    mutation($projectId: ID!, $name: String!, $layout: ProjectV2ViewLayout!) {
        createProjectV2View(input: { projectId: $projectId, name: $name, layout: $layout }) {
            projectV2View { id name layout }
        }
    }
    """
    if dry_run:
        print(f"     [dry-run] create view {name} ({layout})")
        return True
    try:
        gql(query, {"projectId": project_id, "name": name, "layout": layout})
        return True
    except RuntimeError:
        return False


def resolve_ids() -> tuple[str, str]:
    print("1/6  Resolving owner and repo node IDs...")
    ids = gql(
        """
        query($owner: String!, $name: String!) {
            user(login: $owner) { id }
            repository(owner: $owner, name: $name) { id }
        }
        """,
        {"owner": REPO_OWNER, "name": REPO_NAME},
    )
    owner_id = ids["user"]["id"]
    repo_id = ids["repository"]["id"]
    print(f"     owner={owner_id}  repo={repo_id}")
    return owner_id, repo_id


def ensure_project(owner_id: str, dry_run: bool, delete_duplicates: bool) -> dict:
    print(f'\n2/6  Ensuring project "{PROJECT_TITLE}" exists...')
    existing_projects = gql(
        """
        query($login: String!) {
            user(login: $login) {
                projectsV2(first: 30) {
                    nodes { id number url title }
                }
            }
        }
        """,
        {"login": REPO_OWNER},
    )
    existing = [
        p
        for p in existing_projects["user"]["projectsV2"]["nodes"]
        if p.get("title") == PROJECT_TITLE
    ]

    if existing:
        project = sorted(existing, key=lambda p: p["number"])[0]
        print(f"     Reusing #{project['number']}: {project['url']}")
        duplicates = sorted(existing, key=lambda p: p["number"])[1:]
        if duplicates:
            if delete_duplicates:
                for dup in duplicates:
                    maybe_run(
                        dry_run,
                        f"delete duplicate project #{dup['number']}",
                        lambda dup_id=dup["id"]: gql(
                            "mutation($id: ID!) { deleteProjectV2(input:{projectId:$id}) { projectV2 { number } } }",
                            {"id": dup_id},
                        ),
                    )
                    print(f"     - deleted duplicate #{dup['number']}")
            else:
                print("     ! Duplicates detected but preserved (pass --delete-duplicates to remove).")
        return project

    created = maybe_run(
        dry_run,
        f"create project {PROJECT_TITLE}",
        lambda: gql(
            """
            mutation($ownerId: ID!, $title: String!) {
                createProjectV2(input: {ownerId: $ownerId, title: $title}) {
                    projectV2 { id number url title }
                }
            }
            """,
            {"ownerId": owner_id, "title": PROJECT_TITLE},
        ),
    )
    if dry_run:
        return {"id": "DRY_RUN_PROJECT_ID", "number": "dry-run", "url": "(dry-run)", "title": PROJECT_TITLE}
    project = created["createProjectV2"]["projectV2"]
    print(f"     Created: {project['url']}")
    return project


def ensure_link(project_id: str, repo_id: str, dry_run: bool) -> None:
    print("\n3/6  Linking project to repository...")
    try:
        maybe_run(
            dry_run,
            f"link project {project_id} to repository {repo_id}",
            lambda: gql(
                """
                mutation($projectId: ID!, $repositoryId: ID!) {
                    linkProjectV2ToRepository(input: {projectId: $projectId, repositoryId: $repositoryId}) {
                        repository { name }
                    }
                }
                """,
                {"projectId": project_id, "repositoryId": repo_id},
            ),
        )
        print("     Linked.")
    except RuntimeError as exc:
        print(f"     ! Link step returned non-fatal error: {exc}")


def get_existing_fields(project_id: str, dry_run: bool) -> list[dict]:
    print("\n4/6  Checking Status field...")
    if dry_run:
        print("     [dry-run] skipping live field query")
        return []

    fields_data = gql(
        """
        query($id: ID!) {
            node(id: $id) {
                ... on ProjectV2 {
                    fields(first: 30) {
                        nodes {
                            ... on ProjectV2Field { id name dataType }
                            ... on ProjectV2IterationField { id name dataType }
                            ... on ProjectV2SingleSelectField { id name dataType options { id name } }
                        }
                    }
                }
            }
        }
        """,
        {"id": project_id},
    )
    existing_fields = fields_data["node"]["fields"]["nodes"]
    status_field = next((f for f in existing_fields if f.get("name") == "Status"), None)
    if status_field:
        print(f"     Status options: {[o['name'] for o in status_field.get('options', [])]}")
    else:
        print("     ! Status field not found.")
    return existing_fields


def ensure_custom_fields(project_id: str, existing_fields: list[dict], dry_run: bool) -> None:
    print("\n5/6  Ensuring custom fields exist...")
    existing_field_names = {f.get("name") for f in existing_fields}
    for field in CUSTOM_FIELDS:
        if field["name"] in existing_field_names:
            print(f"     ~ {field['name']} already exists")
            continue
        create_field(project_id, field, dry_run)


def ensure_views(project_id: str, dry_run: bool) -> None:
    print("\n6/6  Ensuring views exist...")
    if dry_run:
        existing_view_names = set()
        print("     [dry-run] skipping live views query")
    else:
        existing_views_data = gql(
            """
            query($id: ID!) {
                node(id: $id) {
                    ... on ProjectV2 { views(first: 30) { nodes { id name } } }
                }
            }
            """,
            {"id": project_id},
        )
        existing_view_names = {v["name"] for v in existing_views_data["node"]["views"]["nodes"]}
        print(f"     Existing views: {sorted(existing_view_names)}")

    for name, layout in VIEWS:
        if name in existing_view_names:
            print(f"     ~ {name} already exists")
            continue
        created = create_view(project_id, name, layout, dry_run)
        if created:
            print(f"     + {name} ({layout})")
            continue
        if layout == "ROADMAP_LAYOUT" and create_view(project_id, name, "TABLE_LAYOUT", dry_run):
            print(f"     + {name} (TABLE_LAYOUT fallback)")
            continue
        print(f"     ! Could not create view: {name}")


def main() -> int:
    args = parse_args()
    dry_run = args.dry_run

    owner_id, repo_id = resolve_ids()
    project = ensure_project(owner_id, dry_run, args.delete_duplicates)
    project_id = project["id"]

    ensure_link(project_id, repo_id, dry_run)
    existing_fields = get_existing_fields(project_id, dry_run)
    ensure_custom_fields(project_id, existing_fields, dry_run)
    ensure_views(project_id, dry_run)

    print(f"\n✅ Done! Board URL: {project['url']}")
    print("Manual follow-up:")
    print("  1. Configure Sprint cadence in project settings")
    print("  2. Tune board/table filters for Experiments, Blockers, and Agent Activity")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
