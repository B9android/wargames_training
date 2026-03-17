#!/usr/bin/env python3
"""
setup_project_board.py
One-time script that creates the full GitHub Projects v2 board via GraphQL.

Creates:
  - Project: "Wargames Training — Master Board"
  - Links it to the wargames_training repo
  - Renames Status column options to sprint-board names
  - Adds custom fields: Sprint, Story Points, Experiment Status,
    W&B Run, Git Commit, Version
  - Creates 6 views: Current Sprint, Full Backlog, Roadmap,
    Experiments, Blockers, Agent Activity

Usage:
    python scripts/project_agent/setup_project_board.py
Requires: gh CLI authenticated (gh auth login)
"""
import subprocess
import json
import sys

REPO_OWNER = "B9android"
REPO_NAME = "wargames_training"
PROJECT_TITLE = "Wargames Training \u2014 Master Board"


# ---------------------------------------------------------------------------
# GraphQL helper
# ---------------------------------------------------------------------------

def gql(query: str, variables: dict | None = None) -> dict:
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
        print(f"gh CLI error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(result.stdout)
    if "errors" in data:
        for err in data["errors"]:
            print(f"  GraphQL error: {err['message']}", file=sys.stderr)
        # Non-fatal on some mutations — return data anyway
        return data.get("data") or {}
    return data["data"]


# ---------------------------------------------------------------------------
# Step 1: Resolve node IDs
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Step 2: Create the project
# ---------------------------------------------------------------------------

print(f'\n2/6  Creating project "{PROJECT_TITLE}" (or re-using existing)...')
# Check if a project with this title already exists
existing_projects = gql(
    """
    query($login: String!) {
        user(login: $login) {
            projectsV2(first: 20) {
                nodes { id number url title }
            }
        }
    }
    """,
    {"login": REPO_OWNER},
)
existing = [p for p in existing_projects["user"]["projectsV2"]["nodes"]
            if "Master Board" in p["title"] and "Wargames Training" in p["title"]]

if existing:
    # Use the lowest-numbered (oldest) matching project
    project = sorted(existing, key=lambda p: p["number"])[0]
    project_id = project["id"]
    print(f"     Already exists — reusing #{project['number']}: {project['url']}")
    # Delete any newer duplicates
    for dup in sorted(existing, key=lambda p: p["number"])[1:]:
        gql("mutation($id: ID!) { deleteProjectV2(input:{projectId:$id}) { projectV2 { number } } }",
            {"id": dup["id"]})
        print(f"     Deleted duplicate #{dup['number']}")
else:
    proj_data = gql(
        """
        mutation($ownerId: ID!, $title: String!) {
            createProjectV2(input: {ownerId: $ownerId, title: $title}) {
                projectV2 { id number url title }
            }
        }
        """,
        {"ownerId": owner_id, "title": PROJECT_TITLE},
    )
    project = proj_data["createProjectV2"]["projectV2"]
    project_id = project["id"]
    print(f"     Created: {project['url']}")


# ---------------------------------------------------------------------------
# Step 3: Link project to repo
# ---------------------------------------------------------------------------

print("\n3/6  Linking project to repository...")
gql(
    """
    mutation($projectId: ID!, $repositoryId: ID!) {
        linkProjectV2ToRepository(
            input: {projectId: $projectId, repositoryId: $repositoryId}
        ) { repository { name } }
    }
    """,
    {"projectId": project_id, "repositoryId": repo_id},
)
print("     Linked.")


# ---------------------------------------------------------------------------
# Step 4: Fetch default fields, then rename Status options
# ---------------------------------------------------------------------------

print("\n4/6  Configuring Status field columns...")
fields_data = gql(
    """
    query($id: ID!) {
        node(id: $id) {
            ... on ProjectV2 {
                fields(first: 20) {
                    nodes {
                        ... on ProjectV2Field          { id name dataType }
                        ... on ProjectV2IterationField { id name dataType }
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
existing_fields = fields_data["node"]["fields"]["nodes"]
print(f"     Default fields: {[f.get('name') for f in existing_fields]}")

status_field = next((f for f in existing_fields if f.get("name") == "Status"), None)
if status_field:
    print(f"     Status field found with options: {[o['name'] for o in status_field.get('options', [])]}")
    print("     Note: Rename Status columns to 📥 Backlog / 🔍 Ready / 🔨 In Progress /")
    print("           👀 In Review / ✅ Done / 🚫 Blocked via the board UI (Settings → Fields).")
else:
    print("     Warning: Status field not found.")


# ---------------------------------------------------------------------------
# Step 5: Create custom fields
# ---------------------------------------------------------------------------

print("\n5/6  Creating custom fields...")

existing_field_names = {f.get("name") for f in existing_fields}

CUSTOM_FIELDS = [
    {
        "name": "Story Points",
        "dataType": "NUMBER",
    },
    {
        "name": "W&B Run",
        "dataType": "TEXT",
    },
    {
        "name": "Git Commit",
        "dataType": "TEXT",
    },
    {
        "name": "Experiment Status",
        "dataType": "SINGLE_SELECT",
        "options": [
            {"name": "⏳ Running",  "color": "YELLOW", "description": ""},
            {"name": "✅ Success",  "color": "GREEN",  "description": ""},
            {"name": "❌ Failed",   "color": "RED",    "description": ""},
            {"name": "🤔 Mixed",    "color": "ORANGE", "description": ""},
            {"name": "💥 Crashed",  "color": "RED",    "description": ""},
        ],
    },
    {
        "name": "Version",
        "dataType": "SINGLE_SELECT",
        "options": [
            {"name": "v1",      "color": "BLUE",   "description": "Foundation"},
            {"name": "v2",      "color": "GREEN",  "description": "Multi-agent"},
            {"name": "v3",      "color": "YELLOW", "description": "Hierarchy"},
            {"name": "v4",      "color": "ORANGE", "description": "League"},
            {"name": "vFuture", "color": "GRAY",   "description": "Not scheduled"},
        ],
    },
    {   # Sprint: iteration field — duration configured in the UI after creation
        "name": "Sprint",
        "dataType": "ITERATION",
    },
]

for field in CUSTOM_FIELDS:
    dtype = field["dataType"]
    name = field["name"]

    if name in existing_field_names:
        print(f"     ~ {name} already exists, skipping.")
        continue

    if dtype == "SINGLE_SELECT":
        result = gql(
            """
            mutation($projectId: ID!, $name: String!, $options: [ProjectV2SingleSelectFieldOptionInput!]!) {
                createProjectV2Field(input: {
                    projectId: $projectId
                    dataType: SINGLE_SELECT
                    name: $name
                    singleSelectOptions: $options
                }) {
                    projectV2Field {
                        ... on ProjectV2SingleSelectField { id name }
                    }
                }
            }
            """,
            {"projectId": project_id, "name": name, "options": field["options"]},
        )
    elif dtype == "NUMBER":
        result = gql(
            """
            mutation($projectId: ID!, $name: String!) {
                createProjectV2Field(input: {
                    projectId: $projectId
                    dataType: NUMBER
                    name: $name
                }) {
                    projectV2Field { ... on ProjectV2Field { id name } }
                }
            }
            """,
            {"projectId": project_id, "name": name},
        )
    elif dtype == "TEXT":
        result = gql(
            """
            mutation($projectId: ID!, $name: String!) {
                createProjectV2Field(input: {
                    projectId: $projectId
                    dataType: TEXT
                    name: $name
                }) {
                    projectV2Field { ... on ProjectV2Field { id name } }
                }
            }
            """,
            {"projectId": project_id, "name": name},
        )
    elif dtype == "ITERATION":
        result = gql(
            """
            mutation($projectId: ID!, $name: String!) {
                createProjectV2Field(input: {
                    projectId: $projectId
                    dataType: ITERATION
                    name: $name
                }) {
                    projectV2Field { ... on ProjectV2IterationField { id name } }
                }
            }
            """,
            {"projectId": project_id, "name": name},
        )
    print(f"     + {name} ({dtype})")


# ---------------------------------------------------------------------------
# Step 6: Create views
# ---------------------------------------------------------------------------

print("\n6/6  Creating views...")

VIEWS = [
    ("🗓️ Current Sprint",  "BOARD_LAYOUT"),
    ("📋 Full Backlog",     "TABLE_LAYOUT"),
    ("🧪 Experiments",     "TABLE_LAYOUT"),
    ("🚨 Blockers",        "BOARD_LAYOUT"),
    ("🤖 Agent Activity",  "TABLE_LAYOUT"),
]

# Try ROADMAP_LAYOUT first; fall back to TABLE_LAYOUT if unsupported
def try_create_view(project_id, name, layout):
    result = subprocess.run(
        ["gh", "api", "graphql", "--input", "-"],
        input=json.dumps({
            "query": """
                mutation($projectId: ID!, $name: String!, $layout: ProjectV2ViewLayout!) {
                    createProjectV2View(input: {
                        projectId: $projectId
                        name: $name
                        layout: $layout
                    }) {
                        projectV2View { id name layout }
                    }
                }
            """.strip(),
            "variables": {"projectId": project_id, "name": name, "layout": layout},
        }),
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout) if result.stdout else {}
    if result.returncode != 0 or "errors" in data:
        return None
    return data.get("data", {}).get("createProjectV2View", {}).get("projectV2View")


# Fetch existing views to avoid duplicates
existing_views_data = gql(
    """
    query($id: ID!) {
        node(id: $id) {
            ... on ProjectV2 { views(first: 20) { nodes { id name } } }
        }
    }
    """,
    {"id": project_id},
)
existing_view_names = {v["name"] for v in existing_views_data["node"]["views"]["nodes"]}
print(f"     Existing views: {existing_view_names}")
remaining_views = [v for v in VIEWS if v[0] not in existing_view_names]

for name, layout in remaining_views:
    view = try_create_view(project_id, name, layout)
    if view:
        print(f"     + {view['name']} ({view['layout']})")
    else:
        # Roadmap may not be available; fall back silently
        view = try_create_view(project_id, name, "TABLE_LAYOUT")
        if view:
            print(f"     + {view['name']} (TABLE_LAYOUT — roadmap not available on this plan)")
        else:
            print(f"     ! Could not create view: {name}")

# Roadmap view — try separately since it may require a specific plan
roadmap = try_create_view(project_id, "🗺️ Roadmap", "ROADMAP_LAYOUT")
if roadmap:
    print(f"     + {roadmap['name']} (ROADMAP_LAYOUT)")
else:
    roadmap = try_create_view(project_id, "🗺️ Roadmap", "TABLE_LAYOUT")
    print(f"     + {'🗺️ Roadmap'} (TABLE_LAYOUT — roadmap layout not available)")

print(f"\n✅  Done! Open your board: {project['url']}")
print("\nRemaining manual steps (UI only):")
print("  1. Open the board and assign issues to the project")
print("  2. Configure Sprint field: set start date and 2-week cadence")
print("  3. Set 'Current Sprint' view filter: filter by Sprint = current iteration")
print("  4. Set 'Experiments' view filter: filter by label = 'type: experiment'")
print("  5. Set 'Blockers' view filter: filter by label = 'status: blocked'")
print("  6. Set 'Agent Activity' view filter: filter by label = 'status: agent-created'")
