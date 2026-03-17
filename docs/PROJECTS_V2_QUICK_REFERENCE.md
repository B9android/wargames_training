# GitHub Projects v2 Integration — Quick Reference

## Field Mappings

| GitHub Label | Project Field | Value | Agent |
|---|---|---|---|
| `v1`, `v2`, `v3`, `v4` | `Version` | Single Select | triage_agent, project_syncer |
| `status: triaged` | `Status` | Triaged | project_syncer |
| `status: approved` | `Status` | Approved | project_syncer |
| `status: in-progress` | `Status` | In Progress | project_syncer, pr_linker |
| `status: blocked` | `Status` | Blocked | project_syncer |
| `status: complete` | `Status` | Done | project_syncer |
| `priority: critical` | `Story Points` | 8 | project_syncer |
| `priority: high` | `Story Points` | 5 | project_syncer |
| `priority: medium` | `Story Points` | 3 | project_syncer |
| `priority: low` | `Story Points` | 1 | project_syncer |
| **Milestone** (contains "v1", "v2", etc.) | `Version` | Extracted | project_syncer |
| **PR merge** | `Status` | In Progress | pr_linker |
| **PR merge** | `Git Commit` | Merge SHA | pr_linker |

## Agent Interaction Patterns

### issue_writer.py
```
GitHub Issue Created
    ↓
[trigger: issue_created event]
    ↓
issue_writer.py
    ↓
Query: active_sprint_id, version from milestone
    ↓
Add issue to project board
    ↓
Set: Version field, Sprint field
```

### epic_decomposer.py
```
Epic (<!-- version: v1 -->) Decomposition
    ↓
[trigger: on-demand or scheduled]
    ↓
epic_decomposer.py
    ↓
Create child issues (with markers)
    ↓
Extract parent Version from body
    ↓
Add child issues to project
    ↓
Set: Version (inherited), Sprint (active)
```

### triage_agent.py
```
Issue Labeled with "v1", "v2", etc.
    ↓
[trigger: issues.labeled event]
    ↓
triage_agent.py
    ↓
Apply triage labels
    ↓
Extract version from label
    ↓
Query issue node → project item IDs
    ↓
Update: Version field
```

### sprint_manager.py
```
Sprint Lifecycle: Start Sprint
    ↓
[trigger: manual dispatch or scheduled]
    ↓
sprint_manager.py
    ↓
Query active_sprint_id
    ↓
Query unassigned backlog issues (ordered by priority)
    ↓
Batch assign up to 50 issues
    ↓
Set: Sprint field via projects_v2.update_field_value()
```

### pr_linker.py
```
Pull Request Merged
    ↓
[trigger: pull_request.closed event]
    ↓
pr_linker.py
    ↓
Identify linked issue (from body/commit message)
    ↓
Add status label to issue
    ↓
Query issue node → project item IDs
    ↓
Update: Status field = "In Progress", Git Commit field = SHA
```

### project_syncer.py
```
Issue Labeled / Unlabeled / Milestoned
    ↓
[trigger: issues.labeled, issues.unlabeled, issues.milestoned, issues.demilestoned]
    ↓
project_syncer.py
    ↓
Match label → (field_name, field_value)
    ↓
Map milestone → version
    ↓
Query issue node → project item IDs
    ↓
Update: Matched field values
```

## Invocation Quick Start

### Dry-Run (No Board Changes)
```bash
export GITHUB_TOKEN=<token>
export REPO_NAME=owner/repo
export ISSUE_NUMBER=42
export DRY_RUN=true

python scripts/project_agent/project_syncer.py
# Output: [dry-run] ✅ Issue #42 project fields synced
```

### Live (Board Changes)
```bash
export GITHUB_TOKEN=<token>
export REPO_NAME=owner/repo
export ISSUE_NUMBER=42
export DRY_RUN=false

python scripts/project_agent/project_syncer.py
# Output: ✅ Issue #42 project fields synced
```

### Batch Sprint Assignment
```bash
export GITHUB_TOKEN=<token>
export REPO_NAME=owner/repo
export DRY_RUN=false

python scripts/project_agent/sprint_manager.py
# Assigns max 50 unassigned backlog issues to active sprint
```

### Validation
```bash
export GITHUB_TOKEN=<token>
export REPO_NAME=owner/repo

python scripts/project_agent/validate_projects_v2.py
# Checks prerequisites, token scopes, project board, core module
```

## Field ID Cache

**Location**: In-memory LRU cache in `ProjectsV2Client.get_field_ids()`

**TTL**: Per-process (cache invalidated on process exit)

**Hits**: 
- First call: Cache MISS → GraphQL query → 7-10 API calls
- Subsequent calls: Cache HIT → 0 API calls

**Optimization**: Cache reused across all agents in same process; no deduplication needed

## Retry Logic

**Pattern**: Exponential backoff on transient errors (429, 500, 502, etc.)

```python
# projects_v2.py _gql() method
retries = [1, 2, 4, 8]  # seconds
for attempt, delay in enumerate(retries):
    try:
        result = gh graphql <query>
        return result
    except TransientError:
        if attempt < len(retries) - 1:
            sleep(delay)
        else:
            raise
```

**Permanent Errors** (4xx except 429): Logged, skipped, no retry

**Transient Errors** (5xx, 429): Retry up to 4 times with exponential backoff

## Error Handling

| Error | Handling | Impact |
|-------|----------|--------|
| Missing project field | Graceful skip, log warning | Field not populated |
| Issue not on project | Add it automatically | None |
| GraphQL query fails (transient) | Retry 4x with backoff | Eventual success or timeout |
| GraphQL query fails (permanent) | Log error, skip | Field not updated |
| Token invalid | Fail immediately | Agent exits |
| GITHUB_TOKEN not set | Fail immediately | Agent exits |
| Repo not found | Fail at init | Agent exits |

## Marker-Based Deduplication

**Purpose**: Prevent duplicate decompositions/links on retry

**Format**: HTML comments in issue body
```html
<!-- decomposed-from:epic-#42 -->
<!-- child-issue:#45 -->
<!-- linked-to-pr:#678 -->
```

**Check Pattern**:
```python
from common import has_marker

if has_marker(issue_body, "decomposed-from:epic-#42"):
    print("Already decomposed, skipping")
else:
    # Perform decomposition, add marker
    from common import add_marker
    new_body = add_marker(issue_body, "decomposed-from:epic-#42")
```

## W&B Logging Events

All agents log structured events to W&B:

```python
log_event("issue_created", issue=42, version="v1", sprint="S1")
log_event("field_synced", issue=42, field="Status", value="In Progress")
log_event("project_sync_failed", issue=42, error="field_id_not_found")
log_event("sync_complete", issue=42, fields_updated=3)
```

**Query in W&B**:
```
state.events.issue_created | select(.number, .version, .sprint)
```

## Common Issues & Fixes

### "Field ID not found"
- **Cause**: Project board doesn't have custom field
- **Fix**: Add field to project manually in GitHub UI
- **Verify**: Run `validate_projects_v2.py`

### "Issue not found in project"
- **Cause**: Issue created before project board added
- **Fix**: Manually add issue to project via UI or re-run issue_writer
- **Auto-fix**: `ensure_issue_in_project_with_fields()` adds automatically

### "GraphQL query timeout"
- **Cause**: Rate limited or network issue
- **Fix**: Wait 60s and retry, or check rate limit: `gh api rate-limit`

### "Marker prevents re-decomposition"
- **Cause**: Epic already decomposed (marker present)
- **Fix**: Check issue body for marker comment; remove if false positive
- **Reset**: Edit body to remove marker or create new epic

## Performance Targets

| Operation | Target | Notes |
|---|---|---|
| Field ID cache hit | < 10ms | In-memory lookup |
| Add issue to project | 500-2000ms | 1-2 GraphQL queries |
| Batch sprint assignment (50 issues) | 10-30s | 1 query + 50 mutations |
| Sync 1 label to field | 500-1000ms | Query + 1 mutation |
| Epic decomposition (10 children) | 5-10s | Query + 10 add + 20 mutations |

## Security

- **Token Scope**: `repo` (read/write issues), `project` (read/write)
- **Permissions**: Issue write access required; project board admin recommended
- **Secrets**: GITHUB_TOKEN stored in GitHub Actions secrets, never logged
- **Audit**: All mutations logged to W&B with issue number, field, old/new value

## Troubleshooting Steps

1. **Check token**: `gh auth status`
2. **Check repo**: `gh repo view <repo>`
3. **Check projects**: `gh project list --repo <repo>`
4. **Run validator**: `python scripts/project_agent/validate_projects_v2.py`
5. **Dry-run agent**: `DRY_RUN=true python scripts/project_agent/<agent>.py`
6. **Check W&B logs**: Filter by issue number and agent name
7. **Review GitHub UI**: Verify issue/PR state and labels
8. **Check rate limit**: `gh api rate-limit --format=json | jq '.resources.graphql'`

## Documentation

- **Full Testing Guide**: [PROJECTS_V2_TESTING_GUIDE.md](PROJECTS_V2_TESTING_GUIDE.md)
- **Projects v2 Client API**: [scripts/project_agent/projects_v2.py](../scripts/project_agent/projects_v2.py)
- **Common Utilities**: [scripts/project_agent/common.py](../scripts/project_agent/common.py)
- **Orchestration Config**: [configs/orchestration.yaml](../configs/orchestration.yaml)
