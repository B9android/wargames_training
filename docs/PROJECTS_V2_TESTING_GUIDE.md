# GitHub Projects v2 Integration — Testing & Validation Guide

**Status**: v1 Integration Complete | Ready for Staged Testing  
**Date**: March 17, 2026  
**Scope**: Projects v2 abstraction layer (projects_v2.py) + 6 agent integrations (issue_writer, epic_decomposer, triage_agent, sprint_manager, pr_linker, project_syncer)

---

## Table of Contents

1. [Pre-Flight Checklist](#pre-flight-checklist)
2. [Testing Environments](#testing-environments)
3. [Core Abstraction Testing](#core-abstraction-testing)
4. [Agent Integration Testing](#agent-integration-testing)
5. [End-to-End Scenarios](#end-to-end-scenarios)
6. [Error Handling & Resilience](#error-handling--resilience)
7. [Performance & Rate Limits](#performance--rate-limits)
8. [Troubleshooting](#troubleshooting)

---

## Pre-Flight Checklist

Before running any tests, verify:

- [ ] GitHub Projects v2 board created for test repo with required fields:
  - `Version` (Single Select)
  - `Sprint` (Iteration type)
  - `Status` (Single Select: Triaged, Approved, In Progress, Blocked, Done)
  - `Story Points` (Number)
  - `Experiment Status` (Single Select)
  - `Git Commit` (Text)
  - `W&B Run` (Text)
- [ ] GITHUB_TOKEN set and has `repo`, `read:org`, `project` scopes
- [ ] Test repository has write access for automation
- [ ] Python 3.11+, PyGithub, PyYAML installed
- [ ] All agent scripts and projects_v2.py copied to test environment
- [ ] Orchestration config (orchestration.yaml) in place
- [ ] W&B logging configured (if testing with real logging)

---

## Testing Environments

### Environment 1: Local Dry-Run (Fastest)
```bash
DRY_RUN=true GITHUB_TOKEN=<token> REPO_NAME=B9android/wargames_training \
  python scripts/project_agent/issue_writer.py 2>&1 | grep "agent\|error\|sync"
```
- **Cost**: Free (no API calls beyond read)
- **Speed**: < 1s per agent
- **Coverage**: Logic validation, no board verification
- **Best for**: Quick iteration on code changes

### Environment 2: Staging Test Repo
```bash
REPO_NAME=B9android/wargames-test GITHUB_TOKEN=<token> \
  python scripts/project_agent/issue_writer.py
```
- **Cost**: ~50-100 API calls per run
- **Speed**: 5-30s per agent (depends on GraphQL complexity)
- **Coverage**: Full integration including board mutations
- **Best for**: Comprehensive validation before production

### Environment 3: Production (Monitored)
```bash
REPO_NAME=B9android/wargames_training GITHUB_TOKEN=<prod_token> \
  python scripts/project_agent/issue_writer.py
```
- **Cost**: Production API rate limits apply
- **Speed**: Same as staging
- **Coverage**: Real data, real workflows
- **Best for**: Post-validation, ongoing monitoring

---

## Core Abstraction Testing

### Test 1.1: Field ID Caching

**Objective**: Verify field ID cache works and reduces API calls

```bash
# First run - cache miss
GITHUB_TOKEN=<token> python -c "
from scripts.project_agent.projects_v2 import ProjectsV2Client
client = ProjectsV2Client('<token>', 'B9android/wargames_training')
fields = client.get_field_ids()
print(f'Fields cached: {list(fields.keys())}')
"

# Second run - cache hit (should reuse client instance)
# Expected: No additional GraphQL queries to GitHub
```

**Expected Output**:
```
Fields cached: ['Version', 'Sprint', 'Status', 'Story Points', 'Experiment Status', 'Git Commit', 'W&B Run']
```

**Pass Criteria**:
- ✅ Both fields with id and type present
- ✅ Second run reuses cache (no new queries)
- ✅ Caching works across function calls

---

### Test 1.2: GraphQL Query Execution

**Objective**: Verify gh CLI GraphQL interface works

```bash
cat > /tmp/test_query.graphql << 'EOF'
query {
  repository(owner: "B9android", name: "wargames_training") {
    issues(first: 1) {
      nodes {
        number
        title
      }
    }
  }
}
EOF

GITHUB_TOKEN=<token> gh api graphql -f query=@/tmp/test_query.graphql
```

**Expected Output**:
```json
{
  "repository": {
    "issues": {
      "nodes": [{ "number": 1, "title": "..." }]
    }
  }
}
```

**Pass Criteria**:
- ✅ gh CLI authentication works
- ✅ Query returns valid JSON
- ✅ No syntax errors in response

---

### Test 1.3: Retry Logic (Transient Error)

**Objective**: Verify retry logic handles transient errors

```bash
# Simulate transient error by running against invalid token first, then valid
GITHUB_TOKEN=invalid python -c "
from scripts.project_agent.projects_v2 import ProjectsV2Client
from unittest.mock import patch
import subprocess

client = ProjectsV2Client('invalid', 'B9android/wargames_training')

# Mock subprocess to return transient error once, then succeed
call_count = [0]
original_run = subprocess.run

def mock_run(cmd, **kwargs):
    call_count[0] += 1
    if call_count[0] == 1:
        # Simulate transient error
        class Result:
            returncode = 1
            stdout = ''
            stderr = 'Connection timeout'
        return Result()
    else:
        # Succeed on retry
        return original_run(cmd, **kwargs)

with patch('subprocess.run', mock_run):
    try:
        client.get_field_ids()
        print('Retry logic worked: succeeded after transient error')
    except Exception as e:
        print(f'Retry logic failed: {e}')
" 2>&1 | grep -i "retry\|timeout\|succeeded"
```

**Expected Behavior**:
- First call fails with transient error
- Retry logic kicks in (exponential backoff)
- Second call succeeds
- No exception raised

**Pass Criteria**:
- ✅ Retry attempted after transient error
- ✅ Exponential backoff applied (delay > 0)
- ✅ Succeeds on retry

---

## Agent Integration Testing

### Test 2.1: Issue Writer — Add to Project

**Scenario**: Create new issue, verify it appears on project board with Version + Sprint fields

**Steps**:
```bash
export GITHUB_TOKEN=<token>
export REPO_NAME=B9android/wargames_training
export DRY_RUN=false

# Create test issue
python -c "
import os
import subprocess
from github import Github

gh = Github(os.environ['GITHUB_TOKEN'])
repo = gh.get_repo(os.environ['REPO_NAME'])
issue = repo.create_issue('Test Issue - Projects v2', 'Integration test for projects v2')
print(f'Created: #{issue.number}')
with open('/tmp/test_issue_num.txt', 'w') as f:
    f.write(str(issue.number))
"

# Run issue_writer (which should auto-add to project)
ISSUE_NUMBER=$(cat /tmp/test_issue_num.txt) \
  python scripts/project_agent/issue_writer.py

# Verify issue appears on board
gh project item-list 1 --owner B9android --format=json | jq '.items[] | select(.title | contains("Test Issue"))'
```

**Expected Output**:
```bash
Created: #42
✅ Issue #42 synced to project
# Project query should show issue with Version and Sprint fields set
```

**Pass Criteria**:
- ✅ Issue created in GitHub
- ✅ issue_writer.py runs without errors
- ✅ Issue appears on project board
- ✅ Version field set to current milestone/version
- ✅ Sprint field set to active sprint

---

### Test 2.2: Epic Decomposer — Inherit Version

**Scenario**: Create epic, decompose it, verify child issues inherit parent Version field

**Steps**:
```bash
# Create epic issue
gh issue create --title "Test Epic - v1 Training" \
  --body "<!-- version: v1 -->" \
  --label "epic" \
  --repo B9android/wargames_training

EPIC_NUM=<returned issue number>

# Create child issues via epic_decomposer
export EPIC_NUMBER=$EPIC_NUM
python scripts/project_agent/epic_decomposer.py

# Verify child issues on project board
gh project item-list 1 --owner B9android --format=json | \
  jq '.items[] | select(.parent_issue.number == '$EPIC_NUM') | {title, version}'
```

**Expected Output**:
```json
{
  "title": "Child Issue 1",
  "version": "v1"
}
```

**Pass Criteria**:
- ✅ Epic created
- ✅ Child issues created (no duplicate markers)
- ✅ All child issues appear on project
- ✅ Child issues have Version field = parent Version
- ✅ No decomposition markers cause re-runs

---

### Test 2.3: Triage Agent — Set Version from Labels

**Scenario**: Label issue with version label, verify Version field updated

**Steps**:
```bash
# Create unversioned issue
gh issue create --title "Test Triage Issue" \
  --repo B9android/wargames_training

ISSUE_NUM=<returned>

# Label it with version
gh issue edit $ISSUE_NUM --add-label "v1" \
  --repo B9android/wargames_training

# Run triage agent
export ISSUE_NUMBER=$ISSUE_NUM
python scripts/project_agent/triage_agent.py

# Verify Version field on project board
gh project item-get $ISSUE_NUM --owner B9android --format=json | jq '.version'
```

**Expected Output**:
```
"v1"
```

**Pass Criteria**:
- ✅ Issue labeled with version
- ✅ triage_agent.py runs
- ✅ Version field set on project
- ✅ Correct version value applied

---

### Test 2.4: Sprint Manager — Auto-Assign Backlog

**Scenario**: Start a sprint, verify backlog issues auto-assigned

**Steps**:
```bash
# Create sprint in GitHub Projects v2
gh project field-create --name "Test Sprint" --type=ITERATION \
  --owner B9android --format=json | jq '.field_id' > /tmp/sprint_id.txt

# Create backlog issues
for i in {1..5}; do
  gh issue create --title "Backlog Item $i" \
    --repo B9android/wargames_training
done

# Run sprint manager to assign
python scripts/project_agent/sprint_manager.py

# Verify assignments
gh project item-list 1 --owner B9android --format=json | \
  jq '.items[] | select(.sprint != null) | {title, sprint}'
```

**Expected Output**:
```json
{
  "title": "Backlog Item 1",
  "sprint": "Test Sprint"
}
```

**Pass Criteria**:
- ✅ Backlog issues created
- ✅ sprint_manager.py runs
- ✅ Max 30-50 issues assigned (check config)
- ✅ Only unassigned issues assigned
- ✅ Priority ranking respected

---

### Test 2.5: PR Linker — Status + Git Commit Sync

**Scenario**: Merge PR linked to issue, verify Status and Git Commit fields updated

**Steps**:
```bash
# Create issue
ISSUE_NUM=$(gh issue create --title "Test PR Link" --repo B9android/wargames_training | cut -d' ' -f2)

# Create branch and PR
git checkout -b test-pr-$ISSUE_NUM
echo "test" > test.txt
git add test.txt
git commit -m "Test #$ISSUE_NUM"
gh pr create --title "Fix #$ISSUE_NUM" --body "Closes #$ISSUE_NUM" --head test-pr-$ISSUE_NUM

# Get PR number
PR_NUM=$(gh pr list --repo B9android/wargames_training | grep "Fix" | cut -f1)

# Merge PR
gh pr merge $PR_NUM --merge --repo B9android/wargames_training

# Run pr_linker to sync
export PR_NUMBER=$PR_NUM
export ISSUE_NUMBER=$ISSUE_NUM
python scripts/project_agent/pr_linker.py

# Verify Status and Git Commit on project
gh project item-get $ISSUE_NUM --owner B9android --format=json | \
  jq '{status: .status, git_commit: .git_commit}'
```

**Expected Output**:
```json
{
  "status": "In Progress",
  "git_commit": "abc123def456..."
}
```

**Pass Criteria**:
- ✅ PR created and merged
- ✅ Issue linked to PR
- ✅ pr_linker.py runs
- ✅ Status field set to "In Progress"
- ✅ Git Commit field set to merge commit SHA

---

### Test 2.6: Project Syncer — Label to Field Sync

**Scenario**: Label issue, verify corresponding project fields updated

**Steps**:
```bash
# Create issue
ISSUE_NUM=$(gh issue create --title "Test Sync Labels" --repo B9android/wargames_training | cut -d' ' -f2)

# Label with status and priority
gh issue edit $ISSUE_NUM \
  --add-label "status: approved" \
  --add-label "priority: high" \
  --repo B9android/wargames_training

# Run project syncer
export ISSUE_NUMBER=$ISSUE_NUM
python scripts/project_agent/project_syncer.py

# Verify fields on project
gh project item-get $ISSUE_NUM --owner B9android --format=json | \
  jq '{status: .status, story_points: .story_points}'
```

**Expected Output**:
```json
{
  "status": "Approved",
  "story_points": 5
}
```

**Pass Criteria**:
- ✅ Issue labeled
- ✅ project_syncer.py runs
- ✅ Status field set to "Approved"
- ✅ Story Points field set to 5

---

## End-to-End Scenarios

### Scenario E2E-1: Full Issue Lifecycle

**Flow**: Create epic → Decompose → Triage → Assign to sprint → Link PR → Merge → Verify board

```bash
# 1. Create epic
EPIC=$(gh issue create --title "E2E Test Epic" --label "epic" \
  --body "<!-- version: v1 -->" \
  --repo B9android/wargames_training | awk '{print $NF}')

# 2. Decompose epic
export EPIC_NUMBER=$EPIC
python scripts/project_agent/epic_decomposer.py

# 3. Get first child issue
CHILD=$(gh issue list --repo B9android/wargames_training \
  --search "parent:$EPIC" --json number | jq -r '.[0].number')

# 4. Triage child issue
gh issue edit $CHILD --add-label "status: approved" \
  --repo B9android/wargames_training
export ISSUE_NUMBER=$CHILD
python scripts/project_agent/triage_agent.py

# 5. Run sprint manager
python scripts/project_agent/sprint_manager.py

# 6. Create and merge PR
git checkout -b fix-$CHILD
touch fixed.txt
git add fixed.txt
git commit -m "Fix #$CHILD"
PR=$(gh pr create --title "Fix #$CHILD" --head fix-$CHILD | awk '{print $NF}')
gh pr merge $PR --merge --repo B9android/wargames_training

# 7. Sync PR to project
export PR_NUMBER=$PR
python scripts/project_agent/pr_linker.py

# 8. Verify entire flow on project board
echo "Final state on project board:"
gh project item-get $CHILD --owner B9android --format=json | \
  jq '{number: .number, status: .status, version: .version, sprint: .sprint, git_commit: .git_commit}'
```

**Expected State**:
```json
{
  "number": 43,
  "status": "In Progress",
  "version": "v1",
  "sprint": "Active Sprint",
  "git_commit": "abc123..."
}
```

---

## Error Handling & Resilience

### Test 3.1: Missing Project Board

**Scenario**: Agents continue when projects_v2 board not configured

```bash
# Remove project board access
export GITHUB_TOKEN=<github_token>
export REPO_NAME=invalid-repo/wargames_training

python scripts/project_agent/issue_writer.py 2>&1 | grep -i "project\|sync\|failed"
```

**Expected Behavior**:
- ✅ projects_client initialization fails gracefully
- ✅ Issue still created in GitHub
- ✅ No exception propagated to user

---

### Test 3.2: Rate Limit Handling

**Scenario**: Graceful handling of GraphQL rate limit errors

```bash
# Monitor API usage before test
gh api rate-limit --format=json | jq '.resources.graphql'

# Run multiple agents rapidly
for i in {1..10}; do
  python scripts/project_agent/project_syncer.py &
done
wait

# Check rate limit after
gh api rate-limit --format=json | jq '.resources.graphql'
```

**Expected Behavior**:
- ✅ Rate limit errors caught and logged
- ✅ Transient retries applied (requests < burst rate)
- ✅ No data corruption if rate limited

---

### Test 3.3: Malformed Issue Data

**Scenario**: Handle issues with missing/invalid fields

```bash
# Create issue with minimal data
ISSUE=$(gh issue create --title "" --repo B9android/wargames_training | awk '{print $NF}')

# Run agents against it
export ISSUE_NUMBER=$ISSUE
python scripts/project_agent/project_syncer.py 2>&1
python scripts/project_agent/triage_agent.py 2>&1
```

**Expected Behavior**:
- ✅ No exceptions raised
- ✅ Errors logged
- ✅ Processing continues

---

## Performance & Rate Limits

### Test 4.1: API Efficiency

**Objective**: Verify agents minimize API calls

```bash
# Monitor API calls during test
gh api rate-limit --format=json | jq -r '.resources.graphql | "\(.limit - .remaining)/\(.limit) calls used"' > /tmp/before.txt

# Run batch of operations
for i in {1..5}; do
  ISSUE=$(gh issue create --title "Test $i" --repo B9android/wargames_training | awk '{print $NF}')
  export ISSUE_NUMBER=$ISSUE
  python scripts/project_agent/project_syncer.py
done

# Check API usage
gh api rate-limit --format=json | jq -r '.resources.graphql | "\(.limit - .remaining)/\(.limit) calls used"' > /tmp/after.txt
echo "Cost: $(( $(cat /tmp/after.txt | cut -d/ -f1) - $(cat /tmp/before.txt | cut -d/ -f1) )) API calls for 5 issues"
```

**Expected**:
- ~2-5 API calls per issue (caching should reduce this)
- Clear log output showing what queries were executed

---

### Test 4.2: Bulk Operations

**Objective**: Verify sprint_manager handles 50+ issues

```bash
# Create 50 backlog issues
for i in {1..50}; do
  gh issue create --title "Backlog $i" &
done
wait

# Run sprint manager and time it
time python scripts/project_agent/sprint_manager.py

# Verify all assigned
gh project item-list 1 --owner B9android --format=json | \
  jq '.items | length' > /tmp/count.txt
echo "Assigned: $(cat /tmp/count.txt) issues"
```

**Expected**:
- ✅ All 50 issues processed in < 30s
- ✅ No API timeouts
- ✅ Info logged for each batch

---

## Troubleshooting

### Issue: "Field ID not found"

**Cause**: Project board fields not set up correctly

**Debug**:
```bash
python -c "
from scripts.project_agent.projects_v2 import ProjectsV2Client
client = ProjectsV2Client('<token>', 'B9android/wargames_training')
fields = client.get_field_ids()
print('Available fields:', fields)
"
```

**Fix**: Add missing fields to project board (see Pre-Flight Checklist)

---

### Issue: "Issue not found in project"

**Cause**: Issue not yet added to project board

**Debug**:
```bash
# Manually add issue to board
gh project item-add --id <project_id> <issue_url>

# Retry sync
export ISSUE_NUMBER=<num>
python scripts/project_agent/project_syncer.py
```

---

### Issue: "GraphQL query timeout"

**Cause**: Query too complex or rate limited

**Debug**:
```bash
# Check rate limit
gh api rate-limit --format=json | jq '.resources.graphql'

# Wait for reset if needed
sleep 60

# Retry
python scripts/project_agent/<agent>.py
```

---

## Verification Checklist

- [ ] **All 6 agents pass dry-run tests** (Test 2.1 - 2.6)
- [ ] **E2E issue lifecycle works** (Scenario E2E-1)
- [ ] **Error handling resilient** (Test 3.1 - 3.3)
- [ ] **API efficiency validated** (Test 4.1 - 4.2)
- [ ] **No duplicate issues from retries** (markers working)
- [ ] **Field IDs cached properly** (Test 1.1)
- [ ] **GraphQL queries execute correctly** (Test 1.2)
- [ ] **Rate limits not exceeded** (Test 4.2)
- [ ] **All logs structured and searchable** (grep agent logs)
- [ ] **Dry-run mode respected everywhere** (all agents)

Once all checks pass, projects_v2 integration is production-ready.

---

## Next Steps

1. **Run staging environment tests** (Environment 2) against test repo for 7 days
2. **Monitor W&B logs** for any agent failures or field sync issues
3. **Collect metrics**: API calls, success rates, duration
4. **Production rollout**: Enable workflows when satisfied with coverage
