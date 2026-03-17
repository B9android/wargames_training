# GitHub Projects v2 Integration — Testing Checklist

Use this checklist to track progress through validation, staging, and production phases.

---

## Phase 0: Pre-Flight (15 min)

**Objective**: Verify prerequisites before any testing

- [ ] Fork/clone wargames_training repo to test environment
- [ ] `export GITHUB_TOKEN=<your_token>`
- [ ] `export REPO_NAME=<your_test_repo>` (e.g., username/wargames-test)
- [ ] Verify GitHub Projects v2 board exists on test repo
- [ ] Create required custom fields on board:
  - [ ] Version (Single Select)
  - [ ] Sprint (Iteration)
  - [ ] Status (Single Select: Triaged, Approved, In Progress, Blocked, Done)
  - [ ] Story Points (Number)
  - [ ] Experiment Status (Single Select)
  - [ ] Git Commit (Text)
  - [ ] W&B Run (Text)
- [ ] Install Python 3.11+: `python --version`
- [ ] Install dependencies: `pip install PyGithub PyYAML`
- [ ] Install gh CLI: `brew install gh` (or download from https://cli.github.com)
- [ ] Authenticate gh CLI: `gh auth login`
- [ ] Run validation script: `python scripts/project_agent/validate_projects_v2.py`
  - [ ] All checks pass (green ✅)
  - [ ] Field IDs cached successfully
  - [ ] ProjectsV2Client initialization works

**Blocker?** Stop here. See [PROJECTS_V2_QUICK_REFERENCE.md - Troubleshooting](docs/PROJECTS_V2_QUICK_REFERENCE.md#troubleshooting-steps)

---

## Phase 1: Core Abstraction Testing (30 min)

**Objective**: Validate projects_v2.py GraphQL layer works independently

### Test 1.1: Field ID Caching
- [ ] Run field ID cache test from [PROJECTS_V2_TESTING_GUIDE.md#test-11](docs/PROJECTS_V2_TESTING_GUIDE.md#test-11-field-id-caching)
- [ ] Fields cached successfully
- [ ] Second run reuses cache (no new API calls)
- [ ] Log output shows all 7 expected fields

### Test 1.2: GraphQL Query Execution
- [ ] Run GraphQL query test from Guide
- [ ] Query returns valid JSON
- [ ] No syntax errors in response
- [ ] Can access repository issues

### Test 1.3: Retry Logic
- [ ] Run retry logic test from Guide
- [ ] Transient error triggers retry
- [ ] Exponential backoff applied (delay observed)
- [ ] Eventually succeeds or times out gracefully

**Summary**: Core GraphQL abstraction working? **YES [ ] NO [ ]**

---

## Phase 2: Individual Agent Testing (2-3 hours)

**Objective**: Validate each agent syncs correctly to project board

### Test 2.1: issue_writer.py
- [ ] Create test issue: `gh issue create --title "Test Issue - Projects v2" --label "v1"`
- [ ] Run issue_writer: `ISSUE_NUMBER=<num> python scripts/project_agent/issue_writer.py`
- [ ] Check project board:
  - [ ] Issue appears on board
  - [ ] Version field = "v1"
  - [ ] Sprint field = active sprint
- [ ] W&B log shows `issue_created` or `field_synced` events

### Test 2.2: epic_decomposer.py
- [ ] Create epic: `gh issue create --title "Test Epic - v1" --label "epic" --body "<!-- version: v1 -->"`
- [ ] Run decomposer: `EPIC_NUMBER=<num> python scripts/project_agent/epic_decomposer.py`
- [ ] Check results:
  - [ ] Child issues created (no duplicate markers)
  - [ ] All children on project board
  - [ ] All children have Version = "v1"
- [ ] Re-run same command:
  - [ ] No new children created (deduplication works)

### Test 2.3: triage_agent.py
- [ ] Create issue: `gh issue create --title "Test Triage"`
- [ ] Add label: `gh issue edit <num> --add-label "v1"`
- [ ] Run triage: `ISSUE_NUMBER=<num> python scripts/project_agent/triage_agent.py`
- [ ] Check project board:
  - [ ] Version field = "v1"

### Test 2.4: sprint_manager.py
- [ ] Create 5-10 test issues in backlog
- [ ] Run sprint manager: `python scripts/project_agent/sprint_manager.py`
- [ ] Check project board:
  - [ ] Backlog issues assigned to active sprint
  - [ ] Sprint field populated (max 50 issues)

### Test 2.5: pr_linker.py
- [ ] Create issue: `gh issue create --title "Test PR Link"`
- [ ] Create branch, commit, PR: `git checkout -b test && echo x > f.txt && git add f.txt && git commit -m "Fix #<num>" && gh pr create --title "Fix #<num>"`
- [ ] Get PR number from output
- [ ] Merge PR: `gh pr merge <pr_num> --merge`
- [ ] Run pr_linker: `ISSUE_NUMBER=<num> PR_NUMBER=<pr_num> python scripts/project_agent/pr_linker.py`
- [ ] Check project board:
  - [ ] Status field = "In Progress"
  - [ ] Git Commit field = merge SHA

### Test 2.6: project_syncer.py
- [ ] Create issue: `gh issue create --title "Test Syncer"`
- [ ] Label with status: `gh issue edit <num> --add-label "status: approved"`
- [ ] Label with priority: `gh issue edit <num> --add-label "priority: high"`
- [ ] Run syncer: `ISSUE_NUMBER=<num> python scripts/project_agent/project_syncer.py`
- [ ] Check project board:
  - [ ] Status field = "Approved"
  - [ ] Story Points field = 5

**Summary**: All 6 agents syncing correctly? **YES [ ] NO [ ]**

If NO, which agents failed? _________________________________

---

## Phase 3: Error Handling & Resilience (1 hour)

**Objective**: Validate graceful error handling

### Test 3.1: Missing Project Board
- [ ] Temporarily prevent project access (or use invalid REPO_NAME)
- [ ] Run issue_writer: `python scripts/project_agent/issue_writer.py`
- [ ] Check:
  - [ ] Issue still created in GitHub
  - [ ] No unhandled exceptions
  - [ ] Error logged but doesn't block

### Test 3.2: Rate Limit Handling
- [ ] Check rate limit before: `gh api rate-limit --format=json | jq '.resources.graphql'`
- [ ] Run multiple agents rapidly: `for i in {1..10}; do python scripts/project_agent/project_syncer.py & done`
- [ ] Wait for completion: `wait`
- [ ] Check rate limit after
- [ ] Verify:
  - [ ] No data corruption
  - [ ] API calls made (< total limit)
  - [ ] Backoff applied if rate limited

### Test 3.3: Malformed Data
- [ ] Create issue with empty title: `gh issue create --title ""` 
- [ ] Run agents against it
- [ ] Verify:
  - [ ] No unhandled exceptions
  - [ ] Errors logged
  - [ ] Processing continues

**Summary**: Error handling working? **YES [ ] NO [ ]**

---

## Phase 4: Performance & Efficiency (1 hour)

**Objective**: Validate API efficiency and bulk operation performance

### Test 4.1: API Efficiency
- [ ] Get API call count before: `gh api rate-limit --format=json | jq '.resources.graphql.used'`
- [ ] Create 5 test issues: `for i in {1..5}; do gh issue create --title "Test $i"; done`
- [ ] Run project_syncer 5 times sequentially
- [ ] Get API call count after
- [ ] Calculate cost: (after - before) / 5 = _____ API calls per issue
- [ ] Expected: 2-5 API calls per issue with caching
- [ ] Actual: _____ (acceptable if ≤ 10)

### Test 4.2: Bulk Sprint Assignment
- [ ] Create 50 backlog issues: `for i in {1..50}; do gh issue create --title "Backlog $i" &; done; wait`
- [ ] Time sprint manager: `time python scripts/project_agent/sprint_manager.py`
- [ ] Duration: _____ seconds
- [ ] Expected: 10-30 seconds
- [ ] Verify all 50 assigned (or max 50 if cap enforced)
- [ ] Check W&B logs for any errors

**Summary**: Performance acceptable? **YES [ ] NO [ ]**

---

## Phase 5: End-to-End Scenario (45 min)

**Objective**: Validate full issue lifecycle from creation to merge

### E2E Test: Full Lifecycle

Follow the complete flow from [PROJECTS_V2_TESTING_GUIDE.md#scenario-e2e-1](docs/PROJECTS_V2_TESTING_GUIDE.md#scenario-e2e-1-full-issue-lifecycle):

- [ ] **Step 1**: Create epic (result: EPIC #___)
- [ ] **Step 2**: Decompose epic (result: CHILD #___)
- [ ] **Step 3**: Triage child issue with label
- [ ] **Step 4**: Run sprint manager (auto-assigned: YES [ ] NO [ ])
- [ ] **Step 5**: Create and merge PR (PR #___)
- [ ] **Step 6**: Sync PR to project
- [ ] **Step 7**: Verify final state on board:
  - [ ] Status = "In Progress"
  - [ ] Version = "v1"
  - [ ] Sprint = Active Sprint
  - [ ] Git Commit = Set

**Final Board State**:
```json
{
  "number": ___,
  "status": "___",
  "version": "___",
  "sprint": "___",
  "git_commit": "___"
}
```

**Summary**: E2E flow complete? **YES [ ] NO [ ]**

---

## Phase 6: Production Readiness Check (15 min)

**Objective**: Final verification before enabling on production

### Verification Checklist

- [ ] ✅ All Phase 1-5 tests passed
- [ ] ✅ No unhandled exceptions observed
- [ ] ✅ Field IDs cached (< 100ms per lookup after first)
- [ ] ✅ Markers prevent deduplication (re-run safe)
- [ ] ✅ Dry-run mode prevents board mutations
- [ ] ✅ W&B logging operational (events queryable)
- [ ] ✅ Error logging clear (root cause identifiable)
- [ ] ✅ GitHub token scopes correct (repo, project)
- [ ] ✅ Rate limits not exceeded in bulk tests
- [ ] ✅ No data corruption on errors/retries

### Metrics Summary

| Metric | Target | Actual | Δ |
|--------|--------|--------|---|
| Issue sync latency | < 5s | _____ | _____ |
| Label sync latency | < 2s | _____ | _____ |
| Sprint assignment success | > 98% | _____ | _____ |
| API calls per issue | < 5 | _____ | _____ |
| Error recovery | Graceful | _____ | _____ |

**Ready for Production?** **YES [ ] NO [ ]**

If NO, which phases need work? _________________________________

---

## Production Rollout Steps

Once Phase 6 passes:

- [ ] **Step 1**: Enable agent workflows on main repo
  ```bash
  git checkout main
  git pull origin main
  # Edit .github/workflows entries; set to production repo
  git add .github/workflows
  git commit -m "Enable Projects v2 automation"
  git push origin main
  ```

- [ ] **Step 2**: Set up W&B monitoring & alerts
  - [ ] Create W&B dashboard filtering by agent name
  - [ ] Set alerts for > 5% failure rate
  - [ ] Subscribe to notifications

- [ ] **Step 3**: Enable GitHub Actions
  - [ ] Check: "Settings > Actions > Runners" has available runners
  - [ ] Check: Workflows have execution history

- [ ] **Step 4**: Test on production with monitoring
  - [ ] Create test issue
  - [ ] Watch W&B logs for events
  - [ ] Verify board sync within 5 seconds

- [ ] **Step 5**: Document for team
  - [ ] Share [PROJECTS_V2_QUICK_REFERENCE.md](docs/PROJECTS_V2_QUICK_REFERENCE.md)
  - [ ] Share [PROJECTS_V2_IMPLEMENTATION_SUMMARY.md](docs/PROJECTS_V2_IMPLEMENTATION_SUMMARY.md)
  - [ ] Brief team on label mappings & field sync

- [ ] **Step 6**: Monitor continuously
  - [ ] Weekly W&B metrics review
  - [ ] Month 1: Daily checks
  - [ ] Month 2+: Weekly checks

**Production Enabled**: _____ (date)

---

## Troubleshooting During Testing

**Issue**: "Field ID not found"
- Step 1: Check custom fields exist on board (GitHub UI)
- Step 2: Verify field names match PROJECTS_V2_FIELDS in common.py
- Step 3: Re-run validation script
- Link: [PROJECTS_V2_QUICK_REFERENCE.md#common-issues--fixes](docs/PROJECTS_V2_QUICK_REFERENCE.md#common-issues--fixes)

**Issue**: "Issue doesn't appear on board"
- Step 1: Check issue is in correct project (verify project ID)
- Step 2: Run issue_writer manually to add it
- Step 3: Check W&B logs for sync errors
- Link: [PROJECTS_V2_QUICK_REFERENCE.md#troubleshooting-steps](docs/PROJECTS_V2_QUICK_REFERENCE.md#troubleshooting-steps)

**Issue**: "GraphQL timeout"
- Step 1: Check rate limit: `gh api rate-limit --format=json`
- Step 2: Wait 60 seconds, retry
- Step 3: Check network connectivity
- Link: [PROJECTS_V2_TESTING_GUIDE.md#troubleshooting](docs/PROJECTS_V2_TESTING_GUIDE.md#troubleshooting)

**More help?** See [PROJECTS_V2_QUICK_REFERENCE.md](docs/PROJECTS_V2_QUICK_REFERENCE.md)

---

## Sign-Off

**Test Lead**: _________________________** Date**: __________

**Tested Phases**: 
- [ ] Phase 0: Pre-Flight
- [ ] Phase 1: Core Abstraction
- [ ] Phase 2: Individual Agents
- [ ] Phase 3: Error Handling
- [ ] Phase 4: Performance
- [ ] Phase 5: E2E Scenario
- [ ] Phase 6: Production Ready

**Overall Result**: **PASS [ ] FAIL [ ]**

**Comments**: ________________________________________________________________________

________________________________________________________________________

---

**Questions?** Check [PROJECTS_V2_QUICK_REFERENCE.md](docs/PROJECTS_V2_QUICK_REFERENCE.md)  
**Need to debug?** Run `python scripts/project_agent/validate_projects_v2.py`  
**Have issues?** See [PROJECTS_V2_TESTING_GUIDE.md#troubleshooting](docs/PROJECTS_V2_TESTING_GUIDE.md#troubleshooting)
