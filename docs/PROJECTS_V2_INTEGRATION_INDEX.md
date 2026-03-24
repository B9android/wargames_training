# GitHub Projects v2 Integration — Documentation Index

**Status**: ✅ Implementation Complete | Ready for Staged Testing  
**Version**: v1 Autonomous Project Orchestration  
**Last Updated**: March 17, 2026

---

## Quick Start (5 minutes)

### For Developers (First Time)
1. Read: [PROJECTS_V2_QUICK_REFERENCE.md](PROJECTS_V2_QUICK_REFERENCE.md) (5 min)
2. Run: `python scripts/project_agent/validate_projects_v2.py` (2 min)
3. Check: Are all✅ green? Great, proceed to testing.

### For QA/Testers
1. Read: [PROJECTS_V2_TESTING_CHECKLIST.md](PROJECTS_V2_TESTING_CHECKLIST.md) (start here)
2. Work through: Phases 0-6 sequentially
3. Sign off: When all boxes checked

### For Project Leads
1. Read: [PROJECTS_V2_IMPLEMENTATION_SUMMARY.md](PROJECTS_V2_IMPLEMENTATION_SUMMARY.md) (10 min)
2. Review: Architecture, Files Created, Next Steps
3. Plan: Rollout using Production Rollout Steps

---

## Documentation Map

### 📋 Quick Reference (Start Here)
**[PROJECTS_V2_QUICK_REFERENCE.md](PROJECTS_V2_QUICK_REFERENCE.md)** (5 min read)

What's in it:
- Field mapping table (label → project field)
- 6 agent interaction diagrams
- Quick invocation examples
- Cache & retry logic details
- Error handling matrix
- Marker-based deduplication
- Common issues & fixes
- Performance targets
- Security notes

**Best for**: Quick lookup, troubleshooting, understanding patterns

---

### 🧪 Testing Guide (Comprehensive)
**[PROJECTS_V2_TESTING_GUIDE.md](PROJECTS_V2_TESTING_GUIDE.md)** (30 min read + 6 hours testing)

What's in it:
- Pre-flight checklist (15 min)
- 3 testing environments (local, staging, production)
- Test 1.1-1.3: Core abstraction (caching, GraphQL, retry)
- Test 2.1-2.6: Each agent integration  
- Test 3.1-3.3: Error handling & resilience
- Test 4.1-4.2: Performance & bulk operations
- Scenario E2E-1: Full issue lifecycle
- Troubleshooting section

**Best for**: Complete validation before production, understanding test scenarios

---

### ✅ Testing Checklist (Practical)
**[PROJECTS_V2_TESTING_CHECKLIST.md](PROJECTS_V2_TESTING_CHECKLIST.md)** (Use while testing)

What's in it:
- Phase 0-6 testing phases (15 min - 2 hours each)
- Check-boxes for each step
- Pass/fail tracking
- Metrics table
- Production rollout steps
- Sign-off section

**Best for**: QA teams, tracking progress, rollout readiness

---

### 📚 Implementation Summary (Architecture)
**[PROJECTS_V2_IMPLEMENTATION_SUMMARY.md](PROJECTS_V2_IMPLEMENTATION_SUMMARY.md)** (15 min read)

What's in it:
- Architecture overview (6 agents + 1 core)
- What was built (features checklist)
- Files created/modified
- Data flow examples (3 detailed scenarios)
- Integration points table
- Configuration reference
- Next steps (Phase 1-5)
- Success metrics
- Troubleshooting links

**Best for**: Project leads, high-level understanding, planning rollout

---

## Source Code

### Core Abstraction
- **[projects_v2.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/projects_v2.py)** — GraphQL client with field ID caching, retry logic, batch mutations
- **[sprint_assigner.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/sprint_assigner.py)** — Helper for sprint auto-assignment

### Agents (6 total)
All in `scripts/project_agent/`:

| Agent | Purpose | Trigger | Board Action |
|-------|---------|---------|---------|
| [issue_writer.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/issue_writer.py) | Create issue | Issue created | Add to project + Version/Sprint |
| [epic_decomposer.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/epic_decomposer.py) | Decompose epic | Manual dispatch | Add children + inherit Version |
| [triage_agent.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/triage_agent.py) | Triage issue | Rule-based | Set Version from labels |
| [sprint_manager.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/sprint_manager.py) | Manage sprints | Sprint lifecycle | Auto-assign backlog |
| [pr_linker.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/pr_linker.py) | Link PR to issue | PR merge | Update Status + Git Commit |
| [project_syncer.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/project_syncer.py) | Sync labels | Label applied | Update field(s) |

### Utilities
- **[common.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/common.py)** — Shared constants & helpers (PROJECTS_V2_FIELDS)
- **[validate_projects_v2.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/validate_projects_v2.py)** — Pre-flight validation script

### Workflows
- **[.github/workflows/orchestration.yml](https://github.com/B9android/wargames_training/blob/main/.github/workflows/orchestration.yml)** — Triggered on label/milestone changes

### Configuration
- **[configs/orchestration.yaml](https://github.com/B9android/wargames_training/blob/main/configs/orchestration.yaml)** — State machines, policies, field relationships

---

## How to Use This Documentation

### Scenario 1: "I need to test the integration"
1. Start: [PROJECTS_V2_TESTING_CHECKLIST.md](PROJECTS_V2_TESTING_CHECKLIST.md)
2. Reference: [PROJECTS_V2_TESTING_GUIDE.md](PROJECTS_V2_TESTING_GUIDE.md) for details
3. Troubleshoot: [PROJECTS_V2_QUICK_REFERENCE.md](PROJECTS_V2_QUICK_REFERENCE.md)

### Scenario 2: "Something broke, help!"
1. Check: [PROJECTS_V2_QUICK_REFERENCE.md - Troubleshooting](PROJECTS_V2_QUICK_REFERENCE.md#troubleshooting-steps)
2. Validate: `python scripts/project_agent/validate_projects_v2.py`
3. Debug: [PROJECTS_V2_TESTING_GUIDE.md - Troubleshooting](PROJECTS_V2_TESTING_GUIDE.md#troubleshooting)

### Scenario 3: "What's the architecture?"
1. Read: [PROJECTS_V2_IMPLEMENTATION_SUMMARY.md](PROJECTS_V2_IMPLEMENTATION_SUMMARY.md)
2. Details: [PROJECTS_V2_QUICK_REFERENCE.md - Agent Patterns](PROJECTS_V2_QUICK_REFERENCE.md#agent-interaction-patterns)
3. Code: See [source code links](#source-code) above

### Scenario 4: "How do I invoke an agent?"
1. Quick: [PROJECTS_V2_QUICK_REFERENCE.md - Quick Start](PROJECTS_V2_QUICK_REFERENCE.md#invocation-quick-start)
2. Examples: [PROJECTS_V2_TESTING_GUIDE.md](PROJECTS_V2_TESTING_GUIDE.md) (each Test 2.x has example)

### Scenario 5: "Ready for production?"
1. Checklist: [PROJECTS_V2_TESTING_CHECKLIST.md - Phase 6](PROJECTS_V2_TESTING_CHECKLIST.md#phase-6-production-readiness-check)
2. Summary: [PROJECTS_V2_IMPLEMENTATION_SUMMARY.md - Next Steps](PROJECTS_V2_IMPLEMENTATION_SUMMARY.md#next-steps)
3. Rollout: [PROJECTS_V2_TESTING_CHECKLIST.md - Production Rollout Steps](PROJECTS_V2_TESTING_CHECKLIST.md#production-rollout-steps)

---

## Key Concepts

### Field Mappings
GitHub labels automatically sync to project board fields:
- `v1` → Version = "v1"
- `status: approved` → Status = "Approved"
- `priority: high` → Story Points = 5

See [PROJECTS_V2_QUICK_REFERENCE.md - Field Mappings](PROJECTS_V2_QUICK_REFERENCE.md#field-mappings)

### Agent Interaction Patterns
Each agent follows a pattern:
1. Receive trigger (issue created, label applied, PR merged, etc.)
2. Query GitHub (issue data, active sprint, etc.)
3. Call ProjectsV2Client (add to board, update field)
4. Log event to W&B
5. Graceful error handling

See [PROJECTS_V2_QUICK_REFERENCE.md - Agent Interaction Patterns](PROJECTS_V2_QUICK_REFERENCE.md#agent-interaction-patterns)

### Caching Strategy
- **What**: Field IDs (7 fields used repeatedly)
- **Where**: In-memory LRU cache in ProjectsV2Client
- **When**: First call is cache miss (GraphQL query), subsequent calls are hits (instant)
- **Why**: Reduces API calls from 7-10 per agent to ~1-2 per agent

See [PROJECTS_V2_QUICK_REFERENCE.md - Field ID Cache](PROJECTS_V2_QUICK_REFERENCE.md#field-id-cache)

### Retry Logic
- **Transient errors** (5xx, 429): Retry up to 4 times with exponential backoff (1s, 2s, 4s, 8s)
- **Permanent errors** (4xx except 429): Log and skip (no retry)
- **Result**: Resilient to temporary network issues, rate limits

See [PROJECTS_V2_QUICK_REFERENCE.md - Retry Logic](PROJECTS_V2_QUICK_REFERENCE.md#retry-logic)

### Marker-Based Deduplication
- **Problem**: Retried operations might create duplicates
- **Solution**: Add HTML comments to issue body as markers
- **Example**: `<!-- decomposed-from:epic-#42 -->`
- **Benefit**: Safe to retry; deduplication works without API calls

See [PROJECTS_V2_QUICK_REFERENCE.md - Marker-Based Deduplication](PROJECTS_V2_QUICK_REFERENCE.md#marker-based-deduplication)

---

## Common Workflows

### Add a new field to project board
1. Create field in GitHub Projects v2 UI
2. Add field name to PROJECTS_V2_FIELDS in [common.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/common.py)
3. Update agent scripts to populate field (call `update_field_value()`)
4. Add test case to [PROJECTS_V2_TESTING_GUIDE.md](PROJECTS_V2_TESTING_GUIDE.md)

### Add a new label mapping
1. Add mapping to `load_label_to_field_mappings()` in [project_syncer.py](https://github.com/B9android/wargames_training/blob/main/scripts/project_agent/project_syncer.py)
2. Test: Label issue with new label, run project_syncer, verify field updated
3. Document in [PROJECTS_V2_QUICK_REFERENCE.md - Field Mappings](PROJECTS_V2_QUICK_REFERENCE.md#field-mappings)

### Debug an agent
1. Run: `python scripts/project_agent/validate_projects_v2.py`
2. Run: `DRY_RUN=true python scripts/project_agent/<agent>.py` (dry-run first)
3. Check W&B logs: Filter by issue number and agent name
4. See: [PROJECTS_V2_QUICK_REFERENCE.md - Troubleshooting](PROJECTS_V2_QUICK_REFERENCE.md#troubleshooting-steps)

### Monitor production
1. W&B dashboard: Create view filtering by agent name
2. Alerts: Set up notifications for > 5% failure rate
3. Metrics: Track API calls, latency, success rate
4. Weekly review: Check [Success Metrics](PROJECTS_V2_IMPLEMENTATION_SUMMARY.md#success-metrics)

---

## File Structure Overview

```
wargames_training/
├── docs/
│   ├── PROJECTS_V2_QUICK_REFERENCE.md          ← Start here for quick lookup
│   ├── PROJECTS_V2_TESTING_GUIDE.md            ← Full testing guide
│   ├── PROJECTS_V2_TESTING_CHECKLIST.md        ← Use while testing
│   ├── PROJECTS_V2_IMPLEMENTATION_SUMMARY.md   ← Architecture overview
│   └── PROJECTS_V2_INTEGRATION_INDEX.md        ← This file
│
├── scripts/project_agent/
│   ├── projects_v2.py                          ← Core GraphQL client
│   ├── sprint_assigner.py                      ← Sprint helper
│   ├── validate_projects_v2.py                 ← Validation script
│   ├── common.py                               ← Shared constants (PROJECTS_V2_FIELDS)
│   ├── issue_writer.py                         ← Agent: Create issue
│   ├── epic_decomposer.py                      ← Agent: Decompose epic
│   ├── triage_agent.py                         ← Agent: Triage issue
│   ├── sprint_manager.py                       ← Agent: Manage sprints
│   ├── pr_linker.py                            ← Agent: Link PR to issue
│   └── project_syncer.py                       ← Agent: Sync labels
│
├── .github/workflows/
│   └── agent-project-syncer.yml                ← Trigger project_syncer
│
└── configs/
    └── orchestration.yaml                      ← State machines & policies
```

---

## Support & Help

### Documentation
- **Quick lookup**: [PROJECTS_V2_QUICK_REFERENCE.md](PROJECTS_V2_QUICK_REFERENCE.md)
- **Full reference**: [PROJECTS_V2_IMPLEMENTATION_SUMMARY.md](PROJECTS_V2_IMPLEMENTATION_SUMMARY.md)
- **Testing**: [PROJECTS_V2_TESTING_GUIDE.md](PROJECTS_V2_TESTING_GUIDE.md)

### Validation
- **Pre-flight check**: `python scripts/project_agent/validate_projects_v2.py`
- **Dry-run test**: `DRY_RUN=true python scripts/project_agent/project_syncer.py`

### Troubleshooting
- **First**: Check [PROJECTS_V2_QUICK_REFERENCE.md - Common Issues](PROJECTS_V2_QUICK_REFERENCE.md#common-issues--fixes)
- **Then**: Run validation script
- **Finally**: Check [PROJECTS_V2_TESTING_GUIDE.md - Troubleshooting](PROJECTS_V2_TESTING_GUIDE.md#troubleshooting)

### Questions?
1. Check index: Search this page for your topic
2. Read: Referenced documentation
3. Test: Run validation script
4. Debug: Check W&B logs

---

## Test Status

| Phase | Status | Duration | Documents |
|-------|--------|----------|-----------|
| Pre-Flight | ⏳ Not started | 15 min | [Guide](PROJECTS_V2_TESTING_GUIDE.md#pre-flight-checklist) / [Checklist](PROJECTS_V2_TESTING_CHECKLIST.md#phase-0-pre-flight-15-min) |
| Core Abstraction | ⏳ Not started | 30 min | [Guide](PROJECTS_V2_TESTING_GUIDE.md#core-abstraction-testing) / [Checklist](PROJECTS_V2_TESTING_CHECKLIST.md#phase-1-core-abstraction-testing-30-min) |
| Agent Integration | ⏳ Not started | 2-3 hr | [Guide](PROJECTS_V2_TESTING_GUIDE.md#agent-integration-testing) / [Checklist](PROJECTS_V2_TESTING_CHECKLIST.md#phase-2-individual-agent-testing-2-3-hours) |
| Error Handling | ⏳ Not started | 1 hr | [Guide](PROJECTS_V2_TESTING_GUIDE.md#error-handling--resilience) / [Checklist](PROJECTS_V2_TESTING_CHECKLIST.md#phase-3-error-handling--resilience-1-hour) |
| Performance | ⏳ Not started | 1 hr | [Guide](PROJECTS_V2_TESTING_GUIDE.md#performance--rate-limits) / [Checklist](PROJECTS_V2_TESTING_CHECKLIST.md#phase-4-performance--efficiency-1-hour) |
| E2E Scenario | ⏳ Not started | 45 min | [Guide](PROJECTS_V2_TESTING_GUIDE.md#end-to-end-scenarios) / [Checklist](PROJECTS_V2_TESTING_CHECKLIST.md#phase-5-end-to-end-scenario-45-min) |
| Production Ready | ⏳ Not started | 15 min | [Checklist](PROJECTS_V2_TESTING_CHECKLIST.md#phase-6-production-readiness-check-15-min) |

**Total Testing Time**: ~7 hours

---

## Next Action

**👉 Start Here**: [PROJECTS_V2_TESTING_CHECKLIST.md](PROJECTS_V2_TESTING_CHECKLIST.md)

Follow phases 0-6 sequentially. Check off boxes as you go. Sign off when complete.

**Questions?** See [PROJECTS_V2_QUICK_REFERENCE.md](PROJECTS_V2_QUICK_REFERENCE.md)
