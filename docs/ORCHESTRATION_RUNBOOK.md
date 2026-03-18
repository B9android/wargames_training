# Orchestration Runbook

Operational guide for the agent control-plane workflows.

## Scope

This runbook covers:

- Workflow dispatch actions in .github/workflows/orchestration.yml
- Dry-run procedures before production writes
- Release sync behavior for docs/CHANGELOG.md and docs/ROADMAP.md
- Permanence policy: all agent runs must go through orchestration.yml
- Platform runtime conventions under scripts/project_agent/agent_platform

## Control Plane Entry Point

Primary workflow:

- .github/workflows/orchestration.yml

Dispatch input:

- action: triage | experiment-approve | experiment-kickoff | milestone-check | progress-report | release-sync | epic-decompose | sprint-start | sprint-close | sprint-auto-transition | sync-pr-status | project-sync | training-complete
- issue_number: required for triage, experiment-approve, experiment-kickoff
- pr_number: required for sync-pr-status
- milestone_number: required for release-sync
- sprint_name: required for sprint-start and sprint-close
- wandb_run_id: required for training-complete
- dry_run: true/false

## Permanence Policy

1. Do not create per-agent workflow files.
2. Add new agent actions as jobs in orchestration.yml.
3. Governance CI enforces this contract in .github/workflows/governance.yml.

## Runtime Contract

1. All orchestration-run agents use run_agent from scripts/project_agent/agent_platform/runner.py.
2. Agents publish execution artifacts to agent-artifacts/ (run-report.json and run-report.md).
3. GraphQL board operations are centralized in scripts/project_agent/agent_platform/graphql_gateway.py.

## Standard Operating Procedure

1. Start in dry-run mode for any manual dispatch.
2. Validate logs and ensure no transition/contract errors.
3. Re-run with dry_run=false only after dry-run output matches intent.

## Action Checklist

### triage

Use for manual issue triage when issue-open triggers are bypassed.

Required inputs:

- issue_number

Validation:

- Confirms labels and milestone decisions in logs
- Posts triage comment only when dry_run=false

### experiment-approve

Use to apply approval gate for an [EXP] issue.

Required inputs:

- issue_number

Validation:

- Confirms parent issue reference exists in issue body
- Confirms lifecycle transition is valid before label changes

### experiment-kickoff

Use to move approved experiment into in-progress state.

Required inputs:

- issue_number

Validation:

- Checks required issue sections
- Checks approval and parent linkage policies from configs/orchestration.yaml
- Uses idempotency marker to avoid duplicate kickoff comments

### milestone-check

Use for health sweep of stale, unlabeled, and at-risk milestone issues.

Validation:

- At-risk issue creation is marker-based to avoid duplicates
- Stale label/comment updates skipped during dry-run

### progress-report

Use for weekly status issue generation.

Validation:

- Builds deterministic snapshot first
- Adds AI narrative second (with fallback when AI request fails)

### release-sync

Use when milestone is closed to sync docs.

Required inputs:

- milestone_number

Validation:

- Skips when milestone is not closed
- Uses marker format: <!-- agent:release-sync:<milestone_number> -->
- No-op when marker already exists in both target docs

## Release Sync Verification

After a non-dry-run release-sync:

1. Confirm a marker line was added to docs/CHANGELOG.md.
2. Confirm the same marker line was added to docs/ROADMAP.md.
3. Confirm workflow commit step only commits when docs changed.

Expected no-op message for idempotent rerun:

- Skipped: release sync marker already present for milestone #[number]

## Local Test Command

Run orchestration regression tests:

python -m unittest tests/test_milestone_checker.py tests/test_issue_writer.py tests/test_progress_reporter.py tests/test_triage_agent.py tests/test_training_monitor.py tests/test_release_coordinator.py tests/test_experiment_lifecycle.py tests/test_dependency_resolver.py -v
