
==============================================================
  AGENT EXECUTION REPORT â€” STATIC ANALYSIS
  Target: B9android/wargames_training  |  Elapsed: 0.8s  |  DRY_RUN=False
==============================================================


  âŒ FAILED (1)
    Â· run  â†’  B9android/wargames_training:  Unexpected: 401 {"message": "Bad credentials", "documentation_url": "https://docs.github.com/rest", "status": "401"}

  âŒ FAILED â€” no actions completed successfully
==============================================================


---

```json
{
  "agent": "static_analysis",
  "target": "B9android/wargames_training",
  "dry_run": false,
  "generated_at": "2026-03-18T02:38:32.325689+00:00",
  "elapsed_seconds": 0.81,
  "decisions": [],
  "checkpoints": [],
  "result": {
    "ok": false,
    "partial": false,
    "successes": 0,
    "failures": 1,
    "skipped": 0,
    "actions": [
      {
        "action": "run",
        "resource_id": "B9android/wargames_training",
        "status": "failed",
        "error": "Unexpected: 401 {\"message\": \"Bad credentials\", \"documentation_url\": \"https://docs.github.com/rest\", \"status\": \"401\"}",
        "metadata": {}
      }
    ]
  }
}
```
