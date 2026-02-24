# Continuous Worker Cutover

## Feature flags

Set in `apps/text-analysis/.env`:

```bash
CONTINUOUS_WORKER_ENABLED=true
CONTINUOUS_RING_CAPACITY_SEC=20
CONTINUOUS_WINDOW_SEC=5
CONTINUOUS_HOP_SEC=1
CONTINUOUS_TICK_SEC=1
RESULT_DEDUPE_TTL_SEC=6
```

## Canary rollout procedure

1. Deploy one instance with `CONTINUOUS_WORKER_ENABLED=true`.
2. Keep at least one control instance with `CONTINUOUS_WORKER_ENABLED=false`.
3. Route a small subset of meetings to canary (suggested 10%).
4. Observe for at least 30 minutes under active calls.
5. Increase gradually to 25%, 50%, then 100% only if all gates pass.

## Promotion gates

Promote only if all conditions below hold:

- No schema regression in `text_analysis_result`.
- No duplicate feedback bursts caused by overlapping windows.
- `cycle_total_ms` stays below `CONTINUOUS_TICK_SEC` in steady state.
- No worker/task leak after disconnects and service restart.
- No increase in transcription error logs versus control group.

## Rollback gates

Rollback immediately if any condition below is met:

- Repeated duplicated results for same participant/time window.
- Sustained loop saturation (`cycle_total_ms` > tick for consecutive cycles).
- Drop in semantic signal quality observed in live feedback.
- Unbounded growth of active stream workers.
- Crash loop or repeated failures in deep queue + continuous flow.

## Immediate rollback

Set:

```bash
CONTINUOUS_WORKER_ENABLED=false
```

Then restart the Python service.

