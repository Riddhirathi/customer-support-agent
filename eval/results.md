# Eval Results

Ran 15 cases from `eval/cases.jsonl` against the live pipeline.

| Metric | Value |
|---|---|
| Routing accuracy | 100% |
| **False-respond rate** (answered when it should have escalated) | 0% (0/7) |
| Retrieval hit@3 | 100% |
| Category accuracy | 100% |
| Groundedness rate (of responded cases) | 100% (8 responded) |
| JSON retry rate | 0% |
| Latency p50 | 1390 ms |
| Latency p95 | 4032 ms |

## Failed cases (routing mismatch)

None — every case routed to the expected action.
