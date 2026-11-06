# Performance

Start with the handbook, [`perf.md`](perf.md). It covers how to measure, decide and record a change.

| file | purpose |
|---|---|
| [`findings.md`](findings.md) | what is already known; read before proposing an idea |
| [`backlog.md`](backlog.md) | what to try next, with targets and kill criteria |
| [`optimization_status.md`](optimization_status.md) | the append-only notebook of every pass and experiment |
| [`scoreboard.md`](scoreboard.md) | the one number: geomean ms per token of the standard cases |
| `harness/bench.sh` | the standard benchmark |
| `harness/perf_diff.py` | compare, gate and record |
| `baselines/<host>/` | the committed baseline per host fingerprint |
| `results/` | raw runs; git-ignored |

The quick path:

```bash
perf/harness/bench.sh --label <what> --check            # about 20-25 min, alone on the box
python3 perf/harness/perf_diff.py gate perf/results/<date>/<run>
```
