# Baselines

There is one directory per host fingerprint: `perf_diff.py fingerprint <run>`. Each holds the `run.json`,
`results.jsonl` and `summary.md` of the run that `gate` compares candidates against.

Only `perf_diff.py promote` writes here. It refuses a run whose correctness check did not pass. Promote after
every landed win, so the next experiment's baseline includes it.
