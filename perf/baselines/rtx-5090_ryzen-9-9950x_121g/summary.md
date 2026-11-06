# baseline (2026-11-05T08:15:40)

git f359604 + working tree aa8bd76de618, host `rtx-5090_ryzen-9-9950x_121g`, driver 595.99.02, correctness pass

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode | 3/3 | **32.00** | 31.2 | 31.40 | 33.20 | 5.6% | 36.60 | 14.83 |
| prefill64 | 3/3 | **11.90** | 84.0 | 11.90 | 12.00 | 0.8% | 22.40 | 14.74 |

score (geomean of the case medians, ms/token): **19.514**
