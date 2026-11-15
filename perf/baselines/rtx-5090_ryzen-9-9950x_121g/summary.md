# setd-promo39 (2026-11-15T00:36:32)

git e197dcc + working tree ae76e12e42ee, host `rtx-5090_ryzen-9-9950x_121g`, driver 620.70.06, correctness skipped

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode256k | 2/2 | **39.00** | 25.6 | 38.90 | 39.10 | 0.5% | 64.30 | 0.01 |
| prefill256k | 1/1 | **2.00** | 500.0 | 2.00 | 2.00 | 0.0% | 1.60 | 0.00 |
| serve256k1 | 1/1 | **32.98** | 30.3 | 32.98 | 32.98 | 0.0% | 0.00 | nan |
| serve256k3 | 1/1 | **80.84** | 12.4 | 80.84 | 80.84 | 0.0% | 0.00 | nan |
| decode | 1/1 | **31.00** | 32.3 | 31.00 | 31.00 | 0.0% | 33.00 | 13.34 |
| prefill128 | 1/1 | **10.50** | 95.2 | 10.50 | 10.50 | 0.0% | 20.00 | 13.89 |

score (geomean of the case medians, ms/token): **18.475**
