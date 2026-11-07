# baseline-setd (2026-11-07T00:17:10)

git 642433a + working tree 2f64ec1c5967, host `rtx-5090_ryzen-9-9950x_121g`, driver 595.99.02, correctness pass

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode256k | 3/3 | **52.50** | 19.0 | 52.10 | 52.80 | 1.3% | 107.40 | 0.01 |
| prefill256k | 3/3 | **21.50** | 46.5 | 21.40 | 22.30 | 4.2% | 11.50 | 0.46 |
| serve256k1 | 3/3 | **42.50** | 23.5 | 41.53 | 44.44 | 6.8% | 0.00 | nan |
| serve256k3 | 3/3 | **108.11** | 9.2 | 106.95 | 109.17 | 2.1% | 0.00 | nan |
| decode | 3/3 | **34.20** | 29.2 | 33.80 | 36.00 | 6.4% | 38.10 | 17.47 |
| prefill128 | 3/3 | **11.00** | 90.9 | 11.00 | 11.20 | 1.8% | 25.40 | 15.85 |
| serve1 | 3/3 | **40.08** | 25.0 | 38.51 | 40.50 | 5.0% | 0.00 | nan |
| serve3 | 3/3 | **97.09** | 10.3 | 96.62 | 97.37 | 0.8% | 0.00 | nan |

score (geomean of the case medians, ms/token): **49.601**
