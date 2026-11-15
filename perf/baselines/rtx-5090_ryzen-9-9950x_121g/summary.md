# setd-promo35 (2026-11-14T21:12:20)

git 3d05ab2 + working tree 443232e7902c, host `rtx-5090_ryzen-9-9950x_121g`, driver 620.70.06, correctness skipped

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode256k | 2/2 | **38.60** | 25.9 | 38.50 | 38.70 | 0.5% | 64.25 | 0.01 |
| prefill256k | 1/1 | **2.00** | 500.0 | 2.00 | 2.00 | 0.0% | 1.50 | 0.00 |
| serve256k1 | 1/1 | **34.14** | 29.3 | 34.14 | 34.14 | 0.0% | 0.00 | nan |
| serve256k3 | 1/1 | **83.89** | 11.9 | 83.89 | 83.89 | 0.0% | 0.00 | nan |
| decode | 1/1 | **30.90** | 32.4 | 30.90 | 30.90 | 0.0% | 32.90 | 13.24 |
| prefill128 | 1/1 | **10.50** | 95.2 | 10.50 | 10.50 | 0.0% | 20.00 | 13.90 |

score (geomean of the case medians, ms/token): **18.640**
