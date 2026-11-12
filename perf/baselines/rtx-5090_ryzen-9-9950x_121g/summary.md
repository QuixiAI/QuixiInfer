# setd-promo13 (2026-11-12T16:19:09)

git 967476e + working tree 955828554b0f, host `rtx-5090_ryzen-9-9950x_121g`, driver 620.70.06, correctness skipped

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode256k | 2/2 | **38.55** | 25.9 | 38.50 | 38.60 | 0.3% | 64.10 | 0.01 |
| prefill256k | 1/1 | **2.00** | 500.0 | 2.00 | 2.00 | 0.0% | 1.60 | 0.00 |
| serve256k1 | 1/1 | **38.94** | 25.7 | 38.94 | 38.94 | 0.0% | 0.00 | nan |
| serve256k3 | 1/1 | **91.32** | 11.0 | 91.32 | 91.32 | 0.0% | 0.00 | nan |
| decode | 1/1 | **30.70** | 32.6 | 30.70 | 30.70 | 0.0% | 32.70 | 13.09 |
| prefill128 | 1/1 | **10.60** | 94.3 | 10.60 | 10.60 | 0.0% | 20.10 | 13.89 |

score (geomean of the case medians, ms/token): **19.166**
