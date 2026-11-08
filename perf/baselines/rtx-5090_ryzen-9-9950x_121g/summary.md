# setd-launch (2026-11-07T20:53:38)

git fd2473e, host `rtx-5090_ryzen-9-9950x_121g`, driver 595.99.02, correctness pass
experiment env: `QX_RAW_LAUNCH=1`

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode256k | 3/3 | **44.90** | 22.3 | 44.00 | 45.30 | 2.9% | 124.80 | 0.01 |
| prefill256k | 3/3 | **5.60** | 178.6 | 5.60 | 5.70 | 1.8% | 4.30 | 0.00 |
| serve256k1 | 3/3 | **43.61** | 22.9 | 42.35 | 44.54 | 5.0% | 0.00 | nan |
| serve256k3 | 3/3 | **109.65** | 9.1 | 109.41 | 109.65 | 0.2% | 0.00 | nan |
| decode | 3/3 | **34.70** | 28.8 | 33.60 | 34.90 | 3.7% | 40.60 | 17.51 |
| prefill128 | 3/3 | **10.80** | 92.6 | 10.60 | 10.90 | 2.8% | 27.90 | 16.68 |
| serve1 | 3/3 | **38.54** | 25.9 | 37.29 | 39.40 | 5.5% | 0.00 | nan |
| serve3 | 3/3 | **96.90** | 10.3 | 96.62 | 97.37 | 0.8% | 0.00 | nan |

score (geomean of the case medians, ms/token): **30.210**
