# baseline-setc (2026-11-06T12:45:38)

git b543bce + working tree ff3ca5b81de8, host `rtx-5090_ryzen-9-9950x_121g`, driver 595.99.02, correctness pass

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode | 3/3 | **34.00** | 29.4 | 33.70 | 34.70 | 2.9% | 39.60 | 17.30 |
| prefill128 | 3/3 | **11.00** | 90.9 | 10.80 | 11.10 | 2.7% | 26.50 | 16.56 |
| long16k | 3/3 | **38.80** | 25.8 | 37.90 | 38.90 | 2.6% | 18.40 | 12.15 |
| long16k-prefill | 3/3 | **18.27** | 54.7 | 18.27 | 18.97 | 3.8% | 18.40 | 12.15 |
| serve1 | 3/3 | **40.98** | 24.4 | 40.82 | 42.19 | 3.3% | 0.00 | nan |
| serve3 | 3/3 | **99.01** | 10.1 | 98.04 | 99.01 | 1.0% | 0.00 | nan |

score (geomean of the case medians, ms/token): **34.621**
