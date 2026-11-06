# baseline-setb-ram95 (2026-11-06T09:05:19)

git 4e782fb + working tree bf81ddfd8f0f, host `rtx-5090_ryzen-9-9950x_121g`, driver 595.99.02, correctness pass

| case | ok | median ms/token | tok/s | min | max | spread | mean over all tokens | pool ms/token (all) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| decode | 3/3 | **33.00** | 30.3 | 32.20 | 33.20 | 3.0% | 37.40 | 15.86 |
| prefill64 | 3/3 | **12.40** | 80.6 | 12.10 | 12.40 | 2.4% | 23.20 | 15.02 |
| long16k | 3/3 | **37.30** | 26.8 | 36.30 | 38.00 | 4.6% | 21.90 | 16.71 |
| long16k-prefill | 3/3 | **21.77** | 45.9 | 20.19 | 22.37 | 10.0% | 21.90 | 16.71 |
| serve1 | 3/3 | **39.53** | 25.3 | 38.31 | 39.53 | 3.1% | 0.00 | nan |
| serve3 | 3/3 | **90.91** | 11.0 | 90.91 | 90.91 | 0.0% | 0.00 | nan |

score (geomean of the case medians, ms/token): **34.321**
