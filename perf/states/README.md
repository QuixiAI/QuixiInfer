# Saved sequence states

A state is what `park` produces: the tokens and the used rows of the compressed stores (about 1.3 GB at 256k).
`qx-ds-check` writes one with `QX_DS_SAVE_STATE=perf/states/NAME QX_DS_SAVE_AT=POS[,POS...]` during a pass and
resumes from one with `QX_DS_LOAD_STATE=perf/states/NAME` (the ring is rebuilt by replaying the last window: bit
for bit the straight pass). The files are git-ignored; regenerate them with one full pass of
`perf/data/wiki256k.tokens` (about an hour):

```bash
QX_DS_PACK=... QX_DS_VRAM=20 QX_DS_RAM=100 QX_DS_BENCH=0 QX_DS_PROMOS=32 QX_DS_WINDOW=128 QX_DS_TAIL=64 \
  QX_DS_SAVE_STATE=perf/states/wiki256k QX_DS_SAVE_AT=245760,262144 \
  target/release/qx-ds-check <model> perf/data/wiki256k.tokens 262208
```

That writes `wiki256k-245760.state` and `wiki256k-262144.state`, the inputs of the `prefill256k` and `decode256k`
cases.
