"""Prints a GGUF model's tensor layout from the shard headers alone (the first few MiB of each file suffice).

    python3 -I tools/gguf_layout.py SHARD.gguf...            summary by tensor kind and type
    python3 -I tools/gguf_layout.py --experts OUT.json SHARD.gguf...
        also writes, per layer, the bytes of one routed expert's gate, up and down matrices (for cache_sim3.py)

Headers only: no tensor data is read, so partially downloaded shards work.
"""
import collections
import json
import re
import struct
import sys

TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K',
         12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S',
         20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64',
         29: 'IQ1_M', 30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4'}
# (bytes per block, elements per block)
BLK = {'F32': (4, 1), 'F16': (2, 1), 'BF16': (2, 1), 'F64': (8, 1), 'Q8_0': (34, 32), 'Q4_0': (18, 32), 'Q4_1': (20, 32),
       'Q5_0': (22, 32), 'Q5_1': (24, 32), 'Q4_K': (144, 256), 'Q5_K': (176, 256), 'Q6_K': (210, 256),
       'Q2_K': (84, 256), 'Q3_K': (110, 256), 'IQ2_XXS': (66, 256), 'IQ2_XS': (74, 256), 'IQ3_XXS': (98, 256),
       'IQ1_S': (50, 256), 'IQ1_M': (56, 256), 'IQ4_NL': (18, 32), 'IQ3_S': (110, 256), 'IQ2_S': (82, 256),
       'IQ4_XS': (136, 256), 'MXFP4': (17, 32), 'I8': (1, 1), 'I16': (2, 1), 'I32': (4, 1), 'I64': (8, 1)}
SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<B', 10: '<Q', 11: '<q', 12: '<d'}


class Reader:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.buf = b''

    def take(self, n):
        while len(self.buf) < n:
            more = self.f.read(max(n - len(self.buf), 1 << 20))
            if not more:
                raise EOFError
            self.buf += more
        v, self.buf = self.buf[:n], self.buf[n:]
        return v

    def scalar(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.scalar('<Q')
        return self.take(n).decode('utf-8', 'replace')

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et = self.scalar('<I')
            n = self.scalar('<Q')
            return [self.value(et) for _ in range(n)]
        return self.scalar(SCALAR[t])


def read_header(path):
    r = Reader(path)
    magic, version, n_tensors, n_kv = r.scalar('<I'), r.scalar('<I'), r.scalar('<Q'), r.scalar('<Q')
    assert magic == 0x46554747, f'{path}: not a GGUF file'
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.scalar('<I')
        kv[k] = r.value(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.scalar('<I')
        dims = [r.scalar('<Q') for _ in range(nd)]
        ty = TYPES.get(r.scalar('<I'), '?')
        r.scalar('<Q')  # offset
        n = 1
        for d in dims:
            n *= d
        bb, bs = BLK[ty]
        tensors.append((name, dims, ty, n // bs * bb))
    return kv, tensors


args = sys.argv[1:]
experts_out = None
if args and args[0] == '--experts':
    experts_out, args = args[1], args[2:]
by_kind = collections.Counter()
per_layer = collections.defaultdict(dict)
total = 0
for path in args:
    kv, tensors = read_header(path)
    for k, v in kv.items():
        if k.startswith('general.architecture') or k.endswith('block_count') or k.endswith('expert_count') \
                or k.endswith('expert_used_count'):
            print(f'{k} = {v}')
    for name, dims, ty, nbytes in tensors:
        total += nbytes
        by_kind[(re.sub(r'\d+', '#', name), ty)] += nbytes
        m = re.match(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight', name)
        if m:
            layer, part = int(m.group(1)), m.group(2)
            per_layer[layer][part] = {'type': ty, 'bytes_per_expert': nbytes // dims[-1], 'n_expert': dims[-1]}
print()
print('| GiB | type | tensor kind |')
print('|---|---|---|')
for (kind, ty), v in sorted(by_kind.items(), key=lambda x: -x[1]):
    if v >= 1 << 20:
        print(f'| {v / 2 ** 30:.2f} | {ty} | {kind} |')
print(f'total {total / 2 ** 30:.1f} GiB')
if experts_out:
    layers = sorted(per_layer)
    out = {'layers': [{'layer': l, **per_layer[l]} for l in layers]}
    json.dump(out, open(experts_out, 'w'), indent=1)
    rec = [sum(p['bytes_per_expert'] for p in per_layer[l].values()) for l in layers]
    print(f'{len(layers)} layers with routed experts; record {min(rec) / 2 ** 20:.2f}-{max(rec) / 2 ** 20:.2f} MiB, '
          f'mean {sum(rec) / len(rec) / 2 ** 20:.2f}; wrote {experts_out}')
