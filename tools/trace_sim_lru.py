import numpy as np, json, sys, collections
root='/home/eric/git/QuixiInfer/'
raw=np.fromfile(root+'golden/data/route-dsv41.trace',dtype=np.uint16)
L=40
K=6
for k in []:
    if raw.size%(1+k)==0 and (raw.size//(1+k))%L==0: K=k;break
rows=raw.reshape(-1,1+K); T=rows.shape[0]//L
print('tokens',T,'k',K)
lay=rows[:,0].astype(np.int64); ids=rows[:,1:].astype(np.int64)
NE=384
g=(lay[:,None]*NE+ids).reshape(T,L*K)   # per token, 240 global ids
# golden ranking
rank=[]
for line in open(root+'golden/profile-dsv41.txt'):
    a,b=line.split(); rank.append(int(a)*NE+int(b))
rank=np.array(rank); assert len(rank)==L*NE
pos=np.empty(L*NE,np.int64); pos[rank]=np.arange(L*NE)
# measure on the last 60% of the trace (the ranking was built on wikitext too, so this is in-distribution)
start=int(T*0.4)
sel=g[start:].ravel()
p=pos[sel]
print('selections',sel.size)
for n in [1000,1270,1609,2000,2300,2500,3000,3500,4000,5000]:
    print(f'VRAM top-{n}: hit {np.mean(p<n)*100:.1f}%')
for ram in [10105+1609, 9591+1270, 11000+1609, 12000+1609]:
    print(f'RAM+VRAM top-{ram}: NVMe share {np.mean(p>=ram)*100:.2f}%')
# reuse of host-served records (not in VRAM top-1609): previous-token same-layer repeat, and LRU of N records
V=1609
tok=g[start:]
prev=set(); rep=0; tot=0
for t in range(tok.shape[0]):
    cur=set(int(x) for x in tok[t] if pos[x]>=V)
    tot+=len(cur); rep+=len(cur&prev); prev=cur
print(f'host-served per token {tot/tok.shape[0]:.1f}; repeat of previous token {rep/tot*100:.1f}%')
# LRU over host-served records, capacities
for cap in [128,256,512,1024,2048]:
    lru=collections.OrderedDict(); hits=0; tot=0
    for t in range(tok.shape[0]):
        for x in tok[t]:
            x=int(x)
            if pos[x]<V: continue
            tot+=1
            if x in lru: hits+=1; lru.move_to_end(x)
            else:
                lru[x]=1
                if len(lru)>cap: lru.popitem(last=False)
    print(f'LRU {cap} records over host-served: hit {hits/tot*100:.1f}%  (bytes {cap*9.5/1024:.1f} GiB)')
# a 3-stream serve-like mix: interleave three disjoint segments of the trace token by token
segs=[tok[i*(tok.shape[0]//3):(i+1)*(tok.shape[0]//3)] for i in range(3)]
n=min(s.shape[0] for s in segs)
for cap in [256,512,1024]:
    lru=collections.OrderedDict(); hits=0; tot=0
    for t in range(n):
        for s in segs:
            for x in s[t]:
                x=int(x)
                if pos[x]<1270: continue
                tot+=1
                if x in lru: hits+=1; lru.move_to_end(x)
                else:
                    lru[x]=1
                    if len(lru)>cap: lru.popitem(last=False)
    print(f'3-stream LRU {cap} (VRAM 1270): hit {hits/tot*100:.1f}%, host-served per step {tot/n:.1f}')
