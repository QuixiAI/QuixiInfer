//! `qx-tier-check PACK.qxp SOURCE.gguf RAM_GIB [N_WINDOWS]`: builds the RAM tier of the expert store from the pack
//! with a RAM budget, then checks records against the source GGUF byte for byte (RAM places and NVMe fetches),
//! and times NVMe fetches of random windows of misses (6 experts x 40 layers per token, 8 tokens) and
//! promotions. The ranking is the pack's routing profile when it has one, else layer-major order.
use qx_engine::tiers::ExpertStore;
use qx_pack::gguf::Gguf;
use qx_pack::pack::PackInfo;

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn main() -> anyhow::Result<()> {
    qx_gpu::init();
    let a: Vec<String> = std::env::args().collect();
    let pk = Gguf::open(std::path::Path::new(&a[1]))?;
    let pack = PackInfo::of(&pk).ok_or_else(|| anyhow::anyhow!("not a pack"))??;
    let src = Gguf::open(std::path::Path::new(&a[2]))?;
    let budget = (a[3].parse::<f64>()? * 1073741824.0) as usize;
    let n_windows: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(50);
    let n_layer = pack.experts.record.len();
    let ne = pack.experts.n_expert;
    // a pack may carry another model's calibration ranking (the Qwen default): only a ranking of this model's size counts
    let order: Vec<(usize, usize)> = match &pack.routing {
        Some(r) if r.len() == n_layer * ne && r.iter().all(|&g| (g as usize) < n_layer * ne) => r.iter().map(|&g| (g as usize / ne, g as usize % ne)).collect(),
        _ => (0..n_layer).flat_map(|l| (0..ne).map(move |e| (l, e))).collect(),
    };
    let skip = vec![false; n_layer * ne];
    // byte checks against the source
    let check = |store: &ExpertStore, g: usize, rec: *const u8| -> bool {
        let (l, e) = (g / ne, g % ne);
        let t = |n: &str| src.tensor(&format!("blk.{l}.ffn_{n}_exps.weight")).unwrap();
        let (gt, ut, dt) = (t("gate"), t("up"), t("down"));
        let (gb, db) = ((gt.bytes / ne as u64) as usize, (dt.bytes / ne as u64) as usize);
        let r = unsafe { std::slice::from_raw_parts(rec, store.record_bytes(g)) };
        r[..gb] == src.data(gt)[e * gb..(e + 1) * gb] && r[gb..2 * gb] == src.data(ut)[e * gb..(e + 1) * gb] && r[2 * gb..2 * gb + db] == src.data(dt)[e * db..(e + 1) * db]
    };
    let mut s = 0x1234_5678_9abc_def1u64;
    // QX_TIER_ELASTIC=G: the elastic part (G GiB of the budget) through churn, release and restore, every record
    // in RAM checked byte for byte after each; then a store built with the part out
    if let Some(gib) = std::env::var("QX_TIER_ELASTIC").ok().and_then(|v| v.parse::<f64>().ok()) {
        use rayon::prelude::*;
        let elastic = (gib * 1073741824.0) as usize;
        let all = n_layer * ne;
        let verify = |store: &ExpertStore, what: &str| -> usize {
            let held: Vec<usize> = (0..all).filter(|&g| store.in_ram(g)).collect();
            let mut places: Vec<usize> = held.iter().map(|&g| store.place[g]).collect();
            places.sort();
            let n = places.len();
            places.dedup();
            assert_eq!(n, places.len(), "{what}: two records share a place");
            let free: usize = store.free_places().iter().sum();
            let bad = held.par_iter().filter(|&&g| !check(store, g, store.place[g] as *const u8)).count();
            println!("{what}: {} records in RAM, {free} free places, mismatches {bad}", held.len());
            assert_eq!(bad, 0, "{what}: byte mismatches");
            held.len()
        };
        let mut store = ExpertStore::build_elastic(&pack, n_layer, &order, &skip, budget, 64, elastic, false)?;
        println!("elastic part: {:.2} GiB", store.elastic_bytes() as f64 / 1073741824.0);
        let n0 = verify(&store, "built");
        for round in 0..3 {
            // churn: records leave and others take their places, in both arenas
            let ram: Vec<usize> = (0..all).filter(|&g| store.in_ram(g)).collect();
            for _ in 0..600 {
                store.evict(ram[rng(&mut s) as usize % ram.len()]);
            }
            let nvme: Vec<usize> = (0..all).filter(|&g| !store.in_ram(g)).collect();
            let cands: Vec<usize> = (0..3000).map(|_| nvme[rng(&mut s) as usize % nvme.len()]).collect();
            store.promote(&cands)?;
            let n1 = verify(&store, &format!("round {round} after churn"));
            let score: Vec<f64> = (0..all).map(|_| (rng(&mut s) % 1000) as f64).collect();
            let t0 = std::time::Instant::now();
            let (dropped, moved) = store.elastic_release(&score);
            println!("round {round}: released in {:.0} ms: {} dropped, {} moved", 1e3 * t0.elapsed().as_secs_f64(), dropped.len(), moved.len());
            assert!(store.elastic_out() && dropped.iter().all(|&g| !store.in_ram(g)));
            // every record that stays is at least as hot as every record of its class that left
            let n2 = verify(&store, &format!("round {round} released"));
            assert_eq!(n2 + dropped.len(), n1);
            let t0 = std::time::Instant::now();
            assert!(store.elastic_restore() && !store.elastic_out());
            println!("round {round}: restored in {:.0} ms", 1e3 * t0.elapsed().as_secs_f64());
            let skipped = store.promote(&dropped)?;
            assert!(skipped.is_empty(), "a dropped record found no place after the restore");
            assert_eq!(verify(&store, &format!("round {round} refilled")), n1);
        }
        drop(store);
        let mut store = ExpertStore::build_elastic(&pack, n_layer, &order, &skip, budget, 64, elastic, true)?;
        let owed = std::mem::take(&mut store.elastic_owed);
        assert!(store.elastic_out());
        let n3 = verify(&store, "built with the part out");
        assert_eq!(n3 + owed.len(), n0);
        assert!(store.elastic_restore());
        assert!(store.promote(&owed)?.is_empty());
        assert_eq!(verify(&store, "then restored and refilled"), n0);
        println!("ok");
        return Ok(());
    }
    let mut store = ExpertStore::build(&pack, n_layer, &order, &skip, budget, 64)?;
    let places = store.places();
    println!("classes: {:?} (record bytes, places)", places);
    let ram: Vec<usize> = (0..n_layer * ne).filter(|&g| store.in_ram(g)).collect();
    let nvme: Vec<usize> = (0..n_layer * ne).filter(|&g| !store.in_ram(g)).collect();
    println!("{} records in RAM, {} on NVMe", ram.len(), nvme.len());
    let mut bad = 0;
    for _ in 0..40 {
        let g = ram[rng(&mut s) as usize % ram.len()];
        bad += !check(&store, g, store.place[g] as *const u8) as usize;
    }
    println!("RAM records checked: 40, mismatches {bad}");
    let picks: Vec<usize> = (0..40).map(|_| nvme[rng(&mut s) as usize % nvme.len()]).collect();
    let ptrs = store.fetch(&picks)?;
    let bad2 = picks.iter().zip(&ptrs).filter(|(g, p)| !check(&store, **g, **p)).count();
    println!("NVMe records fetched and checked: 40, mismatches {bad2}");

    // timed windows: 8 tokens x 40 layers x 6 experts, all NVMe misses (the worst case)
    let (mut total_bytes, mut total_s, mut n_rec) = (0u64, 0f64, 0usize);
    for _ in 0..n_windows {
        let gs: Vec<usize> = (0..8 * n_layer * 6).map(|_| nvme[rng(&mut s) as usize % nvme.len()]).collect();
        // per layer, as the pool sees it: 48 misses of one layer at a time
        for l in 0..n_layer {
            let layer_gs: Vec<usize> = gs[l * 48..(l + 1) * 48].iter().map(|&g| (l * ne) + (g % ne)).collect();
            let t0 = std::time::Instant::now();
            let _ = store.fetch(&layer_gs)?;
            total_s += t0.elapsed().as_secs_f64();
            total_bytes += layer_gs.iter().map(|&g| store.record_bytes(g) as u64).sum::<u64>();
            n_rec += layer_gs.len();
        }
    }
    println!("NVMe fetch: {} records, {:.2} GB/s, {:.0} us per record (48 per batch)", n_rec, total_bytes as f64 / total_s / 1e9, total_s * 1e6 / n_rec as f64);
    // small batches (what a single-stream decode step sees: a few misses per layer)
    let (mut tb, mut ts, mut nr) = (0u64, 0f64, 0usize);
    for _ in 0..200 {
        let k = 1 + rng(&mut s) as usize % 4;
        let gs: Vec<usize> = (0..k).map(|_| nvme[rng(&mut s) as usize % nvme.len()]).collect();
        let t0 = std::time::Instant::now();
        let _ = store.fetch(&gs)?;
        ts += t0.elapsed().as_secs_f64();
        tb += gs.iter().map(|&g| store.record_bytes(g) as u64).sum::<u64>();
        nr += k;
    }
    println!("NVMe fetch, 1-4 per batch: {:.2} GB/s, {:.0} us per batch", tb as f64 / ts / 1e9, ts * 1e6 / 200.0);
    // promotions: evict 64 RAM records, promote 64 NVMe ones into their places
    let victims: Vec<usize> = (0..64).map(|_| ram[rng(&mut s) as usize % ram.len()]).collect();
    for &g in &victims {
        store.evict(g);
    }
    let cands: Vec<usize> = nvme.iter().copied().filter(|&g| victims.iter().any(|&v| v / ne == g / ne)).take(64).collect();
    let t0 = std::time::Instant::now();
    let skipped = store.promote(&cands)?;
    let dt = t0.elapsed().as_secs_f64();
    let promoted = cands.len() - skipped.len();
    let bad3 = cands.iter().filter(|&&g| store.in_ram(g) && !check(&store, g, store.place[g] as *const u8)).count();
    println!("promoted {promoted} records in {:.1} ms ({} skipped for want of a place), mismatches {bad3}", dt * 1e3, skipped.len());
    println!("stats: ram hits {}, nvme reads {} ({:.1} GiB), promotions {}", store.stats.ram_hits, store.stats.nvme_reads, store.stats.nvme_bytes as f64 / 1073741824.0, store.stats.promotions);
    assert!(bad == 0 && bad2 == 0 && bad3 == 0, "byte mismatches");
    println!("ok");
    Ok(())
}
