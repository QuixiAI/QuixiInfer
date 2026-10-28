//! The expert ranking the VRAM cache starts from. The swapper saves its learned ranking under ~/.cache/quixiinfer
//! (keyed by the model file), so a start after the first fills the cache with the experts the last run used.
use qx_pack::gguf::Gguf;
use std::path::PathBuf;

/// The saved ranking's path for this model, if HOME is set.
pub fn profile_path(g: &Gguf) -> Option<PathBuf> {
    let p = &g.shards[0].path;
    let meta = std::fs::metadata(p).ok()?;
    let stamp = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    Some(PathBuf::from(std::env::var("HOME").ok()?).join(format!(".cache/quixiinfer/routing-{}-{stamp}.txt", meta.len())))
}

/// Reads a ranking file: one `layer expert ...` line per expert, hottest first.
pub fn read_order(path: &std::path::Path) -> std::io::Result<Vec<(usize, usize)>> {
    Ok(parse(&std::fs::read_to_string(path)?))
}

fn parse(text: &str) -> Vec<(usize, usize)> {
    text.lines()
        .filter_map(|l| {
            let mut v = l.split_whitespace().map(|x| x.parse::<usize>());
            Some((v.next()?.ok()?, v.next()?.ok()?))
        })
        .collect()
}

/// The saved ranking when there is one, else the pack's, else the calibration ranking, else expert-major order.
pub fn default_order(g: &Gguf, n_layer: usize, n_expert: usize) -> Vec<(usize, usize)> {
    let n = n_layer * n_expert;
    if let Some(order) = profile_path(g).and_then(|p| read_order(&p).ok()).filter(|o| o.len() == n) {
        eprintln!("expert cache: starting from the saved routing profile");
        return order;
    }
    if let Some(Ok(p)) = qx_pack::pack::PackInfo::of(g) {
        if let Some(r) = p.routing.filter(|r| r.len() == n) {
            return r.iter().map(|&g| (g as usize / n_expert, g as usize % n_expert)).collect();
        }
    }
    let calib = parse(qx_pack::pack::CALIB_ROUTING);
    if calib.len() == n {
        return calib;
    }
    (0..n_expert).flat_map(|e| (0..n_layer).map(move |l| (l, e))).collect()
}
