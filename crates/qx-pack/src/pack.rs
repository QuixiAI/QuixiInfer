//! QuixiInfer's pack (`.qxp`): one GGUF v3 file the engine loads without repacking anything.
//!
//! - Header: the source's metadata (less `split.*`) plus the `quixi.*` keys, then the tensor index. The routed
//!   experts and the PLE table keep their index entries (shape and type) with offset `PACKED`: their bytes live
//!   in sections of their own.
//! - `dense`: every other tensor byte for byte, each 4 KiB-aligned (`general.alignment` 4096).
//! - `experts` (2 MiB-aligned): per layer, per expert, one [gate | up | down | tail] record padded to 4 KiB. This
//!   is the layout of the engine's pinned host memory, so loading the experts is one O_DIRECT read. The tail
//!   (`tail_bytes`, 16 for NVFP4 experts, else 0) holds the expert's per-tensor scales {gate, up, down, 0} as f32
//!   (ModelOpt's `weight_scale_2`, the GGUF's `ffn_*_exps.scale` tensors; 1 when absent).
//! - `ple` (2 MiB-aligned): the PLE table in 4 KiB pages of `rows_per_page` whole rows, so a row is one page.
//!
//! Tensors are copied byte for byte; only the layout changes. Each section's checksum (`quixi.sha256.*`) is the
//! SHA-256 of the SHA-256s of its 64 MiB pieces, so writing and checking both hash pieces in parallel.
use crate::gguf::{Gguf, TensorInfo};
use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};

pub const PACKED: u64 = u64::MAX;
pub const VERSION: u64 = 1;
pub const SECTION_ALIGN: u64 = 2 << 20;
pub const PAGE: usize = 4096;
pub const PIECE: usize = 64 << 20;
pub const SECTIONS: [&str; 3] = ["dense", "experts", "ple"];
pub const PLE_TENSOR: &str = "per_layer_token_embd.weight";

/// Routing counts on wikitext calibration text, hottest first: `layer expert` lines (`qx-profile`).
pub const CALIB_ROUTING: &str = include_str!("../routing-calib.txt");

/// (layer, part) of a routed-expert tensor: part 0 gate, 1 up, 2 down.
fn expert_part(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("blk.")?;
    let (l, t) = rest.split_once('.')?;
    let part = match t {
        "ffn_gate_exps.weight" => 0,
        "ffn_up_exps.weight" => 1,
        "ffn_down_exps.weight" => 2,
        _ => return None,
    };
    Some((l.parse().ok()?, part))
}

fn is_sectioned(name: &str) -> bool {
    name == PLE_TENSOR || expert_part(name).is_some()
}

/// Where each layer's expert records sit in the `experts` section.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertLayout {
    pub n_expert: usize,
    pub layer_offset: Vec<u64>,
    pub record: Vec<u64>,
    /// Bytes of one expert's gate (= up) rows and down rows, per layer.
    pub gate: Vec<u64>,
    pub down: Vec<u64>,
    /// Bytes of the scale tail after the down rows, per layer (0 or 16).
    pub tail: Vec<u64>,
    pub bytes: u64,
}

/// The per-expert scales of layer `l` as the record tail stores them: {gate, up, down, 0} per expert, 1 where
/// the GGUF has no `blk.l.ffn_*_exps.scale` tensor.
pub fn expert_scales(g: &Gguf, l: usize, n_expert: usize) -> Result<Vec<[f32; 4]>> {
    let mut out = vec![[1f32, 1.0, 1.0, 0.0]; n_expert];
    for (p, n) in ["gate", "up", "down"].iter().enumerate() {
        let Some(t) = g.tensor(&format!("blk.{l}.ffn_{n}_exps.scale")) else { continue };
        if t.ty != crate::ggml::GgmlType::F32 || t.elements() as usize != n_expert {
            bail!("{}: expected F32 [{n_expert}], got {:?} {:?}", t.name, t.ty, t.shape);
        }
        for (e, b) in g.data(t).chunks_exact(4).enumerate() {
            out[e][p] = f32::from_le_bytes(b.try_into().unwrap());
        }
    }
    Ok(out)
}

/// The record tail bytes of expert `e` from `expert_scales`.
pub fn tail_bytes(sc: &[f32; 4]) -> [u8; 16] {
    let mut b = [0u8; 16];
    for (i, v) in sc.iter().enumerate() {
        b[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    b
}

impl ExpertLayout {
    /// Every layer's records back to back, each record 4 KiB-aligned; the section ends 2 MiB-aligned.
    pub fn new(n_expert: usize, parts: &[(u64, u64, u64)]) -> Self {
        let mut l = Self { n_expert, layer_offset: vec![], record: vec![], gate: vec![], down: vec![], tail: vec![], bytes: 0 };
        let mut off = 0u64;
        for &(gb, db, tb) in parts {
            let rec = (2 * gb + db + tb).next_multiple_of(PAGE as u64);
            l.layer_offset.push(off);
            l.record.push(rec);
            l.gate.push(gb);
            l.down.push(db);
            l.tail.push(tb);
            off += rec * n_expert as u64;
        }
        l.bytes = off.next_multiple_of(SECTION_ALIGN);
        l
    }

    fn of(g: &Gguf) -> Result<Self> {
        let mut parts: Vec<[Option<&TensorInfo>; 3]> = Vec::new();
        for t in &g.tensors {
            if let Some((l, p)) = expert_part(&t.name) {
                if parts.len() <= l {
                    parts.resize(l + 1, [None; 3]);
                }
                parts[l][p] = Some(t);
            }
        }
        let mut n_expert = 0;
        let mut sizes = Vec::new();
        for (l, p) in parts.iter().enumerate() {
            let [Some(gt), Some(ut), Some(dt)] = p else { bail!("layer {l}: routed-expert tensors missing") };
            let ne = *gt.shape.last().unwrap() as usize;
            if n_expert != 0 && ne != n_expert {
                bail!("layer {l}: {ne} experts, earlier layers {n_expert}");
            }
            n_expert = ne;
            if ut.bytes != gt.bytes || gt.bytes % ne as u64 != 0 || dt.bytes % ne as u64 != 0 {
                bail!("layer {l}: gate, up and down do not split into {ne} experts");
            }
            let tail = if gt.ty == crate::ggml::GgmlType::NVFP4 { 16 } else { 0 };
            sizes.push((gt.bytes / ne as u64, dt.bytes / ne as u64, tail));
        }
        Ok(Self::new(n_expert, &sizes))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PleLayout {
    pub rows: u64,
    pub row_bytes: u64,
    pub rows_per_page: u64,
    pub bytes: u64,
}

impl PleLayout {
    fn of(t: &TensorInfo) -> Self {
        let row_bytes = t.ty.bytes(t.shape[0] as usize) as u64;
        let rows = t.elements() / t.shape[0];
        let rows_per_page = PAGE as u64 / row_bytes;
        Self { rows, row_bytes, rows_per_page, bytes: (rows.div_ceil(rows_per_page) * PAGE as u64).next_multiple_of(SECTION_ALIGN) }
    }
}

/// A pack's sections and checksums, from its header.
#[derive(Clone, Debug)]
pub struct PackInfo {
    pub path: std::path::PathBuf,
    /// Absolute file offsets and lengths of the sections, in `SECTIONS` order.
    pub offset: [u64; 3],
    pub bytes: [u64; 3],
    pub sha256: [String; 3],
    pub experts: ExpertLayout,
    pub ple: PleLayout,
    /// Expert ranking (`layer * n_expert + e`), hottest first.
    pub routing: Option<Vec<u32>>,
}

impl PackInfo {
    /// None for a plain GGUF.
    pub fn of(g: &Gguf) -> Option<Result<Self>> {
        g.get("quixi.pack.version")?;
        Some(Self::parse(g))
    }

    fn parse(g: &Gguf) -> Result<Self> {
        let v = g.u64("quixi.pack.version")?;
        if v != VERSION {
            bail!("pack version {v}, this build reads {VERSION}");
        }
        let arr = |k: &str| -> Result<Vec<u64>> { g.get(k).and_then(|v| v.as_arr()).ok_or_else(|| anyhow!("missing {k}"))?.iter().map(|x| x.as_u64().ok_or_else(|| anyhow!("{k}: not integers"))).collect() };
        let s = |k: &str| -> Result<String> { Ok(g.get(k).and_then(|v| v.as_str()).ok_or_else(|| anyhow!("missing {k}"))?.to_string()) };
        let mut offset = [0; 3];
        let mut bytes = [0; 3];
        let mut sha256: [String; 3] = Default::default();
        for (i, n) in SECTIONS.iter().enumerate() {
            offset[i] = g.u64(&format!("quixi.{n}.offset"))?;
            bytes[i] = g.u64(&format!("quixi.{n}.bytes"))?;
            sha256[i] = s(&format!("quixi.sha256.{n}"))?;
        }
        let ne = g.u64("quixi.experts.n_expert")? as usize;
        let gate = arr("quixi.experts.gate_bytes")?;
        let down = arr("quixi.experts.down_bytes")?;
        let tail = if g.get("quixi.experts.tail_bytes").is_some() { arr("quixi.experts.tail_bytes")? } else { vec![0; gate.len()] };
        let parts: Vec<(u64, u64, u64)> = (0..gate.len()).map(|l| (gate[l], down[l], tail[l])).collect();
        let experts = ExpertLayout::new(ne, &parts);
        if experts.layer_offset != arr("quixi.experts.layer_offset")? || experts.record != arr("quixi.experts.record_bytes")? || experts.bytes != bytes[1] {
            bail!("the pack's expert layout is not the one this build computes");
        }
        let ple = PleLayout { rows: g.u64("quixi.ple.rows")?, row_bytes: g.u64("quixi.ple.row_bytes")?, rows_per_page: g.u64("quixi.ple.rows_per_page")?, bytes: bytes[2] };
        let routing = g.get("quixi.routing").and_then(|v| v.as_arr()).map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect());
        Ok(Self { path: g.shards[0].path.clone(), offset, bytes, sha256, experts, ple, routing })
    }

    /// Byte offset (in the file) of PLE row `r`.
    pub fn ple_row(&self, r: u64) -> u64 {
        self.offset[2] + r / self.ple.rows_per_page * PAGE as u64 + r % self.ple.rows_per_page * self.ple.row_bytes
    }
}

// ---------------------------------------------------------------------------------------------- header

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}
fn kv_u64(b: &mut Vec<u8>, k: &str, v: u64) {
    put_str(b, k);
    b.extend(10u32.to_le_bytes());
    b.extend(v.to_le_bytes());
}
fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
    put_str(b, k);
    b.extend(4u32.to_le_bytes());
    b.extend(v.to_le_bytes());
}
fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
    put_str(b, k);
    b.extend(8u32.to_le_bytes());
    put_str(b, v);
}
fn kv_arr_u64(b: &mut Vec<u8>, k: &str, v: &[u64]) {
    put_str(b, k);
    b.extend(9u32.to_le_bytes());
    b.extend(10u32.to_le_bytes());
    b.extend((v.len() as u64).to_le_bytes());
    v.iter().for_each(|x| b.extend(x.to_le_bytes()));
}
fn kv_arr_u32(b: &mut Vec<u8>, k: &str, v: &[u32]) {
    put_str(b, k);
    b.extend(9u32.to_le_bytes());
    b.extend(4u32.to_le_bytes());
    b.extend((v.len() as u64).to_le_bytes());
    v.iter().for_each(|x| b.extend(x.to_le_bytes()));
}

/// Where everything goes. Header length does not depend on any offset or checksum value (all fixed width),
/// so the header is built once for its length and again with the real values.
struct Plan {
    /// Offset of each dense tensor from the data start (None: in a section).
    tensor_offset: Vec<Option<u64>>,
    data_start: u64,
    offset: [u64; 3],
    bytes: [u64; 3],
    experts: ExpertLayout,
    ple: PleLayout,
}

fn header(g: &Gguf, p: &Plan, sha: &[String; 3], routing: Option<&[u32]>) -> Result<Vec<u8>> {
    let kvs: Vec<_> = g.raw_kvs()?.into_iter().filter(|(k, _)| !k.starts_with("split.") && k != "general.alignment" && !k.starts_with("quixi.")).collect();
    let mut kv = Vec::new();
    let mut n_kv = kvs.len();
    for (k, raw) in &kvs {
        put_str(&mut kv, k);
        kv.extend_from_slice(raw);
    }
    let mut add = |f: &mut dyn FnMut(&mut Vec<u8>)| {
        f(&mut kv);
        n_kv += 1;
    };
    add(&mut |b| kv_u32(b, "general.alignment", PAGE as u32));
    add(&mut |b| kv_u64(b, "quixi.pack.version", VERSION));
    add(&mut |b| kv_str(b, "quixi.source", &g.shards[0].path.file_name().unwrap().to_string_lossy()));
    for (i, n) in SECTIONS.iter().enumerate() {
        add(&mut |b| kv_u64(b, &format!("quixi.{n}.offset"), p.offset[i]));
        add(&mut |b| kv_u64(b, &format!("quixi.{n}.bytes"), p.bytes[i]));
        add(&mut |b| kv_str(b, &format!("quixi.sha256.{n}"), &sha[i]));
    }
    let e = &p.experts;
    add(&mut |b| kv_u64(b, "quixi.experts.n_expert", e.n_expert as u64));
    add(&mut |b| kv_arr_u64(b, "quixi.experts.layer_offset", &e.layer_offset));
    add(&mut |b| kv_arr_u64(b, "quixi.experts.record_bytes", &e.record));
    add(&mut |b| kv_arr_u64(b, "quixi.experts.gate_bytes", &e.gate));
    add(&mut |b| kv_arr_u64(b, "quixi.experts.down_bytes", &e.down));
    add(&mut |b| kv_arr_u64(b, "quixi.experts.tail_bytes", &e.tail));
    add(&mut |b| kv_u64(b, "quixi.ple.rows", p.ple.rows));
    add(&mut |b| kv_u64(b, "quixi.ple.row_bytes", p.ple.row_bytes));
    add(&mut |b| kv_u64(b, "quixi.ple.rows_per_page", p.ple.rows_per_page));
    if let Some(r) = routing {
        add(&mut |b| kv_arr_u32(b, "quixi.routing", r));
    }
    let mut h = Vec::new();
    h.extend(b"GGUF");
    h.extend(3u32.to_le_bytes());
    h.extend((g.tensors.len() as u64).to_le_bytes());
    h.extend((n_kv as u64).to_le_bytes());
    h.extend(kv);
    for (t, off) in g.tensors.iter().zip(&p.tensor_offset) {
        put_str(&mut h, &t.name);
        h.extend((t.shape.len() as u32).to_le_bytes());
        t.shape.iter().for_each(|d| h.extend(d.to_le_bytes()));
        h.extend(t.ty.id().to_le_bytes());
        h.extend(off.unwrap_or(PACKED).to_le_bytes());
    }
    Ok(h)
}

fn plan(g: &Gguf, routing: Option<&[u32]>) -> Result<Plan> {
    let experts = ExpertLayout::of(g)?;
    let ple = PleLayout::of(g.tensor(PLE_TENSOR).ok_or_else(|| anyhow!("no {PLE_TENSOR}"))?);
    let mut cur = 0u64;
    let tensor_offset: Vec<Option<u64>> = g
        .tensors
        .iter()
        .map(|t| {
            (!is_sectioned(&t.name)).then(|| {
                let o = cur.next_multiple_of(PAGE as u64);
                cur = o + t.bytes;
                o
            })
        })
        .collect();
    let mut p = Plan { tensor_offset, data_start: 0, offset: [0; 3], bytes: [0; 3], experts, ple };
    let zero: [String; 3] = std::array::from_fn(|_| "0".repeat(64));
    p.data_start = (header(g, &p, &zero, routing)?.len() as u64).next_multiple_of(PAGE as u64);
    p.offset[0] = p.data_start;
    p.offset[1] = (p.data_start + cur).next_multiple_of(SECTION_ALIGN);
    p.bytes[0] = p.offset[1] - p.offset[0];
    p.offset[2] = p.offset[1] + p.experts.bytes;
    p.bytes[1] = p.experts.bytes;
    p.bytes[2] = p.ple.bytes;
    Ok(p)
}

// ---------------------------------------------------------------------------------------------- I/O

/// A page-aligned buffer, for O_DIRECT.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}
impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let len = len.next_multiple_of(PAGE).max(PAGE);
        let ptr = unsafe { std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align(len, PAGE).unwrap()) };
        assert!(!ptr.is_null(), "allocating {len} bytes");
        Self { ptr, len }
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.len, PAGE).unwrap()) };
    }
}

fn open_direct(path: &Path, write: bool) -> Result<File> {
    let mut o = std::fs::OpenOptions::new();
    if write {
        o.write(true).create(true).truncate(true);
    } else {
        o.read(true);
    }
    o.custom_flags(libc::O_DIRECT).open(path).with_context(|| format!("opening {} with O_DIRECT", path.display()))
}

fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> Result<()> {
    f.read_exact_at(buf, off).with_context(|| format!("reading {} bytes at {off}", buf.len()))
}

/// Reads `len` bytes at `offset` (both page multiples) into `dst` (page-aligned) with O_DIRECT, 4 MiB pieces
/// in parallel (fresh anonymous memory faults in as the reads land; larger pieces measured slower).
///
/// # Safety
/// `dst` must be valid for writes of `len` bytes and page-aligned.
pub unsafe fn read_into(path: &Path, offset: u64, dst: *mut u8, len: usize) -> Result<()> {
    if offset % PAGE as u64 != 0 || len % PAGE != 0 || dst as usize % PAGE != 0 {
        bail!("read_into: offset, length and destination must be page-aligned");
    }
    let f = open_direct(path, false)?;
    let d = dst as usize;
    let piece = 4 << 20;
    (0..len.div_ceil(piece)).into_par_iter().try_for_each(|i| {
        let n = piece.min(len - i * piece);
        let s = unsafe { std::slice::from_raw_parts_mut((d + i * piece) as *mut u8, n) };
        read_exact_at(&f, s, offset + (i * piece) as u64)
    })
}

fn root(pieces: &[[u8; 32]]) -> String {
    let mut h = Sha256::new();
    pieces.iter().for_each(|p| h.update(p));
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// A sequential writer: fills 64 MiB buffers, and hands each to a rayon task that writes it at its offset and
/// hashes it, so copying, writing and hashing overlap.
struct Stream {
    file: Arc<File>,
    free_rx: mpsc::Receiver<AlignedBuf>,
    free_tx: mpsc::Sender<AlignedBuf>,
    n_bufs: usize,
    buf: Option<AlignedBuf>,
    len: usize,
    pos: u64,
    section: usize,
    piece: usize,
    hashes: Arc<Mutex<Vec<Vec<(usize, [u8; 32])>>>>,
    err: Arc<Mutex<Option<String>>>,
    written: Arc<std::sync::atomic::AtomicU64>,
}

impl Stream {
    fn new(file: File, start: u64) -> Self {
        let (free_tx, free_rx) = mpsc::channel();
        let n_bufs = 8;
        for _ in 0..n_bufs {
            free_tx.send(AlignedBuf::new(PIECE)).unwrap();
        }
        Self { file: Arc::new(file), free_rx, free_tx, n_bufs, buf: None, len: 0, pos: start, section: 0, piece: 0, hashes: Arc::new(Mutex::new(vec![Vec::new(); 3])), err: Default::default(), written: Default::default() }
    }

    fn put(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let len = self.len;
            let buf = self.buf.get_or_insert_with(|| self.free_rx.recv().unwrap());
            let n = (PIECE - len).min(data.len());
            buf.as_mut()[len..len + n].copy_from_slice(&data[..n]);
            self.len += n;
            data = &data[n..];
            if self.len == PIECE {
                self.flush();
            }
        }
    }

    fn zeros(&mut self, mut n: u64) {
        const Z: [u8; PAGE] = [0; PAGE];
        while n > 0 {
            let k = n.min(PAGE as u64) as usize;
            self.put(&Z[..k]);
            n -= k as u64;
        }
    }

    fn at(&self) -> u64 {
        self.pos + self.len as u64
    }

    fn pad_to(&mut self, abs: u64) {
        assert!(abs >= self.at(), "layout overlap: at {} padding to {abs}", self.at());
        self.zeros(abs - self.at());
    }

    fn flush(&mut self) {
        let Some(buf) = self.buf.take() else { return };
        let (len, pos, section, piece) = (self.len, self.pos, self.section, self.piece);
        let (file, hashes, err, free, written) = (self.file.clone(), self.hashes.clone(), self.err.clone(), self.free_tx.clone(), self.written.clone());
        rayon::spawn(move || {
            let data = &buf.as_slice()[..len];
            if let Err(e) = file.write_all_at(data, pos) {
                *err.lock().unwrap() = Some(format!("write at {pos}: {e}"));
            }
            let h: [u8; 32] = Sha256::digest(data).into();
            hashes.lock().unwrap()[section].push((piece, h));
            written.fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
            let _ = free.send(buf);
        });
        self.pos += len as u64;
        self.len = 0;
        self.piece += 1;
    }

    fn begin(&mut self, section: usize, abs: u64) {
        self.pad_to(abs);
        self.flush();
        self.section = section;
        self.piece = 0;
    }

    /// Waits for every write; the section checksums.
    fn finish(&mut self) -> Result<[String; 3]> {
        self.flush();
        let bufs: Vec<AlignedBuf> = (0..self.n_bufs).map(|_| self.free_rx.recv().unwrap()).collect();
        drop(bufs);
        if let Some(e) = self.err.lock().unwrap().take() {
            bail!(e);
        }
        let mut hs = self.hashes.lock().unwrap();
        Ok(std::array::from_fn(|i| {
            hs[i].sort_by_key(|x| x.0);
            root(&hs[i].iter().map(|x| x.1).collect::<Vec<_>>())
        }))
    }
}

/// Writes the pack of `g` to `out`. `routing`: an expert ranking to store (`layer * n_expert + e`, hottest first).
pub fn write(g: &Gguf, out: &Path, routing: Option<&[u32]>) -> Result<PackInfo> {
    let p = plan(g, routing)?;
    let t0 = std::time::Instant::now();
    let mut s = Stream::new(open_direct(out, true)?, p.data_start);
    let progress = s.written.clone();
    let total: u64 = p.bytes.iter().sum();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reporter = {
        let (progress, done) = (progress.clone(), done.clone());
        std::thread::spawn(move || {
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let w = progress.load(std::sync::atomic::Ordering::Relaxed);
                eprintln!("  {:.1} / {:.1} GiB", w as f64 / 1073741824.0, total as f64 / 1073741824.0);
            }
        })
    };
    // dense
    s.begin(0, p.offset[0]);
    for (t, off) in g.tensors.iter().zip(&p.tensor_offset) {
        if let Some(o) = off {
            s.pad_to(p.data_start + o);
            s.put(g.data(t));
        }
    }
    // experts
    s.begin(1, p.offset[1]);
    let ne = p.experts.n_expert;
    for l in 0..p.experts.layer_offset.len() {
        let t = |n: &str| g.tensor(&format!("blk.{l}.ffn_{n}_exps.weight")).unwrap();
        let (gd, ud, dd) = (g.data(t("gate")), g.data(t("up")), g.data(t("down")));
        let (gb, db, rec) = (p.experts.gate[l] as usize, p.experts.down[l] as usize, p.experts.record[l]);
        let tails = (p.experts.tail[l] != 0).then(|| expert_scales(g, l, ne)).transpose()?;
        for e in 0..ne {
            s.pad_to(p.offset[1] + p.experts.layer_offset[l] + e as u64 * rec);
            s.put(&gd[e * gb..(e + 1) * gb]);
            s.put(&ud[e * gb..(e + 1) * gb]);
            s.put(&dd[e * db..(e + 1) * db]);
            if let Some(t) = &tails {
                s.put(&tail_bytes(&t[e]));
            }
        }
    }
    // ple: page k holds rows k*rpp .. k*rpp+rpp, which are contiguous in the source
    s.begin(2, p.offset[2]);
    let pd = g.data(g.tensor(PLE_TENSOR).unwrap());
    let span = (p.ple.rows_per_page * p.ple.row_bytes) as usize;
    for (k, rows) in pd.chunks(span).enumerate() {
        s.pad_to(p.offset[2] + (k * PAGE) as u64);
        s.put(rows);
    }
    s.pad_to(p.offset[2] + p.bytes[2]);
    let sha = s.finish()?;
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = reporter.join();
    // the header last, with the offsets and checksums
    let h = header(g, &p, &sha, routing)?;
    assert!(h.len() as u64 <= p.data_start, "header grew");
    let mut hb = AlignedBuf::new(p.data_start as usize);
    hb.as_mut()[..h.len()].copy_from_slice(&h);
    s.file.write_all_at(&hb.as_slice()[..p.data_start as usize], 0)?;
    s.file.sync_all()?;
    eprintln!("wrote {} ({:.1} GiB) in {:.1}s", out.display(), (p.offset[2] + p.bytes[2]) as f64 / 1073741824.0, t0.elapsed().as_secs_f64());
    let pk = Gguf::open(out)?;
    PackInfo::of(&pk).ok_or_else(|| anyhow!("the written file is not a pack"))?
}

/// Recomputes every section checksum; with `src`, also checks that every tensor (and every metadata entry) of
/// the source is byte-equal in the pack, and that padding is zero.
pub fn verify(pack: &Path, src: Option<&Gguf>) -> Result<()> {
    let pk = Gguf::open(pack)?;
    let info = PackInfo::of(&pk).ok_or_else(|| anyhow!("{}: not a pack", pack.display()))??;
    if info.offset[0] != pk.shards[0].data_start {
        bail!("the dense section does not start at the data start");
    }
    let f = open_direct(pack, false)?;
    let t0 = std::time::Instant::now();
    for (i, n) in SECTIONS.iter().enumerate() {
        let (off, len) = (info.offset[i], info.bytes[i] as usize);
        let pieces: Vec<[u8; 32]> = (0..len.div_ceil(PIECE))
            .into_par_iter()
            .map(|k| -> Result<[u8; 32]> {
                let n = PIECE.min(len - k * PIECE);
                let mut b = AlignedBuf::new(n);
                read_exact_at(&f, &mut b.as_mut()[..n], off + (k * PIECE) as u64)?;
                Ok(Sha256::digest(&b.as_slice()[..n]).into())
            })
            .collect::<Result<_>>()?;
        let got = root(&pieces);
        if got != info.sha256[i] {
            bail!("section {n}: checksum {got}, header says {}", info.sha256[i]);
        }
        eprintln!("  {n}: {:.2} GiB, checksum ok", len as f64 / 1073741824.0);
    }
    eprintln!("checksums: {:.1}s", t0.elapsed().as_secs_f64());
    let Some(g) = src else { return Ok(()) };
    let t0 = std::time::Instant::now();
    // metadata: every source entry but the split and alignment keys, as stored
    let theirs = g.raw_kvs()?;
    let ours: std::collections::HashMap<_, _> = pk.raw_kvs()?.into_iter().collect();
    for (k, raw) in theirs.iter().filter(|(k, _)| !k.starts_with("split.") && k != "general.alignment") {
        if ours.get(k) != Some(raw) {
            bail!("metadata {k} differs");
        }
    }
    if pk.tensors.len() != g.tensors.len() {
        bail!("{} tensors in the pack, {} in the source", pk.tensors.len(), g.tensors.len());
    }
    let read = |off: u64, len: usize| -> Result<(AlignedBuf, usize)> {
        let a = off / PAGE as u64 * PAGE as u64;
        let skip = (off - a) as usize;
        let mut b = AlignedBuf::new(skip + len);
        let n = b.len;
        read_exact_at(&f, &mut b.as_mut()[..n], a)?;
        Ok((b, skip))
    };
    // dense
    pk.tensors.par_iter().zip(&g.tensors).try_for_each(|(a, b)| -> Result<()> {
        if a.name != b.name || a.shape != b.shape || a.ty != b.ty {
            bail!("tensor index differs at {}", b.name);
        }
        if a.packed() != is_sectioned(&a.name) {
            bail!("{}: in the wrong place", a.name);
        }
        if !a.packed() {
            let (buf, skip) = read(pk.shards[0].data_start + a.offset, a.bytes as usize)?;
            if &buf.as_slice()[skip..skip + a.bytes as usize] != g.data(b) {
                bail!("{}: bytes differ", a.name);
            }
        }
        Ok(())
    })?;
    eprintln!("  dense: {} tensors equal", pk.tensors.iter().filter(|t| !t.packed()).count());
    // experts, a layer at a time
    let e = &info.experts;
    (0..e.layer_offset.len()).into_par_iter().try_for_each(|l| -> Result<()> {
        let rec = e.record[l] as usize;
        let (buf, _) = read(info.offset[1] + e.layer_offset[l], rec * e.n_expert)?;
        let t = |n: &str| g.data(g.tensor(&format!("blk.{l}.ffn_{n}_exps.weight")).unwrap());
        let (gd, ud, dd) = (t("gate"), t("up"), t("down"));
        let (gb, db, tb) = (e.gate[l] as usize, e.down[l] as usize, e.tail[l] as usize);
        let tails = (tb != 0).then(|| expert_scales(g, l, e.n_expert)).transpose()?;
        for x in 0..e.n_expert {
            let r = &buf.as_slice()[x * rec..(x + 1) * rec];
            let tail_ok = match &tails {
                Some(t) => r[2 * gb + db..2 * gb + db + tb] == tail_bytes(&t[x]),
                None => true,
            };
            if r[..gb] != gd[x * gb..(x + 1) * gb] || r[gb..2 * gb] != ud[x * gb..(x + 1) * gb] || r[2 * gb..2 * gb + db] != dd[x * db..(x + 1) * db] || !tail_ok || r[2 * gb + db + tb..].iter().any(|&b| b != 0) {
                bail!("expert ({l}, {x}) differs");
            }
        }
        Ok(())
    })?;
    eprintln!("  experts: {} layers x {} equal", e.layer_offset.len(), e.n_expert);
    // ple, 64 MiB of pages at a time
    let pd = g.data(g.tensor(PLE_TENSOR).unwrap());
    let (rpp, rb) = (info.ple.rows_per_page as usize, info.ple.row_bytes as usize);
    let pages = (info.ple.rows as usize).div_ceil(rpp);
    let per = PIECE / PAGE;
    (0..pages.div_ceil(per)).into_par_iter().try_for_each(|c| -> Result<()> {
        let (p0, p1) = (c * per, ((c + 1) * per).min(pages));
        let (buf, _) = read(info.offset[2] + (p0 * PAGE) as u64, (p1 - p0) * PAGE)?;
        for k in p0..p1 {
            let page = &buf.as_slice()[(k - p0) * PAGE..(k - p0 + 1) * PAGE];
            let src = &pd[k * rpp * rb..((k + 1) * rpp * rb).min(pd.len())];
            if page[..src.len()] != *src || page[src.len()..].iter().any(|&b| b != 0) {
                bail!("PLE page {k} differs");
            }
        }
        Ok(())
    })?;
    eprintln!("  ple: {} rows equal", info.ple.rows);
    eprintln!("round trip: every tensor byte-equal ({:.1}s)", t0.elapsed().as_secs_f64());
    Ok(())
}
