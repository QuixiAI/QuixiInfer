//! GGUF v3 reader: metadata, tensor index and memory-mapped tensor data, across split shards.
use crate::ggml::GgmlType;
use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U64(v) => Some(v),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F64(v) => Some(v),
            Value::U64(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: shape[0] is the contiguous (row) dimension.
    pub shape: Vec<u64>,
    pub ty: GgmlType,
    /// Which shard holds the data, and the byte offset from that shard's data start.
    pub shard: usize,
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    /// In a pack, the tensor lives in a section of its own (`pack::PACKED` offset): its index entry gives the
    /// shape and type, and `Gguf::data` returns no bytes.
    pub fn packed(&self) -> bool {
        self.offset == crate::pack::PACKED
    }

    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

pub struct Shard {
    pub path: PathBuf,
    pub map: Mmap,
    pub data_start: u64,
}

pub struct Gguf {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub shards: Vec<Shard>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.p..self.p + n).ok_or_else(|| anyhow!("GGUF header truncated at byte {}", self.p))?;
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn scalar(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U64(self.take(1)?[0] as u64),
            1 => Value::I64(self.take(1)?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.scalar(et)?);
                }
                Value::Arr(v)
            }
            _ => bail!("unknown GGUF value type {t}"),
        })
    }
}

fn parse_shard(path: &Path, shard_idx: usize, md: &mut BTreeMap<String, Value>, tensors: &mut Vec<TensorInfo>) -> Result<Shard> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: the shard files are read-only inputs and are not modified while mapped.
    let map = unsafe { Mmap::map(&f)? };
    let mut c = Cursor { b: &map, p: 0 };
    if c.take(4)? != b"GGUF" {
        bail!("{}: not a GGUF file", path.display());
    }
    let version = c.u32()?;
    if version != 3 {
        bail!("{}: GGUF v{version}, expected v3", path.display());
    }
    let n_tensors = c.u64()? as usize;
    let n_kv = c.u64()? as usize;
    let mut local = BTreeMap::new();
    for _ in 0..n_kv {
        let k = c.string()?;
        let t = c.u32()?;
        local.insert(k, c.scalar(t)?);
    }
    let align = local.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()? as usize;
        let shape = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
        let tid = c.u32()?;
        let offset = c.u64()?;
        let ty = GgmlType::from_id(tid).ok_or_else(|| anyhow!("{name}: unsupported ggml type id {tid}"))?;
        let n: u64 = shape.iter().product();
        let bytes = ty.bytes(n as usize) as u64;
        tensors.push(TensorInfo { name, shape, ty, shard: shard_idx, offset, bytes });
    }
    let data_start = (c.p as u64).div_ceil(align) * align;
    if shard_idx == 0 {
        *md = local;
    }
    Ok(Shard { path: path.to_path_buf(), map, data_start })
}

impl Gguf {
    /// Opens a GGUF, following `-00001-of-0000N` split naming to the other shards.
    pub fn open(first: &Path) -> Result<Self> {
        let mut metadata = BTreeMap::new();
        let mut tensors = Vec::new();
        let mut shards = vec![parse_shard(first, 0, &mut metadata, &mut tensors)?];
        let count = metadata.get("split.count").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        if count > 1 {
            let name = first.file_name().unwrap().to_str().unwrap();
            let tag = format!("-00001-of-{count:05}.gguf");
            let stem = name.strip_suffix(&tag).ok_or_else(|| anyhow!("{name}: expected a name ending in {tag}"))?;
            for i in 2..=count {
                let p = first.with_file_name(format!("{stem}-{i:05}-of-{count:05}.gguf"));
                let mut ignored = BTreeMap::new();
                shards.push(parse_shard(&p, i - 1, &mut ignored, &mut tensors)?);
            }
        }
        let g = Self { metadata, tensors, shards };
        for t in g.tensors.iter().filter(|t| !t.packed()) {
            let s = &g.shards[t.shard];
            let end = s.data_start + t.offset + t.bytes;
            if end > s.map.len() as u64 {
                bail!("{}: data ends at {end}, past the end of {}", t.name, s.path.display());
            }
        }
        Ok(g)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        if t.packed() {
            return &[];
        }
        let s = &self.shards[t.shard];
        let o = (s.data_start + t.offset) as usize;
        &s.map[o..o + t.bytes as usize]
    }

    /// The first shard's metadata entries as stored: key, then the value's type id and encoding.
    pub fn raw_kvs(&self) -> Result<Vec<(String, &[u8])>> {
        let b: &[u8] = &self.shards[0].map;
        let mut c = Cursor { b, p: 8 };
        let _ = c.u64()?;
        let n_kv = c.u64()? as usize;
        let mut out = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let k = c.string()?;
            let start = c.p;
            let t = c.u32()?;
            c.scalar(t)?;
            out.push((k, &b[start..c.p]));
        }
        Ok(out)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.get(key).and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("missing integer metadata {key}"))
    }
}
