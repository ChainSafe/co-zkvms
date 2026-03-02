//! Backing store for persisted preprocessing data (edaBits / daBits).
//!
//! Provides [`BackingStore`] — a dual-mode container that holds field elements
//! either in memory (`Vec<F>`) or as a memory-mapped file (`MmapMut`).
//!
//! When the `reuse-preproc` feature is **disabled** (production), consumed
//! elements are zeroed on disk and the cursor is persisted crash-safely.
//! When the feature is **enabled** (testing), data files remain intact and
//! can be loaded multiple times.

use crate::protocols::rep3::PartyID;
use memmap2::MmapMut;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Compile-time layout assertion
// ---------------------------------------------------------------------------

/// Asserts at const-eval time that `F` has a layout compatible with raw
/// byte-level save/load (size multiple of 8, alignment ≤ 8).
///
/// Sound for ark-ff `Fp<P, N>` which is `BigInt<N>` (`[u64; N]`) plus a
/// zero-sized `PhantomData`.  All parties run the same binary on the same
/// architecture so endianness is consistent.
pub(crate) const fn assert_field_layout<F>() {
    assert!(
        std::mem::size_of::<F>() % 8 == 0,
        "Field element size must be a multiple of 8 bytes"
    );
    assert!(
        std::mem::align_of::<F>() <= 8,
        "Field element alignment must not exceed u64 alignment"
    );
}

// ---------------------------------------------------------------------------
// BackingStore<F>
// ---------------------------------------------------------------------------

/// Dual-mode backing store: either in-memory `Vec` or mutable memory-mapped file.
pub(crate) enum BackingStore<F> {
    /// In-memory storage (after preprocessing, before save to disk).
    InMemory(Vec<F>),
    /// Memory-mapped file with optional consume-on-read zeroing.
    Mapped {
        mmap: MmapMut,
        /// Number of `F` elements in the mapping.
        len: usize,
        _phantom: PhantomData<F>,
    },
    /// No data (P0/P1 where no stored elements exist).
    Empty,
}

impl<F> BackingStore<F> {
    /// Return a slice over all elements.
    ///
    /// # Safety (internal)
    ///
    /// For the `Mapped` variant this performs an `unsafe` pointer cast.
    /// Soundness relies on:
    /// - `mmap` is page-aligned (≥ 4096), satisfying `align_of::<F>() ≤ 8`
    /// - `F` has the same in-memory representation that was written to disk
    ///   (guaranteed: same binary, same architecture)
    /// - No padding bytes in `F` (true for `Fp<P, N>` = `[u64; N]` + ZST)
    pub(crate) fn as_slice(&self) -> &[F] {
        match self {
            BackingStore::InMemory(v) => v.as_slice(),
            BackingStore::Mapped { mmap, len, .. } => unsafe {
                std::slice::from_raw_parts(mmap.as_ptr() as *const F, *len)
            },
            BackingStore::Empty => &[],
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            BackingStore::InMemory(v) => v.len(),
            BackingStore::Mapped { len, .. } => *len,
            BackingStore::Empty => 0,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wrap a `Vec<F>`, returning `Empty` if the vec is empty.
    pub(crate) fn from_vec(v: Vec<F>) -> Self {
        if v.is_empty() {
            BackingStore::Empty
        } else {
            BackingStore::InMemory(v)
        }
    }

    /// Write raw bytes to a file.  No-op for `Empty`.
    pub(crate) fn save_to_file(&self, path: &Path) -> io::Result<()> {
        let slice = match self {
            BackingStore::InMemory(v) => v.as_slice(),
            BackingStore::Mapped { mmap, len, .. } => unsafe {
                std::slice::from_raw_parts(mmap.as_ptr() as *const F, *len)
            },
            BackingStore::Empty => return Ok(()),
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
        };
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        w.write_all(bytes)?;
        w.flush()?;
        w.into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        Ok(())
    }

    /// Open a file and memory-map it (read-write).
    ///
    /// Returns `Empty` if the file is zero-length.
    pub(crate) fn load_from_file(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let elem_size = std::mem::size_of::<F>();
        if file_len == 0 {
            return Ok(BackingStore::Empty);
        }
        if file_len % elem_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file size {} is not a multiple of element size {}",
                    file_len, elem_size
                ),
            ));
        }
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        let len = file_len / elem_size;
        Ok(BackingStore::Mapped {
            mmap,
            len,
            _phantom: PhantomData,
        })
    }

    /// Zero out consumed elements in the backing store and flush to disk.
    ///
    /// `start..end` are element indices (not byte offsets).
    /// No-op for `InMemory` / `Empty` (in-memory data dies with the process).
    /// No-op when the `reuse-preproc` feature is enabled.
    pub(crate) fn consume(&mut self, start: usize, end: usize) {
        #[cfg(not(feature = "reuse-preproc"))]
        if let BackingStore::Mapped { mmap, .. } = self {
            let elem_size = std::mem::size_of::<F>();
            let byte_start = start * elem_size;
            let byte_end = end * elem_size;
            mmap[byte_start..byte_end].fill(0);
            // Best-effort flush; ignore errors (crash safety handled by cursor).
            let _ = mmap.flush_range(byte_start, byte_end - byte_start);
        }
    }
}

// ---------------------------------------------------------------------------
// MetaData — binary header for LazyEdaBits / LazyDaBits
// ---------------------------------------------------------------------------

/// Fixed-layout metadata serialised alongside each lazy struct.
pub(crate) struct MetaData {
    pub seed1: [u8; crate::SEED_SIZE],
    pub pos1: u128,
    pub seed2: [u8; crate::SEED_SIZE],
    pub pos2: u128,
    pub total: usize,
    pub party_id_byte: u8,
    pub cursor: usize,
    pub field_bytes: usize,
}

/// Byte offset of the `cursor` field inside the meta file.
const CURSOR_OFFSET: u64 = {
    // seed1(32) + pos1(16) + seed2(32) + pos2(16) + total(8) + party_id(1)
    (crate::SEED_SIZE as u64) + 16 + (crate::SEED_SIZE as u64) + 16 + 8 + 1
};

/// Write a complete meta file (atomically: write + fsync).
pub(crate) fn write_meta(path: &Path, meta: &MetaData) -> io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(&meta.seed1)?;
    w.write_all(&meta.pos1.to_le_bytes())?;
    w.write_all(&meta.seed2)?;
    w.write_all(&meta.pos2.to_le_bytes())?;
    w.write_all(&(meta.total as u64).to_le_bytes())?;
    w.write_all(&[meta.party_id_byte])?;
    w.write_all(&(meta.cursor as u64).to_le_bytes())?;
    w.write_all(&(meta.field_bytes as u32).to_le_bytes())?;
    w.flush()?;
    w.into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    Ok(())
}

/// Read a meta file.
pub(crate) fn read_meta(path: &Path) -> io::Result<MetaData> {
    let mut f = File::open(path)?;
    let mut seed1 = [0u8; crate::SEED_SIZE];
    f.read_exact(&mut seed1)?;
    let mut buf16 = [0u8; 16];
    f.read_exact(&mut buf16)?;
    let pos1 = u128::from_le_bytes(buf16);
    let mut seed2 = [0u8; crate::SEED_SIZE];
    f.read_exact(&mut seed2)?;
    f.read_exact(&mut buf16)?;
    let pos2 = u128::from_le_bytes(buf16);
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8)?;
    let total = u64::from_le_bytes(buf8) as usize;
    let mut buf1 = [0u8; 1];
    f.read_exact(&mut buf1)?;
    let party_id_byte = buf1[0];
    f.read_exact(&mut buf8)?;
    let cursor = u64::from_le_bytes(buf8) as usize;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let field_bytes = u32::from_le_bytes(buf4) as usize;
    Ok(MetaData {
        seed1,
        pos1,
        seed2,
        pos2,
        total,
        party_id_byte,
        cursor,
        field_bytes,
    })
}

/// Update only the cursor field inside an existing meta file (seek + write + fsync).
///
/// No-op when the `reuse-preproc` feature is enabled.
pub(crate) fn update_cursor(path: &Path, cursor: usize) -> io::Result<()> {
    #[cfg(not(feature = "reuse-preproc"))]
    {
        let mut f = OpenOptions::new().write(true).open(path)?;
        f.seek(SeekFrom::Start(CURSOR_OFFSET))?;
        f.write_all(&(cursor as u64).to_le_bytes())?;
        f.sync_all()?;
    }
    Ok(())
}

/// Convert a `PartyID` to its byte representation.
pub(crate) fn party_id_to_byte(pid: PartyID) -> u8 {
    match pid {
        PartyID::ID0 => 0,
        PartyID::ID1 => 1,
        PartyID::ID2 => 2,
    }
}

/// Convert a byte to `PartyID`.
pub(crate) fn byte_to_party_id(b: u8) -> io::Result<PartyID> {
    match b {
        0 => Ok(PartyID::ID0),
        1 => Ok(PartyID::ID1),
        2 => Ok(PartyID::ID2),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid party_id byte: {b}"),
        )),
    }
}
