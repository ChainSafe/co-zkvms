//! Backing store for persisted preprocessing data (edaBits / daBits).
//!
//! Provides [`BackingStore`] — a dual-mode container that holds field elements
//! either in memory (`Vec<F>`) or as a file-backed store (`File` + range reads).
//!
//! When the `reuse-preproc` feature is **disabled** (production), consumed
//! elements are zeroed on disk and the cursor is persisted crash-safely.
//! When the feature is **enabled** (testing), data files remain intact and
//! can be loaded multiple times.

use crate::protocols::rep3::PartyID;
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
    assert!(std::mem::size_of::<F>() % 8 == 0, "Field element size must be a multiple of 8 bytes");
    assert!(std::mem::align_of::<F>() <= 8, "Field element alignment must not exceed u64 alignment");
}

// ---------------------------------------------------------------------------
// BackingStore<F>
// ---------------------------------------------------------------------------

/// Dual-mode backing store: either in-memory `Vec` or file-backed range reads.
pub(crate) enum BackingStore<F> {
    /// In-memory storage (after preprocessing, before save to disk).
    InMemory(Vec<F>),
    /// File-backed store for sequential / range reads without mmap'ing the file.
    FileBacked {
        file: File,
        path: PathBuf,
        /// Number of `F` elements in the file.
        len: usize,
        /// Next byte offset used for append writes.
        append_offset: u64,
        _phantom: PhantomData<F>,
    },
    /// No data (P0/P1 where no stored elements exist).
    Empty,
}

pub(crate) enum BackingStoreReadView<'a, F> {
    InMemory(&'a [F]),
    FileBacked { file: File, len: usize },
    Empty,
}

pub(crate) struct FileBackedWriter<F> {
    file: File,
    len: usize,
    _phantom: PhantomData<F>,
}

impl<F> Clone for BackingStoreReadView<'_, F> {
    fn clone(&self) -> Self {
        match self {
            Self::InMemory(slice) => Self::InMemory(slice),
            Self::FileBacked { file, len } => {
                Self::FileBacked { file: file.try_clone().expect("cloning file-backed read view"), len: *len }
            }
            Self::Empty => Self::Empty,
        }
    }
}

impl<F> Clone for FileBackedWriter<F> {
    fn clone(&self) -> Self {
        Self { file: self.file.try_clone().expect("cloning file-backed writer"), len: self.len, _phantom: PhantomData }
    }
}

impl<F> BackingStore<F> {
    fn vec_from_bytes(bytes: &[u8]) -> io::Result<Vec<F>> {
        const { assert_field_layout::<F>() };
        let elem_size = Self::elem_size_bytes();
        if bytes.len() % elem_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("raw payload length {} is not divisible by element size {}", bytes.len(), elem_size,),
            ));
        }
        let elems = bytes.len() / elem_size;
        if elems == 0 {
            return Ok(Vec::new());
        }

        let mut out: Vec<std::mem::MaybeUninit<F>> = Vec::with_capacity(elems);
        unsafe { out.set_len(elems) };
        let out_bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, bytes.len()) };
        out_bytes.copy_from_slice(bytes);
        let out: Vec<F> = unsafe { std::mem::transmute(out) };
        Ok(out)
    }

    pub(crate) fn as_slice(&self) -> &[F] {
        match self {
            BackingStore::InMemory(v) => v.as_slice(),
            BackingStore::FileBacked { .. } => {
                panic!("BackingStore::FileBacked does not support as_slice(); use read_reuse/read_consume")
            }
            BackingStore::Empty => &[],
        }
    }

    pub(crate) fn read_view(&self) -> io::Result<BackingStoreReadView<'_, F>> {
        match self {
            BackingStore::InMemory(v) => Ok(BackingStoreReadView::InMemory(v.as_slice())),
            BackingStore::FileBacked { file, len, .. } => {
                Ok(BackingStoreReadView::FileBacked { file: file.try_clone()?, len: *len })
            }
            BackingStore::Empty => Ok(BackingStoreReadView::Empty),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            BackingStore::InMemory(v) => v.len(),
            BackingStore::FileBacked { len, .. } => *len,
            BackingStore::Empty => 0,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn validate_range(&self, start: usize, end: usize) -> io::Result<()> {
        if start > end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid range: start({start}) > end({end})"),
            ));
        }
        if end > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("range end({end}) exceeds len({})", self.len()),
            ));
        }
        Ok(())
    }

    fn elem_size_bytes() -> usize {
        std::mem::size_of::<F>()
    }

    fn byte_range(start: usize, end: usize) -> (u64, usize) {
        let elem_size = Self::elem_size_bytes();
        let byte_start = start * elem_size;
        let byte_len = (end - start) * elem_size;
        (byte_start as u64, byte_len)
    }

    fn file_read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_exact_at(buf, offset)?;
            return Ok(());
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            file.seek_read(buf, offset)?;
            return Ok(());
        }
        #[cfg(not(any(unix, windows)))]
        {
            let mut f = file.try_clone()?;
            f.seek(SeekFrom::Start(offset))?;
            f.read_exact(buf)?;
            Ok(())
        }
    }

    fn file_write_all_at(file: &File, offset: u64, buf: &[u8]) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.write_all_at(buf, offset)?;
            return Ok(());
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            file.seek_write(buf, offset)?;
            return Ok(());
        }
        #[cfg(not(any(unix, windows)))]
        {
            let mut f = file.try_clone()?;
            f.seek(SeekFrom::Start(offset))?;
            f.write_all(buf)?;
            Ok(())
        }
    }

    fn read_file_backed_range(&self, file: &File, start: usize, end: usize) -> io::Result<Vec<F>> {
        self.validate_range(start, end)?;
        let count = end - start;
        if count == 0 {
            return Ok(Vec::new());
        }

        let (byte_offset, byte_len) = Self::byte_range(start, end);
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(start, end, count, byte_offset, byte_len, "BackingStore file-backed read");
        }

        // SAFETY: We read exactly `byte_len` bytes from disk into a
        // `Vec<MaybeUninit<F>>`, then transmute to `Vec<F>`. This is sound
        // under the existing `assert_field_layout::<F>()` contract and because
        // we fully initialize all elements via `read_exact_at`.
        let mut out: Vec<std::mem::MaybeUninit<F>> = Vec::with_capacity(count);
        unsafe { out.set_len(count) };
        let out_bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };

        Self::file_read_exact_at(file, byte_offset, out_bytes)?;
        let out: Vec<F> = unsafe { std::mem::transmute(out) };
        Ok(out)
    }

    fn read_file_backed_range_into(&self, file: &File, start: usize, end: usize, out: &mut Vec<F>) -> io::Result<()> {
        self.validate_range(start, end)?;
        let count = end - start;
        out.clear();
        if count == 0 {
            return Ok(());
        }

        let (byte_offset, byte_len) = Self::byte_range(start, end);
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(start, end, count, byte_offset, byte_len, "BackingStore file-backed read_into");
        }

        out.reserve(count.saturating_sub(out.capacity()));
        // SAFETY: `F` obeys `assert_field_layout::<F>()` and the read fills the
        // entire byte range for `count` initialized elements.
        unsafe { out.set_len(count) };
        let out_bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };
        if let Err(err) = Self::file_read_exact_at(file, byte_offset, out_bytes) {
            out.clear();
            return Err(err);
        }
        Ok(())
    }

    /// Read a range of elements in **reuse** mode (never mutates the file).
    #[cfg(feature = "reuse-preproc")]
    pub(crate) fn read_reuse(&self, start: usize, end: usize) -> io::Result<Vec<F>>
    where
        F: Copy,
    {
        match self {
            BackingStore::InMemory(v) => {
                self.validate_range(start, end)?;
                Ok(v[start..end].to_vec())
            }
            BackingStore::FileBacked { file, path, .. } => {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    tracing::debug!(
                        path = %path.display(),
                        start,
                        end,
                        "BackingStore::read_reuse"
                    );
                }
                self.read_file_backed_range(file, start, end)
            }
            BackingStore::Empty => Ok(Vec::new()),
        }
    }

    /// Read a range of elements in **reuse** mode into an existing vec.
    #[cfg(feature = "reuse-preproc")]
    pub(crate) fn read_reuse_into(&self, start: usize, end: usize, out: &mut Vec<F>) -> io::Result<()>
    where
        F: Copy,
    {
        match self {
            BackingStore::InMemory(v) => {
                self.validate_range(start, end)?;
                out.clear();
                out.extend_from_slice(&v[start..end]);
                Ok(())
            }
            BackingStore::FileBacked { file, path, .. } => {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    tracing::debug!(
                        path = %path.display(),
                        start,
                        end,
                        "BackingStore::read_reuse_into"
                    );
                }
                self.read_file_backed_range_into(file, start, end, out)
            }
            BackingStore::Empty => {
                out.clear();
                Ok(())
            }
        }
    }

    /// Read a range of elements in **reuse** mode into an existing slice.
    #[cfg(feature = "reuse-preproc")]
    pub(crate) fn read_reuse_into_slice(&self, start: usize, end: usize, out: &mut [F]) -> io::Result<()>
    where
        F: Copy,
    {
        let count = end.saturating_sub(start);
        if out.len() != count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("slice length {} does not match requested element count {}", out.len(), count),
            ));
        }
        match self {
            BackingStore::InMemory(v) => {
                self.validate_range(start, end)?;
                out.copy_from_slice(&v[start..end]);
                Ok(())
            }
            BackingStore::FileBacked { file, path, .. } => {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    tracing::debug!(
                        path = %path.display(),
                        start,
                        end,
                        "BackingStore::read_reuse_into_slice"
                    );
                }
                let (byte_offset, byte_len) = Self::byte_range(start, end);
                let out_bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };
                Self::file_read_exact_at(file, byte_offset, out_bytes)
            }
            BackingStore::Empty => {
                if !out.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "attempted to read from empty backing store",
                    ));
                }
                Ok(())
            }
        }
    }

    /// Read a range of elements in **consume** mode (call `consume()` after use).
    #[cfg(not(feature = "reuse-preproc"))]
    pub(crate) fn read_consume(&self, start: usize, end: usize) -> io::Result<Vec<F>>
    where
        F: Copy,
    {
        match self {
            BackingStore::InMemory(v) => {
                self.validate_range(start, end)?;
                Ok(v[start..end].to_vec())
            }
            BackingStore::FileBacked { file, path, .. } => {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    tracing::debug!(
                        path = %path.display(),
                        start,
                        end,
                        "BackingStore::read_consume"
                    );
                }
                self.read_file_backed_range(file, start, end)
            }
            BackingStore::Empty => Ok(Vec::new()),
        }
    }

    /// Read a range of elements in **consume** mode into an existing vec.
    #[cfg(not(feature = "reuse-preproc"))]
    pub(crate) fn read_consume_into(&self, start: usize, end: usize, out: &mut Vec<F>) -> io::Result<()>
    where
        F: Copy,
    {
        match self {
            BackingStore::InMemory(v) => {
                self.validate_range(start, end)?;
                out.clear();
                out.extend_from_slice(&v[start..end]);
                Ok(())
            }
            BackingStore::FileBacked { file, path, .. } => {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    tracing::debug!(
                        path = %path.display(),
                        start,
                        end,
                        "BackingStore::read_consume_into"
                    );
                }
                self.read_file_backed_range_into(file, start, end, out)
            }
            BackingStore::Empty => {
                out.clear();
                Ok(())
            }
        }
    }

    /// Read a range of elements in **consume** mode into an existing slice.
    #[cfg(not(feature = "reuse-preproc"))]
    pub(crate) fn read_consume_into_slice(&self, start: usize, end: usize, out: &mut [F]) -> io::Result<()>
    where
        F: Copy,
    {
        let count = end.saturating_sub(start);
        if out.len() != count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("slice length {} does not match requested element count {}", out.len(), count),
            ));
        }
        match self {
            BackingStore::InMemory(v) => {
                self.validate_range(start, end)?;
                out.copy_from_slice(&v[start..end]);
                Ok(())
            }
            BackingStore::FileBacked { file, path, .. } => {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    tracing::debug!(
                        path = %path.display(),
                        start,
                        end,
                        "BackingStore::read_consume_into_slice"
                    );
                }
                let (byte_offset, byte_len) = Self::byte_range(start, end);
                let out_bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };
                Self::file_read_exact_at(file, byte_offset, out_bytes)
            }
            BackingStore::Empty => {
                if !out.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "attempted to read from empty backing store",
                    ));
                }
                Ok(())
            }
        }
    }

    /// Wrap a `Vec<F>`, returning `Empty` if the vec is empty.
    pub(crate) fn from_vec(v: Vec<F>) -> Self {
        if v.is_empty() { BackingStore::Empty } else { BackingStore::InMemory(v) }
    }

    /// Create a file-backed store at `path` for incremental `extend()` appends.
    ///
    /// The file is created (or truncated if it exists) and opened read+write.
    pub(crate) fn create_file_backed(path: &Path) -> io::Result<Self> {
        Self::create_file_backed_sized(path, 0)
    }

    /// Create a file-backed store at `path` with space reserved for `capacity_elems`.
    pub(crate) fn create_file_backed_sized(path: &Path, capacity_elems: usize) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).truncate(true).read(true).write(true).open(path)?;
        let reserved_bytes = capacity_elems.saturating_mul(Self::elem_size_bytes()) as u64;
        file.set_len(reserved_bytes)?;
        Ok(BackingStore::FileBacked {
            file,
            path: path.to_path_buf(),
            len: capacity_elems,
            append_offset: 0,
            _phantom: PhantomData,
        })
    }

    pub(crate) fn writer(&self) -> io::Result<Option<FileBackedWriter<F>>> {
        match self {
            BackingStore::FileBacked { file, len, .. } => {
                Ok(Some(FileBackedWriter { file: file.try_clone()?, len: *len, _phantom: PhantomData }))
            }
            _ => Ok(None),
        }
    }

    /// Pre-extend the file to hold `additional_elems` more elements and return
    /// a writer covering the full new range.  Updates internal len/append_offset.
    /// Returns `None` for non-file-backed stores.
    pub(crate) fn pre_extended_writer(&mut self, additional_elems: usize) -> io::Result<Option<FileBackedWriter<F>>> {
        match self {
            BackingStore::FileBacked { file, len, append_offset, .. } => {
                let elem_size = Self::elem_size_bytes();
                let new_len = *len + additional_elems;
                let new_file_size = (new_len * elem_size) as u64;
                file.set_len(new_file_size)?;
                *append_offset = new_file_size;
                *len = new_len;
                Ok(Some(FileBackedWriter { file: file.try_clone()?, len: new_len, _phantom: PhantomData }))
            }
            _ => Ok(None),
        }
    }

    /// Write raw bytes to a file.  No-op for `Empty`.
    pub(crate) fn save_to_file(&self, path: &Path) -> io::Result<()> {
        match self {
            BackingStore::InMemory(v) => {
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice())) };
                let file = File::create(path)?;
                let mut w = BufWriter::new(file);
                w.write_all(bytes)?;
                w.flush()?;
                Ok(())
            }
            BackingStore::FileBacked { path: src, .. } => {
                if src == path {
                    return Ok(());
                }
                std::fs::copy(src, path)?;
                Ok(())
            }
            BackingStore::Empty => Ok(()),
        }
    }

    /// Open a file for file-backed range reads.
    ///
    /// Returns `Empty` if the file is zero-length. The file is opened read+write
    /// (even with `reuse-preproc`) so pool extension can append to it if needed.
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
                format!("file size {} is not a multiple of element size {}", file_len, elem_size),
            ));
        }

        let len = file_len / elem_size;
        if tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(
                path = %path.display(),
                file_len_bytes = file_len,
                elem_size,
                len_elems = len,
                "BackingStore::load_from_file"
            );
        }
        Ok(BackingStore::FileBacked {
            file,
            path: path.to_path_buf(),
            len,
            append_offset: file_len as u64,
            _phantom: PhantomData,
        })
    }

    /// Append additional elements to this store.
    ///
    /// Only used during pool extension, not normal consumption.
    pub(crate) fn extend(&mut self, additional: &[F])
    where
        F: Copy,
    {
        if additional.is_empty() {
            return;
        }
        match self {
            BackingStore::InMemory(v) => v.extend_from_slice(additional),
            BackingStore::FileBacked { file, len, append_offset, .. } => {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(additional.as_ptr() as *const u8, std::mem::size_of_val(additional))
                };
                // Best-effort append; panic is fine here because preprocessing pools
                // can't recover from persistence failure anyway.
                Self::file_write_all_at(file, *append_offset, bytes).unwrap_or_else(|e| {
                    panic!("BackingStore::extend file append failed: {e}");
                });
                *append_offset += bytes.len() as u64;
                *len += additional.len();
            }
            BackingStore::Empty => {
                *self = BackingStore::InMemory(additional.to_vec());
            }
        }
    }

    /// Write `data` at the exact element offset `start_elem`.
    ///
    /// Unlike `extend()`, this does not depend on append state and is therefore
    /// suitable for deterministic chunk writes into pre-sized file-backed stores.
    pub(crate) fn write_at(&mut self, start_elem: usize, data: &[F]) -> io::Result<()>
    where
        F: Copy,
    {
        if data.is_empty() {
            return Ok(());
        }
        let end_elem = start_elem.saturating_add(data.len());
        match self {
            BackingStore::InMemory(v) => {
                if end_elem > v.len() {
                    v.resize(end_elem, data[0]);
                }
                v[start_elem..end_elem].copy_from_slice(data);
                Ok(())
            }
            BackingStore::FileBacked { file, len, append_offset, .. } => {
                if end_elem > *len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("BackingStore::write_at range end({end_elem}) exceeds len({len})"),
                    ));
                }
                let (byte_offset, _) = Self::byte_range(start_elem, end_elem);
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
                Self::file_write_all_at(file, byte_offset, bytes)?;
                // Track high-water mark so that subsequent extend() appends
                // after all written data instead of overwriting it.
                let end_byte = byte_offset + bytes.len() as u64;
                if end_byte > *append_offset {
                    *append_offset = end_byte;
                }
                Ok(())
            }
            BackingStore::Empty => {
                if start_elem != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("BackingStore::write_at on Empty with non-zero offset {start_elem}"),
                    ));
                }
                *self = BackingStore::InMemory(data.to_vec());
                Ok(())
            }
        }
    }

    pub(crate) fn write_bytes_at(&mut self, start_elem: usize, bytes: &[u8]) -> io::Result<()>
    where
        F: Copy,
    {
        if bytes.is_empty() {
            return Ok(());
        }
        const { assert_field_layout::<F>() };
        let elem_size = Self::elem_size_bytes();
        if bytes.len() % elem_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "BackingStore::write_bytes_at byte length {} is not divisible by element size {}",
                    bytes.len(),
                    elem_size,
                ),
            ));
        }

        let elems = bytes.len() / elem_size;
        let end_elem = start_elem.saturating_add(elems);
        match self {
            BackingStore::InMemory(v) => {
                let decoded = Self::vec_from_bytes(bytes)?;
                if end_elem > v.len() {
                    v.resize_with(end_elem, || decoded[0]);
                }
                v[start_elem..end_elem].copy_from_slice(&decoded);
                Ok(())
            }
            BackingStore::FileBacked { file, len, append_offset, .. } => {
                if end_elem > *len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("BackingStore::write_bytes_at range end({end_elem}) exceeds len({len})"),
                    ));
                }
                let (byte_offset, _) = Self::byte_range(start_elem, end_elem);
                Self::file_write_all_at(file, byte_offset, bytes)?;
                let end_byte = byte_offset + bytes.len() as u64;
                if end_byte > *append_offset {
                    *append_offset = end_byte;
                }
                Ok(())
            }
            BackingStore::Empty => {
                if start_elem != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("BackingStore::write_bytes_at on Empty with non-zero offset {start_elem}"),
                    ));
                }
                *self = BackingStore::InMemory(Self::vec_from_bytes(bytes)?);
                Ok(())
            }
        }
    }

    /// Write interleaved `[left[i], right[i]]` pairs starting at pair index `start_pair`.
    pub(crate) fn write_interleaved_at(&mut self, start_pair: usize, left: &[F], right: &[F]) -> io::Result<()>
    where
        F: Copy,
    {
        if left.len() != right.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "BackingStore::write_interleaved_at length mismatch: left={} right={}",
                    left.len(),
                    right.len()
                ),
            ));
        }
        if left.is_empty() {
            return Ok(());
        }
        let mut interleaved = Vec::with_capacity(left.len() * 2);
        for i in 0..left.len() {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }
        self.write_at(start_pair.saturating_mul(2), &interleaved)
    }

    /// Consume a range and reclaim backing storage in non-reuse mode.
    ///
    /// `start..end` are element indices (not byte offsets).
    /// No-op for `InMemory` / `Empty` (in-memory data dies with the process).
    ///
    pub(crate) fn consume(&mut self, start: usize, end: usize) {
        let (file, path, len) = match self {
            BackingStore::FileBacked { file, path, len, .. } => (file, path, *len),
            _ => return,
        };

        #[cfg(feature = "reuse-preproc")]
        {
            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    path = %path.display(),
                    start,
                    end,
                    "BackingStore::consume noop (reuse-preproc)"
                );
            }
            let _ = (file, path, start, end);
            return;
        }

        #[cfg(not(feature = "reuse-preproc"))]
        {
            if start > end || end > len {
                if tracing::enabled!(tracing::Level::DEBUG) {
                    tracing::debug!(
                        path = %path.display(),
                        start,
                        end,
                        len,
                        "BackingStore::consume invalid range"
                    );
                }
                return;
            }

            let (byte_offset, byte_len) = Self::byte_range(start, end);
            if byte_len == 0 {
                return;
            }

            #[cfg(target_os = "linux")]
            {
                use std::os::unix::io::AsRawFd;
                let fd = file.as_raw_fd();
                // SAFETY: libc call; parameters are validated and come from file offsets.
                let rc = unsafe {
                    libc::fallocate(
                        fd,
                        libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                        byte_offset as libc::off_t,
                        byte_len as libc::off_t,
                    )
                };
                if rc == 0 {
                    if tracing::enabled!(tracing::Level::TRACE) {
                        tracing::trace!(
                            path = %path.display(),
                            start,
                            end,
                            byte_offset,
                            byte_len,
                            "BackingStore::consume punched hole"
                        );
                    }
                    return;
                }

                let err = io::Error::last_os_error();
                let errno = err.raw_os_error().unwrap_or(0);
                // If unsupported, fall back to explicit zeroing.
                if matches!(errno, libc::EOPNOTSUPP | libc::ENOSYS | libc::EINVAL | libc::ENODEV) {
                    if tracing::enabled!(tracing::Level::DEBUG) {
                        tracing::debug!(
                            path = %path.display(),
                            start,
                            end,
                            byte_offset,
                            byte_len,
                            errno,
                            "BackingStore::consume punch-hole unsupported; falling back to zero"
                        );
                    }
                } else {
                    if tracing::enabled!(tracing::Level::DEBUG) {
                        tracing::debug!(
                            path = %path.display(),
                            start,
                            end,
                            byte_offset,
                            byte_len,
                            errno,
                            err = %err,
                            "BackingStore::consume punch-hole failed"
                        );
                    }
                }
            }

            // Fallback: overwrite with zeros in fixed-size chunks.
            let mut remaining = byte_len;
            let mut cur = byte_offset;
            let zero_buf = [0u8; 64 * 1024];
            while remaining > 0 {
                let chunk = remaining.min(zero_buf.len());
                if let Err(e) = Self::file_write_all_at(file, cur, &zero_buf[..chunk]) {
                    if tracing::enabled!(tracing::Level::DEBUG) {
                        tracing::debug!(
                            path = %path.display(),
                            start,
                            end,
                            byte_offset,
                            byte_len,
                            cur,
                            chunk,
                            err = %e,
                            "BackingStore::consume zero fallback failed"
                        );
                    }
                    break;
                }
                cur += chunk as u64;
                remaining -= chunk;
            }

            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    path = %path.display(),
                    start,
                    end,
                    byte_offset,
                    byte_len,
                    "BackingStore::consume zeroed range"
                );
            }
        }
    }
}

impl<F> BackingStoreReadView<'_, F> {
    fn validate_range(&self, start: usize, end: usize) -> io::Result<()> {
        let len = match self {
            BackingStoreReadView::InMemory(v) => v.len(),
            BackingStoreReadView::FileBacked { len, .. } => *len,
            BackingStoreReadView::Empty => 0,
        };
        if start > end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid range: start({start}) > end({end})"),
            ));
        }
        if end > len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("range end({end}) exceeds len({len})"),
            ));
        }
        Ok(())
    }

    pub(crate) fn read_into_slice(&self, start: usize, end: usize, out: &mut [F]) -> io::Result<()>
    where
        F: Copy,
    {
        let count = end.saturating_sub(start);
        if out.len() != count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("slice length {} does not match requested element count {}", out.len(), count),
            ));
        }
        match self {
            BackingStoreReadView::InMemory(v) => {
                self.validate_range(start, end)?;
                out.copy_from_slice(&v[start..end]);
                Ok(())
            }
            BackingStoreReadView::FileBacked { file, .. } => {
                self.validate_range(start, end)?;
                let (byte_offset, byte_len) = BackingStore::<F>::byte_range(start, end);
                let out_bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, byte_len) };
                BackingStore::<F>::file_read_exact_at(file, byte_offset, out_bytes)
            }
            BackingStoreReadView::Empty => {
                if !out.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "attempted to read from empty backing store",
                    ));
                }
                Ok(())
            }
        }
    }
}

impl<F> FileBackedWriter<F> {
    pub(crate) fn write_at(&self, start_elem: usize, data: &[F]) -> io::Result<()>
    where
        F: Copy,
    {
        if data.is_empty() {
            return Ok(());
        }
        let end_elem = start_elem.saturating_add(data.len());
        if end_elem > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("FileBackedWriter::write_at range end({end_elem}) exceeds len({})", self.len),
            ));
        }
        let (byte_offset, _) = BackingStore::<F>::byte_range(start_elem, end_elem);
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        BackingStore::<F>::file_write_all_at(&self.file, byte_offset, bytes)
    }

    pub(crate) fn write_interleaved_at(&self, start_pair: usize, left: &[F], right: &[F]) -> io::Result<()>
    where
        F: Copy,
    {
        if left.len() != right.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "FileBackedWriter::write_interleaved_at length mismatch: left={} right={}",
                    left.len(),
                    right.len()
                ),
            ));
        }
        if left.is_empty() {
            return Ok(());
        }
        let mut interleaved = Vec::with_capacity(left.len() * 2);
        for i in 0..left.len() {
            interleaved.push(left[i]);
            interleaved.push(right[i]);
        }
        self.write_at(start_pair.saturating_mul(2), &interleaved)
    }

    pub(crate) fn write_bytes_at(&self, start_elem: usize, bytes: &[u8]) -> io::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        const { assert_field_layout::<F>() };
        let elem_size = BackingStore::<F>::elem_size_bytes();
        if bytes.len() % elem_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "FileBackedWriter::write_bytes_at byte length {} is not divisible by element size {}",
                    bytes.len(),
                    elem_size,
                ),
            ));
        }
        let elems = bytes.len() / elem_size;
        let end_elem = start_elem.saturating_add(elems);
        if end_elem > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("FileBackedWriter::write_bytes_at range end({end_elem}) exceeds len({})", self.len),
            ));
        }
        let (byte_offset, _) = BackingStore::<F>::byte_range(start_elem, end_elem);
        BackingStore::<F>::file_write_all_at(&self.file, byte_offset, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::BackingStore;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("{}_{}", name, nanos))
    }

    #[test]
    fn write_at_file_backed_correct() {
        let path = temp_path("backing_store_write_at");
        let mut store = BackingStore::<u64>::create_file_backed_sized(&path, 8).unwrap();
        store.write_at(0, &[10, 11, 12]).unwrap();
        store.write_at(3, &[20, 21, 22, 23, 24]).unwrap();

        let loaded = BackingStore::<u64>::load_from_file(&path).unwrap();
        #[cfg(feature = "reuse-preproc")]
        let data = loaded.read_reuse(0, 8).unwrap();
        #[cfg(not(feature = "reuse-preproc"))]
        let data = loaded.read_consume(0, 8).unwrap();
        assert_eq!(data, vec![10, 11, 12, 20, 21, 22, 23, 24]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_interleaved_at_file_backed_correct() {
        let path = temp_path("backing_store_write_interleaved");
        let mut store = BackingStore::<u64>::create_file_backed_sized(&path, 6).unwrap();
        store.write_interleaved_at(0, &[1, 2], &[11, 12]).unwrap();
        store.write_interleaved_at(2, &[3], &[13]).unwrap();

        let loaded = BackingStore::<u64>::load_from_file(&path).unwrap();
        #[cfg(feature = "reuse-preproc")]
        let data = loaded.read_reuse(0, 6).unwrap();
        #[cfg(not(feature = "reuse-preproc"))]
        let data = loaded.read_consume(0, 6).unwrap();
        assert_eq!(data, vec![1, 11, 2, 12, 3, 13]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_bytes_at_file_backed_correct() {
        let path = temp_path("backing_store_write_bytes_at");
        let mut store = BackingStore::<u64>::create_file_backed_sized(&path, 4).unwrap();
        let src = [7u64, 8, 9, 10];
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(&src)) };
        store.write_bytes_at(0, bytes).unwrap();

        let loaded = BackingStore::<u64>::load_from_file(&path).unwrap();
        #[cfg(feature = "reuse-preproc")]
        let data = loaded.read_reuse(0, 4).unwrap();
        #[cfg(not(feature = "reuse-preproc"))]
        let data = loaded.read_consume(0, 4).unwrap();
        assert_eq!(data, src);

        let _ = std::fs::remove_file(path);
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
    w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
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
    Ok(MetaData { seed1, pos1, seed2, pos2, total, party_id_byte, cursor, field_bytes })
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
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, format!("invalid party_id byte: {b}"))),
    }
}
