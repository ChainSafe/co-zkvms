/// A `Send + Sync` wrapper around a raw mutable pointer for use in parallel
/// scatter operations where the caller guarantees disjoint write indices.
#[derive(Copy, Clone)]
pub(crate) struct SendPtr<T>(pub *mut T);

// SAFETY: the caller must ensure that distinct threads write to distinct
// memory locations (disjoint index sets).
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}
