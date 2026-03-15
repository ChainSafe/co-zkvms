//! RandOHV preprocessing for instruction-chunk one-hot masking.
//!
//! This stores full precomputed material for the current `u8` / `K_CHUNK=16`
//! use case:
//! - `r_share`: replicated ring share of the mask index
//! - `e_field`: replicated field-sharing of the one-hot vector
//!
//! Unlike edaBits/daBits, the required count is tiny, so this module keeps the
//! implementation simple and persists the full material via `BackingStore`.

use crate::IoResult;
use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::Rep3PrimeFieldShare;
use crate::protocols::rep3::network::{IoContext, Rep3Network};
use crate::protocols::rep3_ring::Rep3RingShare;
use crate::protocols::rep3_ring::conversion;
use crate::protocols::rep3_ring::gadgets;
use crate::protocols::rep3_ring::ring::bit::Bit;
use crate::field::PrimeField;
use rand::distributions::Standard;
use rand::prelude::Distribution;

use super::backing_store;

pub const RAND_OHV_LOG_K: usize = 4;
pub const RAND_OHV_K: usize = 1 << RAND_OHV_LOG_K;

pub struct RandOhvBatch<F: PrimeField> {
    pub r_shares: Vec<Rep3RingShare<u8>>,
    pub e_fields_flat: Vec<Rep3PrimeFieldShare<F>>,
}

pub struct LazyRandOhvs<F: PrimeField> {
    total: usize,
    cursor: usize,
    party_id: PartyID,
    meta_path: Option<std::path::PathBuf>,
    r_a: backing_store::BackingStore<u8>,
    r_b: backing_store::BackingStore<u8>,
    e_a_flat: backing_store::BackingStore<F>,
    e_b_flat: backing_store::BackingStore<F>,
}

impl<F: PrimeField> LazyRandOhvs<F> {
    pub fn empty(party_id: PartyID) -> Self {
        Self {
            total: 0,
            cursor: 0,
            party_id,
            meta_path: None,
            r_a: backing_store::BackingStore::Empty,
            r_b: backing_store::BackingStore::Empty,
            e_a_flat: backing_store::BackingStore::Empty,
            e_b_flat: backing_store::BackingStore::Empty,
        }
    }

    pub fn new(
        total: usize,
        party_id: PartyID,
        r_a: Vec<u8>,
        r_b: Vec<u8>,
        e_a_flat: Vec<F>,
        e_b_flat: Vec<F>,
    ) -> Self {
        Self {
            total,
            cursor: 0,
            party_id,
            meta_path: None,
            r_a: backing_store::BackingStore::from_vec(r_a),
            r_b: backing_store::BackingStore::from_vec(r_b),
            e_a_flat: backing_store::BackingStore::from_vec(e_a_flat),
            e_b_flat: backing_store::BackingStore::from_vec(e_b_flat),
        }
    }

    pub fn remaining(&self) -> usize {
        self.total - self.cursor
    }

    #[cfg(feature = "reuse-preproc")]
    pub(crate) fn reset_cursor_for_reuse(&mut self) {
        self.cursor = 0;
    }

    pub fn take_batch(&mut self, n: usize) -> eyre::Result<RandOhvBatch<F>> {
        eyre::ensure!(
            self.cursor + n <= self.total,
            "LazyRandOhvs: need {n}, have {} (cursor={}, total={})",
            self.remaining(),
            self.cursor,
            self.total
        );

        if n == 0 {
            return Ok(RandOhvBatch {
                r_shares: Vec::new(),
                e_fields_flat: Vec::new(),
            });
        }

        let cursor_base = self.cursor;
        let e_start = cursor_base * RAND_OHV_K;
        let e_end = e_start + n * RAND_OHV_K;

        let read_ring = |store: &backing_store::BackingStore<u8>,
                         start: usize,
                         end: usize|
         -> eyre::Result<Vec<u8>> {
            #[cfg(feature = "reuse-preproc")]
            {
                Ok(store.read_reuse(start, end)?)
            }
            #[cfg(not(feature = "reuse-preproc"))]
            {
                Ok(store.read_consume(start, end)?)
            }
        };

        let read_field =
            |store: &backing_store::BackingStore<F>, start: usize, end: usize| -> eyre::Result<Vec<F>> {
                #[cfg(feature = "reuse-preproc")]
                {
                    Ok(store.read_reuse(start, end)?)
                }
                #[cfg(not(feature = "reuse-preproc"))]
                {
                    Ok(store.read_consume(start, end)?)
                }
            };

        let r_a = read_ring(&self.r_a, cursor_base, cursor_base + n)?;
        let r_b = read_ring(&self.r_b, cursor_base, cursor_base + n)?;
        let e_a_flat = read_field(&self.e_a_flat, e_start, e_end)?;
        let e_b_flat = read_field(&self.e_b_flat, e_start, e_end)?;

        let r_shares = (0..n).map(|i| Rep3RingShare::new(r_a[i], r_b[i])).collect();
        let e_fields_flat = (0..n * RAND_OHV_K)
            .map(|i| Rep3PrimeFieldShare::new(e_a_flat[i], e_b_flat[i]))
            .collect();

        self.cursor += n;
        self.persist_cursor();
        self.r_a.consume(cursor_base, cursor_base + n);
        self.r_b.consume(cursor_base, cursor_base + n);
        self.e_a_flat.consume(e_start, e_end);
        self.e_b_flat.consume(e_start, e_end);

        Ok(RandOhvBatch {
            r_shares,
            e_fields_flat,
        })
    }

    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        const { backing_store::assert_field_layout::<F>() };
        std::fs::create_dir_all(dir)?;

        if !self.r_a.is_empty() {
            self.r_a.save_to_file(&dir.join("rand_ohv_u8_k4.r_a"))?;
        }
        if !self.r_b.is_empty() {
            self.r_b.save_to_file(&dir.join("rand_ohv_u8_k4.r_b"))?;
        }
        if !self.e_a_flat.is_empty() {
            self.e_a_flat.save_to_file(&dir.join("rand_ohv_u8_k4.e_a"))?;
        }
        if !self.e_b_flat.is_empty() {
            self.e_b_flat.save_to_file(&dir.join("rand_ohv_u8_k4.e_b"))?;
        }

        backing_store::write_meta(
            &dir.join("rand_ohv_u8_k4.meta"),
            &backing_store::MetaData {
                seed1: [0u8; crate::SEED_SIZE],
                pos1: 0,
                seed2: [0u8; crate::SEED_SIZE],
                pos2: 0,
                total: self.total,
                party_id_byte: backing_store::party_id_to_byte(self.party_id),
                cursor: self.cursor,
                field_bytes: std::mem::size_of::<F>(),
            },
        )?;
        Ok(())
    }

    pub fn load(dir: &std::path::Path, party_id: PartyID) -> std::io::Result<Self> {
        const { backing_store::assert_field_layout::<F>() };

        let meta_path = dir.join("rand_ohv_u8_k4.meta");
        if !meta_path.exists() {
            return Ok(Self::empty(party_id));
        }

        let meta = backing_store::read_meta(&meta_path)?;
        assert_eq!(
            meta.party_id_byte,
            backing_store::party_id_to_byte(party_id)
        );

        let r_a = if meta.total > 0 {
            backing_store::BackingStore::load_from_file(&dir.join("rand_ohv_u8_k4.r_a"))?
        } else {
            backing_store::BackingStore::Empty
        };
        let r_b = if meta.total > 0 {
            backing_store::BackingStore::load_from_file(&dir.join("rand_ohv_u8_k4.r_b"))?
        } else {
            backing_store::BackingStore::Empty
        };
        let e_a_flat = if meta.total > 0 {
            backing_store::BackingStore::load_from_file(&dir.join("rand_ohv_u8_k4.e_a"))?
        } else {
            backing_store::BackingStore::Empty
        };
        let e_b_flat = if meta.total > 0 {
            backing_store::BackingStore::load_from_file(&dir.join("rand_ohv_u8_k4.e_b"))?
        } else {
            backing_store::BackingStore::Empty
        };

        Ok(Self {
            total: meta.total,
            cursor: meta.cursor,
            party_id,
            meta_path: Some(meta_path),
            r_a,
            r_b,
            e_a_flat,
            e_b_flat,
        })
    }

    fn persist_cursor(&self) {
        if let Some(ref path) = self.meta_path {
            let _ = backing_store::update_cursor(path, self.cursor);
        }
    }

    pub(crate) fn apply_extension(
        &mut self,
        deficit: usize,
        r_a_ext: Vec<u8>,
        r_b_ext: Vec<u8>,
        e_a_ext: Vec<F>,
        e_b_ext: Vec<F>,
    ) {
        if deficit == 0 {
            return;
        }
        if !r_a_ext.is_empty() {
            self.r_a.extend(&r_a_ext);
        }
        if !r_b_ext.is_empty() {
            self.r_b.extend(&r_b_ext);
        }
        if !e_a_ext.is_empty() {
            self.e_a_flat.extend(&e_a_ext);
        }
        if !e_b_ext.is_empty() {
            self.e_b_flat.extend(&e_b_ext);
        }
        self.total += deficit;
    }
}

impl<F: PrimeField> Drop for LazyRandOhvs<F> {
    fn drop(&mut self) {
        #[cfg(not(feature = "reuse-preproc"))]
        self.persist_cursor();
    }
}

pub fn generate_rand_ohvs_lazy<F: PrimeField, N: Rep3Network>(
    n: usize,
    io: &mut IoContext<N>,
) -> IoResult<LazyRandOhvs<F>>
where
    Standard: Distribution<u8>,
{
    let party_id = io.id;
    if n == 0 {
        return Ok(LazyRandOhvs::empty(party_id));
    }

    let mut r_a = Vec::with_capacity(n);
    let mut r_b = Vec::with_capacity(n);
    let mut e_a_flat = Vec::with_capacity(n * RAND_OHV_K);
    let mut e_b_flat = Vec::with_capacity(n * RAND_OHV_K);

    for _ in 0..n {
        let (r_share, e_bits): (Rep3RingShare<u8>, Vec<Rep3RingShare<Bit>>) =
            gadgets::ohv::rand_ohv::<u8, _>(RAND_OHV_LOG_K, io)?;
        let e_field = conversion::bit_inject_from_bits_to_field_many(&e_bits, io)?;
        r_a.push(r_share.a.0);
        r_b.push(r_share.b.0);
        for share in e_field {
            e_a_flat.push(share.a);
            e_b_flat.push(share.b);
        }
    }

    Ok(LazyRandOhvs::new(
        n, party_id, r_a, r_b, e_a_flat, e_b_flat,
    ))
}
