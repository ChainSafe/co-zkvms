use crate::zkvm::suffixes::suffix_edabit_ring_bits;
use jolt2_common::constants::XLEN;
use jolt_core::zkvm::instruction_lookups::LOG_M;
use jolt_core::zkvm::lookup_table::LookupTables;
use strum::IntoEnumIterator;

const PHASES: usize = 8;

/// Per-ring-type EdaBit counts needed for the ReadRaf sumcheck.
///
/// Computed from the actual lookup table structure, not hardcoded multipliers.
#[derive(Clone, Default)]
pub struct PreprocessingBudget {
    pub u8: usize,
    pub u16: usize,
    pub u32: usize,
    pub u64: usize,
    pub u128: usize,
}

impl std::fmt::Debug for PreprocessingBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "EdaBits: u8={}, u16={}, u32={}, u64={}, u128={}",
            self.u8, self.u16, self.u32, self.u64, self.u128
        ))
    }
}

/// Compute the EdaBit budget needed for the given number of non-noop cycles.
///
/// Pass the un-padded trace length (upper bound on real non-NoOp cycles).
/// NoOp padding cycles produce `None` in `instruction_ra` / `masked_indices_c`
/// and are excluded from `non_noop_cycles` in ReadRaf, so the budget only needs
/// to cover real instruction cycles.
///
/// Two consumers:
/// 1. **Suffix eval** (per-table): each cycle belongs to exactly one table.
///    For that table, each B2A suffix consumes 1 edaBit of the ring type given
///    by `suffix_edabit_ring_bits`. The worst case per ring bucket is
///    `max_over_tables(count_of_B2A_suffixes_in_bucket) × n`.
/// 2. **Operand Q**: `n` edaBits of type T (identity) + `2n` of T::Half
///    (left + right) per phase.
///
/// Phase ring types: suffix_len = (7 - phase) * 16
///   Phase 0-2: T=u128, T::Half=u64
///   Phase 3-4: T=u64,  T::Half=u32
///   Phase 5:   T=u32,  T::Half=u16
///   Phase 6:   T=u16,  T::Half=u8
///   Phase 7:   suffix_len=0, no B2A needed
pub fn compute_edabit_budget(non_noop_cycles: usize) -> PreprocessingBudget {
    let n = non_noop_cycles;
    let mut budget = PreprocessingBudget::default();

    for phase in 0..PHASES {
        let suffix_len = (PHASES - 1 - phase) * LOG_M;
        if suffix_len == 0 {
            continue;
        }

        let (t_k, t_half_k) = match suffix_len {
            65..=128 => (128usize, 64usize),
            33..=64 => (64, 32),
            17..=32 => (32, 16),
            1..=16 => (16, 8),
            _ => unreachable!(),
        };

        // Operand Q: n × T + 2n × T::Half
        add_to_budget(&mut budget, t_k, n);
        add_to_budget(&mut budget, t_half_k, 2 * n);

        // Suffix eval: each cycle belongs to exactly one table, so the per-bucket
        // consumption is bounded by the table with the most B2A suffixes in that
        // bucket. Compute max_per_table_count for each ring bucket, then budget
        // max_count × n.
        let mut max_per_bucket = [0usize; 5];
        for table in LookupTables::<XLEN>::iter() {
            let mut table_counts = [0usize; 5];
            for suffix in table.suffixes() {
                if let Some(ring_bits) = suffix_edabit_ring_bits(&suffix, t_k, t_half_k) {
                    table_counts[ring_bucket(ring_bits)] += 1;
                }
            }
            for b in 0..5 {
                max_per_bucket[b] = max_per_bucket[b].max(table_counts[b]);
            }
        }
        const BUCKET_BITS: [usize; 5] = [8, 16, 32, 64, 128];
        for (b, &bits) in BUCKET_BITS.iter().enumerate() {
            if max_per_bucket[b] > 0 {
                add_to_budget(&mut budget, bits, max_per_bucket[b] * n);
            }
        }
    }

    budget
}

fn ring_bucket(ring_bits: usize) -> usize {
    match ring_bits {
        1..=8 => 0,
        9..=16 => 1,
        17..=32 => 2,
        33..=64 => 3,
        65..=128 => 4,
        _ => unreachable!("unsupported ring bit-width: {}", ring_bits),
    }
}

fn add_to_budget(budget: &mut PreprocessingBudget, ring_bits: usize, count: usize) {
    match ring_bucket(ring_bits) {
        0 => budget.u8 += count,
        1 => budget.u16 += count,
        2 => budget.u32 += count,
        3 => budget.u64 += count,
        4 => budget.u128 += count,
        _ => unreachable!(),
    }
}
