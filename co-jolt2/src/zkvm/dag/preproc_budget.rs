use crate::zkvm::suffixes::suffix_edabit_ring_bits;
use jolt_common::constants::XLEN;
use jolt_core::zkvm::instruction_lookups::LOG_M;
use jolt_core::zkvm::lookup_table::suffixes::Suffixes;
use jolt_core::zkvm::lookup_table::LookupTables;
use strum::IntoEnumIterator;

const PHASES: usize = 8;

/// Per-ring-type EdaBit counts and daBit count needed for the ReadRaf sumcheck.
///
/// Computed from the actual lookup table structure, not hardcoded multipliers.
#[derive(Clone, Default)]
pub struct PreprocessingBudget {
    pub u8: usize,
    pub u16: usize,
    pub u32: usize,
    pub u64: usize,
    pub u128: usize,
    /// daBits for BitInject (single-bit → field) suffix conversions.
    pub dabits: usize,
    /// daPoints for Dory U64Scalars wrap correction (2 per committed coefficient).
    pub dapoints: usize,
    /// Wrap masks for DaBit-based wrap-m extraction (1 per committed coefficient).
    pub wrap_masks: usize,
    /// Ring edaBits (U66) for ring-domain B2A in Dory wrap correction (1 per committed coefficient).
    pub ring_edabits_u66: usize,
}

impl std::fmt::Debug for PreprocessingBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "EdaBits: u8={}, u16={}, u32={}, u64={}, u128={}; daBits: {}; daPoints: {}; wrapMasks: {}; ringEdaBitsU66: {}",
            self.u8, self.u16, self.u32, self.u64, self.u128, self.dabits, self.dapoints, self.wrap_masks, self.ring_edabits_u66
        ))
    }
}

/// Compute the EdaBit budget needed for the padded trace.
///
/// Pass the **padded** trace length (`trace.len()`), which is always a power of 2.
/// This must be the padded length because witness-gen consumers (rd_inc, ram_inc)
/// operate on the full padded trace, not just non-NoOp cycles.
///
/// Three consumer groups:
/// 1. **Suffix eval** (per-table, per-phase): each non-NoOp cycle belongs to
///    exactly one table.  Worst case per ring bucket:
///    `max_over_tables(B2A_suffix_count_in_bucket) × n`.
/// 2. **Operand Q** (per-phase): `n` edaBits of type T (identity) + `2n` of
///    T::Half (left + right).  Overestimates because identity and interleaved
///    cycles are disjoint, but the split is unknown at budget time.
/// 3. **Witness gen**: `5n` XlenInt (sparse operand cast, worst case) +
///    `4n` XlenInt (rd_inc + ram_inc, each `2n`).
///
/// Phase ring types: suffix_len = (7 - phase) * LOG_M
///
///   rv32 (LOG_M=8):
///     Phase 0-2 (suffix 56,48,40): T=u64,  T::Half=u32
///     Phase 3-4 (suffix 32,24):    T=u32,  T::Half=u16
///     Phase 5-6 (suffix 16, 8):    T=u16,  T::Half=u8
///     Phase 7   (suffix  0):       skip
///
///   rv64 (LOG_M=16):
///     Phase 0-2 (suffix 112,96,80): T=u128, T::Half=u64
///     Phase 3-4 (suffix 64,48):     T=u64,  T::Half=u32
///     Phase 5   (suffix 32):        T=u32,  T::Half=u16
///     Phase 6   (suffix 16):        T=u16,  T::Half=u8
///     Phase 7   (suffix  0):        skip
pub fn compute_edabit_budget(trace_len: usize) -> PreprocessingBudget {
    let n = trace_len;
    let mut budget = PreprocessingBudget::default();

    // daBit budget: per cycle, sum BitInject suffixes across all phases.
    // Each cycle belongs to exactly one table, so budget = max_over_tables × n.
    let max_dabits_per_cycle = LookupTables::<XLEN>::iter()
        .map(|table| {
            let mut total = 0usize;
            for phase in 0..PHASES {
                let suffix_len = (PHASES - 1 - phase) * LOG_M;
                if suffix_len == 0 {
                    continue;
                }
                let t_k = match suffix_len {
                    65..=128 => 128usize,
                    33..=64 => 64,
                    17..=32 => 32,
                    1..=16 => 16,
                    _ => unreachable!(),
                };
                for suffix in table.suffixes() {
                    if suffix_is_bitinject(&suffix, t_k, suffix_len) {
                        total += 1;
                    }
                }
            }
            total
        })
        .max()
        .unwrap_or(0);
    budget.dabits = max_dabits_per_cycle * n;
    // Dory U64Scalars wrap correction: 2 daPoints per committed coefficient.
    budget.dapoints = 2 * n;
    // Wrap masks for DaBit-based wrap-m extraction: 1 per committed coefficient.
    budget.wrap_masks = n;
    // Ring edaBits (U66) for ring-domain B2A: 1 per committed coefficient.
    budget.ring_edabits_u66 = n;

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

    // Witness generation: binary→field B2A for operand columns and inc polynomials.
    // All three consumers use XlenInt (u32 for rv32, u64 for rv64):
    // - Sparse operand cast (5 columns × n, worst case): 5n EdaBits
    // - rd_inc/ram_inc (2 pre + 2 post × n): 4n EdaBits
    add_to_budget(&mut budget, XLEN, 9 * n);

    budget
}

/// Returns true if this suffix produces a BitInject result (consuming 1 daBit)
/// for the given phase parameters.
///
/// - `t_k`: bit-width of ring T for this phase (16, 32, 64, or 128)
/// - `suffix_len`: suffix length in bits for this phase
fn suffix_is_bitinject(suffix: &Suffixes, t_k: usize, suffix_len: usize) -> bool {
    match suffix {
        // Always BitInject regardless of phase
        Suffixes::Lsb
        | Suffixes::TwoLsb
        | Suffixes::LessThan
        | Suffixes::GreaterThan
        | Suffixes::Eq
        | Suffixes::LeftOperandIsZero
        | Suffixes::RightOperandIsZero
        | Suffixes::DivByZero
        | Suffixes::ChangeDivisor
        | Suffixes::ChangeDivisorW => true,

        // BitInject only when suffix_len >= XLEN/2 (sign bit is within the suffix window)
        Suffixes::SignExtensionUpperHalf => suffix_len >= XLEN / 2,

        // BitInject only when T::K > XLEN (upper bits exist to check for zero)
        Suffixes::OverflowBitsZero => t_k > XLEN,

        // All other suffixes are either Ready or B2A edaBit conversions
        _ => false,
    }
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
