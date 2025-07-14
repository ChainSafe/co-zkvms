use co_spartan::mpc::rep3::Rep3PrimeFieldShare;
use jolt_core::poly::field::JoltField;
use mpc_core::protocols::rep3;


/// Concatenates `C` `vals` field elements each of max size 2^`operand_bits`-1
/// into a single field element. `operand_bits` is the number of bits required to represent
/// each element in `vals`. If an element of `vals` is larger it will not be truncated, which
/// is commonly used by the collation functions of instructions.
pub fn concatenate_lookups<F: JoltField>(vals: &[F], C: usize, operand_bits: usize) -> F {
    assert_eq!(vals.len(), C);

    let mut sum = F::zero();
    let mut weight = F::one();
    let shift = F::from_u64(1u64 << operand_bits).unwrap();
    for i in 0..C {
        sum += weight * vals[C - i - 1];
        weight *= shift;
    }
    sum
}

pub fn concatenate_lookups_rep3<F: JoltField>(vals: &[Rep3PrimeFieldShare<F>], C: usize, operand_bits: usize) -> Rep3PrimeFieldShare<F> {
    assert_eq!(vals.len(), C);

    let mut sum = Rep3PrimeFieldShare::zero_share();
    let mut weight = F::one();
    let shift = F::from_u64(1u64 << operand_bits).unwrap();
    for i in 0..C {
        sum += rep3::arithmetic::mul_public(vals[C - i - 1], weight);
        weight *= shift;
    }
    sum
}

