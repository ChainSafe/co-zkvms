use itertools::{izip, Itertools};
pub use jolt_core::utils::instruction_utils::*;

use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};
use num_bigint::BigUint;
use std::{collections::HashMap, ops::Shr};

use crate::jolt::instruction::JoltInstruction;

pub fn concatenate_lookups_rep3<F: JoltField>(
    vals: &[Rep3PrimeFieldShare<F>],
    C: usize,
    operand_bits: usize,
) -> Rep3PrimeFieldShare<F> {
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

pub fn concatenate_lookups_rep3_batched<F: JoltField>(
    vals: impl IntoIterator<
        Item = Vec<Rep3PrimeFieldShare<F>>,
        IntoIter: DoubleEndedIterator + ExactSizeIterator,
    >,
    C: usize,
    operand_bits: usize,
) -> Vec<Rep3PrimeFieldShare<F>> {
    let mut vals_rev = vals.into_iter().rev();
    assert_eq!(vals_rev.len(), C);
    let mut sums = vals_rev.next().unwrap();
    let shift = F::from_u64(1u64 << operand_bits).unwrap();
    let mut weight = shift;
    for val in vals_rev {
        // sum += rep3::arithmetic::mul_public(vals[C - i - 1], weight);
        izip!(sums.iter_mut(), val).for_each(|(sum, val)| {
            *sum += rep3::arithmetic::mul_public(val, weight);
        });
        weight *= shift;
    }
    sums
}

pub fn rep3_chunk_and_concatenate_operands<F: JoltField>(
    x: Rep3BigUintShare<F>,
    y: Rep3BigUintShare<F>,
    C: usize,
    log_M: usize,
) -> Vec<Rep3BigUintShare<F>> {
    let operand_bits: usize = log_M / 2;

    let operand_bit_mask = BigUint::from(((1 << operand_bits) - 1) as u64);
    (0..C)
        .map(|i| {
            let shift = (C - i - 1) * operand_bits;
            let left = x.clone().shr(shift) & operand_bit_mask.clone();
            let right = y.clone().shr(shift) & operand_bit_mask.clone();
            // (left << operand_bits) | right
            // since we performed left shift the right part are all zero bits so we can do XOR instead of OR
            (left << operand_bits) ^ right
        })
        .collect()
}

pub fn transpose<I, T>(matrix: I) -> Vec<Vec<T>>
where
    I: IntoIterator<Item = Vec<T>>,
{
    let mut it = matrix.into_iter();
    let first_row = match it.next() {
        Some(r) => r,
        None => return Vec::new(),
    };
    let cols = first_row.len();
    let (low, _) = it.size_hint();
    let mut out: Vec<Vec<T>> = (0..cols).map(|_| Vec::with_capacity(low + 1)).collect();

    // push first row
    for (c, v) in first_row.into_iter().enumerate() {
        out[c].push(v);
    }
    // push remaining rows
    for row in it {
        assert_eq!(row.len(), cols, "ragged matrix");
        for (c, v) in row.into_iter().enumerate() {
            out[c].push(v);
        }
    }
    out
}

pub fn transpose_flatten<I, T>(matrix: I) -> Vec<Vec<T>>
where
    I: IntoIterator<Item = Vec<Vec<T>>>, // [R][C][D] with D possibly var-length
{
    let mut rows = matrix.into_iter();
    let first = match rows.next() {
        Some(r) => r,
        None => return Vec::new(),
    };
    let cols = first.len();
    let (low, _) = rows.size_hint();
    // estimate avg depth from first row
    let avg_depth = if cols > 0 {
        first.iter().map(Vec::len).sum::<usize>() / cols
    } else {
        0
    };
    // pre-allocate each column to (rows_est × avg_depth)
    let mut out: Vec<Vec<T>> = (0..cols)
        .map(|_| Vec::with_capacity((low + 1) * avg_depth))
        .collect();

    // flatten first row
    for (c, dv) in first.into_iter().enumerate() {
        out[c].extend(dv);
    }
    // flatten remaining rows
    for row in rows {
        assert_eq!(row.len(), cols, "ragged cols");
        for (c, dv) in row.into_iter().enumerate() {
            out[c].extend(dv);
        }
    }
    out
}

pub fn transpose_hashmap<T>(rows: Vec<HashMap<usize, T>>) -> HashMap<usize, Vec<T>> {
    let mut out: HashMap<usize, Vec<T>> = HashMap::new();
    for (i, row) in rows.into_iter().enumerate() {
        for (k, v) in row {
            out.entry(k).or_default().push(v);
        }
    }
    out
}

pub fn chunks_take_nth<'a, T>(
    data: &'a [T],
    chunk_len: usize,
    step: usize,
) -> impl Iterator<Item = impl Iterator<Item = &'a T>> {
    // for each offset 0‥step-1 build a strided view
    (0..step).map(move |off| data.iter().skip(off).step_by(step).take(chunk_len))
}

#[cfg(test)]
mod test {
    use super::*;

    use ark_std::test_rng;

    type F = ark_bn254::Fr;

    #[test]
    fn test_transpose_flatten() {
        let matrix = vec![vec![vec![(); 8]; 4], vec![vec![(); 8]; 4]];
        let transposed = transpose_flatten(matrix);
        assert_eq!(
            transposed.iter().map(|v| v.len()).collect::<Vec<_>>(),
            vec![16; 4]
        );

        let matrix = vec![vec![vec![(); 7]; 4], vec![vec![(); 8]; 4]];
        let transposed = transpose_flatten(matrix);
        assert_eq!(
            transposed.iter().map(|v| v.len()).collect::<Vec<_>>(),
            vec![15; 4]
        );
    }

    #[test]
    fn test_chunk_and_concatenate_operands() {
        let x = 0b10101010101010;
        let y = 0b11001100110011;
        let C = 4;
        let log_M = 8;
        let indices = chunk_and_concatenate_operands(x, y, C, log_M);
        let indices_alt = chunk_and_concatenate_operands_alt(x, y, C, log_M);
        assert_eq!(indices, indices_alt);
        println!("{:?}", indices);
    }

    fn chunk_and_concatenate_operands_alt(x: u64, y: u64, C: usize, log_M: usize) -> Vec<usize> {
        let operand_bits: usize = log_M / 2;

        #[cfg(test)]
        {
            let max_operand_bits = C * log_M / 2;
            println!("max_operand_bits: {}", max_operand_bits);
            if max_operand_bits != 64 {
                // if 64, handled by normal overflow checking
                let max_operand: u64 = (1 << max_operand_bits) - 1;
                assert!(x <= max_operand);
                assert!(y <= max_operand);
            }
        }

        let operand_bit_mask: u64 = (1 << operand_bits) - 1;
        (0..C)
            .map(|i| {
                let shift = ((C - i - 1) * operand_bits) as u32;
                let left = x.shr(shift) & operand_bit_mask;
                let right = y.shr(shift) & operand_bit_mask;
                ((left << operand_bits) ^ right) as usize
            })
            .collect()
    }

    #[test]
    fn test_transpose() {
        let matrix = vec![
            vec![vec![1; 4], vec![2; 4]],
            vec![vec![3; 4], vec![4; 4]],
            vec![vec![5; 4], vec![6; 4]],
        ];
        let transposed = transpose(matrix);
        println!("{:?}", transposed);
        let x = transpose(transposed[1].clone());
        println!("\n{:?}", x);
    }
}
