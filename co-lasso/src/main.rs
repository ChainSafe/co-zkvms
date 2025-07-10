use std::iter;

use co_lasso::{
    lasso::{polynomialize, LassoPolynomials, LassoPreprocessing, MemoryCheckingProver},
    subtables::range_check::RangeLookup,
};
use itertools::Itertools;
use jolt_core::{poly::field::JoltField, utils::transcript::ProofTranscript};

const LIMB_BITS: usize = 8;
const C: usize = 1;
const M: usize = 1 << LIMB_BITS;
type F = ark_bn254::Fr;

fn main() {
    let preprocessing = LassoPreprocessing::preprocess::<C, M>([RangeLookup::new_boxed(256)]);

    let mut transcript = ProofTranscript::new(b"Memory checking");
    let inputs = iter::repeat_with(|| F::from(rand::random::<u64>() % 256)).take(64).collect_vec();
    println!("inputs: {:?}", inputs);
    let polynomials = polynomialize(
        &preprocessing,
        &inputs,
        &core::array::from_fn::<_, 64, _>(|_| RangeLookup::<F>::id_for(256)),
        M,
        C,
    );
    let proof = MemoryCheckingProver::<C, M, F, LassoPolynomials<F>>::prove(
        &preprocessing,
        &polynomials,
        &mut transcript,
    );

    // println!("--------------verify------------------");

    // let mut transcript = ProofTranscript::new(b"Memory checking");

    // Rep3MemoryCheckingProver::<C, M, F, LassoPolynomials<F>>::verify_memory_checking(
    //     &preprocessing,
    //     proof,
    //     &mut transcript,
    // ).unwrap();
}
