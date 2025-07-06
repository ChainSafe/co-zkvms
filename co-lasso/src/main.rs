use co_lasso::{
    memory_checking::Rep3MemoryCheckingProver, polynomialize, subtables::range_check::RangeLookup, LassoPolynomials, LassoPreprocessing
};
use jolt_core::{poly::field::JoltField, utils::transcript::ProofTranscript};

const LIMB_BITS: usize = 8;
const C: usize = 1;
const M: usize = 1 << LIMB_BITS;
type F = ark_bn254::Fr;

fn main() {
    let preprocessing = LassoPreprocessing::preprocess::<C, M>([RangeLookup::new_boxed(256)]);

    let mut transcript = ProofTranscript::new(b"Memory checking");
    let inputs = [F::from_u64(1).unwrap(); 64];
    let polynomials = polynomialize(
        &preprocessing,
        &inputs,
        &core::array::from_fn::<_, 64, _>(|_| RangeLookup::<F>::id_for(256)),
        M,
        C,
    );
    // let proof = Rep3MemoryCheckingProver::<C, M, F, LassoPolynomials<F>>::prove_rep3(
    //     &preprocessing,
    //     &polynomials,
    //     &mut transcript,
    // );

    // println!("--------------verify------------------");

    // let mut transcript = ProofTranscript::new(b"Memory checking");

    // Rep3MemoryCheckingProver::<C, M, F, LassoPolynomials<F>>::verify_memory_checking(
    //     &preprocessing,
    //     proof,
    //     &mut transcript,
    // ).unwrap();
}
