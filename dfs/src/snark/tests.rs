use ark_ff::PrimeField;
use ark_linear_sumcheck::gkr_round_sumcheck::start_phase1_sumcheck;
use ark_poly::MultilinearExtension;
use ark_std::{end_timer, start_timer};
use rand::Rng;

use crate::math::Math;
use crate::snark::indexer::Indexer;
use crate::snark::verifier;
use crate::snark::zk::{SpecMultiCommit, SRS};
use crate::utils::{
    aggregate_comm, aggregate_proof, distributed_open, eq_eval, merge_proof, split_ck, split_poly,
};
use crate::{R1CSInstance, SparseMatPolynomial};
use ark_ff::One;
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;

// This sample was mostly taken from https://github.com/arkworks-rs/spartan/blob/master/examples/cubic.rs
// Demonstrates how to produces a proof for canonical cubic equation: `x^3 + x + 5 = y`.
// The example is described in detail [here]
//
// The R1CS for this problem consists of the following 4 constraints:
// `Z0 * Z0 - Z1 = 0`
// `Z1 * Z0 - Z2 = 0`
// `(Z2 + Z0) * 1 - Z3 = 0`
// `(Z3 + 5) * 1 - I0 = 0`
//
// [here]: https://medium.com/@VitalikButerin/quadratic-arithmetic-programs-from-zero-to-hero-f6d558cea649
fn simple_r1cs<F: PrimeField>(prng: &mut impl Rng) -> (R1CSInstance<F>, Vec<F>, Vec<F>) {
    // parameters of the R1CS instance
    let num_cons = 4;
    let num_vars = 4;
    let num_inputs = 1;
    let num_non_zero_entries = 8;

    // We will encode the above constraints into three matrices, where
    // the coefficients in the matrix are in the little-endian byte order
    let mut A: Vec<(usize, usize, F)> = Vec::new();
    let mut B: Vec<(usize, usize, F)> = Vec::new();
    let mut C: Vec<(usize, usize, F)> = Vec::new();

    let one = F::one();

    // R1CS is a set of three sparse matrices A B C, where is a row for every
    // constraint and a column for every entry in z = (vars, 1, inputs)
    // An R1CS instance is satisfiable iff:
    // Az \circ Bz = Cz, where z = (vars, 1, inputs)

    // constraint 0 entries in (A,B,C)
    // constraint 0 is Z0 * Z0 - Z1 = 0.
    A.push((0, 0, one));
    B.push((0, 0, one));
    C.push((0, 1, one));

    // constraint 1 entries in (A,B,C)
    // constraint 1 is Z1 * Z0 - Z2 = 0.
    A.push((1, 1, one));
    B.push((1, 0, one));
    C.push((1, 2, one));

    // constraint 2 entries in (A,B,C)
    // constraint 2 is (Z2 + Z0) * 1 - Z3 = 0.
    A.push((2, 2, one));
    A.push((2, 0, one));
    B.push((2, num_vars, one));
    C.push((2, 3, one));

    // constraint 3 entries in (A,B,C)
    // constraint 3 is (Z3 + 5) * 1 - I0 = 0.
    A.push((3, 3, one));
    A.push((3, num_vars, F::from(5u32)));
    B.push((3, num_vars, one));
    C.push((3, num_vars + 1, one));

    let r1cs = R1CSInstance::new(num_cons, num_vars, num_inputs, &A, &B, &C);

    // compute a satisfying assignment
    let z0 = F::rand(prng);
    let z1 = z0 * z0; // constraint 0
    let z2 = z1 * z0; // constraint 1
    let z3 = z2 + z0; // constraint 2
    let i0 = z3 + F::from(5u32); // constraint 3

    // create a VarsAssignment
    let mut vars = vec![F::zero(); num_vars];
    vars[0] = z0;
    vars[1] = z1;
    vars[2] = z2;
    vars[3] = z3;
    let assignment_vars = vars.clone();

    // create an InputsAssignment
    let mut inputs = vec![F::zero(); num_inputs];
    inputs[0] = i0;
    let assignment_inputs = inputs.clone();

    (r1cs, assignment_vars, assignment_inputs)
}

/*fn test_snark_correctness() {
    use crate::snark::R1CSProof;
    use ark_bn254::{Bn254, Fr};
    use ark_poly_commit::multilinear_pc::MultilinearPC;
    use ark_std::test_rng;
    let rng = &mut test_rng();
    let (r1cs, vars, inputs) = simple_r1cs::<Fr>(rng);

    let params = MultilinearPC::<Bn254>::setup(r1cs.num_vars, rng);

    //let (pk, vk) = Indexer::index_for_prover_and_verifier(&r1cs, &params);

    let proof = R1CSProof::new(&r1cs, &pk, &vars, &inputs);

    assert!(proof.verify(&vk, &inputs).is_ok());
}

fn generate_rand_vec<F: PrimeField>(num_variables: usize, rng: &mut impl Rng) -> Vec<F> {
    let mut v = Vec::with_capacity(num_variables.pow2());
    for _ in 0..num_variables.pow2() {
        let temp = F::rand(rng);
        if (temp != F::zero()) {
            v.push(temp)
        };
    }
    v
}
fn generate_rand_mat<F: PrimeField>(
    num_variables: usize,
    rng: &mut impl Rng,
) -> SparseMatPolynomial<F> {
    todo!();
}

fn compute_z_M<F: PrimeField>(M: &SparseMatPolynomial<F>, z: &Vec<F>) {
    todo!();
}*/
#[test]
fn test_snark_correctness() {
    use crate::snark::R1CSProof;
    use ark_bn254::{Bn254, Fr};
    use ark_poly_commit::multilinear_pc::MultilinearPC;
    use ark_std::test_rng;
    let rng = &mut test_rng();
    let r1cs_time = start_timer!(|| "Generating r1cs");
    println!("Generating r1cs");
    let (r1cs, io, witness) = R1CSInstance::<Fr>::produce_synthetic_r1cs(65536, 8192, 7168, rng);
    end_timer!(r1cs_time);

    let srs_time = start_timer!(|| "Generating SRS");
    println!("Generating SRS");
    let srs = SRS::<Bn254, _>::generate_srs(r1cs.num_cons.log_2() + 2, 4, rng);
    end_timer!(srs_time);

    let index_time = start_timer!(|| "Indexing");
    println!("Indexing");
    let (pk, vk) = Indexer::index_for_prover_and_verifier(&r1cs, &srs);
    end_timer!(index_time);

    let prove_time = start_timer!(|| "Proving");
    println!("Proving");
    let proof = R1CSProof::new(&r1cs, &pk, &vk, &witness, &io);
    end_timer!(prove_time);

    let verifier_time = start_timer!(|| "Verifying");
    assert!(proof.verify(&vk, &io).is_ok());
    end_timer!(verifier_time);
}

#[test]
fn test_distributed_poly_commit() {
    use ark_bn254::{Bn254, Fr};
    use ark_poly_commit::multilinear_pc::MultilinearPC;
    use ark_std::test_rng;
    let rng = &mut test_rng();

    const D: usize = 11;
    const S: usize = 10;
    const P: usize = 2;

    let params = MultilinearPC::<Bn254>::setup(D, rng);
    let (ck, vk) = MultilinearPC::trim(&params, S);

    let rand_poly = DenseMultilinearExtension::<Fr>::rand(10, rng);
    let rand_poly_list = split_poly(&rand_poly, P);
    let ck_list = split_ck(&ck, P);

    let point: Vec<_> = (0..10).map(|_| Fr::one()).collect();
    let commitment_ml = MultilinearPC::<Bn254>::commit(&ck, &rand_poly);

    let mut comms = Vec::new();
    for i in 0..1 << P {
        comms.push(MultilinearPC::<Bn254>::commit(
            &ck_list.0[i],
            // &ck,
            &rand_poly_list[i],
        ));
        // println!("{} {}", ck_list.0[i].nv, rand_poly_list[i].num_vars)
    }

    let comm = aggregate_comm(Fr::one(), &comms);

    assert!(comm.g_product == commitment_ml.g_product);

    // let proof = MultilinearPC::<Bn254>::open(&ck, &rand_poly, &point);
    // let check = MultilinearPC::<Bn254>::check(
    //     &vk,
    //     &commitment_ml,
    //     &point,
    //     rand_poly.evaluate(&point).unwrap(),
    //     &proof,
    // );

    // assert!(check);
}

#[test]
fn test_distributed_poly_open() {
    use ark_bn254::{Bn254, Fr};
    use ark_poly_commit::multilinear_pc::MultilinearPC;
    use ark_std::test_rng;
    let rng = &mut test_rng();

    const D: usize = 11;
    const S: usize = 10;
    const P: usize = 2;

    let params = MultilinearPC::<Bn254>::setup(D, rng);
    let (ck, vk) = MultilinearPC::trim(&params, S);

    let rand_poly = DenseMultilinearExtension::<Fr>::rand(10, rng);
    let rand_poly_list = split_poly(&rand_poly, P);
    let (ck_list, merge_ck) = split_ck(&ck, P);

    let point: Vec<_> = (0..10).map(|_| Fr::rand(rng)).collect();
    let commitment_ml = MultilinearPC::<Bn254>::commit(&ck, &rand_poly);

    let mut comms = Vec::new();
    for i in 0..1 << P {
        comms.push(MultilinearPC::<Bn254>::commit(
            &ck_list[i],
            &rand_poly_list[i],
        ));
    }

    let comm = aggregate_comm(Fr::one(), &comms);

    assert!(comm.g_product == commitment_ml.g_product);

    let proof = MultilinearPC::<Bn254>::open(&ck, &rand_poly, &point);
    let check = MultilinearPC::<Bn254>::check(
        &vk,
        &commitment_ml,
        &point,
        rand_poly.evaluate(&point).unwrap(),
        &proof,
    );

    assert!(check);

    let mut pfs = Vec::new();
    let mut rs = Vec::new();
    for i in 0..rand_poly_list.len() {
        let (p, r) = distributed_open(&ck_list[i], &rand_poly_list[i], &point[0..S - P]);
        pfs.push(p);
        rs.push(r);
    }

    let pf1 = aggregate_proof(Fr::one(), &pfs);
    let rp = DenseMultilinearExtension::from_evaluations_vec(P, rs);

    let pf2 = MultilinearPC::<Bn254>::open(&merge_ck, &rp, &point[S - P..S]);

    let pf = merge_proof(&pf1, &pf2);

    let mut e = Vec::new();
    for p in rand_poly_list {
        e.push(p.evaluate(&point[0..S - P]).unwrap())
    }
    let ep = DenseMultilinearExtension::from_evaluations_vec(P, e);

    let check = MultilinearPC::<Bn254>::check(
        &vk,
        &comm,
        &point,
        // rand_poly.evaluate(&point).unwrap(),
        ep.evaluate(&point[S - P..S]).unwrap(),
        &pf,
    );

    assert!(check);
}
