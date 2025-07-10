use std::marker::PhantomData;

use ark_ff::{PrimeField, Zero};
use ark_linear_sumcheck::{
    ml_sumcheck::protocol::{prover::ProverMsg, verifier::VerifierMsg},
    rng::FeedableRNG,
};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_iter_mut;
use rand::RngCore;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::mpc::{
    additive::get_mask_scalar_additive,
    rep3::{get_mask_scalar_rep3, Rep3DensePolynomial, Rep3PrimeFieldShare},
    SSRandom,
};

pub struct Rep3Sumcheck<F: PrimeField> {
    _pairing: PhantomData<F>,
}

// 1st round: pub * priv * priv
// 2nd round:
pub struct ProverState<F: PrimeField> {
    pub secret_polys: Vec<Rep3DensePolynomial<F>>,
    pub pub_polys: Vec<DenseMultilinearExtension<F>>,
    pub randomness: Vec<F>,
    pub round: usize,
    pub num_vars: usize,
    // pub party: usize,
    pub coef: Vec<F>,
}

pub trait Rep3SumcheckProverMsg<F: PrimeField>:
    Sized + Default + CanonicalSerialize + CanonicalDeserialize + Clone
{
    fn open(msgs: &Vec<Self>) -> Vec<F>;
    fn open_to_msg(msgs: &Vec<Self>) -> ProverMsg<F>;
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct ProverFirstMsg<F: PrimeField> {
    pub evaluations: Vec<F>,
}

impl<F: PrimeField> Default for ProverFirstMsg<F> {
    fn default() -> Self {
        ProverFirstMsg {
            evaluations: vec![F::zero(); 4],
        }
    }
}

impl<F: PrimeField> Rep3SumcheckProverMsg<F> for ProverFirstMsg<F> {
    fn open(msgs: &Vec<Self>) -> Vec<F> {
        assert!(msgs.len() == 3);
        let mut sum = vec![F::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i];
            }
        }
        sum
    }

    fn open_to_msg(msgs: &Vec<Self>) -> ProverMsg<F> {
        assert!(msgs.len() == 3);
        let mut sum = vec![F::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i];
            }
        }
        ProverMsg { evaluations: sum }
    }
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct ProverSecondMsg<F: PrimeField> {
    pub evaluations: Vec<Rep3PrimeFieldShare<F>>,
}

impl<F: PrimeField> Default for ProverSecondMsg<F> {
    fn default() -> Self {
        ProverSecondMsg {
            evaluations: vec![
                Rep3PrimeFieldShare {
                    // party: 0,
                    a: F::zero(),
                    b: F::zero(),
                };
                3
            ],
        }
    }
}

impl<F: PrimeField> Rep3SumcheckProverMsg<F> for ProverSecondMsg<F> {
    fn open(msgs: &Vec<Self>) -> Vec<F> {
        assert!(msgs.len() == 3);
        let mut sum = vec![F::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].a;
            }
        }
        sum
    }

    fn open_to_msg(msgs: &Vec<Self>) -> ProverMsg<F> {
        assert!(msgs.len() == 3);
        let mut sum = vec![F::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].a;
            }
        }
        ProverMsg { evaluations: sum }
    }
}

impl<F: PrimeField> Rep3Sumcheck<F> {
    pub fn first_sumcheck_init(
        v_a: &Rep3DensePolynomial<F>,
        v_b: &Rep3DensePolynomial<F>,
        v_c: &Rep3DensePolynomial<F>,
        pub1: &DenseMultilinearExtension<F>,
    ) -> ProverState<F> {
        let secret_polys = vec![v_a.clone(), v_b.clone(), v_c.clone()];
        let pub_polys = vec![pub1.clone()];
        ProverState {
            secret_polys,
            pub_polys,
            randomness: Vec::with_capacity(pub1.num_vars),
            round: 0,
            num_vars: pub1.num_vars,
            // party: v_a.party_id,
            coef: vec![],
        }
    }
    pub fn second_sumcheck_init(
        v_a: &DenseMultilinearExtension<F>,
        v_b: &DenseMultilinearExtension<F>,
        v_c: &DenseMultilinearExtension<F>,
        z: &Rep3DensePolynomial<F>,
        v_msg: &Vec<F>,
    ) -> ProverState<F> {
        let pub_polys = vec![v_a.clone(), v_b.clone(), v_c.clone()];
        ProverState {
            secret_polys: vec![z.clone()],
            pub_polys,
            randomness: Vec::with_capacity(v_a.num_vars),
            round: 0,
            num_vars: v_a.num_vars,
            // party: z.party_id,
            coef: v_msg.clone(),
        }
    }

    pub fn first_sumcheck_prove_round<R: RngCore + FeedableRNG>(
        prover_state: &mut ProverState<F>,
        v_msg: &Option<VerifierMsg<F>>,
        rng: &mut SSRandom<R>,
    ) -> ProverFirstMsg<F> {
        if let Some(msg) = v_msg {
            if prover_state.round == 0 {
                panic!("first round should be prover first.");
            }
            prover_state.randomness.push(msg.randomness);

            // fix argument
            let i = prover_state.round;
            let r = prover_state.randomness[i - 1];
            cfg_iter_mut!(prover_state.secret_polys).for_each(|multiplicand| {
                *multiplicand = multiplicand.fix_variables(&[r]);
            });
            cfg_iter_mut!(prover_state.pub_polys).for_each(|multiplicand| {
                *multiplicand = multiplicand.fix_variables(&[r]);
            });

            if prover_state.round == prover_state.num_vars {
                prover_state.round += 1;
                return ProverFirstMsg {
                    evaluations: Vec::new(),
                };
            }
        } else if prover_state.round > 0 {
            panic!("verifier message is empty");
        }

        prover_state.round += 1;

        if prover_state.round > prover_state.num_vars {
            panic!("Prover is not active");
        }

        let i = prover_state.round;
        let nv = prover_state.num_vars;
        let degree = 3; // the degree of univariate polynomial sent by prover at this round
                        // let party = prover_state.party;

        #[cfg(not(feature = "parallel"))]
        let zeros = (vec![F::zero(); degree + 1], vec![F::zero(); degree + 1]);
        #[cfg(feature = "parallel")]
        let zeros = || (vec![F::zero(); degree + 1], vec![F::zero(); degree + 1]);

        // generate sum
        let fold_result = ark_std::cfg_into_iter!(0..1 << (nv - i), 1 << 10).fold(
            zeros,
            |(mut products_sum, mut product), b| {
                // In effect, this fold is essentially doing simply:
                // for b in 0..1 << (nv - i) {

                let mut start_a = prover_state.secret_polys[0].get_share_by_idx(b << 1);
                let step_a = prover_state.secret_polys[0].get_share_by_idx((b << 1) + 1) - start_a;
                let mut start_b = prover_state.secret_polys[1].get_share_by_idx(b << 1);
                let step_b = prover_state.secret_polys[1].get_share_by_idx((b << 1) + 1) - start_b;
                let mut start_pub1 = prover_state.pub_polys[0][b << 1];
                let step_pub1 = prover_state.pub_polys[0][(b << 1) + 1] - start_pub1;

                for p in product.iter_mut() {
                    *p = (&start_a * &start_b) * &start_pub1;
                    start_a += step_a;
                    start_b += step_b;
                    start_pub1 += step_pub1;
                }

                let mut start_c = prover_state.secret_polys[2].get_share_by_idx(b << 1);
                let step_c = prover_state.secret_polys[2].get_share_by_idx((b << 1) + 1) - start_c;
                let mut start_pub1 = prover_state.pub_polys[0][b << 1];
                let step_pub1 = prover_state.pub_polys[0][(b << 1) + 1] - start_pub1;

                for p in product.iter_mut() {
                    *p -= (start_c * start_pub1).into_additive();
                    start_c += step_c;
                    start_pub1 += step_pub1;
                }

                for t in 0..degree + 1 {
                    products_sum[t] += product[t];
                }

                (products_sum, product)
            },
        );

        #[cfg(not(feature = "parallel"))]
        let products_sum = fold_result.0;

        // When rayon is used, the `fold` operation results in a iterator of `Vec<F>` rather than a single `Vec<F>`. In this case, we simply need to sum them.
        #[cfg(feature = "parallel")]
        let mut products_sum = fold_result.map(|scratch| scratch.0).reduce(
            || vec![F::zero(); degree + 1],
            |mut overall_products_sum, sublist_sum| {
                overall_products_sum
                    .iter_mut()
                    .zip(sublist_sum.iter())
                    .for_each(|(f, s)| *f += s);
                overall_products_sum
            },
        );
        for i in products_sum.iter_mut() {
            *i += get_mask_scalar_additive::<F, _>(rng);
        }

        ProverFirstMsg {
            evaluations: products_sum,
        }
    }

    pub fn second_sumcheck_prove_round<R: RngCore + FeedableRNG>(
        prover_state: &mut ProverState<F>,
        v_msg: &Option<VerifierMsg<F>>,
        rng: &mut SSRandom<R>, // TODO: correlate randomness
    ) -> ProverSecondMsg<F> {
        if let Some(msg) = v_msg {
            if prover_state.round == 0 {
                panic!("first round should be prover first.");
            }
            prover_state.randomness.push(msg.randomness);

            // fix argument
            let i = prover_state.round;
            let r = prover_state.randomness[i - 1];
            cfg_iter_mut!(prover_state.secret_polys).for_each(|multiplicand| {
                *multiplicand = multiplicand.fix_variables(&[r]);
            });
            cfg_iter_mut!(prover_state.pub_polys).for_each(|multiplicand| {
                *multiplicand = multiplicand.fix_variables(&[r]);
            });

            if prover_state.round == prover_state.num_vars {
                prover_state.round += 1;
                return ProverSecondMsg {
                    evaluations: Vec::new(),
                };
            }
        } else if prover_state.round > 0 {
            panic!("verifier message is empty");
        }

        prover_state.round += 1;

        if prover_state.round > prover_state.num_vars {
            panic!("Prover is not active");
        }

        let i = prover_state.round;
        let nv = prover_state.num_vars;
        let degree = 2; // the degree of univariate polynomial sent by prover at this round

        #[cfg(not(feature = "parallel"))]
        let zeros = (
            vec![Rep3PrimeFieldShare::<F>::zero(); degree + 1],
            vec![Rep3PrimeFieldShare::<F>::zero(); degree + 1],
        );
        #[cfg(feature = "parallel")]
        let zeros = || {
            (
                vec![Rep3PrimeFieldShare::<F>::zero(); degree + 1],
                vec![Rep3PrimeFieldShare::<F>::zero(); degree + 1],
            )
        };

        // generate sum
        let fold_result = ark_std::cfg_into_iter!(0..1 << (nv - i), 1 << 10).fold(
            zeros,
            |(mut products_sum, mut product), b| {
                // In effect, this fold is essentially doing simply:
                // for b in 0..1 << (nv - i) {

                let mut start_a = prover_state.pub_polys[0][b << 1];
                let step_a = prover_state.pub_polys[0][(b << 1) + 1] - start_a;
                let mut start_b = prover_state.pub_polys[1][b << 1];
                let step_b = prover_state.pub_polys[1][(b << 1) + 1] - start_b;
                let mut start_c = prover_state.pub_polys[2][b << 1];
                let step_c = prover_state.pub_polys[2][(b << 1) + 1] - start_c;
                let mut start_z = prover_state.secret_polys[0].get_share_by_idx(b << 1);
                let step_z = prover_state.secret_polys[0].get_share_by_idx((b << 1) + 1) - start_z;

                for p in product.iter_mut() {
                    *p = start_z
                        * (start_a * prover_state.coef[0]
                            + start_b * prover_state.coef[1]
                            + start_c * prover_state.coef[2]);
                    start_a += step_a;
                    start_b += step_b;
                    start_c += step_c;
                    start_z += step_z;
                }

                for t in 0..degree + 1 {
                    products_sum[t] += &product[t];
                }

                (products_sum, product)
            },
        );

        #[cfg(not(feature = "parallel"))]
        let products_sum = fold_result.0;

        // When rayon is used, the `fold` operation results in a iterator of `Vec<F>` rather than a single `Vec<F>`. In this case, we simply need to sum them.
        #[cfg(feature = "parallel")]
        let mut products_sum = fold_result.map(|scratch| scratch.0).reduce(
            || vec![Rep3PrimeFieldShare::<F>::zero(); degree + 1],
            |mut overall_products_sum, sublist_sum| {
                overall_products_sum
                    .iter_mut()
                    .zip(sublist_sum.iter())
                    .for_each(|(f, s)| *f += s);
                overall_products_sum
            },
        );
        for i in products_sum.iter_mut() {
            let (mask_0, mask_1) = get_mask_scalar_rep3::<F, _>(rng);
            i.a += mask_0;
            i.b += mask_1;
        }
        ProverSecondMsg {
            evaluations: products_sum,
        }
    }
}

// #[cfg(test)]
// mod test {
//     use ark_linear_sumcheck::Error;

//     use super::*;

//     #[test]
//     pub(crate) fn test() {
//         test_first_sumcheck();
//         test_second_sumcheck();
//     }

//     #[test]
//     pub(crate) fn test_first_sumcheck() {
//         let num_vars = 10;
//         let mut rng = test_rng();
//         use ark_bn254::{Bn254, Fr};
//         type PAIR = Bn254;
//         type SCALAR = <Bn254 as Pairing>::ScalarField;
//         let v_a = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let v_b = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let v_c = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let pub1 = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);

//         let mut product_list = ListOfProductsOfPolynomials::new(num_vars);
//         let a_b_hat = vec![
//             Rc::new(v_a.clone()),
//             Rc::new(v_b.clone()),
//             Rc::new(pub1.clone()),
//         ];
//         let c_hat = vec![Rc::new(v_c.clone()), Rc::new(pub1.clone())];

//         product_list.add_product(a_b_hat, SCALAR::one());
//         product_list.add_product(c_hat, SCALAR::one().neg());

//         let (v_a_share_0, v_a_share_1, v_a_share_2) = generate_poly_shares_rss(&v_a, &mut rng);
//         let (v_b_share_0, v_b_share_1, v_b_share_2) = generate_poly_shares_rss(&v_b, &mut rng);
//         let (v_c_share_0, v_c_share_1, v_c_share_2) = generate_poly_shares_rss(&v_c, &mut rng);

//         let va_0 = Rep3Poly::<PAIR>::new(0, v_a_share_0.clone(), v_a_share_1.clone());
//         let vb_0 = Rep3Poly::<PAIR>::new(0, v_b_share_0.clone(), v_b_share_1.clone());
//         let vc_0 = Rep3Poly::<PAIR>::new(0, v_c_share_0.clone(), v_c_share_1.clone());

//         let va_1 = Rep3Poly::<PAIR>::new(1, v_a_share_1.clone(), v_a_share_2.clone());
//         let vb_1 = Rep3Poly::<PAIR>::new(1, v_b_share_1.clone(), v_b_share_2.clone());
//         let vc_1 = Rep3Poly::<PAIR>::new(1, v_c_share_1.clone(), v_c_share_2.clone());

//         let va_2 = Rep3Poly::<PAIR>::new(2, v_a_share_2.clone(), v_a_share_0.clone());
//         let vb_2 = Rep3Poly::<PAIR>::new(2, v_b_share_2.clone(), v_b_share_0.clone());
//         let vc_2 = Rep3Poly::<PAIR>::new(2, v_c_share_2.clone(), v_c_share_0.clone());

//         let mut prover_state_0 =
//             RssSumcheck::<PAIR>::first_sumcheck_init(&va_0, &vb_0, &vc_0, &pub1);
//         let mut prover_state_1 =
//             RssSumcheck::<PAIR>::first_sumcheck_init(&va_1, &vb_1, &vc_1, &pub1);
//         let mut prover_state_2 =
//             RssSumcheck::<PAIR>::first_sumcheck_init(&va_2, &vb_2, &vc_2, &pub1);

//         let mut prover_state = vec![prover_state_0, prover_state_1, prover_state_2];

//         let mut vec_random = generate_rss_share_randomness::<Blake2s512Rng>();

//         let mut fs_rng_ss = Blake2s512Rng::setup();
//         fs_rng_ss.feed(&"fs_rng".as_bytes());
//         let mut fs_rng_regular = Blake2s512Rng::setup();
//         fs_rng_regular.feed(&"fs_rng".as_bytes());

//         let (prover_msg_1, _) =
//             prove_as_subprotocol_test(&mut fs_rng_regular, &product_list).unwrap();
//         let (prover_msg_2, _) =
//             prove_as_subprotocol_first_round(&mut fs_rng_ss, prover_state, &mut vec_random)
//                 .unwrap();

//         assert_eq!(prover_msg_1.len(), prover_msg_2.len());
//         //assert_eq!(prover_msg_1[0].evaluations, prover_msg_2[0].evaluations);
//         prover_msg_1
//             .iter()
//             .zip(prover_msg_2.iter())
//             .for_each(|(x, y)| assert_eq!(x.evaluations, y.evaluations));
//         //let point: Vec<_> = (0..10).map(|_| Fr::one()).collect();
//     }

//     #[test]
//     pub(crate) fn test_second_sumcheck() {
//         let num_vars = 10;
//         let mut rng = test_rng();
//         use ark_bn254::{Bn254, Fr};
//         type PAIR = Bn254;
//         type SCALAR = <Bn254 as Pairing>::ScalarField;
//         let v_a = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let v_b = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let v_c = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let z = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let mut v_msg = Vec::with_capacity(3);
//         for i in 0..3 {
//             v_msg.push(SCALAR::rand(&mut rng));
//         }

//         let mut product_list = ListOfProductsOfPolynomials::new(num_vars);
//         let a_hat = vec![Rc::new(v_a.clone()), Rc::new(z.clone())];
//         let b_hat = vec![Rc::new(v_b.clone()), Rc::new(z.clone())];
//         let c_hat = vec![Rc::new(v_c.clone()), Rc::new(z.clone())];

//         product_list.add_product(a_hat, v_msg[0]);
//         product_list.add_product(b_hat, v_msg[1]);
//         product_list.add_product(c_hat, v_msg[2]);

//         let z_share_0 = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let z_share_1 = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
//         let z_share_2 = z - z_share_0.clone() - z_share_1.clone();

//         let z_0 = Rep3Poly::<PAIR>::new(0, z_share_0.clone(), z_share_1.clone());

//         let z_1 = Rep3Poly::<PAIR>::new(1, z_share_1.clone(), z_share_2.clone());

//         let z_2 = Rep3Poly::<PAIR>::new(2, z_share_2.clone(), z_share_0.clone());

//         let mut prover_state_0 =
//             RssSumcheck::<PAIR>::second_sumcheck_init(&v_a, &v_b, &v_c, &z_0, &v_msg);
//         let mut prover_state_1 =
//             RssSumcheck::<PAIR>::second_sumcheck_init(&v_a, &v_b, &v_c, &z_1, &v_msg);
//         let mut prover_state_2 =
//             RssSumcheck::<PAIR>::second_sumcheck_init(&v_a, &v_b, &v_c, &z_2, &v_msg);

//         let mut prover_state = vec![prover_state_0, prover_state_1, prover_state_2];

//         let mut seed_0 = Blake2s512Rng::setup();
//         seed_0.feed(&"seed 0".as_bytes());
//         let mut seed_1 = Blake2s512Rng::setup();
//         seed_1.feed(&"seed 1".as_bytes());
//         let mut random_0 = SSRandom::<Blake2s512Rng>::new(seed_0, seed_1);

//         let mut seed_1 = Blake2s512Rng::setup();
//         seed_1.feed(&"seed 1".as_bytes());
//         let mut seed_2 = Blake2s512Rng::setup();
//         seed_2.feed(&"seed 2".as_bytes());
//         let mut random_1 = SSRandom::<Blake2s512Rng>::new(seed_1, seed_2);

//         let mut seed_2 = Blake2s512Rng::setup();
//         seed_2.feed(&"seed 2".as_bytes());
//         let mut seed_0 = Blake2s512Rng::setup();
//         seed_0.feed(&"seed 0".as_bytes());
//         let mut random_2 = SSRandom::<Blake2s512Rng>::new(seed_2, seed_0);

//         let mut vec_random = vec![random_0, random_1, random_2];

//         let mut fs_rng_ss = Blake2s512Rng::setup();
//         fs_rng_ss.feed(&"fs_rng".as_bytes());
//         let mut fs_rng_regular = Blake2s512Rng::setup();
//         fs_rng_regular.feed(&"fs_rng".as_bytes());

//         let (prover_msg_1, _) =
//             prove_as_subprotocol_test(&mut fs_rng_regular, &product_list).unwrap();
//         let (prover_msg_2, _) =
//             prove_as_subprotocol_second_round(&mut fs_rng_ss, prover_state, &mut vec_random)
//                 .unwrap();

//         assert_eq!(prover_msg_1.len(), prover_msg_2.len());
//         //assert_eq!(prover_msg_1[0].evaluations, prover_msg_2[0].evaluations);
//         prover_msg_1
//             .iter()
//             .zip(prover_msg_2.iter())
//             .for_each(|(x, y)| assert_eq!(x.evaluations, y.evaluations));
//         //let point: Vec<_> = (0..10).map(|_| Fr::one()).collect();
//     }
// }
