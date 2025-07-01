use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_ff::Zero;
use ark_linear_sumcheck::{
    ml_sumcheck::{
        protocol::{
            prover, prover::ProverMsg, verifier::VerifierMsg, IPForMLSumcheck,
            ListOfProductsOfPolynomials,
        },
        Proof,
    },
    rng::FeedableRNG,
};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_iter_mut;
use rand::{Rng, RngCore};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::mpc::{
    additive::AdditiveShare,
    rep3::{Rep3Share, RssPoly},
    SSOpen, SSRandom,
};

pub struct RssSumcheck<E: Pairing> {
    _pairing: PhantomData<E>,
}

// 1st round: pub * priv * priv
// 2nd round:
pub struct ProverState<E: Pairing> {
    pub secret_polys: Vec<RssPoly<E>>,
    pub pub_polys: Vec<DenseMultilinearExtension<E::ScalarField>>,
    pub randomness: Vec<E::ScalarField>,
    pub round: usize,
    pub num_vars: usize,
    pub party: usize,
    pub coef: Vec<E::ScalarField>,
}

pub trait Rep3SumcheckProverMsg<E: Pairing>:
    Sized + Default + CanonicalSerialize + CanonicalDeserialize + Clone
{
    fn open(msgs: &Vec<Self>) -> Vec<E::ScalarField>;
    fn open_to_msg(msgs: &Vec<Self>) -> ProverMsg<E::ScalarField>;
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct ProverFirstMsg<E: Pairing> {
    pub evaluations: Vec<AdditiveShare<E>>,
}

impl<E: Pairing> Default for ProverFirstMsg<E> {
    fn default() -> Self {
        ProverFirstMsg {
            evaluations: vec![
                AdditiveShare {
                    party: 0,
                    share_0: E::ScalarField::zero(),
                };
                4
            ],
        }
    }
}

impl<E: Pairing> Rep3SumcheckProverMsg<E> for ProverFirstMsg<E> {
    fn open(msgs: &Vec<Self>) -> Vec<E::ScalarField> {
        assert!(msgs.len() == 3);
        let mut sum = vec![E::ScalarField::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].share_0;
            }
        }
        sum
    }

    fn open_to_msg(msgs: &Vec<Self>) -> ProverMsg<E::ScalarField> {
        assert!(msgs.len() == 3);
        let mut sum = vec![E::ScalarField::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].share_0;
            }
        }
        ProverMsg { evaluations: sum }
    }
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct ProverSecondMsg<E: Pairing> {
    pub evaluations: Vec<Rep3Share<E>>,
}

impl<E: Pairing> Default for ProverSecondMsg<E> {
    fn default() -> Self {
        ProverSecondMsg {
            evaluations: vec![
                Rep3Share {
                    party: 0,
                    share_0: E::ScalarField::zero(),
                    share_1: E::ScalarField::zero(),
                };
                3
            ],
        }
    }
}

impl<E: Pairing> Rep3SumcheckProverMsg<E> for ProverSecondMsg<E> {
    fn open(msgs: &Vec<Self>) -> Vec<E::ScalarField> {
        assert!(msgs.len() == 3);
        let mut sum = vec![E::ScalarField::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].share_0;
            }
        }
        sum
    }

    fn open_to_msg(msgs: &Vec<Self>) -> ProverMsg<E::ScalarField> {
        assert!(msgs.len() == 3);
        let mut sum = vec![E::ScalarField::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].share_0;
            }
        }
        ProverMsg { evaluations: sum }
    }
}

impl<E: Pairing> RssSumcheck<E> {
    pub fn first_sumcheck_init(
        v_a: &RssPoly<E>,
        v_b: &RssPoly<E>,
        v_c: &RssPoly<E>,
        pub1: &DenseMultilinearExtension<E::ScalarField>,
    ) -> ProverState<E> {
        let secret_polys = vec![v_a.clone(), v_b.clone(), v_c.clone()];
        let pub_polys = vec![pub1.clone()];
        ProverState {
            secret_polys,
            pub_polys,
            randomness: Vec::with_capacity(pub1.num_vars),
            round: 0,
            num_vars: pub1.num_vars,
            party: v_a.party,
            coef: vec![],
        }
    }
    pub fn second_sumcheck_init(
        v_a: &DenseMultilinearExtension<E::ScalarField>,
        v_b: &DenseMultilinearExtension<E::ScalarField>,
        v_c: &DenseMultilinearExtension<E::ScalarField>,
        z: &RssPoly<E>,
        v_msg: &Vec<E::ScalarField>,
    ) -> ProverState<E> {
        let pub_polys = vec![v_a.clone(), v_b.clone(), v_c.clone()];
        ProverState {
            secret_polys: vec![z.clone()],
            pub_polys,
            randomness: Vec::with_capacity(v_a.num_vars),
            round: 0,
            num_vars: v_a.num_vars,
            party: z.party,
            coef: v_msg.clone(),
        }
    }

    pub fn first_sumcheck_prove_round<R: RngCore + FeedableRNG>(
        prover_state: &mut ProverState<E>,
        v_msg: &Option<VerifierMsg<E::ScalarField>>,
        rng: &mut SSRandom<R>,
    ) -> ProverFirstMsg<E> {
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
        let party = prover_state.party;

        #[cfg(not(feature = "parallel"))]
        let zeros = (
            vec![AdditiveShare::<E>::zero().set_party(party); degree + 1],
            vec![AdditiveShare::<E>::zero().set_party(party); degree + 1],
        );
        #[cfg(feature = "parallel")]
        let zeros = || {
            (
                vec![AdditiveShare::<E>::zero().with_party(party); degree + 1],
                vec![AdditiveShare::<E>::zero().with_party(party); degree + 1],
            )
        };

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
                    *p = AdditiveShare {
                        party: party,
                        share_0: Rep3Share::<E>::mul_wo_zero(&start_a, &start_b) * &start_pub1,
                    };
                    start_a += step_a;
                    start_b += step_b;
                    start_pub1 += step_pub1;
                }

                let mut start_c = prover_state.secret_polys[2].get_share_by_idx(b << 1);
                let step_c = prover_state.secret_polys[2].get_share_by_idx((b << 1) + 1) - start_c;
                let mut start_pub1 = prover_state.pub_polys[0][b << 1];
                let step_pub1 = prover_state.pub_polys[0][(b << 1) + 1] - start_pub1;

                for p in product.iter_mut() {
                    *p -= (start_c * start_pub1).to_additive();
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
            || vec![AdditiveShare::<E>::zero().with_party(party); degree + 1],
            |mut overall_products_sum, sublist_sum| {
                overall_products_sum
                    .iter_mut()
                    .zip(sublist_sum.iter())
                    .for_each(|(f, s)| *f += s);
                overall_products_sum
            },
        );
        for i in products_sum.iter_mut() {
            i.share_0 += AdditiveShare::<E>::get_mask_scalar(rng);
        }

        ProverFirstMsg {
            evaluations: products_sum,
        }
    }

    pub fn second_sumcheck_prove_round<R: RngCore + FeedableRNG>(
        prover_state: &mut ProverState<E>,
        v_msg: &Option<VerifierMsg<E::ScalarField>>,
        rng: &mut SSRandom<R>,
    ) -> ProverSecondMsg<E> {
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
        let party = prover_state.party;

        #[cfg(not(feature = "parallel"))]
        let zeros = (
            vec![RssShare::<E>::zero().set_party(party); degree + 1],
            vec![RssShare::<E>::zero().set_party(party); degree + 1],
        );
        #[cfg(feature = "parallel")]
        let zeros = || {
            (
                vec![Rep3Share::<E>::zero().with_party(party); degree + 1],
                vec![Rep3Share::<E>::zero().with_party(party); degree + 1],
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
            || vec![Rep3Share::<E>::zero().with_party(party); degree + 1],
            |mut overall_products_sum, sublist_sum| {
                overall_products_sum
                    .iter_mut()
                    .zip(sublist_sum.iter())
                    .for_each(|(f, s)| *f += s);
                overall_products_sum
            },
        );
        for i in products_sum.iter_mut() {
            let (mask_0, mask_1) = Rep3Share::<E>::get_mask_scalar(rng);
            i.share_0 += mask_0;
            i.share_1 += mask_1;
        }
        ProverSecondMsg {
            evaluations: products_sum,
        }
    }
}

mod test {
    use super::*;
    use ark_linear_sumcheck::Error;

    #[test]
    pub(crate) fn test() {
        test_first_sumcheck();
        test_second_sumcheck();
    }

    #[test]
    pub(crate) fn test_first_sumcheck() {
        let num_vars = 10;
        let mut rng = test_rng();
        use ark_bn254::{Bn254, Fr};
        type PAIR = Bn254;
        type SCALAR = <Bn254 as Pairing>::ScalarField;
        let v_a = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let v_b = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let v_c = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let pub1 = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);

        let mut product_list = ListOfProductsOfPolynomials::new(num_vars);
        let A_B_hat = vec![
            Rc::new(v_a.clone()),
            Rc::new(v_b.clone()),
            Rc::new(pub1.clone()),
        ];
        let C_hat = vec![Rc::new(v_c.clone()), Rc::new(pub1.clone())];

        product_list.add_product(A_B_hat, SCALAR::one());
        product_list.add_product(C_hat, SCALAR::one().neg());

        let (v_a_share_0, v_a_share_1, v_a_share_2) = generate_poly_shares_rss(&v_a, &mut rng);
        let (v_b_share_0, v_b_share_1, v_b_share_2) = generate_poly_shares_rss(&v_b, &mut rng);
        let (v_c_share_0, v_c_share_1, v_c_share_2) = generate_poly_shares_rss(&v_c, &mut rng);

        let va_0 = RssPoly::<PAIR>::new(0, v_a_share_0.clone(), v_a_share_1.clone());
        let vb_0 = RssPoly::<PAIR>::new(0, v_b_share_0.clone(), v_b_share_1.clone());
        let vc_0 = RssPoly::<PAIR>::new(0, v_c_share_0.clone(), v_c_share_1.clone());

        let va_1 = RssPoly::<PAIR>::new(1, v_a_share_1.clone(), v_a_share_2.clone());
        let vb_1 = RssPoly::<PAIR>::new(1, v_b_share_1.clone(), v_b_share_2.clone());
        let vc_1 = RssPoly::<PAIR>::new(1, v_c_share_1.clone(), v_c_share_2.clone());

        let va_2 = RssPoly::<PAIR>::new(2, v_a_share_2.clone(), v_a_share_0.clone());
        let vb_2 = RssPoly::<PAIR>::new(2, v_b_share_2.clone(), v_b_share_0.clone());
        let vc_2 = RssPoly::<PAIR>::new(2, v_c_share_2.clone(), v_c_share_0.clone());

        let mut prover_state_0 =
            RssSumcheck::<PAIR>::first_sumcheck_init(&va_0, &vb_0, &vc_0, &pub1);
        let mut prover_state_1 =
            RssSumcheck::<PAIR>::first_sumcheck_init(&va_1, &vb_1, &vc_1, &pub1);
        let mut prover_state_2 =
            RssSumcheck::<PAIR>::first_sumcheck_init(&va_2, &vb_2, &vc_2, &pub1);

        let mut prover_state = vec![prover_state_0, prover_state_1, prover_state_2];

        let mut vec_random = generate_rss_share_randomness::<Blake2s512Rng>();

        let mut fs_rng_ss = Blake2s512Rng::setup();
        fs_rng_ss.feed(&"fs_rng".as_bytes());
        let mut fs_rng_regular = Blake2s512Rng::setup();
        fs_rng_regular.feed(&"fs_rng".as_bytes());

        let (prover_msg_1, _) =
            prove_as_subprotocol_test(&mut fs_rng_regular, &product_list).unwrap();
        let (prover_msg_2, _) =
            prove_as_subprotocol_first_round(&mut fs_rng_ss, prover_state, &mut vec_random)
                .unwrap();

        assert_eq!(prover_msg_1.len(), prover_msg_2.len());
        //assert_eq!(prover_msg_1[0].evaluations, prover_msg_2[0].evaluations);
        prover_msg_1
            .iter()
            .zip(prover_msg_2.iter())
            .for_each(|(x, y)| assert_eq!(x.evaluations, y.evaluations));
        //let point: Vec<_> = (0..10).map(|_| Fr::one()).collect();
    }

    #[test]
    pub(crate) fn test_second_sumcheck() {
        let num_vars = 10;
        let mut rng = test_rng();
        use ark_bn254::{Bn254, Fr};
        type PAIR = Bn254;
        type SCALAR = <Bn254 as Pairing>::ScalarField;
        let v_a = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let v_b = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let v_c = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let z = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let mut v_msg = Vec::with_capacity(3);
        for i in 0..3 {
            v_msg.push(SCALAR::rand(&mut rng));
        }

        let mut product_list = ListOfProductsOfPolynomials::new(num_vars);
        let A_hat = vec![Rc::new(v_a.clone()), Rc::new(z.clone())];
        let B_hat = vec![Rc::new(v_b.clone()), Rc::new(z.clone())];
        let C_hat = vec![Rc::new(v_c.clone()), Rc::new(z.clone())];

        product_list.add_product(A_hat, v_msg[0]);
        product_list.add_product(B_hat, v_msg[1]);
        product_list.add_product(C_hat, v_msg[2]);

        let z_share_0 = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let z_share_1 = DenseMultilinearExtension::<Fr>::rand(num_vars, &mut rng);
        let z_share_2 = z - z_share_0.clone() - z_share_1.clone();

        let z_0 = RssPoly::<PAIR>::new(0, z_share_0.clone(), z_share_1.clone());

        let z_1 = RssPoly::<PAIR>::new(1, z_share_1.clone(), z_share_2.clone());

        let z_2 = RssPoly::<PAIR>::new(2, z_share_2.clone(), z_share_0.clone());

        let mut prover_state_0 =
            RssSumcheck::<PAIR>::second_sumcheck_init(&v_a, &v_b, &v_c, &z_0, &v_msg);
        let mut prover_state_1 =
            RssSumcheck::<PAIR>::second_sumcheck_init(&v_a, &v_b, &v_c, &z_1, &v_msg);
        let mut prover_state_2 =
            RssSumcheck::<PAIR>::second_sumcheck_init(&v_a, &v_b, &v_c, &z_2, &v_msg);

        let mut prover_state = vec![prover_state_0, prover_state_1, prover_state_2];

        let mut seed_0 = Blake2s512Rng::setup();
        seed_0.feed(&"seed 0".as_bytes());
        let mut seed_1 = Blake2s512Rng::setup();
        seed_1.feed(&"seed 1".as_bytes());
        let mut random_0 = SSRandom::<Blake2s512Rng>::new(seed_0, seed_1);

        let mut seed_1 = Blake2s512Rng::setup();
        seed_1.feed(&"seed 1".as_bytes());
        let mut seed_2 = Blake2s512Rng::setup();
        seed_2.feed(&"seed 2".as_bytes());
        let mut random_1 = SSRandom::<Blake2s512Rng>::new(seed_1, seed_2);

        let mut seed_2 = Blake2s512Rng::setup();
        seed_2.feed(&"seed 2".as_bytes());
        let mut seed_0 = Blake2s512Rng::setup();
        seed_0.feed(&"seed 0".as_bytes());
        let mut random_2 = SSRandom::<Blake2s512Rng>::new(seed_2, seed_0);

        let mut vec_random = vec![random_0, random_1, random_2];

        let mut fs_rng_ss = Blake2s512Rng::setup();
        fs_rng_ss.feed(&"fs_rng".as_bytes());
        let mut fs_rng_regular = Blake2s512Rng::setup();
        fs_rng_regular.feed(&"fs_rng".as_bytes());

        let (prover_msg_1, _) =
            prove_as_subprotocol_test(&mut fs_rng_regular, &product_list).unwrap();
        let (prover_msg_2, _) =
            prove_as_subprotocol_second_round(&mut fs_rng_ss, prover_state, &mut vec_random)
                .unwrap();

        assert_eq!(prover_msg_1.len(), prover_msg_2.len());
        //assert_eq!(prover_msg_1[0].evaluations, prover_msg_2[0].evaluations);
        prover_msg_1
            .iter()
            .zip(prover_msg_2.iter())
            .for_each(|(x, y)| assert_eq!(x.evaluations, y.evaluations));
        //let point: Vec<_> = (0..10).map(|_| Fr::one()).collect();
    }

    pub fn prove_as_subprotocol_first_round<E: Pairing, R: RngCore + FeedableRNG>(
        fs_rng: &mut impl FeedableRNG<Error = Error>,
        mut prover_state: Vec<ProverState<E>>,
        random_rng: &mut Vec<SSRandom<R>>,
    ) -> Result<(Proof<E::ScalarField>, Vec<ProverState<E>>), Error> {
        //fs_rng.feed(&polynomial.info())?;

        let num_vars = prover_state[0].num_vars;
        let mut verifier_msg = None;
        let mut prover_msgs = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            let mut prover_msg = vec![];
            for party in 0..=2 {
                prover_msg.push(RssSumcheck::<E>::first_sumcheck_prove_round(
                    &mut prover_state[party],
                    &verifier_msg,
                    &mut random_rng[party],
                ));
            }
            // let prover_msg = ProverFirstMsg::<E>::open(&prover_msg);
            // let prover_msg = ProverMsg {
            //     evaluations: prover_msg,
            // };
            let prover_msg = ProverFirstMsg::<E>::open_to_msg(&prover_msg);
            fs_rng.feed(&prover_msg)?;
            prover_msgs.push(prover_msg);
            verifier_msg = Some(IPForMLSumcheck::sample_round(fs_rng));
        }
        if let Some(msg) = verifier_msg {
            prover_state
                .iter_mut()
                .for_each(|x| x.randomness.push(msg.randomness));
        }
        Ok((prover_msgs, prover_state))
    }

    pub fn prove_as_subprotocol_second_round<E: Pairing, R: RngCore + FeedableRNG>(
        fs_rng: &mut impl FeedableRNG<Error = Error>,
        mut prover_state: Vec<ProverState<E>>,
        random_rng: &mut Vec<SSRandom<R>>,
    ) -> Result<(Proof<E::ScalarField>, Vec<ProverState<E>>), Error> {
        //fs_rng.feed(&polynomial.info())?;

        let num_vars = prover_state[0].num_vars;
        let mut verifier_msg = None;
        let mut prover_msgs = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            let mut prover_msg = vec![];
            for party in 0..=2 {
                prover_msg.push(RssSumcheck::<E>::second_sumcheck_prove_round(
                    &mut prover_state[party],
                    &verifier_msg,
                    &mut random_rng[party],
                ));
            }
            let prover_msg = ProverSecondMsg::<E>::open(&prover_msg);
            let prover_msg = ProverMsg {
                evaluations: prover_msg,
            };
            fs_rng.feed(&prover_msg)?;
            prover_msgs.push(prover_msg);
            verifier_msg = Some(IPForMLSumcheck::sample_round(fs_rng));
        }
        if let Some(msg) = verifier_msg {
            prover_state
                .iter_mut()
                .for_each(|x| x.randomness.push(msg.randomness));
        }
        Ok((prover_msgs, prover_state))
    }

    pub fn prove_as_subprotocol_test<F: Field>(
        fs_rng: &mut impl FeedableRNG<Error = Error>,
        polynomial: &ListOfProductsOfPolynomials<F>,
    ) -> Result<
        (
            Proof<F>,
            ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverState<F>,
        ),
        Error,
    > {
        let mut prover_state = IPForMLSumcheck::prover_init(polynomial);
        let mut verifier_msg = None;
        let mut prover_msgs = Vec::with_capacity(polynomial.num_variables);
        for _ in 0..polynomial.num_variables {
            let prover_msg = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
            fs_rng.feed(&prover_msg)?;
            prover_msgs.push(prover_msg);
            verifier_msg = Some(IPForMLSumcheck::sample_round(fs_rng));
        }
        if let Some(msg) = verifier_msg {
            prover_state.randomness.push(msg.randomness);
        }
        Ok((prover_msgs, prover_state))
    }
}
