use ark_ff::Field;
use ark_ff::{PrimeField, Zero};
use ark_linear_sumcheck::ml_sumcheck::{
    data_structures::{ListOfProductsOfPolynomials, PolynomialInfo},
    protocol::{prover::ProverState as SumcheckProverState, IPForMLSumcheck},
};
use ark_linear_sumcheck::{
    ml_sumcheck::protocol::{prover::ProverMsg, verifier::VerifierMsg},
    rng::FeedableRNG,
};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::cfg_iter_mut;
use ark_std::rc::Rc;
use mpc_core::protocols::{
    additive::get_mask_scalar_additive,
    rep3::{
        arithmetic::get_mask_scalar_rep3, poly::Rep3DensePolynomial, rngs::SSRandom,
        Rep3PrimeFieldShare,
    },
};
use rand::RngCore;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use spartan::utils::{partial_generate_eq, two_pow_n};
use std::marker::PhantomData;

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
                    *p = (&start_a * &start_b).into_fe() * &start_pub1;
                    start_a += step_a;
                    start_b += step_b;
                    start_pub1 += step_pub1;
                }

                let mut start_c = prover_state.secret_polys[2].get_share_by_idx(b << 1);
                let step_c = prover_state.secret_polys[2].get_share_by_idx((b << 1) + 1) - start_c;
                let mut start_pub1 = prover_state.pub_polys[0][b << 1];
                let step_pub1 = prover_state.pub_polys[0][(b << 1) + 1] - start_pub1;

                for p in product.iter_mut() {
                    *p -= (start_c * start_pub1).into_additive().into_fe();
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

/// Prover State
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistrbutedSumcheckProverState<F: Field> {
    /// Stores the list of products that is meant to be added together. Each multiplicand is represented by
    /// the index in flattened_ml_extensions
    pub list_of_products: Vec<(F, Vec<F>)>,
}

pub fn merge_list_of_distributed_poly<F: Field>(
    prover_states: &Vec<DistrbutedSumcheckProverState<F>>,
    poly_info: &PolynomialInfo,
    log_num_parties: usize,
) -> ListOfProductsOfPolynomials<F> {
    let mut merge_poly = ListOfProductsOfPolynomials::new(log_num_parties);
    merge_poly.max_multiplicands = poly_info.max_multiplicands;
    for j in 0..prover_states[0].list_of_products.len() {
        let mut evals: Vec<Vec<F>> = vec![Vec::new(); prover_states[0].list_of_products[j].1.len()];
        for i in 0..(1 << log_num_parties) {
            let (_, prods) = &prover_states[i].list_of_products[j];
            for k in 0..prods.len() {
                evals[k].push(prods[k]);
            }
        }
        let mut prod: Vec<Rc<DenseMultilinearExtension<F>>> = Vec::new();
        for e in &evals {
            prod.push(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                log_num_parties,
                e.clone(),
            )))
        }

        merge_poly.add_product(prod.into_iter(), prover_states[0].list_of_products[j].0);
    }

    merge_poly
}

pub fn obtain_distrbuted_sumcheck_prover_state<F: Field>(
    prover_state: &SumcheckProverState<F>,
) -> DistrbutedSumcheckProverState<F> {
    let mut distributed_state = DistrbutedSumcheckProverState {
        list_of_products: Vec::new(),
    };
    for i in 0..prover_state.list_of_products.len() {
        let coeff = prover_state.list_of_products[i].0;
        let mut prod = Vec::new();
        for j in &prover_state.list_of_products[i].1 {
            prod.push(prover_state.flattened_ml_extensions[*j].evaluations[0]);
        }
        distributed_state.list_of_products.push((coeff, prod));
    }
    distributed_state
}

pub fn poly_list_to_prover_state<F: Field>(
    q_polys: &ListOfProductsOfPolynomials<F>,
) -> DistrbutedSumcheckProverState<F> {
    let state = IPForMLSumcheck::prover_init(&q_polys);
    let res = obtain_distrbuted_sumcheck_prover_state(&state);
    res
}

pub fn append_sumcheck_polys<F: Field>(
    h: (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
    phi: (DenseMultilinearExtension<F>, DenseMultilinearExtension<F>),
    m: DenseMultilinearExtension<F>,
    degree_diff: usize,
    q_polys: &mut ListOfProductsOfPolynomials<F>,
    z: &Vec<F>,
    lambda: &F,
    start: usize,
    log_chunk_size: usize,
) {
    assert_eq!(h.0.num_vars, phi.0.num_vars);
    assert_eq!(h.0.num_vars, m.num_vars);

    let mut eta: F = *lambda;

    let lagrange = partial_generate_eq(&z, start, log_chunk_size);

    let q_0_h = vec![Rc::new(h.0.clone())];
    let q_0_h_times_phi = vec![Rc::new(lagrange.clone()), Rc::new(h.0), Rc::new(phi.0)];
    let q_0_m = vec![Rc::new(lagrange.clone()), Rc::new(m)];

    q_polys.add_product(q_0_h, *lambda);
    eta = eta * lambda;
    q_polys.add_product(q_0_h_times_phi, eta);
    q_polys.add_product(
        q_0_m,
        two_pow_n::<F>(degree_diff).inverse().unwrap() * eta.neg(),
    );

    let q_1_h = vec![Rc::new(h.1.clone())];
    let q_1_h_times_phi = vec![Rc::new(lagrange.clone()), Rc::new(h.1), Rc::new(phi.1)];
    let q_1_m = vec![Rc::new(lagrange)];

    // eta = eta * lambda;
    q_polys.add_product(q_1_h, lambda.neg());
    eta = eta * lambda;
    q_polys.add_product(q_1_h_times_phi, eta);
    q_polys.add_product(q_1_m, eta.neg());
}

pub fn default_sumcheck_poly_list<F: Field>(
    lambda: &F,
    degree_diff: usize,
    q_polys: &mut ListOfProductsOfPolynomials<F>,
) {
    let mut eta: F = *lambda;
    // let mut q_polys = ListOfProductsOfPolynomials::new(1);

    let default_poly =
        DenseMultilinearExtension::from_evaluations_vec(1, vec![F::zero(), F::zero()]);
    let q_0_h = vec![Rc::new(default_poly.clone())];
    let q_0_h_times_phi = vec![
        Rc::new(default_poly.clone()),
        Rc::new(default_poly.clone()),
        Rc::new(default_poly.clone()),
    ];
    let q_0_m = vec![Rc::new(default_poly.clone()), Rc::new(default_poly.clone())];

    q_polys.add_product(q_0_h, *lambda);
    eta = eta * lambda;
    q_polys.add_product(q_0_h_times_phi, eta);
    q_polys.add_product(
        q_0_m,
        two_pow_n::<F>(degree_diff).inverse().unwrap() * eta.neg(),
    );

    let q_1_h = vec![Rc::new(default_poly.clone())];
    let q_1_h_times_phi = vec![
        Rc::new(default_poly.clone()),
        Rc::new(default_poly.clone()),
        Rc::new(default_poly.clone()),
    ];
    let q_1_m = vec![Rc::new(default_poly.clone())];

    // eta = eta * lambda;
    q_polys.add_product(q_1_h, lambda.neg());
    eta = eta * lambda;
    q_polys.add_product(q_1_h_times_phi, eta);
    q_polys.add_product(q_1_m, eta.neg());
}
