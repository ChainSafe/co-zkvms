use ark_ff::Field;
use ark_linear_sumcheck::ml_sumcheck::{
    data_structures::{ListOfProductsOfPolynomials, PolynomialInfo},
    protocol::{prover::ProverState as SumcheckProverState, IPForMLSumcheck},
};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rc::Rc;
use spartan::utils::{partial_generate_eq, two_pow_n};

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
