
#[test]
fn test_multiplicites() {
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use rand::thread_rng;

    let random_elts = <[Fr; 8]>::rand(&mut thread_rng());

    let query_evals = vec![
        random_elts[0],
        random_elts[1],
        random_elts[0],
        random_elts[1],
        random_elts[3],
        random_elts[4],
        random_elts[2],
        Fr::zero(),
    ];
    let query = DenseMultilinearExtension::<Fr>::from_evaluations_vec(3, query_evals);

    let table_evals = random_elts[..4].to_vec();
    let table = DenseMultilinearExtension::<Fr>::from_evaluations_vec(2, table_evals);

    println!(
        "{:?}",
        normalized_multiplicities(&query, &table).evaluations
    );
}

#[test]
fn test_end_to_end() {
    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

    use crate::PROTOCOL_NAME;

    const D: usize = 8;
    const S: usize = 8;

    let mut rng = StdRng::seed_from_u64(12);

    let params = MultilinearPC::<Bn254>::setup(D, &mut rng);
    let (ck, vk) = MultilinearPC::trim(&params, D);

    let table_evals = (0..1 << S).map(|_| Fr::rand(&mut rng)).collect();
    let table = DenseMultilinearExtension::from_evaluations_vec(S, table_evals);

    let query_valid_evals = (0..(1 << D))
        .map(|_| *table.evaluations.choose(&mut rng).unwrap())
        .collect();

    let query_valid = DenseMultilinearExtension::from_evaluations_vec(D, query_valid_evals);
    // let alpha = rng.gen::<[Fr; S]>().to_vec();
    // let eval0 = table.evaluate(&alpha).unwrap();
    // let boosted_table = boost_degree(&table, D);

    // let mut alpha_with_random_extension = alpha.clone();
    // alpha_with_random_extension.extend(vec![Fr::rand(&mut rng); D - S]);
    // let eval1 = boosted_table
    //     .evaluate(&alpha_with_random_extension)
    //     .unwrap();

    let mut prover_transcript = Blake2s512Rng::setup();
    let mut verifier_transcript = Blake2s512Rng::setup();

    let freq = normalized_multiplicities(&query_valid, &table);

    let mut q_polys = ListOfProductsOfPolynomials::new(D);

    let x = get_scalar_challenge(&mut prover_transcript);
    let pf = LogLookupProof::<Bn254>::prove(&query_valid, &table, &freq, &ck, &x);

    _ = prover_transcript.feed(&pf.1[0]);
    _ = prover_transcript.feed(&pf.1[1]);

    let z = get_vector_challenge(&mut prover_transcript, query_valid.num_vars);
    let lambda = get_scalar_challenge(&mut prover_transcript);
    append_sumcheck_polys(
        (pf.0[0].clone(), pf.0[1].clone()),
        (pf.2[0].clone(), pf.2[1].clone()),
        freq.clone(),
        query_valid.num_vars - table.num_vars,
        &mut q_polys,
        &z,
        &lambda,
        0,
        z.len(),
    );

    let (sc_pf, final_state) =
        MLSumcheck::prove_as_subprotocol(&mut prover_transcript, &q_polys).unwrap();
    let final_point = final_state.randomness;

    let eta = get_scalar_challenge(&mut prover_transcript);
    let batch_oracle = batch_open_poly(
        &[&pf.0[0], &pf.0[1], &freq, &query_valid, &table],
        &[pf.1[0].clone(), pf.1[1].clone()],
        &ck,
        &final_point,
        eta,
    );

    // let query_valid_oracle = OracleEval {
    //     val: query_valid.evaluate(&pf.point).unwrap(),
    //     commitment: MultilinearPC::commit(&ck, &query_valid),
    //     proof: MultilinearPC::open(&ck, &query_valid, &pf.point),
    // };

    // let boosted_table = boost_degree(&table, D);

    // let table_valid_oracle = OracleEval {
    //     val: boosted_table.evaluate(&pf.point).unwrap(),
    //     commitment: MultilinearPC::commit(&ck, &boosted_table),
    //     proof: MultilinearPC::open(&ck, &boosted_table, &pf.point),
    // };

    let degree_diff = query_valid.num_vars - table.num_vars;

    let (lookup_x, z, lambda) = LogLookupProof::<Bn254>::verify_before_sumcheck(
        &q_polys.info(),
        &batch_oracle,
        1,
        &mut verifier_transcript,
    );

    assert!(LogLookupProof::<Bn254>::verify(
        &q_polys.info(),
        &sc_pf,
        &batch_oracle,
        degree_diff,
        &lookup_x,
        &z,
        &lambda,
        &vk,
        &mut verifier_transcript,
        <Bn254 as Pairing>::ScalarField::zero(),
        <Bn254 as Pairing>::ScalarField::zero(),
    )
    .is_ok())
}

#[test]
fn test_distributed_end_to_end() {
    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

    use crate::PROTOCOL_NAME;

    const D: usize = 8;
    const S: usize = 8;
    const P: usize = 2;

    let mut rng = StdRng::seed_from_u64(12);

    let params = MultilinearPC::<Bn254>::setup(D, &mut rng);
    let (ck, vk) = MultilinearPC::trim(&params, D);
    let (ck_list, merge_ck) = split_ck(&ck, P);

    let table_evals = (0..1 << S).map(|_| Fr::rand(&mut rng)).collect();
    let table = DenseMultilinearExtension::from_evaluations_vec(S, table_evals);
    let table_list = split_poly(&table, P);

    let query_valid_evals = (0..(1 << D))
        .map(|_| *table.evaluations.choose(&mut rng).unwrap())
        .collect();

    let query_valid = DenseMultilinearExtension::from_evaluations_vec(D, query_valid_evals);
    let query_list = split_poly(&query_valid, P);

    let freq = normalized_multiplicities(&query_valid, &table);
    let freq_list = split_poly(&freq, P);

    let mut prover_transcript = Blake2s512Rng::setup();
    // let mut q_polys: ListOfProductsOfPolynomials<ark_ff::Fp<ark_ff::MontBackend<ark_Bn254::FrConfig, 4>, 4>> = ListOfProductsOfPolynomials::new(D);
    let mut distributed_q_polys = vec![ListOfProductsOfPolynomials::new(D - P); 1 << P];

    let x = get_scalar_challenge(&mut prover_transcript);

    let mut distributed_pf = Vec::new();
    let mut comm_0 = Vec::new();
    let mut comm_1 = Vec::new();

    let tmp_pf = LogLookupProof::<Bn254>::prove(&query_valid, &table, &freq, &ck, &x);

    for i in 0..1 << P {
        let pf = LogLookupProof::<Bn254>::prove(
            &query_list[i],
            &table_list[i],
            &freq_list[i],
            &ck_list[i],
            &x,
        );
        comm_0.push(pf.1[0].clone());
        comm_1.push(pf.1[1].clone());
        distributed_pf.push(pf);
    }
    _ = prover_transcript.feed(&combine_comm(&comm_0));
    _ = prover_transcript.feed(&combine_comm(&comm_1));

    let z = get_vector_challenge(&mut prover_transcript, query_valid.num_vars);
    let lambda = get_scalar_challenge(&mut prover_transcript);

    let mut start = 0;
    let log_chunk_size = z.len() - P;
    for i in 0..1 << P {
        append_sumcheck_polys(
            (
                distributed_pf[i].0[0].clone(),
                distributed_pf[i].0[1].clone(),
            ),
            (
                distributed_pf[i].2[0].clone(),
                distributed_pf[i].2[1].clone(),
            ),
            freq_list[i].clone(),
            query_valid.num_vars - table.num_vars,
            &mut distributed_q_polys[i],
            &z,
            &lambda,
            start,
            log_chunk_size,
        );
        start = start + (1 << log_chunk_size);
    }

    let mut prover_states = Vec::new();
    let mut prover_msgs = Vec::new();
    for i in 0..1 << P {
        prover_states.push(IPForMLSumcheck::prover_init(&distributed_q_polys[i]))
    }
    let poly_info = PolynomialInfo {
        max_multiplicands: distributed_q_polys[0].max_multiplicands,
        num_variables: D,
    };
    let mut verifier_state = IPForMLSumcheck::verifier_init(&poly_info);
    let mut verifier_msg = None;

    let mut final_point = Vec::new();

    _ = prover_transcript.feed(&poly_info.clone());

    for _ in 0..D - P {
        let mut prover_message = IPForMLSumcheck::prove_round(&mut prover_states[0], &verifier_msg);
        for i in 1..1 << P {
            let tmp = IPForMLSumcheck::prove_round(&mut prover_states[i], &verifier_msg);
            // Aggregate results from different parties
            for j in 0..prover_message.evaluations.len() {
                prover_message.evaluations[j] = prover_message.evaluations[j] + tmp.evaluations[j]
            }
        }
        prover_transcript.feed(&prover_message);
        prover_msgs.push(prover_message.clone());
        // Using the aggregate results to generate the verifier's message.
        let verifier_msg2 = IPForMLSumcheck::verify_round(
            prover_message,
            &mut verifier_state,
            &mut prover_transcript,
        );
        verifier_msg = verifier_msg2;
        final_point.push(verifier_msg.clone().unwrap().randomness);
        // println!("{:?}", verifier_msg);
    }

    if P != 0 {
        for i in 0..1 << P {
            let _ = IPForMLSumcheck::prove_round(&mut prover_states[i], &verifier_msg);
        }
        // println!("start");
        let merge_poly = merge_list_of_distributed_poly(prover_states, poly_info, P);

        // println!("pass");
        let mut prover_state = IPForMLSumcheck::prover_init(&merge_poly);
        // assert!(prover_states[0].round == nv - log_num_parties);
        // prover_state.round = nv - log_num_parties;
        let mut verifier_msg = None;
        for _ in D - P..D {
            let prover_message = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
            prover_transcript.feed(&prover_message);
            prover_msgs.push(prover_message.clone());
            let verifier_msg2 = IPForMLSumcheck::verify_round(
                prover_message,
                &mut verifier_state,
                &mut prover_transcript,
            );
            verifier_msg = verifier_msg2;
            final_point.push(verifier_msg.clone().unwrap().randomness);
        }
    }

    let eta = get_scalar_challenge(&mut prover_transcript);
    let batch_oracle = batch_open_poly(
        &[&tmp_pf.0[0], &tmp_pf.0[1], &freq, &query_valid, &table],
        &[tmp_pf.1[0].clone(), tmp_pf.1[1].clone()],
        &ck,
        &final_point,
        eta,
    );

    let mut verifier_transcript = Blake2s512Rng::setup();

    let degree_diff = query_valid.num_vars - table.num_vars;

    let poly_info = PolynomialInfo {
        max_multiplicands: distributed_q_polys[0].max_multiplicands,
        num_variables: D,
    };
    let (lookup_x, z, lambda) = LogLookupProof::<Bn254>::verify_before_sumcheck(
        &poly_info,
        &batch_oracle,
        1,
        &mut verifier_transcript,
    );

    assert!(LogLookupProof::<Bn254>::verify(
        &poly_info,
        &prover_msgs,
        &batch_oracle,
        degree_diff,
        &lookup_x,
        &z,
        &lambda,
        &vk,
        &mut verifier_transcript,
        <Bn254 as Pairing>::ScalarField::zero(),
        <Bn254 as Pairing>::ScalarField::zero(),
    )
    .is_ok())
}

#[test]
fn test_batch_end_to_end() {
    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

    use crate::PROTOCOL_NAME;

    const D: usize = 8;
    const S: usize = 8;

    let mut rng = StdRng::seed_from_u64(12);

    let params = MultilinearPC::<Bn254>::setup(D, &mut rng);
    let (ck, vk) = MultilinearPC::trim(&params, D);

    let table_evals = (0..1 << S).map(|_| Fr::rand(&mut rng)).collect();
    let table = DenseMultilinearExtension::from_evaluations_vec(S, table_evals);

    let table_evals_2 = (0..1 << S).map(|_| Fr::rand(&mut rng)).collect();
    let table_2 = DenseMultilinearExtension::from_evaluations_vec(S, table_evals_2);

    let query_valid_evals = (0..(1 << D))
        .map(|_| *table.evaluations.choose(&mut rng).unwrap())
        .collect();

    let query_valid = DenseMultilinearExtension::from_evaluations_vec(D, query_valid_evals);

    let query_valid_evals_2 = (0..(1 << D))
        .map(|_| *table_2.evaluations.choose(&mut rng).unwrap())
        .collect();

    let query_valid_2 = DenseMultilinearExtension::from_evaluations_vec(D, query_valid_evals_2);
    // let alpha = rng.gen::<[Fr; S]>().to_vec();
    // let eval0 = table.evaluate(&alpha).unwrap();
    // let boosted_table = boost_degree(&table, D);

    // let mut alpha_with_random_extension = alpha.clone();
    // alpha_with_random_extension.extend(vec![Fr::rand(&mut rng); D - S]);
    // let eval1 = boosted_table
    //     .evaluate(&alpha_with_random_extension)
    //     .unwrap();

    let mut prover_transcript = Blake2s512Rng::setup();
    let mut verifier_transcript = Blake2s512Rng::setup();

    let freq = normalized_multiplicities(&query_valid, &table);

    let mut q_polys = ListOfProductsOfPolynomials::new(D);

    let x_1 = get_scalar_challenge(&mut prover_transcript);
    let x_2 = get_scalar_challenge(&mut prover_transcript);

    let pf = LogLookupProof::<Bn254>::prove(&query_valid, &table, &freq, &ck, &x_1);

    _ = prover_transcript.feed(&pf.1[0]);
    _ = prover_transcript.feed(&pf.1[1]);

    let freq_2 = normalized_multiplicities(&query_valid_2, &table_2);
    let pf_2 = LogLookupProof::<Bn254>::prove(&query_valid_2, &table_2, &freq_2, &ck, &x_2);

    _ = prover_transcript.feed(&pf_2.1[0]);
    _ = prover_transcript.feed(&pf_2.1[1]);

    let z = get_vector_challenge(&mut prover_transcript, query_valid.num_vars);
    let lambda = get_scalar_challenge(&mut prover_transcript);
    append_sumcheck_polys(
        (pf.0[0].clone(), pf.0[1].clone()),
        (pf.2[0].clone(), pf.2[1].clone()),
        freq.clone(),
        query_valid.num_vars - table.num_vars,
        &mut q_polys,
        &z,
        &lambda,
        0,
        z.len(),
    );

    let z = get_vector_challenge(&mut prover_transcript, query_valid.num_vars);
    let lambda = get_scalar_challenge(&mut prover_transcript);
    append_sumcheck_polys(
        (pf_2.0[0].clone(), pf_2.0[1].clone()),
        (pf_2.2[0].clone(), pf_2.2[1].clone()),
        freq_2.clone(),
        query_valid_2.num_vars - table_2.num_vars,
        &mut q_polys,
        &z,
        &lambda,
        0,
        z.len(),
    );

    let (sc_pf, final_state) =
        MLSumcheck::prove_as_subprotocol(&mut prover_transcript, &q_polys).unwrap();
    let final_point = final_state.randomness;

    let eta = get_scalar_challenge(&mut prover_transcript);
    let batch_oracle = batch_open_poly(
        &[
            &pf.0[0],
            &pf.0[1],
            &pf_2.0[0],
            &pf_2.0[1],
            &freq,
            &query_valid,
            &table,
            &freq_2,
            &query_valid_2,
            &table_2,
        ],
        &[
            pf.1[0].clone(),
            pf.1[1].clone(),
            pf_2.1[0].clone(),
            pf_2.1[1].clone(),
        ],
        &ck,
        &final_point,
        eta,
    );

    // let query_valid_oracle = OracleEval {
    //     val: query_valid.evaluate(&pf.point).unwrap(),
    //     commitment: MultilinearPC::commit(&ck, &query_valid),
    //     proof: MultilinearPC::open(&ck, &query_valid, &pf.point),
    // };

    // let boosted_table = boost_degree(&table, D);

    // let table_valid_oracle = OracleEval {
    //     val: boosted_table.evaluate(&pf.point).unwrap(),
    //     commitment: MultilinearPC::commit(&ck, &boosted_table),
    //     proof: MultilinearPC::open(&ck, &boosted_table, &pf.point),
    // };

    let degree_diff = query_valid.num_vars - table.num_vars;

    let (lookup_x, z, lambda) = LogLookupProof::<Bn254>::verify_before_sumcheck(
        &q_polys.info(),
        &batch_oracle,
        2,
        &mut verifier_transcript,
    );

    assert!(LogLookupProof::<Bn254>::verify(
        &q_polys.info(),
        &sc_pf,
        &batch_oracle,
        degree_diff,
        &lookup_x,
        &z,
        &lambda,
        // &[&lookup_x_2],
        // &[&z_2].to_vec(),
        // &[&lambda_2],
        &vk,
        &mut verifier_transcript,
        <Bn254 as Pairing>::ScalarField::zero(),
        <Bn254 as Pairing>::ScalarField::zero(),
    )
    .is_ok())
}

#[test]
fn test_mpi_distributed_end_to_end() {
    use ark_bn254::{Bn254, Fr};
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

    use crate::PROTOCOL_NAME;

    const D: usize = 8;
    const S: usize = 8;
    const P: usize = 2;

    let mut rng = StdRng::seed_from_u64(12);

    let params = MultilinearPC::<Bn254>::setup(D, &mut rng);
    let (ck, vk) = MultilinearPC::trim(&params, D);
    let (ck_list, merge_ck) = split_ck(&ck, P);

    let table_evals = (0..1 << S).map(|_| Fr::rand(&mut rng)).collect();
    let table = DenseMultilinearExtension::from_evaluations_vec(S, table_evals);
    let table_list = split_poly(&table, P);

    let query_valid_evals = (0..(1 << D))
        .map(|_| *table.evaluations.choose(&mut rng).unwrap())
        .collect();

    let query_valid = DenseMultilinearExtension::from_evaluations_vec(D, query_valid_evals);
    let query_list = split_poly(&query_valid, P);

    let freq = normalized_multiplicities(&query_valid, &table);
    let freq_list = split_poly(&freq, P);

    let mut prover_transcript = Blake2s512Rng::setup();
    let mut verifier_transcript = Blake2s512Rng::setup();
    // let mut q_polys: ListOfProductsOfPolynomials<ark_ff::Fp<ark_ff::MontBackend<ark_Bn254::FrConfig, 4>, 4>> = ListOfProductsOfPolynomials::new(D);
    let mut distributed_q_polys = vec![ListOfProductsOfPolynomials::new(D - P); 1 << P];

    let x = get_scalar_challenge(&mut prover_transcript);

    let mut distributed_pf = Vec::new();
    let mut comm_0 = Vec::new();
    let mut comm_1 = Vec::new();

    let tmp_pf = LogLookupProof::<Bn254>::prove(&query_valid, &table, &freq, &ck, &x);

    for i in 0..1 << P {
        let pf = LogLookupProof::<Bn254>::prove(
            &query_list[i],
            &table_list[i],
            &freq_list[i],
            &ck_list[i],
            &x,
        );
        comm_0.push(pf.1[0].clone());
        comm_1.push(pf.1[1].clone());
        distributed_pf.push(pf);
    }
    _ = prover_transcript.feed(&combine_comm(&comm_0));
    _ = prover_transcript.feed(&combine_comm(&comm_1));

    let z = get_vector_challenge(&mut prover_transcript, query_valid.num_vars);
    let lambda = get_scalar_challenge(&mut prover_transcript);

    let mut start = 0;
    let log_chunk_size = z.len() - P;
    for i in 0..1 << P {
        append_sumcheck_polys(
            (
                distributed_pf[i].0[0].clone(),
                distributed_pf[i].0[1].clone(),
            ),
            (
                distributed_pf[i].2[0].clone(),
                distributed_pf[i].2[1].clone(),
            ),
            freq_list[i].clone(),
            query_valid.num_vars - table.num_vars,
            &mut distributed_q_polys[i],
            &z,
            &lambda,
            start,
            log_chunk_size,
        );
        start = start + (1 << log_chunk_size);
    }

    let mut prover_states = Vec::new();
    let mut prover_msgs = Vec::new();
    for i in 0..1 << P {
        prover_states.push(IPForMLSumcheck::prover_init(&distributed_q_polys[i]))
    }
    let poly_info = PolynomialInfo {
        max_multiplicands: distributed_q_polys[0].max_multiplicands,
        num_variables: D,
    };
    let mut verifier_state = IPForMLSumcheck::verifier_init(&poly_info);
    let mut verifier_msg = None;

    let mut final_point = Vec::new();

    _ = prover_transcript.feed(&poly_info.clone());

    for _ in 0..D - P {
        let mut prover_message = IPForMLSumcheck::prove_round(&mut prover_states[0], &verifier_msg);
        for i in 1..1 << P {
            let tmp = IPForMLSumcheck::prove_round(&mut prover_states[i], &verifier_msg);
            // Aggregate results from different parties
            for j in 0..prover_message.evaluations.len() {
                prover_message.evaluations[j] = prover_message.evaluations[j] + tmp.evaluations[j]
            }
        }
        prover_transcript.feed(&prover_message);
        prover_msgs.push(prover_message.clone());
        // Using the aggregate results to generate the verifier's message.
        let verifier_msg2 = IPForMLSumcheck::verify_round(
            prover_message,
            &mut verifier_state,
            &mut prover_transcript,
        );
        verifier_msg = verifier_msg2;
        final_point.push(verifier_msg.clone().unwrap().randomness);
        // println!("{:?}", verifier_msg);
    }

    if P != 0 {
        let mut distributed_states = Vec::new();
        for i in 0..1 << P {
            let _ = IPForMLSumcheck::prove_round(&mut prover_states[i], &verifier_msg);
            distributed_states.push(obtain_distrbuted_sumcheck_prover_state(&prover_states[i]))
        }

        // println!("start");
        let merge_poly =
            crate::mpi_utils::merge_list_of_distributed_poly(&distributed_states, &poly_info, P);

        // println!("pass");
        let mut prover_state = IPForMLSumcheck::prover_init(&merge_poly);
        // assert!(prover_states[0].round == nv - log_num_parties);
        // prover_state.round = nv - log_num_parties;
        let mut verifier_msg = None;
        for _ in D - P..D {
            let prover_message = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
            prover_transcript.feed(&prover_message);
            prover_msgs.push(prover_message.clone());
            let verifier_msg2 = IPForMLSumcheck::verify_round(
                prover_message,
                &mut verifier_state,
                &mut prover_transcript,
            );
            verifier_msg = verifier_msg2;
            final_point.push(verifier_msg.clone().unwrap().randomness);
        }
    }

    // println!("finish subclaim");

    // let final_point = final_state.randomness;

    let eta = get_scalar_challenge(&mut prover_transcript);
    let batch_oracle = batch_open_poly(
        &[&tmp_pf.0[0], &tmp_pf.0[1], &freq, &query_valid, &table],
        &[tmp_pf.1[0].clone(), tmp_pf.1[1].clone()],
        &ck,
        &final_point,
        eta,
    );

    let degree_diff = query_valid.num_vars - table.num_vars;

    let poly_info = PolynomialInfo {
        max_multiplicands: distributed_q_polys[0].max_multiplicands,
        num_variables: D,
    };
    let (lookup_x, z, lambda) = LogLookupProof::<Bn254>::verify_before_sumcheck(
        &poly_info,
        &batch_oracle,
        1,
        &mut verifier_transcript,
    );

    assert!(LogLookupProof::<Bn254>::verify(
        &poly_info,
        &prover_msgs,
        &batch_oracle,
        degree_diff,
        &lookup_x,
        &z,
        &lambda,
        &vk,
        &mut verifier_transcript,
        <Bn254 as Pairing>::ScalarField::zero(),
        <Bn254 as Pairing>::ScalarField::zero(),
    )
    .is_ok())
}
