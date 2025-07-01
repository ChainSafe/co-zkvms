use std::{cmp::max, collections::HashMap, marker::PhantomData};

use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, CryptographicSponge};
use ark_ec::pairing::Pairing;
use ark_ff::{Field, Zero};
use ark_poly::{
    multivariate::{SparsePolynomial, SparseTerm},
    DenseMultilinearExtension,
};
use ark_poly_commit::{
    marlin_pst13_pc::{
        CommitterKey as MaskCommitterKey, MarlinPST13, UniversalParams as MaskParam,
        VerifierKey as MaskVerifierKey,
    },
    multilinear_pc::{
        data_structures::{Commitment, CommitterKey, UniversalParams, VerifierKey},
        MultilinearPC,
    },
    PolynomialCommitment,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;

use super::zk::{ZKMLCommit, ZKMLCommitterKey, ZKMLVerifierKey, SRS};
use crate::{
    utils::{hash_usize, normalized_multiplicities},
    math::Math, R1CSInstance,
};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct IndexProverKey<E: Pairing> {
    pub row: Vec<usize>,
    pub col: Vec<usize>,
    pub rows_indexed: Vec<usize>,
    pub cols_indexed: Vec<usize>,
    pub val_a_indexed: DenseMultilinearExtension<E::ScalarField>,
    pub val_b_indexed: DenseMultilinearExtension<E::ScalarField>,
    pub val_c_indexed: DenseMultilinearExtension<E::ScalarField>,
    pub val_a: DenseMultilinearExtension<E::ScalarField>,
    pub val_b: DenseMultilinearExtension<E::ScalarField>,
    pub val_c: DenseMultilinearExtension<E::ScalarField>,
    pub freq_r: DenseMultilinearExtension<E::ScalarField>,
    pub freq_c: DenseMultilinearExtension<E::ScalarField>,
    pub real_len_val: usize,
    pub num_variables_val: usize,
    pub padded_num_var: usize,
    pub ck_w: ZKMLCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    pub ck_index: CommitterKey<E>,
    pub ck_mask: MaskCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct IndexVerifierKey<E: Pairing> {
    // pub row: Vec<usize>,
    // pub col: Vec<usize>,
    pub val_a_oracle: Commitment<E>,
    pub val_b_oracle: Commitment<E>,
    pub val_c_oracle: Commitment<E>,
    pub real_len_val: usize,
    pub num_variables_val: usize,
    pub padded_num_var: usize,
    pub vk_w: ZKMLVerifierKey<E>,
    pub vk_index: VerifierKey<E>,
    pub vk_mask: MaskVerifierKey<E>,
}

pub struct Indexer<E: Pairing> {
    _marker: PhantomData<E>,
}

impl<E: Pairing> Indexer<E> {
    ///Before execute prover and verifier, call indexer to gain pk, vk and arthimetization (row, col) of the R1CS instance.
    /// Output pk, vk which will be feeded into prover_init and verifier_check.
    /// For the time the padding is not implemented.
    #[allow(non_snake_case)]
    pub fn index_for_prover_and_verifier(
        relation: &R1CSInstance<E::ScalarField>,
        srs: &SRS<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    ) -> (IndexProverKey<E>, IndexVerifierKey<E>) {
        let mut padded_num_var = max(relation.num_cons, relation.num_vars);
        println!("padded_num_var: {:?}", padded_num_var);
        let param = &srs.poly_srs;
        let param_index = &srs.poly_srs.0;
        let param_mask = &srs.mask_srs;
        padded_num_var = padded_num_var.log_2();
        let mut row: Vec<usize> = Vec::new();
        let mut col: Vec<usize> = Vec::new();
        let mut v_a: Vec<E::ScalarField> = Vec::new();
        let mut v_b: Vec<E::ScalarField> = Vec::new();
        let mut v_c: Vec<E::ScalarField> = Vec::new();
        let mut count: usize = 0;
        let mut evaluation_point: HashMap<(usize, usize), usize> = HashMap::new();
        let mat_A = &relation.A.M;
        let mat_B = &relation.B.M;
        let mat_C = &relation.C.M;
        println!(
            "mat_A.len: {:?}",
            (relation.A.num_vars_x, relation.A.num_vars_y)
        );
        println!(
            "mat_B.len: {:?}",
            (relation.B.num_vars_x, relation.B.num_vars_y)
        );
        println!(
            "mat_C.len: {:?}",
            (relation.C.num_vars_x, relation.C.num_vars_y)
        );
        // seems like this the only place where second sumcheck can be corrupted
        // eq_rx is generated from first sumcheck challange
        // checksum2 is val_a(r2) * r2 + val_b(r2) * r2 + val_c(r2) * r2 ?check?
        let mut iter = 0;
        println!("mat_C.len: {:?}", mat_C.len());
        let max_len = max(mat_A.len(), max(mat_B.len(), mat_C.len()));
        mat_A
            .iter()
            .map(|e| Some(e.clone()))
            .pad_using(max_len, |_| None)
            .zip(
                mat_B
                    .iter()
                    .map(|e| Some(e.clone()))
                    .pad_using(max_len, |_| None),
            )
            .zip(
                mat_C
                    .iter()
                    .map(|e| Some(e.clone()))
                    .pad_using(max_len, |_| None),
            )
            .for_each(|((e_a, e_b), e_c)| {
                if let Some(e_a) = e_a {
                    if !evaluation_point.contains_key(&(e_a.row, e_a.col)) {
                        v_a.push(e_a.val);
                        v_b.push(E::ScalarField::zero());
                        v_c.push(E::ScalarField::zero());
                        row.push(e_a.row);
                        col.push(e_a.col);
                        evaluation_point.insert((e_a.row, e_a.col), count);
                        count += 1;
                    } else {
                        if e_a.val != v_a[evaluation_point[&(e_a.row, e_a.col)]] {
                            println!(
                                "e_a.val: {:?} v_a: {:?}",
                                e_a.val,
                                v_a[evaluation_point[&(e_a.row, e_a.col)]]
                            );
                        }
                        v_a[evaluation_point[&(e_a.row, e_a.col)]] = e_a.val;
                    }
                }
                if let Some(e_b) = e_b {
                    if !evaluation_point.contains_key(&(e_b.row, e_b.col)) {
                        v_a.push(E::ScalarField::zero());
                        v_b.push(e_b.val);
                        v_c.push(E::ScalarField::zero());
                        row.push(e_b.row);
                        col.push(e_b.col);
                        evaluation_point.insert((e_b.row, e_b.col), count);
                        count += 1;
                    } else {
                        // if e_b.val != v_b[evaluation_point[&(e_b.row, e_b.col)]] {
                        //     println!("e_b.val: {:?} v_b: {:?}", e_b.val, v_b[evaluation_point[&(e_b.row, e_b.col)]]);
                        // }
                        v_b[evaluation_point[&(e_b.row, e_b.col)]] = e_b.val;
                    }
                }

                if let Some(e_c) = e_c {
                    if !evaluation_point.contains_key(&(e_c.row, e_c.col)) {
                        v_a.push(E::ScalarField::zero());
                        v_b.push(E::ScalarField::zero());
                        v_c.push(e_c.val);
                        row.push(e_c.row);
                        col.push(e_c.col);
                        evaluation_point.insert((e_c.row, e_c.col), count);
                        count += 1;
                    } else {
                        // if e_c.val != v_c[evaluation_point[&(e_c.row, e_c.col)]] {
                        //     println!("e_c.val: {:?} v_c: {:?}", e_c.val, v_c[evaluation_point[&(e_c.row, e_c.col)]]);
                        // }
                        v_c[evaluation_point[&(e_c.row, e_c.col)]] = e_c.val;
                    }
                }
            });
        println!("iter: {:?}", iter);
        let real_len_val = count;
        println!("real_len_val: {:?}", real_len_val);
        count = count.next_power_of_two();
        let num_non_zero_var = Math::log_2(count) as usize;
        println!("v_a.len: {:?}", v_a.len());
        println!("v_b.len: {:?}", v_b.len());
        println!("v_c.len: {:?}", v_c.len());
        println!("row.len: {:?}", row.len());
        println!("col.len: {:?}", col.len());
        println!("count: {:?}", count);
        v_a.resize(count, E::ScalarField::zero());
        v_b.resize(count, E::ScalarField::zero());
        v_c.resize(count, E::ScalarField::zero());
        row.resize(count, usize::MAX);
        col.resize(count, usize::MAX);
        println!("param.0.num_vars: {:?}", param.0.num_vars);
        let (ck_w, vk_w) = ZKMLCommit::<E, SparsePolynomial<E::ScalarField, SparseTerm>>::trim(
            param,
            padded_num_var,
            2,
        );
        let (ck_index, vk_index) = MultilinearPC::trim(param_index, num_non_zero_var);
        let val_a = DenseMultilinearExtension {
            evaluations: (v_a),
            num_vars: (num_non_zero_var),
        };
        let val_b = DenseMultilinearExtension {
            evaluations: (v_b),
            num_vars: (num_non_zero_var),
        };
        let val_c = DenseMultilinearExtension {
            evaluations: (v_c),
            num_vars: (num_non_zero_var),
        };
        let val_a_oracle = MultilinearPC::commit(&ck_index, &val_a);
        let val_b_oracle = MultilinearPC::commit(&ck_index, &val_b);
        let val_c_oracle = MultilinearPC::commit(&ck_index, &val_c);
        let (ck_mask, vk_mask) = MarlinPST13::<_, _>::trim(param_mask, 4, 1, None).unwrap();

        let domain = (0usize..1 << num_non_zero_var).collect::<Vec<_>>();
        let domain_vec = hash_usize(&domain.as_slice());
        let domain_poly =
            DenseMultilinearExtension::from_evaluations_vec(num_non_zero_var, domain_vec);

        let row_vec = hash_usize(&row[..real_len_val]);
        let col_vec = hash_usize(&col[..real_len_val]);
        let row_poly = DenseMultilinearExtension::from_evaluations_vec(num_non_zero_var, row_vec);
        let col_poly = DenseMultilinearExtension::from_evaluations_vec(num_non_zero_var, col_vec);
        let freq_row = normalized_multiplicities(&row_poly, &domain_poly);
        let freq_col = normalized_multiplicities(&col_poly, &domain_poly);

        //let (ck, vk) = MultilinearPC::trim(param, num_non_zero_var);
        (
            IndexProverKey {
                row: (row.clone()),
                col: (col.clone()),
                real_len_val,
                padded_num_var,
                val_a: val_a.clone(),
                val_b: val_b.clone(),
                val_c: val_c.clone(),
                freq_r: freq_row.clone(),
                freq_c: freq_col.clone(),
                num_variables_val: (num_non_zero_var),
                ck_w,
                ck_index,
                ck_mask,

                rows_indexed: Vec::new(),
                cols_indexed: Vec::new(),
                val_a_indexed: Default::default(),
                val_b_indexed: Default::default(),
                val_c_indexed: Default::default(),
            },
            IndexVerifierKey {
                // row: (row),
                // col: (col),
                real_len_val,
                padded_num_var,
                val_a_oracle,
                val_b_oracle,
                val_c_oracle,
                num_variables_val: (num_non_zero_var),
                vk_w,
                vk_index,
                vk_mask,
            },
        )
    }
}
