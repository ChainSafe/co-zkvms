use jolt_common::constants::{ArithmeticWideInt, XLEN};
use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare, arithmetic::sub_shared_by_public};
use mpc_core::protocols::rep3_ring::casts;
use mpc_core::protocols::rep3_ring::conversion as ring_conv;
use mpc_core::protocols::rep3_ring::edabits::{EdaBitsRangeView, ForkedB2aScratch, PreprocessingPool};
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rand::distributions::{Distribution, Standard};

pub(crate) fn biased_inc_b2a_many<F, N>(
    biased_arith: &[Rep3RingShare<ArithmeticWideInt>],
    io_ctx: &mut IoContextPool<N>,
    preproc: &mut PreprocessingPool<F>,
    chunk_size: usize,
    max_forks: usize,
    party_id: PartyID,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    F: JoltField,
    N: Rep3NetworkWorker,
    Standard: Distribution<ArithmeticWideInt>,
{
    let n = biased_arith.len();
    let bias_f = F::from_u64(1u64 << XLEN);
    let mut inc = Vec::with_capacity(n);

    for off in (0..n).step_by(chunk_size.max(1)) {
        let end = (off + chunk_size.max(1)).min(n);
        let chunk = &biased_arith[off..end];
        let biased_bin = ring_conv::a2b_many(chunk, io_ctx.main())?;
        let biased_field: Vec<Rep3PrimeFieldShare<F>> = if max_forks <= 1 {
            let batch_eda = preproc.take_edabits::<ArithmeticWideInt>(chunk.len())?;
            casts::r2f_b2a_preproc_many::<ArithmeticWideInt, F, _>(&biased_bin, &batch_eda, io_ctx.main())?
        } else {
            let fork_chunk_size = biased_bin.len().div_ceil(max_forks);
            let mut session = preproc.begin_forkable_session();
            let reserved = session.reserve_edabits::<ArithmeticWideInt>(chunk.len())?;
            let mut scratch: Vec<ForkedB2aScratch<ArithmeticWideInt, F>> =
                (0..max_forks).map(|_| ForkedB2aScratch::default()).collect();
            let biased_field = io_ctx.par_chunks_preproc(
                &biased_bin,
                Some(fork_chunk_size),
                &mut scratch,
                |start, len| reserved.range_view(start, len),
                |_, xs, view: EdaBitsRangeView<'_, ArithmeticWideInt, F>, ctx, scratch| {
                    view.fill_into_par_safe(&mut scratch.batch)?;
                    casts::r2f_b2a_preproc_many_into::<ArithmeticWideInt, F, _>(
                        xs,
                        scratch.batch.as_ref(),
                        ctx,
                        &mut scratch.cast,
                    )?;
                    Ok::<_, eyre::Report>(scratch.cast.take_output())
                },
            )?;
            session.finalize_success();
            biased_field
        };
        inc.extend(biased_field.into_iter().map(|share| sub_shared_by_public(share, bias_f, party_id)));
    }

    Ok(inc)
}
