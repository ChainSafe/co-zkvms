use crate::{
    field::JoltField,
    jolt::vm::read_write_memory::witness::Rep3ReadWriteMemoryPolynomials,
};

use jolt_core::jolt::vm::timestamp_range_check::{TimestampRangeCheckPolynomials, TimestampRangeCheckStuff};

use crate::poly::Rep3MultilinearPolynomial;

pub fn get_timestamp_range_check_polynomials<F: JoltField, PCS, ProofTranscript>(
    rw_polys: &mut Rep3ReadWriteMemoryPolynomials<F>,
) -> TimestampRangeCheckPolynomials<F>
where
    PCS: crate::poly::commitment::Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: jolt_core::utils::transcript::Transcript,
{
    // Take timestamp polys to use for witness generation
    let mut polys = jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryPolynomials {
        t_read_rd: std::mem::take(rw_polys.t_read_rd.as_public_mut()),
        t_read_rs1: std::mem::take(rw_polys.t_read_rs1.as_public_mut()),
        t_read_rs2: std::mem::take(rw_polys.t_read_rs2.as_public_mut()),
        t_read_ram: std::mem::take(rw_polys.t_read_ram.as_public_mut()),
        ..Default::default()
    };

    let res = jolt_core::jolt::vm::timestamp_range_check::TimestampValidityProof::<
        F,
        PCS,
        ProofTranscript,
    >::generate_witness(&polys);

    // Put back
    std::mem::swap(&mut polys.t_read_rd, rw_polys.t_read_rd.as_public_mut());
    std::mem::swap(&mut polys.t_read_rs1, rw_polys.t_read_rs1.as_public_mut());
    std::mem::swap(&mut polys.t_read_rs2, rw_polys.t_read_rs2.as_public_mut());
    std::mem::swap(&mut polys.t_read_ram, rw_polys.t_read_ram.as_public_mut());
    res
}

pub fn get_timestamp_range_check_polynomials_rep3<F: JoltField, PCS, ProofTranscript>(
    rw_polys: &mut Rep3ReadWriteMemoryPolynomials<F>,
) -> TimestampRangeCheckStuff<Rep3MultilinearPolynomial<F>>
where
    PCS: crate::poly::commitment::Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: jolt_core::utils::transcript::Transcript,
{
    let TimestampRangeCheckStuff {
        read_cts_read_timestamp,
        read_cts_global_minus_read,
        final_cts_read_timestamp,
        final_cts_global_minus_read,
        ..
    } = get_timestamp_range_check_polynomials::<F, PCS, ProofTranscript>(rw_polys);
    TimestampRangeCheckStuff {
        read_cts_read_timestamp: read_cts_read_timestamp.map(Rep3MultilinearPolynomial::public),
        read_cts_global_minus_read: read_cts_global_minus_read
            .map(Rep3MultilinearPolynomial::public),
        final_cts_read_timestamp: final_cts_read_timestamp.map(Rep3MultilinearPolynomial::public),
        final_cts_global_minus_read: final_cts_global_minus_read
            .map(Rep3MultilinearPolynomial::public),
        identity: None,
    }
}
