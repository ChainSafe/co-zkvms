use crate::jolt::vm::witness::{Rep3JoltPolynomials, Rep3Polynomials};
use crate::poly::{
    generate_poly_shares_rep3, generate_poly_shares_rep3_vec, Rep3MultilinearPolynomial,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_common::rv_trace::MemoryLayout;
use jolt_core::jolt::vm::read_write_memory::{
    ReadWriteMemoryPolynomials, ReadWriteMemoryPreprocessing, ReadWriteMemoryStuff,
};
use jolt_core::{
    field::JoltField, jolt::vm::timestamp_range_check::TimestampRangeCheckPolynomials,
};

use jolt_tracer::JoltDevice;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{self, PartyID, Rep3BigUintShare, Rep3PrimeFieldShare};

pub type Rep3ReadWriteMemoryPolynomials<F: JoltField> =
    ReadWriteMemoryStuff<Rep3MultilinearPolynomial<F>>;

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3ProgramIO<F: JoltField> {
    pub input_words: Vec<Rep3PrimeFieldShare<F>>,
    pub output_words: Vec<Rep3PrimeFieldShare<F>>,
    pub panic: Rep3PrimeFieldShare<F>, // 0 if not panicked, 1 if panicked
    pub memory_layout: MemoryLayout,
}

impl<F: JoltField> Rep3Polynomials<F, ReadWriteMemoryPreprocessing>
    for Rep3ReadWriteMemoryPolynomials<F>
{
    type PublicPolynomials = ReadWriteMemoryPolynomials<F>;

    fn generate_secret_shares<R: rand::Rng>(
        _: &ReadWriteMemoryPreprocessing,
        polynomials: Self::PublicPolynomials,
        rng: &mut R,
    ) -> Vec<Self> {
        let mut v_read_rd_shares = generate_poly_shares_rep3(&polynomials.v_read_rd, rng);
        let mut v_read_rs1_shares = generate_poly_shares_rep3(&polynomials.v_read_rs1, rng);
        let mut v_read_rs2_shares = generate_poly_shares_rep3(&polynomials.v_read_rs2, rng);
        let mut v_read_ram_shares = generate_poly_shares_rep3(&polynomials.v_read_ram, rng);
        let mut v_write_rd_shares = generate_poly_shares_rep3(&polynomials.v_write_rd, rng);
        let mut v_write_ram_shares = generate_poly_shares_rep3(&polynomials.v_write_ram, rng);
        let mut v_final_shares = generate_poly_shares_rep3(&polynomials.v_final, rng);
        let mut v_init_shares = if let Some(v_init) = polynomials.v_init {
            generate_poly_shares_rep3(&v_init, rng)
                .into_iter()
                .map(Some)
                .collect()
        } else {
            vec![None; 3]
        };

        (0..3)
            .map(|i| Rep3ReadWriteMemoryPolynomials {
                a_ram: Rep3MultilinearPolynomial::public(polynomials.a_ram.clone()),
                v_read_rd: std::mem::take(&mut v_read_rd_shares[i]),
                v_read_rs1: std::mem::take(&mut v_read_rs1_shares[i]),
                v_read_rs2: std::mem::take(&mut v_read_rs2_shares[i]),
                v_read_ram: std::mem::take(&mut v_read_ram_shares[i]),
                v_write_rd: std::mem::take(&mut v_write_rd_shares[i]),
                v_write_ram: std::mem::take(&mut v_write_ram_shares[i]),
                v_final: std::mem::take(&mut v_final_shares[i]),
                t_read_rd: Rep3MultilinearPolynomial::public(polynomials.t_read_rd.clone()),
                t_read_rs1: Rep3MultilinearPolynomial::public(polynomials.t_read_rs1.clone()),
                t_read_rs2: Rep3MultilinearPolynomial::public(polynomials.t_read_rs2.clone()),
                t_read_ram: Rep3MultilinearPolynomial::public(polynomials.t_read_ram.clone()),
                t_final: Rep3MultilinearPolynomial::public(polynomials.t_final.clone()),
                a_init_final: polynomials
                    .a_init_final
                    .as_ref()
                    .map(|poly| Rep3MultilinearPolynomial::public(poly.clone())),
                v_init: std::mem::take(&mut v_init_shares[i]),
                identity: polynomials
                    .identity
                    .as_ref()
                    .map(|poly| Rep3MultilinearPolynomial::public(poly.clone())),
            })
            .collect()
    }

    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &ReadWriteMemoryPreprocessing,
        ops: &mut [crate::jolt::vm::JoltTraceStep<F, Instructions>],
        M: usize,
        network: rep3::network::IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: crate::jolt::instruction::JoltInstructionSet<F>
            + crate::jolt::instruction::Rep3JoltInstructionSet<F>,
        Network: rep3::network::Rep3Network,
    {
        todo!()
    }

    fn combine_polynomials(
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials> {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3ProgramIOInput<F: JoltField> {
    pub input: Vec<Rep3BigUintShare<F>>,
    pub output: Vec<Rep3BigUintShare<F>>,
    pub panic: Rep3PrimeFieldShare<F>, // 0 if not panicked, 1 if panicked
    pub memory_layout: MemoryLayout,
}

impl<F: JoltField> Rep3ProgramIO<F> {
    pub fn generate_witness_rep3<Network>(
        preprocessing: &ReadWriteMemoryPreprocessing,
        program_io: Rep3ProgramIOInput<F>,
        network: &mut IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Network: Rep3Network,
    {
        todo!()
    }

    pub fn generate_secret_shares<R: rand::Rng>(program_io: JoltDevice, rng: &mut R) -> Vec<Self> {
        let mut input_words = vec![Vec::with_capacity(program_io.inputs.len() / 4); 3];
        // Convert input bytes into words
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            rep3::arithmetic::generate_shares_rep3(F::from_u32(word), rng)
                .into_iter()
                .enumerate()
                .for_each(|(i, share)| input_words[i].push(share));
        }

        let mut output_words = vec![Vec::with_capacity(program_io.outputs.len() / 4); 3];
        for chunk in program_io.outputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            rep3::arithmetic::generate_shares_rep3(F::from_u32(word), rng)
                .into_iter()
                .enumerate()
                .for_each(|(i, share)| output_words[i].push(share));
        }

        let mut panic =
            rep3::arithmetic::generate_shares_rep3(F::from_u8(program_io.panic as u8), rng);

        (0..3)
            .map(|i| Self {
                input_words: std::mem::take(&mut input_words[i]),
                output_words: std::mem::take(&mut output_words[i]),
                panic: std::mem::take(&mut panic[i]),
                memory_layout: program_io.memory_layout.clone(),
            })
            .collect()
    }
}
