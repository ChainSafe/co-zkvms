use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::vm::witness::Rep3Polynomials;
use crate::jolt::vm::JoltTraceStep;
use crate::poly::{generate_poly_shares_rep3, Rep3MultilinearPolynomial};
use crate::utils::transpose;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::izip;
use jolt_common::rv_trace::{MemoryLayout, MemoryOp};
use jolt_core::jolt::vm::read_write_memory::{
    memory_address_to_witness_index, remap_address, ReadWriteMemoryPolynomials,
    ReadWriteMemoryPreprocessing, ReadWriteMemoryStuff,
};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;

use jolt_tracer::JoltDevice;
use mpc_core::protocols::rep3::network::{
    IoContext, Rep3Network, Rep3NetworkCoordinator, Rep3NetworkWorker, WorkerIoContext,
};
use mpc_core::protocols::rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use serde::{Deserialize, Serialize};

use rayon::prelude::*;

pub type Rep3ReadWriteMemoryPolynomials<F> = ReadWriteMemoryStuff<Rep3MultilinearPolynomial<F>>;

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3ProgramIO<F: JoltField> {
    pub v_io: Rep3MultilinearPolynomial<F>,
    pub memory_layout: MemoryLayout,
}

impl<F: JoltField> Rep3Polynomials<F, ReadWriteMemoryPreprocessing>
    for Rep3ReadWriteMemoryPolynomials<F>
{
    type PublicPolynomials = ReadWriteMemoryPolynomials<F>;

    #[tracing::instrument(
        skip_all,
        name = "Rep3ReadWriteMemoryPolynomials::stream_secret_shares",
        level = "trace"
    )]
    fn stream_secret_shares<R: rand::Rng, Network: Rep3NetworkCoordinator>(
        _: &ReadWriteMemoryPreprocessing,
        polynomials: Self::PublicPolynomials,
        rng: &mut R,
        network: &mut Network,
    ) -> eyre::Result<()> {
        let public_polynomials = (0..3)
            .map(|_| Rep3ReadWriteMemoryPolynomials {
                a_ram: Rep3MultilinearPolynomial::public(polynomials.a_ram.clone()),
                v_read_rd: Default::default(),
                v_read_rs1: Default::default(),
                v_read_rs2: Default::default(),
                v_read_ram: Default::default(),
                v_write_rd: Default::default(),
                v_write_ram: Default::default(),
                v_final: Default::default(),
                t_read_rd: Rep3MultilinearPolynomial::public(polynomials.t_read_rd.clone()),
                t_read_rs1: Rep3MultilinearPolynomial::public(polynomials.t_read_rs1.clone()),
                t_read_rs2: Rep3MultilinearPolynomial::public(polynomials.t_read_rs2.clone()),
                t_read_ram: Rep3MultilinearPolynomial::public(polynomials.t_read_ram.clone()),
                t_final: Rep3MultilinearPolynomial::public(polynomials.t_final.clone()),
                a_init_final: polynomials
                    .a_init_final
                    .as_ref()
                    .map(|poly| Rep3MultilinearPolynomial::public(poly.clone())),
                v_init: Default::default(),
                identity: polynomials
                    .identity
                    .as_ref()
                    .map(|poly| Rep3MultilinearPolynomial::public(poly.clone())),
            })
            .collect();
        network.send_requests_blocking(public_polynomials)?;

        let v_read_rd_shares = generate_poly_shares_rep3(&polynomials.v_read_rd, rng);
        network.send_requests_blocking(v_read_rd_shares)?;
        let v_read_rs1_shares = generate_poly_shares_rep3(&polynomials.v_read_rs1, rng);
        network.send_requests_blocking(v_read_rs1_shares)?;
        let v_read_rs2_shares = generate_poly_shares_rep3(&polynomials.v_read_rs2, rng);
        network.send_requests_blocking(v_read_rs2_shares)?;
        let v_read_ram_shares = generate_poly_shares_rep3(&polynomials.v_read_ram, rng);
        network.send_requests_blocking(v_read_ram_shares)?;
        let v_write_rd_shares = generate_poly_shares_rep3(&polynomials.v_write_rd, rng);
        network.send_requests_blocking(v_write_rd_shares)?;
        let v_write_ram_shares = generate_poly_shares_rep3(&polynomials.v_write_ram, rng);
        network.send_requests_blocking(v_write_ram_shares)?;
        let v_final_shares = generate_poly_shares_rep3(&polynomials.v_final, rng);
        network.send_requests_blocking(v_final_shares)?;
        let v_init_shares: Vec<_> = if let Some(v_init) = polynomials.v_init {
            generate_poly_shares_rep3(&v_init, rng)
                .into_iter()
                .map(Some)
                .collect()
        } else {
            panic!("v_init is not set");
        };
        network.send_requests_blocking(v_init_shares)?;

        Ok(())
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3ReadWriteMemoryPolynomials::receive_witness_share",
        level = "trace"
    )]
    fn receive_witness_share<Network: rep3::network::Rep3NetworkWorker>(
        _: &ReadWriteMemoryPreprocessing,
        io_ctx: &mut rep3::network::IoContextPool<Network>,
    ) -> eyre::Result<Self> {
        let mut partial: Self = io_ctx.network().receive_request()?;
        partial.v_read_rd = io_ctx.network().receive_request()?;
        partial.v_read_rs1 = io_ctx.network().receive_request()?;
        partial.v_read_rs2 = io_ctx.network().receive_request()?;
        partial.v_read_ram = io_ctx.network().receive_request()?;
        partial.v_write_rd = io_ctx.network().receive_request()?;
        partial.v_write_ram = io_ctx.network().receive_request()?;
        partial.v_final = io_ctx.network().receive_request()?;
        partial.v_init = io_ctx.network().receive_request()?;
        Ok(partial)
    }

    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &ReadWriteMemoryPreprocessing,
        ops: &mut [crate::jolt::vm::JoltTraceStep<Instructions>],
        M: usize,
        network: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: crate::jolt::instruction::JoltInstructionSet
            + crate::jolt::instruction::Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rep3ProgramIOInput {
    pub inputs: Vec<Rep3RingShare<u8>>,
    pub outputs: Vec<Rep3RingShare<u8>>,
    pub panic: Rep3RingShare<Bit>, // 0 if not panicked, 1 if panicked
    pub memory_layout: MemoryLayout,
}

impl Rep3ProgramIOInput {
    pub fn generate_secret_shares<R: rand::Rng>(program_io: JoltDevice, rng: &mut R) -> Vec<Self> {
        let JoltDevice {
            inputs,
            outputs,
            panic,
            memory_layout,
        } = program_io;

        let inputs = transpose(
            inputs
                .into_iter()
                .map(|byte| rep3_ring::binary::generate_shares_rep3(byte.into(), rng))
                .collect::<Vec<_>>(),
        );

        let outputs = transpose(
            outputs
                .into_iter()
                .map(|byte| rep3_ring::binary::generate_shares_rep3(byte.into(), rng))
                .collect::<Vec<_>>(),
        );

        let panic = rep3_ring::binary::generate_shares_rep3(panic.into(), rng);

        izip!(inputs, outputs, panic)
            .map(|(inputs, outputs, panic)| Self {
                inputs,
                outputs,
                panic,
                memory_layout,
            })
            .collect()
    }
}

impl<F: JoltField> Rep3ProgramIO<F> {
    pub fn generate_witness_rep3<Network, Instruction>(
        program_io: Rep3ProgramIOInput,
        trace: &[JoltTraceStep<Instruction>],
        program_io: &Rep3ProgramIO<F>,
        io_ctx: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Network: Rep3NetworkWorker,
        Instruction: JoltInstructionSet,
    {
        const RAM: usize = 3;

        let Rep3ProgramIOInput {
            inputs,
            outputs,
            panic,
            memory_layout,
        } = program_io;

        let max_trace_address = trace
            .par_iter()
            .map(|step| match step.memory_ops[RAM] {
                MemoryOp::Read(a) => remap_address(a, &program_io.memory_layout),
                MemoryOp::Write(a, _) => remap_address(a, &program_io.memory_layout),
            })
            .max()
            .unwrap();

        let padded_memory_size = max_trace_address.next_power_of_two() as usize;
        let input_words_len = inputs.len() / 4;
        let output_words_len = outputs.len() / 4;

        // Convert input bytes into words and populate `v_io`
        let input_words = inputs
            .par_chunks(4)
            .map(|word| Rep3RingShare::<u32>::from_le_bytes(word));

        // Convert output bytes into words and populate `v_io`
        let output_words = outputs
            .par_chunks(4)
            .map(|word| Rep3RingShare::<u32>::from_le_bytes(word));

        let (mut input_shares, mut output_shares) = {
            let mut words =
                io_ctx.par_chunks(input_words.chain(output_words), None, |words, io_ctx| {
                    rep3_ring::casts::binary_ring_to_field_many(&words, io_ctx)
                })?;
            let output_words = words.split_off(input_words_len);
            assert_eq!(output_words.len(), output_words_len);
            (words, output_words)
        };

        let termination_bits = [panic, !panic];
        let [panic, termination] = rep3_ring::conversion::bit_inject_from_bits_to_field_many(
            &termination_bits,
            io_ctx.main(),
        )?
        .try_into()
        .unwrap();

        let mut v_io: Vec<_> = vec![Rep3PrimeFieldShare::zero_share(); padded_memory_size];
        let input_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        );
        let output_index = memory_address_to_witness_index(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        );
        v_io[input_index..input_index + input_words_len].swap_with_slice(&mut input_shares[..]);
        v_io[output_index..output_index + output_words_len].swap_with_slice(&mut output_shares[..]);

        v_io[memory_address_to_witness_index(
            program_io.memory_layout.panic,
            &program_io.memory_layout,
        )] = panic;
        v_io[memory_address_to_witness_index(
            program_io.memory_layout.termination,
            &program_io.memory_layout,
        )] = termination;

        Ok(Self {
            v_io: Rep3MultilinearPolynomial::from(v_io),
            memory_layout,
        })
    }

    pub fn generate_secret_shares<R: rand::Rng>(
        program_io: JoltDevice,
        memory_size: usize,
        rng: &mut R,
    ) -> Vec<Self> {
        let mut v_io: Vec<u32> = vec![0; memory_size];
        let mut input_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        );
        // Convert input bytes into words and populate `v_io`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            v_io[input_index] = word;
            input_index += 1;
        }
        let mut output_index = memory_address_to_witness_index(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        );
        // Convert output bytes into words and populate `v_io`
        for chunk in program_io.outputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            v_io[output_index] = word;
            output_index += 1;
        }

        // Copy panic bit
        v_io[memory_address_to_witness_index(
            program_io.memory_layout.panic,
            &program_io.memory_layout,
        )] = program_io.panic as u32;
        if !program_io.panic {
            // Set termination bit
            v_io[memory_address_to_witness_index(
                program_io.memory_layout.termination,
                &program_io.memory_layout,
            )] = 1;
        }

        let v_io = MultilinearPolynomial::<F>::from(v_io);
        let mut v_io_shares = generate_poly_shares_rep3(&v_io, rng);

        (0..3)
            .map(|i| Self {
                v_io: std::mem::take(&mut v_io_shares[i]),
                memory_layout: program_io.memory_layout.clone(),
            })
            .collect()
    }
}
