use std::mem;

use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::trace::mem_op::MemoryOp;
use crate::jolt::vm::witness::Rep3Polynomials;
use crate::jolt::vm::JoltTraceStep;
use crate::poly::{generate_poly_shares_rep3, Rep3MultilinearPolynomial};
use crate::utils::future_ring::{FutureRep3Ring, Rep3RingFutureExt};
use crate::utils::transpose;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::izip;
use jolt_common::constants::REGISTER_COUNT;
use jolt_common::rv_trace::MemoryLayout;
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

const RS1: usize = 0;
const RS2: usize = 1;
const RD: usize = 2;
const RAM: usize = 3;

pub type Rep3ReadWriteMemoryPolynomials<F> = ReadWriteMemoryStuff<Rep3MultilinearPolynomial<F>>;

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3ProgramIO<F: JoltField> {
    pub v_io: Rep3MultilinearPolynomial<F>,
    pub memory_layout: MemoryLayout,
    pub memory_size: usize,
    input_words_len: usize,
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
        trace: &mut [JoltTraceStep<Instructions>],
        program_io: &Rep3ProgramIO<F>,
        M: usize,
        io_ctx: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: crate::jolt::instruction::Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker,
    {
        let m = trace.len();
        assert!(m.is_power_of_two());

        let memory_size = program_io.memory_size;
        let mut v_init: Vec<_> = vec![Rep3PrimeFieldShare::zero_share(); memory_size];
        // Copy bytecode
        let v_init_index = memory_address_to_witness_index(
            preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        );
        let v_inputs_index = v_init_index + preprocessing.bytecode_words.len();
        let party_id = io_ctx.party_id();
        v_init[v_init_index..v_inputs_index]
            .par_iter_mut()
            .zip_eq(&preprocessing.bytecode_words)
            .for_each(|(v_init, word)| {
                *v_init = rep3::arithmetic::promote_to_trivial_share(party_id, F::from_u32(*word))
            });
        // Copy input bytes
        let v_inputs_range = v_inputs_index..v_inputs_index + program_io.input_words_len;
        v_init[v_inputs_range.clone()]
            .par_iter_mut()
            .zip_eq(&program_io.v_io.as_shared().coeffs_ref()[v_inputs_range])
            .for_each(|(v_init, word)| {
                *v_init = *word;
            });

        let mut a_ram: Vec<u32> = Vec::with_capacity(m);

        let mut v_read_rs1: Vec<_> = Vec::with_capacity(m);
        let mut v_read_rs2: Vec<_> = Vec::with_capacity(m);
        let mut v_read_rd: Vec<_> = Vec::with_capacity(m);
        let mut v_read_ram: Vec<_> = Vec::with_capacity(m);

        let mut t_read_rs1: Vec<u32> = Vec::with_capacity(m);
        let mut t_read_rs2: Vec<u32> = Vec::with_capacity(m);
        let mut t_read_rd: Vec<u32> = Vec::with_capacity(m);
        let mut t_read_ram: Vec<u32> = Vec::with_capacity(m);

        let mut v_write_rd: Vec<_> = Vec::with_capacity(m);
        let mut v_write_ram: Vec<_> = Vec::with_capacity(m);

        let mut t_final = vec![0; memory_size];
        let mut v_final = v_init
            .par_iter()
            .copied()
            .map(FutureRep3Ring::<u64, Rep3PrimeFieldShare<F>>::Ready)
            .collect::<Vec<_>>();

        let span = tracing::span!(tracing::Level::DEBUG, "memory_trace_processing");
        let _enter = span.enter();

        for (i, step) in trace.iter().enumerate() {
            let timestamp = i as u32;

            match step.memory_ops[RS1] {
                MemoryOp::Read(a) => {
                    assert!(a < REGISTER_COUNT);
                    let a = a as usize;

                    v_read_rs1.push(a);
                    t_read_rs1.push(t_final[a]);
                    t_final[a] = timestamp;
                }
                MemoryOp::Write(a, v) => {
                    panic!("Unexpected rs1 MemoryOp::Write({a}, {v:?})");
                }
            };

            match step.memory_ops[RS2] {
                MemoryOp::Read(a) => {
                    assert!(a < REGISTER_COUNT);
                    let a = a as usize;

                    v_read_rs2.push(a);
                    t_read_rs2.push(t_final[a]);
                    t_final[a] = timestamp;
                }
                MemoryOp::Write(a, v) => {
                    panic!("Unexpected rs2 MemoryOp::Write({a}, {v:?})")
                }
            };

            match step.memory_ops[RD] {
                MemoryOp::Read(a) => {
                    panic!("Unexpected rd MemoryOp::Read({a})")
                }
                MemoryOp::Write(a, v_new) => {
                    assert!(a < REGISTER_COUNT);
                    let a = a as usize;

                    v_read_rd.push(a);
                    t_read_rd.push(t_final[a]);
                    v_final[a] = FutureRep3Ring::cast_to_field_b2a(*v_new.as_shared());
                    v_write_rd.push(a);
                    t_final[a] = timestamp;
                }
            };

            match step.memory_ops[RAM] {
                MemoryOp::Read(a) => {
                    debug_assert!(a % 4 == 0);
                    let remapped_a = remap_address(a, &program_io.memory_layout) as usize;

                    a_ram.push(remapped_a as u32);
                    v_read_ram.push(remapped_a);
                    t_read_ram.push(t_final[remapped_a]);
                    v_write_ram.push(remapped_a);
                    t_final[remapped_a] = timestamp;
                }
                MemoryOp::Write(a, v_new) => {
                    debug_assert!(a % 4 == 0);
                    let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
                    a_ram.push(remapped_a as u32);
                    v_read_ram.push(remapped_a);
                    t_read_ram.push(t_final[remapped_a]);
                    v_final[remapped_a] = FutureRep3Ring::cast_to_field_b2a(*v_new.as_shared());
                    v_write_ram.push(remapped_a);
                    t_final[remapped_a] = timestamp;
                }
            }
        }

        let v_final = v_final.fulfill_batched(io_ctx, |res, _| res)?;
        let v_read_rd = v_read_rd
            .into_par_iter()
            .map(|addr| v_final[addr])
            .collect::<Vec<_>>();
        let v_read_rs1 = v_read_rs1
            .into_par_iter()
            .map(|addr| v_final[addr])
            .collect::<Vec<_>>();
        let v_read_rs2 = v_read_rs2
            .into_par_iter()
            .map(|addr| v_final[addr])
            .collect::<Vec<_>>();
        let v_read_ram = v_read_ram
            .into_par_iter()
            .map(|addr| v_final[addr])
            .collect::<Vec<_>>();
        let v_write_ram = v_write_ram
            .into_par_iter()
            .map(|addr| v_final[addr])
            .collect::<Vec<_>>();
        let v_write_rd = v_write_rd
            .into_par_iter()
            .map(|addr| v_final[addr])
            .collect::<Vec<_>>();

        let [a_ram, t_read_rd, t_read_rs1, t_read_rs2, t_read_ram, t_final] =
            map_to_polys_public([
                a_ram, t_read_rd, t_read_rs1, t_read_rs2, t_read_ram, t_final,
            ]);

        let [v_read_rd, v_read_rs1, v_read_rs2, v_read_ram, v_write_rd, v_write_ram, v_final, v_init] =
            map_to_polys_shared([
                v_read_rd,
                v_read_rs1,
                v_read_rs2,
                v_read_ram,
                v_write_rd,
                v_write_ram,
                v_final,
                v_init,
            ]);

        Ok(Self {
            a_ram,
            v_read_rd,
            v_read_rs1,
            v_read_rs2,
            v_read_ram,
            v_write_rd,
            v_write_ram,
            v_final,
            t_read_rd,
            t_read_rs1,
            t_read_rs2,
            t_read_ram,
            t_final,
            v_init: Some(v_init),
            a_init_final: None,
            identity: None,
        })
    }

    fn combine_polynomials(
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials> {
        todo!()
    }
}

fn map_to_polys_public<F: JoltField, const N: usize>(
    vals: [Vec<u32>; N],
) -> [Rep3MultilinearPolynomial<F>; N] {
    vals.into_par_iter()
        .map(Rep3MultilinearPolynomial::from)
        .collect::<Vec<Rep3MultilinearPolynomial<F>>>()
        .try_into()
        .unwrap()
}

fn map_to_polys_shared<F: JoltField, const N: usize>(
    vals: [Vec<Rep3PrimeFieldShare<F>>; N],
) -> [Rep3MultilinearPolynomial<F>; N] {
    vals.into_par_iter()
        .map(Rep3MultilinearPolynomial::from)
        .collect::<Vec<Rep3MultilinearPolynomial<F>>>()
        .try_into()
        .unwrap()
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
        io_ctx: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Network: Rep3NetworkWorker,
        Instruction: JoltInstructionSet,
    {
        assert!(program_io.inputs.len() <= program_io.memory_layout.max_input_size as usize);
        assert!(program_io.outputs.len() <= program_io.memory_layout.max_output_size as usize);

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

        let memory_size = max_trace_address.next_power_of_two() as usize;
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

        let mut v_io: Vec<_> = vec![Rep3PrimeFieldShare::zero_share(); memory_size];
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
            memory_size,
            input_words_len,
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
                memory_size,
                input_words_len: program_io.inputs.len() / 4,
            })
            .collect()
    }
}
