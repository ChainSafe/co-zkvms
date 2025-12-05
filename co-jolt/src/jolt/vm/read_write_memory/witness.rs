use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::trace::mem_op::MemoryOp;
use crate::jolt::vm::witness::Rep3Polynomials;
use crate::jolt::vm::JoltTraceStep;
use crate::poly::{generate_poly_shares_rep3, Rep3MultilinearPolynomial};
use crate::utils::transpose;
use crate::utils::types::Either;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::izip;
use jolt_common::constants::REGISTER_COUNT;
use jolt_common::rv_trace::MemoryLayout;
#[cfg(feature = "debug")]
use jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryPolynomials;
use jolt_core::jolt::vm::read_write_memory::{
    memory_address_to_witness_index, remap_address, ReadWriteMemoryPreprocessing,
    ReadWriteMemoryStuff,
};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;

use jolt_tracer::JoltDevice;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, Rep3PrimeFieldShare};
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
    #[cfg(feature = "debug")]
    type PublicPolynomials = ReadWriteMemoryPolynomials<F>;

    #[tracing::instrument(
        skip_all,
        name = "ReadWriteMemory::generate_witness_rep3",
        level = "info"
    )]
    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &ReadWriteMemoryPreprocessing,
        trace: &mut [JoltTraceStep<Instructions>],
        program_io: &Rep3ProgramIO<F>,
        _: usize,
        io_ctx: &mut IoContextPool<Network>,
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
        let v_inputs_range = v_init_index..v_init_index + preprocessing.bytecode_words.len();
        let id = io_ctx.party_id();
        v_init[v_inputs_range]
            .par_iter_mut()
            .zip_eq(&preprocessing.bytecode_words)
            .for_each(|(v_init, word)| {
                *v_init = rep3::arithmetic::promote_to_trivial_share(id, F::from_u32(*word))
            });
        let v_inputs_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        );
        // Copy input words
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
        let mut v_final = v_init.clone();

        let mut write_vals_f = vec![
            (
                Rep3PrimeFieldShare::<F>::zero_share(), // RD
                Rep3PrimeFieldShare::<F>::zero_share()  // RAM
            );
            m
        ];

        let (write_vals_ring, write_refs): (Vec<_>, Vec<_>) = trace
            .par_iter_mut()
            .zip_eq(write_vals_f.par_iter_mut())
            .flat_map(|(step, ops_f)| {
                let mut new_vals = vec![];
                let mut mut_refs = vec![];

                if let MemoryOp::Write(_, Either::Shared(v_new)) = step.memory_ops[RD] {
                    new_vals.push(v_new);
                    mut_refs.push(&mut ops_f.0);
                }

                if let MemoryOp::Write(_, Either::Shared(v_new)) = step.memory_ops[RAM] {
                    new_vals.push(v_new);
                    mut_refs.push(&mut ops_f.1);
                }

                (new_vals, mut_refs)
            })
            .unzip();

        let _guard = tracing::trace_span!("cast_writes").entered();
        io_ctx
            .par_chunks(write_vals_ring, None, |chunk, io_ctx| {
                rep3_ring::casts::binary_ring_to_field_many(&chunk, io_ctx)
            })?
            .into_par_iter()
            .zip_eq(write_refs)
            .for_each(|(new_val, write)| {
                *write = new_val;
            });
        drop(_guard);

        for (i, (step, writes)) in trace.iter().zip(write_vals_f).enumerate() {
            let timestamp = i as u32;

            match step.memory_ops[RS1] {
                MemoryOp::Read(a) => {
                    assert!(a < REGISTER_COUNT);
                    let a = a as usize;
                    let v = v_final[a];

                    v_read_rs1.push(v);
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
                    let v = v_final[a];

                    v_read_rs2.push(v);
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
                    let v_old = v_final[a];
                    let v_new = match v_new {
                        Either::Public(_) => Rep3PrimeFieldShare::<F>::zero_share(), // zero pad
                        Either::Shared(_) => writes.0,
                    };

                    v_read_rd.push(v_old);
                    t_read_rd.push(t_final[a]);
                    v_final[a] = v_new;
                    v_write_rd.push(v_new);
                    t_final[a] = timestamp;
                }
            };

            match step.memory_ops[RAM] {
                MemoryOp::Read(a) => {
                    debug_assert!(a % 4 == 0);
                    let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
                    let v = v_final[remapped_a];

                    a_ram.push(remapped_a as u32);
                    v_read_ram.push(v);
                    t_read_ram.push(t_final[remapped_a]);
                    v_write_ram.push(v);
                    t_final[remapped_a] = timestamp;
                }
                MemoryOp::Write(a, v_new) => {
                    debug_assert!(a % 4 == 0);
                    let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
                    let v_old = v_final[remapped_a];
                    let v_new = match v_new {
                        Either::Public(_) => Rep3PrimeFieldShare::<F>::zero_share(), // zero pad
                        Either::Shared(_) => writes.1,
                    };

                    a_ram.push(remapped_a as u32);
                    v_read_ram.push(v_old);
                    t_read_ram.push(t_final[remapped_a]);
                    v_final[remapped_a] = v_new;
                    v_write_ram.push(v_new);
                    t_final[remapped_a] = timestamp;
                }
            }
        }

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

    #[cfg(feature = "debug")]
    fn combine_polynomials(
        _: &ReadWriteMemoryPreprocessing,
        polynomials_shares: Vec<Self>,
    ) -> Self::PublicPolynomials {
        let [share1, share2, share3] = polynomials_shares
            .try_into()
            .map_err(|_| "Expected 3 shares".to_string())
            .unwrap();

        let v_final = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.v_final,
            share2.v_final,
            share3.v_final,
        ]);

        let v_read_rd = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.v_read_rd,
            share2.v_read_rd,
            share3.v_read_rd,
        ]);

        let v_read_rs1 = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.v_read_rs1,
            share2.v_read_rs1,
            share3.v_read_rs1,
        ]);

        let v_read_rs2 = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.v_read_rs2,
            share2.v_read_rs2,
            share3.v_read_rs2,
        ]);

        let v_read_ram = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.v_read_ram,
            share2.v_read_ram,
            share3.v_read_ram,
        ]);

        let v_write_rd = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.v_write_rd,
            share2.v_write_rd,
            share3.v_write_rd,
        ]);

        let v_write_ram = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.v_write_ram,
            share2.v_write_ram,
            share3.v_write_ram,
        ]);

        let v_init = Rep3MultilinearPolynomial::combine_shares(vec![
            share1.v_init.unwrap(),
            share2.v_init.unwrap(),
            share3.v_init.unwrap(),
        ]);

        Self::PublicPolynomials {
            a_ram: share1.a_ram.try_into().unwrap(),
            v_read_rd,
            v_read_rs1,
            v_read_rs2,
            v_read_ram,
            v_write_rd,
            v_write_ram,
            v_final,
            t_read_rd: share1.t_read_rd.try_into().unwrap(),
            t_read_rs1: share1.t_read_rs1.try_into().unwrap(),
            t_read_rs2: share1.t_read_rs2.try_into().unwrap(),
            t_read_ram: share1.t_read_ram.try_into().unwrap(),
            t_final: share1.t_final.try_into().unwrap(),
            v_init: Some(v_init),
            a_init_final: None,
            identity: None,
        }
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
    #[tracing::instrument(skip_all, name = "ProgramIO::generate_witness_rep3")]
    pub fn generate_witness_rep3<Network, Instruction>(
        program_io: Rep3ProgramIOInput,
        trace: &[JoltTraceStep<Instruction>],
        io_ctx: &mut IoContextPool<Network>,
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
        let input_words_len = inputs.len().div_ceil(4);
        let output_words_len = outputs.len().div_ceil(4);

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
