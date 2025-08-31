#![allow(clippy::type_complexity)]

use core::str::FromStr;
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    iter,
    marker::PhantomData,
    path::PathBuf,
    process::Command,
};

use itertools::{izip, Itertools};
use jolt_tracer::{RVTraceRow, RV32IM};
use mpc_core::protocols::rep3::{self, Rep3PrimeFieldShare};
use rand::{RngCore, SeedableRng};
use rand_chacha::{ChaCha12Core, ChaCha12Rng};
use rayon::prelude::*;

use jolt_common::{
    self as common,
    constants::{
        DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE, DEFAULT_MEMORY_SIZE, DEFAULT_STACK_SIZE,
    },
    rv_trace::JoltDevice,
};
pub use jolt_tracer::{self as tracer, ELFInstruction};

use crate::{field::JoltField, jolt::vm::read_write_memory::witness::Rep3ProgramIOInput};
use crate::{
    jolt::{
        instruction::{Rep3JoltInstruction, Rep3Operand},
        vm::{
            bytecode::{BytecodeRow, BytecodeRowExt},
            instruction_lookups,
            rv32i_vm::RV32I,
            JoltTraceStep,
        },
    },
    utils::instruction_utils::transpose,
};

// use self::analyze::ProgramSummary;
use jolt_core::{
    host::toolchain::install_toolchain,
    jolt::instruction::{
        div::DIVInstruction, divu::DIVUInstruction, lb::LBInstruction, lbu::LBUInstruction,
        lh::LHInstruction, lhu::LHUInstruction, mulh::MULHInstruction, mulhsu::MULHSUInstruction,
        rem::REMInstruction, remu::REMUInstruction, sb::SBInstruction, sh::SHInstruction,
        VirtualInstructionSequence,
    },
};

// pub mod analyze;

pub const DEFAULT_TARGET_DIR: &str = "/tmp/jolt-guest-targets";

#[derive(Clone)]
pub struct Program {
    guest: String,
    func: Option<String>,
    memory_size: u64,
    stack_size: u64,
    max_input_size: u64,
    max_output_size: u64,
    std: bool,
    pub elf: Option<PathBuf>,
}

impl Program {
    pub fn new(guest: &str) -> Self {
        Self {
            guest: guest.to_string(),
            func: None,
            memory_size: DEFAULT_MEMORY_SIZE,
            stack_size: DEFAULT_STACK_SIZE,
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
            std: false,
            elf: None,
        }
    }

    pub fn set_std(&mut self, std: bool) {
        self.std = std;
    }

    pub fn set_func(&mut self, func: &str) {
        self.func = Some(func.to_string())
    }

    pub fn set_memory_size(&mut self, len: u64) {
        self.memory_size = len;
    }

    pub fn set_stack_size(&mut self, len: u64) {
        self.stack_size = len;
    }

    pub fn set_max_input_size(&mut self, size: u64) {
        self.max_input_size = size;
    }

    pub fn set_max_output_size(&mut self, size: u64) {
        self.max_output_size = size;
    }

    #[tracing::instrument(skip_all, name = "Program::build")]
    pub fn build(&mut self, target_dir: &str) {
        if self.elf.is_none() {
            install_toolchain().unwrap();
            // install_no_std_toolchain().unwrap();

            self.save_linker();

            let rust_flags = [
                "-C",
                &format!("link-arg=-T{}", self.linker_path()),
                "-C",
                "passes=lower-atomic",
                "-C",
                "panic=abort",
                "-C",
                "strip=symbols",
                "-C",
                "opt-level=z",
            ];

            let toolchain = if self.std {
                "riscv32im-jolt-zkvm-elf"
            } else {
                "riscv32im-unknown-none-elf"
            };

            let mut envs = vec![("CARGO_ENCODED_RUSTFLAGS", rust_flags.join("\x1f"))];

            if self.std {
                envs.push(("RUSTUP_TOOLCHAIN", toolchain.to_string()));
            }

            if let Some(func) = &self.func {
                envs.push(("JOLT_FUNC_NAME", func.to_string()));
            }

            let target = format!(
                "{}/{}-{}",
                target_dir,
                self.guest,
                self.func.as_ref().unwrap_or(&"".to_string())
            );

            let output = Command::new("cargo")
                .envs(envs)
                .args([
                    "build",
                    "--release",
                    "--features",
                    "guest",
                    "-p",
                    &self.guest,
                    "--target-dir",
                    &target,
                    "--target",
                    toolchain,
                ])
                .output()
                .expect("failed to build guest");

            if !output.status.success() {
                io::stderr().write_all(&output.stderr).unwrap();
                panic!("failed to compile guest");
            }

            let elf = format!("{}/{}/release/{}", target, toolchain, self.guest);
            self.elf = Some(PathBuf::from_str(&elf).unwrap());
        }
    }

    pub fn decode(&self) -> (Vec<ELFInstruction>, Vec<(u64, u8)>) {
        let elf = self.elf.as_ref().unwrap();
        let mut elf_file =
            File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));
        let mut elf_contents = Vec::new();
        elf_file.read_to_end(&mut elf_contents).unwrap();
        tracer::decode(&elf_contents)
    }

    // TODO(moodlezoup): Make this generic over InstructionSet
    #[tracing::instrument(skip_all, name = "Program::trace")]
    pub fn trace<F: JoltField>(
        &mut self,
        inputs: &[u8],
    ) -> (JoltDevice, Vec<JoltTraceStep<F, RV32I<F>>>) {
        self.build(DEFAULT_TARGET_DIR);

        let elf = self.elf.as_ref().unwrap();
        let mut elf_file =
            File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));

        let mut elf_contents = Vec::new();
        elf_file.read_to_end(&mut elf_contents).unwrap();
        let memory_config = common::rv_trace::MemoryConfig {
            memory_size: self.memory_size,
            stack_size: self.stack_size,
            max_input_size: self.max_input_size,
            max_output_size: self.max_output_size,
        };
        let (raw_trace, io_device) = tracer::trace(elf_contents, inputs, &memory_config);

        // Self::print_used_instructions(&raw_trace);

        let trace = raw_trace
            .into_par_iter()
            .flat_map(|row| match row.instruction.opcode {
                // RV32IM::MULH => MULHInstruction::<32>::virtual_trace(row),
                // RV32IM::MULHSU => MULHSUInstruction::<32>::virtual_trace(row),
                // RV32IM::DIV => DIVInstruction::<32>::virtual_trace(row),
                // RV32IM::DIVU => DIVUInstruction::<32>::virtual_trace(row),
                // RV32IM::REM => REMInstruction::<32>::virtual_trace(row),
                // RV32IM::REMU => REMUInstruction::<32>::virtual_trace(row),
                // RV32IM::SH => SHInstruction::<32>::virtual_trace(row),
                // RV32IM::SB => SBInstruction::<32>::virtual_trace(row),
                // RV32IM::LBU => LBUInstruction::<32>::virtual_trace(row),
                // RV32IM::LHU => LHUInstruction::<32>::virtual_trace(row),
                // RV32IM::LB => LBInstruction::<32>::virtual_trace(row),
                // RV32IM::LH => LHInstruction::<32>::virtual_trace(row),
                // _ => vec![row],

                // ["ADD", "AND", "BEQ", "BGEU", "BNE", "MUL", "OR", "SLL", "SLT", "SLTU", "SRA", "SRL", "SUB", "VIRTUAL_ADVICE", "XOR"]
                RV32IM::OR
                | RV32IM::AND
                | RV32IM::XOR
                | RV32IM::MUL
                | RV32IM::BEQ
                | RV32IM::BGEU
                | RV32IM::BNE
                | RV32IM::SUB => vec![row],
                _ => vec![],
            })
            .map(|row| {
                let instruction_lookup = RV32I::try_from(&row)
                    .map_err(|_| match row.instruction.opcode {
                        RV32IM::MULH
                        | RV32IM::MULHSU
                        | RV32IM::DIV
                        | RV32IM::DIVU
                        | RV32IM::REM
                        | RV32IM::REMU
                        | RV32IM::SH
                        | RV32IM::SB
                        | RV32IM::LBU
                        | RV32IM::LHU
                        | RV32IM::LB
                        | RV32IM::LH
                        | RV32IM::LW
                        | RV32IM::SW => {}
                        _ => {
                            tracing::warn!(
                                "Failed to map opcode {:?} to RV32I",
                                row.instruction.opcode
                            );
                        }
                    })
                    .ok();

                JoltTraceStep {
                    instruction_lookup,
                    bytecode_row: BytecodeRow::from_instruction_ext::<F, RV32I<F>>(
                        &row.instruction,
                    ),
                    memory_ops: (&row).into(),
                    circuit_flags: row.instruction.to_circuit_flags(),
                    _field: PhantomData,
                }
            })
            .collect();

        (io_device, trace)
    }

    pub fn generate_trace_shares<F: JoltField, R: RngCore>(
        &mut self,
        inputs: &[u8],
        rng: &mut R,
    ) -> Vec<(Rep3ProgramIOInput<F>, Vec<JoltTraceStep<F, RV32I<F>>>)> {
        let (bytecode, memory_init) = self.decode();
        let (program_io, trace) = self.trace::<F>(inputs);

        let program_io = Rep3ProgramIOInput::<F> {
            input: vec![],
            output: vec![],
            panic: Rep3PrimeFieldShare::zero_share(),
            memory_layout: program_io.memory_layout,
        };

        let program_io_shares = vec![program_io; 3];

        let root = rng.next_u64();
        let trace_shares = trace
            .into_par_iter()
            .map_init(
                move || {
                    let tid = rayon::current_thread_index().unwrap_or(0) as u64;
                    ChaCha12Rng::seed_from_u64(root ^ tid)
                },
                |rng, row| {
                    let instruction_shares = if let Some(r) = row.instruction_lookup {
                        let op1_shares =
                            rep3::binary::generate_shares_rep3(r.lhs().as_public().into(), rng);
                        let op2_shares = if let Some(op2) = r.rhs() {
                            match r {
                                RV32I::SLL(..)
                                | RV32I::SRA(..)
                                | RV32I::SRL(..)
                                | RV32I::VIRTUAL_POW2(..)
                                | RV32I::VIRTUAL_SRA_PADDING(..) => {
                                    vec![Some(op2.clone()); 3]
                                }
                                _ => {
                                    rep3::binary::generate_shares_rep3(op2.as_public().into(), rng)
                                        .into_iter()
                                        .map(|share| Some(Rep3Operand::from(share)))
                                        .collect()
                                }
                            }
                        } else {
                            vec![None; 3]
                        };
                        let mut instruction_shares: Vec<Option<RV32I<F>>> =
                            vec![Some(r.clone()); 3];
                        izip!(instruction_shares.iter_mut(), op1_shares, op2_shares).for_each(
                            |(r, op1_share, op2_share)| {
                                let (op1, op2) = r.as_mut().unwrap().operands_mut();
                                *op1 = op1_share.into();
                                if let Some(op2) = op2 {
                                    *op2 = op2_share.unwrap();
                                }
                            },
                        );
                        instruction_shares
                    } else {
                        vec![None; 3]
                    };

                    instruction_shares
                        .into_iter()
                        .map(|instruction_lookup| JoltTraceStep {
                            instruction_lookup,
                            bytecode_row: row.bytecode_row.clone(),
                            memory_ops: row.memory_ops.clone(),
                            circuit_flags: row.circuit_flags.clone(),
                            _field: PhantomData,
                        })
                        .collect()
                },
            )
            .collect::<Vec<_>>();

        let trace_shares = transpose(trace_shares);

        izip!(program_io_shares, trace_shares).collect()
    }

    // pub fn trace_analyze<F: JoltField>(mut self, inputs: &[u8]) -> ProgramSummary {
    //     self.build(DEFAULT_TARGET_DIR);
    //     let elf = self.elf.as_ref().unwrap();
    //     let mut elf_file =
    //         File::open(elf).unwrap_or_else(|_| panic!("could not open elf file: {elf:?}"));
    //     let mut elf_contents = Vec::new();
    //     elf_file.read_to_end(&mut elf_contents).unwrap();
    //     let memory_config = common::rv_trace::MemoryConfig {
    //         memory_size: self.memory_size,
    //         stack_size: self.stack_size,
    //         max_input_size: self.max_input_size,
    //         max_output_size: self.max_output_size,
    //     };
    //     let (raw_trace, _) = tracer::trace(elf_contents, inputs, &memory_config);

    //     let (bytecode, memory_init) = self.decode();
    //     let (io_device, processed_trace) = self.trace(inputs);

    //     ProgramSummary {
    //         raw_trace,
    //         bytecode,
    //         memory_init,
    //         io_device,
    //         processed_trace,
    //     }
    // }

    fn print_used_instructions(instruction_trace: &[RVTraceRow]) {
        let opcodes_used = instruction_trace
            .par_iter()
            .map(|step| step.instruction.opcode.as_ref())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .unique()
            .sorted()
            .collect::<Vec<_>>();
        println!("opcodes_used: {:?}", opcodes_used);
    }

    fn save_linker(&self) {
        let linker_path = PathBuf::from_str(&self.linker_path()).unwrap();
        if let Some(parent) = linker_path.parent() {
            fs::create_dir_all(parent).expect("could not create linker file");
        }

        let linker_script = LINKER_SCRIPT_TEMPLATE
            .replace("{MEMORY_SIZE}", &self.memory_size.to_string())
            .replace("{STACK_SIZE}", &self.stack_size.to_string());

        let mut file = File::create(linker_path).expect("could not create linker file");
        file.write_all(linker_script.as_bytes())
            .expect("could not save linker");
    }

    fn linker_path(&self) -> String {
        format!("/tmp/jolt-guest-linkers/{}.ld", self.guest)
    }
}

const LINKER_SCRIPT_TEMPLATE: &str = r#"
MEMORY {
  program (rwx) : ORIGIN = 0x80000000, LENGTH = {MEMORY_SIZE}
}

SECTIONS {
  .text.boot : {
    *(.text.boot)
  } > program

  .text : {
    *(.text)
  } > program

  .data : {
    *(.data)
  } > program

  .bss : {
    *(.bss)
  } > program

  . = ALIGN(8);
  . = . + {STACK_SIZE};
  _STACK_PTR = .;
  . = ALIGN(8);
  _HEAP_PTR = .;
}
"#;
