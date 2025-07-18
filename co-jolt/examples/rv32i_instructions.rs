mod three_party;

use clap::Parser;
use jolt_core::field::JoltField;
use co_jolt::{host, jolt::{instruction::JoltInstructionSet, vm::rv32i_vm::RV32I}};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use itertools::Itertools;
use mpc_net::config::{NetworkConfig, NetworkConfigFile};

use crate::three_party::{init_tracing, run_coordinator, run_party, Args};

type Instructions = co_jolt::jolt::vm::rv32i_vm::RV32I<F>;
type Subtables = co_jolt::jolt::vm::rv32i_vm::RV32ISubtables<F>;

const C: usize = co_jolt::jolt::vm::rv32i_vm::C;
const M: usize = co_jolt::jolt::vm::rv32i_vm::M;
type F = ark_bn254::Fr;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| eyre!("Could not install default rustls crypto provider"))?;

    let config: NetworkConfigFile =
        toml::from_str(&std::fs::read_to_string(&args.config_file).context("opening config file")?)
            .context("parsing config file")?;
    let config = NetworkConfig::try_from(config).context("converting network config")?;

    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&9u32).unwrap();
    let (_, instruction_trace) = program.trace::<F>(&inputs);

    if config.is_coordinator {
        print_used_instructions(&instruction_trace);
        run_coordinator::<C, M, Instructions, Subtables>(args, config, instruction_trace, 1, 1)?;
    } else {
        run_party::<C, M, Instructions, Subtables>(args, config, instruction_trace, 1, 1)?;
    }

    Ok(())
}

fn print_used_instructions<F: JoltField>(instruction_trace: &[Option<RV32I<F>>]) {
    let opcodes_used = instruction_trace
        .par_iter()
        .filter_map(|op| match op {
            Some(op) => Some(op.name()),
            None => None,
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .unique()
        .sorted()
        .collect::<Vec<_>>();
    tracing::info!("opcodes_used: {:?}", opcodes_used);
}
