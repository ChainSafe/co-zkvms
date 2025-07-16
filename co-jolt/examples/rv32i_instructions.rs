mod three_party;

use clap::Parser;
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use co_jolt::host;
use mpc_net::config::{NetworkConfig, NetworkConfigFile};

use crate::three_party::{init_tracing, run_coordinator, run_party, Args};

type Instructions = co_jolt::jolt::vm::rv32i_vm::RV32I<F>;
type Subtables = co_jolt::jolt::vm::rv32i_vm::RV32ISubtables<F>;

const C: usize = co_jolt::jolt::vm::rv32i_vm::C;
const M: usize = co_jolt::jolt::vm::rv32i_vm::M;
type F = ark_bn254::Fr;

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
    program.set_input(&9u32);
    let (bytecode, memory_init) = program.decode();
    let (io_device, bytecode_trace, instruction_trace, memory_trace, circuit_flags) =
        program.trace::<F>();

    if config.is_coordinator {
        run_coordinator::<C, M, Instructions, Subtables>(args, config, vec![None], 1, 1)?;
    } else {
        run_party::<C, M, Instructions, Subtables>(args, config, vec![None], 1, 1)?;
    }

    Ok(())
}
