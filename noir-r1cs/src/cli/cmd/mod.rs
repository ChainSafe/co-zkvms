mod circuit_stats;
// mod generate_gnark_inputs;
mod prepare;
mod solve_witness;
// mod verify;

use anyhow::Result;
use argh::FromArgs;

pub trait Command {
    fn run(&self) -> Result<()>;
}

/// Prove & verify a compiled Noir program using R1CS.
#[derive(FromArgs, PartialEq, Debug)]
pub struct Args {
    #[argh(subcommand)]
    subcommand: Commands,
}

#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand)]
enum Commands {
    Prepare(prepare::Args),
    SolveWitness(solve_witness::Args),
    CircuitStats(circuit_stats::Args),
}

impl Command for Args {
    fn run(&self) -> Result<()> {
        self.subcommand.run()
    }
}

impl Command for Commands {
    fn run(&self) -> Result<()> {
        match self {
            Commands::Prepare(args) => args.run(),
            Commands::SolveWitness(args) => args.run(),
            Commands::CircuitStats(args) => args.run(),
        }
    }
}
