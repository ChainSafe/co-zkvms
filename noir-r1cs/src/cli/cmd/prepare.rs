use std::path::PathBuf;

use anyhow::{Context, Result};
use argh::FromArgs;
use noir_r1cs::{write, NoirProofScheme};
use tracing::instrument;

use super::Command;

/// Prepare a Noir program for proving
#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "prepare")]
pub struct Args {
    /// path to the compiled Noir program
    #[argh(positional)]
    program_path: PathBuf,

    /// output path for the prepared proof scheme
    #[argh(
        option,
        long = "out",
        short = 'o',
        default = "PathBuf::from(\"noir_proof_scheme.json\")"
    )]
    output_path: PathBuf,
}

impl Command for Args {
    #[instrument(skip_all)]
    fn run(&self) -> Result<()> {
        let scheme = NoirProofScheme::from_file(&self.program_path)
            .context("while compiling Noir program")?;
        write(&scheme, &self.output_path).context("while writing Noir proof scheme")?;
        Ok(())
    }
}
