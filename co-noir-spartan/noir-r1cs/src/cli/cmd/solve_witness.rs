use std::{
    fs::File,
    io::{BufWriter, Read},
    path::PathBuf,
};

use anyhow::{Context, Result};
use argh::FromArgs;
use ark_serialize::CanonicalSerialize;
use noir_r1cs::{self, read, NoirProofScheme};
use tracing::{info, instrument};

use super::Command;

/// Prove a prepared Noir program
#[derive(FromArgs, PartialEq, Debug)]
#[argh(subcommand, name = "solve-witness")]
pub struct Args {
    /// path to the compiled Noir program
    #[argh(positional)]
    scheme_path: PathBuf,

    /// path to the input values
    #[argh(positional)]
    input_path: PathBuf,

    /// path to store witness file
    #[argh(
        option,
        long = "out",
        short = 'o',
        default = "PathBuf::from(\"./witness.np\")"
    )]
    witness_path: PathBuf,
}

impl Command for Args {
    #[instrument(skip_all)]
    fn run(&self) -> Result<()> {
        // Read the scheme
        let scheme: NoirProofScheme =
            read(&self.scheme_path).context("while reading Noir proof scheme")?;
        let (constraints, witnesses) = scheme.size();
        info!(constraints, witnesses, "Read Noir proof scheme");

        // Read the input toml
        let mut file = File::open(&self.input_path).context("while opening input file")?;
        let mut input_toml =
            String::with_capacity(file.metadata().map(|m| m.len() as usize).unwrap_or(0));
        file.read_to_string(&mut input_toml)
            .context("while reading input file")?;

        // Generate the proof
        let witness = scheme
            .solve_witness(&self.input_path)
            .context("While solving witness")?;

        let mut buf = BufWriter::new(
            File::create(&self.witness_path).context("while creating witness file")?,
        );
        witness
            .serialize_compressed(&mut buf)
            .context("while serializing witness")?;

        Ok(())
    }
}
