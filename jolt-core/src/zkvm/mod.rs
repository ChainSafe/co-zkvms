use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::{
    curve::Bn254Curve,
    field::JoltField,
    poly::commitment::{
        commitment_scheme::{CommitmentScheme, ZkEvalCommitment},
        pedersen::PedersenGenerators,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::{
        bytecode::BytecodePreprocessing, dag::jolt_dag::JoltDAG, dag::proof_serialization::JoltProof,
        ram::RAMPreprocessing, witness::DTH_ROOT_OF_K,
    },
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use tracer::{instruction::Instruction, JoltDevice};

pub mod bytecode;
pub mod dag;
pub mod instruction;
pub mod instruction_lookups;
pub mod lookup_table;
pub mod r1cs;
pub mod ram;
pub mod registers;
pub mod spartan;
pub mod witness;

// Scoped CPU profiler for performance analysis. Feature-gated by "pprof".
// Usage: let _guard = pprof_scope!("label");
//
// Writes pprof/label.pb on scope exit
// View with: go tool pprof -http=:8080 pprof/label.pb

// Public type for the profiling guard
#[cfg(feature = "pprof")]
pub struct PprofGuard {
    guard: pprof::ProfilerGuard<'static>,
    label: &'static str,
}

#[cfg(not(feature = "pprof"))]
pub struct PprofGuard;

#[cfg(feature = "pprof")]
impl Drop for PprofGuard {
    fn drop(&mut self) {
        if let Ok(report) = self.guard.report().build() {
            let prefix = std::env::var("PPROF_PREFIX").unwrap_or_else(|_| String::from("benchmark-runs/pprof/"));
            let filename = format!("{}{}.pb", prefix, self.label);
            // Extract directory from prefix for creation
            if let Some(dir) = std::path::Path::new(&filename).parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(mut f) = std::fs::File::create(&filename) {
                use pprof::protos::Message;
                if let Ok(p) = report.pprof() {
                    let mut buf = Vec::new();
                    if p.encode(&mut buf).is_ok() {
                        let _ = std::io::Write::write_all(&mut f, &buf);
                        tracing::info!("Wrote pprof profile to {}", filename);
                    }
                }
            }
        }
    }
}

#[macro_export]
macro_rules! pprof_scope {
    ($label:expr) => {{
        #[cfg(feature = "pprof")]
        {
            Some($crate::zkvm::PprofGuard {
                guard: pprof::ProfilerGuardBuilder::default()
                    .frequency(std::env::var("PPROF_FREQ").unwrap_or("100".to_string()).parse::<i32>().unwrap())
                    .blocklist(&["libc", "libgcc", "pthread", "vdso"])
                    .build()
                    .expect("Failed to initialize profiler"),
                label: $label,
            })
        }
        #[cfg(not(feature = "pprof"))]
        None::<$crate::zkvm::PprofGuard>
    }};
    () => {
        pprof_scope!("default");
    };
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

#[cfg(feature = "zk")]
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BlindfoldSetup(pub PedersenGenerators<Bn254Curve>);

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
    #[cfg(feature = "zk")]
    pub blindfold_setup: Option<BlindfoldSetup>,
}

impl<F, PCS> Serializable for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }

    #[cfg(feature = "zk")]
    pub fn pedersen_generators(&self, count: usize) -> PedersenGenerators<Bn254Curve> {
        let gens = &self.blindfold_setup.as_ref().expect("BlindfoldSetup required for ZK mode").0;
        assert!(
            count <= gens.message_generators.len(),
            "requested {count} Pedersen generators but only {} are available",
            gens.message_generators.len()
        );
        PedersenGenerators::new(gens.message_generators[..count].to_vec(), gens.blinding_generator)
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::ProverSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> Serializable for JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    #[cfg(feature = "zk")]
    pub fn blindfold_setup(&self) -> BlindfoldSetup
    where
        PCS: ZkEvalCommitment<Bn254Curve>,
    {
        let (message_generators, blinding_generator) = PCS::zk_generators(&self.generators, usize::MAX)
            .expect("PCS does not support BlindFold Pedersen generators");
        BlindfoldSetup(PedersenGenerators::new(message_generators, blinding_generator))
    }

    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

impl<F, PCS> From<&JoltProverPreprocessing<F, PCS>> for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<Bn254Curve>,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&preprocessing.generators);
        JoltVerifierPreprocessing {
            generators,
            shared: preprocessing.shared.clone(),
            #[cfg(feature = "zk")]
            blindfold_setup: Some(preprocessing.blindfold_setup()),
        }
    }
}

pub trait Jolt<F: JoltField, PCS, FS: Transcript>
where
    PCS: CommitmentScheme<Field = F>,
{
    fn shared_preprocess(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
    ) -> JoltSharedPreprocessing {
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);
        let ram_preprocessing = RAMPreprocessing::preprocess(memory_init);

        JoltSharedPreprocessing { memory_layout, bytecode: bytecode_preprocessing, ram: ram_preprocessing }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prover_preprocess")]
    fn prover_preprocess(
        bytecode: Vec<Instruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS> {
        let shared = Self::shared_preprocess(bytecode, memory_layout, memory_init);

        let max_T: usize = max_trace_length.next_power_of_two();

        let generators = PCS::setup_prover(DTH_ROOT_OF_K.log_2() + max_T.log_2());

        JoltProverPreprocessing { generators, shared }
    }

    #[tracing::instrument(skip_all, level = "trace", name = "Jolt::verify")]
    fn verify(
        preprocessing: &JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<F, Bn254Curve, PCS, FS>,
        mut program_io: JoltDevice,
        trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
        _debug_info: Option<()>,
    ) -> Result<(), ProofVerifyError>
    where
        PCS: ZkEvalCommitment<Bn254Curve>,
    {
        let _pprof_verify = pprof_scope!("verify");

        #[cfg(test)]
        let T = proof.trace_length.next_power_of_two();
        #[cfg(test)]
        let _guard = DoryGlobals::initialize(DTH_ROOT_OF_K, T);

        if program_io.memory_layout != preprocessing.shared.memory_layout {
            return Err(ProofVerifyError::MemoryLayoutMismatch);
        }
        if program_io.inputs.len() > preprocessing.shared.memory_layout.max_input_size as usize {
            return Err(ProofVerifyError::InputTooLarge);
        }
        if program_io.outputs.len() > preprocessing.shared.memory_layout.max_output_size as usize {
            return Err(ProofVerifyError::OutputTooLarge);
        }

        program_io.outputs.truncate(program_io.outputs.iter().rposition(|&b| b != 0).map_or(0, |pos| pos + 1));

        let mut state_manager = proof.to_verifier_state_manager(preprocessing, program_io);
        state_manager.trusted_advice_commitment = trusted_advice_commitment;

        JoltDAG::verify(state_manager).map_err(|err| ProofVerifyError::DoryError(err.to_string()))?;

        Ok(())
    }
}

pub struct JoltRV32IM;
impl Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltRV32IM {}

pub struct JoltRV64IMAC;
impl Jolt<Fr, DoryCommitmentScheme, Blake2bTranscript> for JoltRV64IMAC {}
#[cfg(not(feature = "rv64"))]
pub type JoltRVArch = JoltRV32IM;
#[cfg(feature = "rv64")]
pub type JoltRVArch = JoltRV64IMAC;
pub type RV64IMACJoltProof = JoltProof<Fr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;

use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Blake2bTranscript;
use eyre::Result;
use std::io::Cursor;
use std::path::PathBuf;

pub trait Serializable: CanonicalSerialize + CanonicalDeserialize + Sized {
    /// Gets the byte size of the serialized data
    fn size(&self) -> Result<usize> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer.len())
    }

    /// Saves the data to a file
    fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<()> {
        let file = File::create(path.into())?;
        self.serialize_compressed(file)?;
        Ok(())
    }

    /// Reads data from a file
    fn from_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let file = File::open(path.into())?;
        Ok(Self::deserialize_compressed(file)?)
    }

    /// Serializes the data to a byte vector
    fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer)
    }

    /// Deserializes data from a byte vector
    fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        Ok(Self::deserialize_compressed(cursor)?)
    }

    /// Deserializes data from bytes but skips checks for performance
    fn deserialize_from_bytes_unchecked(bytes: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        Ok(Self::deserialize_with_mode(cursor, ark_serialize::Compress::Yes, ark_serialize::Validate::No)?)
    }
}

impl Serializable for RV64IMACJoltProof {}
impl Serializable for JoltDevice {}
