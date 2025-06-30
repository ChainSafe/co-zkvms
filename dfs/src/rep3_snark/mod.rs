use ark_bn254::Bn254;
use mpc_core::protocols::rep3::network::{Rep3MpcNet, Rep3Network};
use serde::{Deserialize, Serialize};
use mpc_types::{protocols::rep3::Rep3ShareVecType, serde_compat::{ark_de, ark_se}};
use ark_ff::PrimeField;

use crate::mpc_snark::worker::DistributedRSSProverKey;


/// This type represents the serialized version of a Rep3 witness. Its share can be either additive or replicated, and in both cases also compressed.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedRep3SharedWitness<F: PrimeField> {
    /// The public inputs (which are the outputs of the circom circuit).
    /// This also includes the constant 1 at position 0.
    #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
    pub public_inputs: Vec<F>,
    /// The secret-shared witness elements.
    pub witness: Rep3ShareVecType<F>,
}

/// State with a rep3 shared witness
pub struct Rep3ProvingKeyState {
    net: Rep3MpcNet,
    /// The shared witness
    // pub witness: Vec<Rep3AcvmType<ark_bn254::Fr>>,
    proving_key: DistributedRSSProverKey<Bn254>,

}

impl Rep3ProvingKeyState {
    /// Create a new [Rep3SharedWitnessState ]
    pub fn new(net: Rep3MpcNet, proving_key: DistributedRSSProverKey<Bn254>) -> Self {
        Self { net, proving_key }
    }

    // /// Generate the proving key and advance to the next state
    // pub fn generate_proving_key(
    //     self,
    //     constraint_system: &AcirFormat<ark_bn254::Fr>,
    //     recursive: bool,
    // ) -> eyre::Result<Rep3ProvingKeyState> {
    //     let (proving_key, net) =
    //         generate_proving_key_rep3(self.net, constraint_system, self.witness, recursive)?;
    //     Ok(Rep3ProvingKeyState { net, proving_key })
    // }

    // pub fn prove<H: TranscriptHasher<TranscriptFieldType>>(
    //     self,
    //     prover_crs: &ProverCrs<Bn254>,
    // ) -> eyre::Result<HonkProof<ark_bn254::Fr>> {
    //     let (proof, _public_inputs, _net) =
    //         Rep3CoUltraHonk::<_, _, H>::prove(self.net, self.proving_key, prover_crs, has_zk)?;
    //     Ok(proof)
    // }
}
