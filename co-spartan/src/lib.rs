pub mod coordinator;
pub mod mpc;
pub mod network;
pub mod setup;
pub mod sumcheck;
pub mod utils;
pub mod worker;

pub use coordinator::SpartanProverCoordinator;
pub use setup::split_ipk;
pub use worker::{Rep3ProverKey, SpartanProverWorker};

use noir_r1cs::FieldElement;
use spartan::{
    math::{SparseMatEntry, SparseMatPolynomial},
    R1CSInstance,
};
