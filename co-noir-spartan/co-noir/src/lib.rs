use std::collections::BTreeMap;

use co_acvm::{solver::Rep3CoSolver, Rep3AcvmType};
use mpc_core::protocols::rep3::network::Rep3MpcNet;
use noirc_artifacts::program::ProgramArtifact;

// pub struct CoNoirRep3 {
//     net: Rep3MpcNet,
// }

// impl CoNoirRep3 {
//     /// Create a new initial state
//     pub fn new(net: Rep3MpcNet) -> Self {
//         Self { net }
//     }

//     /// Perform the witness generation advance to the next state
//     pub fn generate_witness(
//         self,
//         compiled_program: ProgramArtifact,
//         input_share: BTreeMap<String, Rep3AcvmType<ark_bn254::Fr>>,
//     ) -> anyhow::Result<co_spartan::witness::Rep3WitnessShare<ark_bn254::Bn254>> {
//         let input_share = Rep3CoSolver::<ark_bn254::Fr, Rep3MpcNet>::witness_map_from_string_map(
//             input_share,
//             &compiled_program.abi,
//         )
//         .map_err(|e| anyhow::anyhow!(e))?;

//         // init MPC protocol
//         let rep3_vm = Rep3CoSolver::from_network_with_witness(self.net, compiled_program, input_share)
//             .map_err(|e| anyhow::anyhow!(e))?;

//         // execute witness generation in MPC
//         let (result_witness_share, _, _) = rep3_vm
//             .solve_with_output()
//             .map_err(|e| anyhow::anyhow!(e))?;

//         todo!()
//     }
// }
