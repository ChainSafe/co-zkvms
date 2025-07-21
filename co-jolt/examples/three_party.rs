// use ark_ff::Zero;
// use ark_poly_commit::multilinear_pc::MultilinearPC;
// use ark_std::test_rng;
// use clap::Parser;
// use color_eyre::{
//     eyre::{eyre, Context},
//     Result,
// };
// use itertools::Itertools;
// use jolt_core::{jolt::vm::JoltVerifierPreprocessing, utils::transcript::KeccakTranscript};
// use mpc_core::protocols::rep3::{
//     self,
//     network::{IoContext, Rep3Network},
// };
// use mpc_net::{
//     config::{NetworkConfig, NetworkConfigFile},
//     mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker},
//     rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator},
// };
// use rand::Rng;
// use std::{iter, path::PathBuf};
// use tracing_forest::ForestLayer;
// use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

// use co_jolt::{
//     poly::opening_proof::Rep3ProverOpeningAccumulator,
//     subprotocols::commitment::{PST13Setup, PST13},
// };
// use co_jolt::{
//     jolt::{
//         instruction::{
//             self,
//             range_check::RangeLookup,
//             xor::{self, XORInstruction},
//             JoltInstruction, JoltInstructionSet, Rep3JoltInstructionSet,
//         },
//         subtable::{self, JoltSubtableSet},
//         vm::{
//             instruction_lookups::{
//                 witness::Rep3InstructionLookupPolynomials, worker::Rep3InstructionLookupsProver,
//                 InstructionLookupCommitments, InstructionLookupsPreprocessing,
//                 InstructionLookupsProof,
//             },
//             witness::Rep3JoltPolynomials,
//             JoltCommitments, JoltPolynomials, JoltTraceStep,
//         },
//     },
//     poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
//     utils::transcript::Transcript,
// };

// type Instructions = instruction::TestInstructions<F>;
// type Subtables = subtable::TestInstructionSubtables<F>;

// #[derive(Parser)]
// pub struct Args {
//     /// The config file path
//     #[clap(short, long, value_name = "FILE")]
//     pub config_file: PathBuf,

//     #[clap(short, long, value_name = "NUM_INPUTS", default_value = "8")]
//     pub log_num_inputs: usize,

//     #[arg(short, long, value_name = "SOLVE_WITNESS", env = "SOLVE_WITNESS")]
//     pub solve_witness: bool,

//     #[clap(short, long, value_name = "DEBUG", env = "DEBUG")]
//     pub debug: bool,

//     #[clap(short, long, value_name = "TRACE_PARTIES", env = "TRACE_PARTIES")]
//     pub trace_parties: bool,
// }

// const C: usize = 2; // num chunks
// const M: usize = 1 << 8;
// type F = ark_bn254::Fr;
// type E = ark_bn254::Bn254;

// fn main() -> Result<()> {
//     // init_tracing();
//     let args = Args::parse();
//     let num_inputs = 1 << args.log_num_inputs;
//     rustls::crypto::aws_lc_rs::default_provider()
//         .install_default()
//         .map_err(|_| eyre!("Could not install default rustls crypto provider"))?;

//     let config: NetworkConfigFile =
//         toml::from_str(&std::fs::read_to_string(&args.config_file).context("opening config file")?)
//             .context("parsing config file")?;
//     let config = NetworkConfig::try_from(config).context("converting network config")?;

//     let lookups = {
//         let mut rng = test_rng();
//         iter::repeat_with(|| {
//             let a = rng.gen_range::<u64, _>(0..256);
//             let b = rng.gen_range::<u64, _>(0..256);
//             Some(Instructions::XOR(XORInstruction::<F>::public(a, b)))
//         })
//         .take(num_inputs)
//         .collect_vec()
//     };

//     // let lookups = chain!(
//     //     iter::repeat_with(|| {
//     //         let value = F::from(rng.gen_range(0..256));
//     //         Lookups::Range256(RangeLookup::shared(
//     //             rep3::arithmetic::promote_to_trivial_share(party_id, value),
//     //         ))
//     //     })
//     //     .take(num_inputs / 2)
//     //     .collect_vec(),
//     //     iter::repeat_with(|| {
//     //         let value = F::from(rng.gen_range(0..320));
//     //         Lookups::Range320(RangeLookup::shared(
//     //             rep3::arithmetic::promote_to_trivial_share(party_id, value),
//     //         ))
//     //     })
//     //     .take(num_inputs / 2)
//     //     .collect_vec(),
//     // )
//     // .collect_vec();

//     if config.is_coordinator {
//         run_coordinator::<C, M, Instructions, Subtables>(args, config, lookups, 1, 1)?;
//     } else {
//         run_party::<C, M, Instructions, Subtables>(args, config, lookups, 1, 1)?;
//     }

//     Ok(())
// }

// pub fn run_party<
//     const C: usize,
//     const M: usize,
//     Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
//     Subtables: JoltSubtableSet<F>,
// >(
//     args: Args,
//     config: NetworkConfig,
//     lookups: Vec<Option<Instructions>>, // TODO: make this an Option<Vec<Option<Instructions>>>
//     log_num_workers_per_party: usize,
//     log_num_pub_workers: usize,
// ) -> Result<()> {
//     if args.trace_parties && config.my_id == 0 {
//         init_tracing();
//     }
//     type LassoProof<const C: usize, const M: usize, Instructions, Subtables> =
//         InstructionLookupsProof<C, M, F, PST13<E>, Instructions, Subtables, KeccakTranscript>;

//     let my_id = config.my_id;

//     let span = tracing::info_span!("run_party", id = my_id);
//     let _enter = span.enter();

//     let mut rep3_net = Rep3QuicMpcNetWorker::new_with_coordinator(
//         config,
//         log_num_workers_per_party,
//         log_num_pub_workers,
//     )
//     .unwrap();

//     let setup = {
//         let max_len = std::cmp::max(lookups.len().next_power_of_two(), M);
//         let mut rng = test_rng();
//         PST13::setup(max_len, &mut rng)
//     };

//     let preprocessing = {
//         let instruction_lookups_preprocessing =
//             LassoProof::<C, M, Instructions, Subtables>::preprocess();
//         JoltVerifierPreprocessing::<C, F, PST13<E>, KeccakTranscript> {
//             instruction_lookups: instruction_lookups_preprocessing,
//             generators: setup,
//             bytecode: Default::default(),
//             read_write_memory: Default::default(),
//             memory_layout: Default::default(),
//         }
//     };

//     let mut io_ctx = IoContext::init(rep3_net).unwrap();

//     let mut polynomials = if args.solve_witness {
//         let mut ops = lookups
//             .into_iter()
//             .map(|op| JoltTraceStep::<F, Instructions>::from_instruction_lookup(op))
//             .collect_vec();

//         let polynomials = Rep3InstructionLookupPolynomials::generate_witness_rep3(
//             &preprocessing.instruction_lookups,
//             &mut ops,
//             M,
//             io_ctx.fork().unwrap(),
//         )?;

//         Rep3JoltPolynomials {
//             instruction_lookups: polynomials,
//         }
//     } else {
//         let polynomials = io_ctx.network.receive_request()?;
//         polynomials
//     };

//     polynomials.commit::<C, PST13<E>, KeccakTranscript, _>(&preprocessing, &mut io_ctx)?;

//     if args.debug && args.solve_witness {
//         // rep3_net.send_response(polynomials.clone())?;
//     }

//     if args.debug {
//         return Ok(());
//     }

//     let mut opening_accumulator = Rep3ProverOpeningAccumulator::<F>::new();
//     Rep3InstructionLookupsProver::<C, M, F, Instructions, Subtables, _>::prove::<
//         PST13<E>,
//         KeccakTranscript,
//     >(
//         &preprocessing.instruction_lookups,
//         &mut polynomials,
//         &mut opening_accumulator,
//         &preprocessing.generators,
//         &mut io_ctx,
//     )?;

//     io_ctx.network.log_connection_stats();
//     drop(_enter);
//     Ok(())
// }

// #[tracing::instrument(skip_all)]
// pub fn run_coordinator<
//     const C: usize,
//     const M: usize,
//     Instructions: JoltInstructionSet<F>,
//     Subtables: JoltSubtableSet<F>,
// >(
//     args: Args,
//     config: NetworkConfig,
//     lookups: Vec<Option<Instructions>>,
//     log_num_workers_per_party: usize,
//     log_num_pub_workers: usize,
// ) -> Result<()> {
//     init_tracing();
//     type LassoProof<
//         const C: usize,
//         const M: usize,
//         Instructions: JoltInstructionSet<F>,
//         Subtables: JoltSubtableSet<F>,
//     > = InstructionLookupsProof<C, M, F, PST13<E>, Instructions, Subtables, KeccakTranscript>;

//     let num_inputs = lookups.len();
//     if args.solve_witness {
//         tracing::info!("Witness solving enabled");
//     } else {
//         tracing::warn!("Witness solving disabled");
//     }

//     let mut rep3_net =
//         Rep3QuicNetCoordinator::new(config, log_num_workers_per_party, log_num_pub_workers)
//             .unwrap();

//     let setup = {
//         let max_len = std::cmp::max(num_inputs.next_power_of_two(), M);
//         let mut rng = test_rng();
//         PST13::setup(max_len, &mut rng)
//         // PST13Setup::default()
//     };
//     let preprocessing = {
//         let instruction_lookups_preprocessing =
//             LassoProof::<C, M, Instructions, Subtables>::preprocess();
//         JoltVerifierPreprocessing::<C, F, PST13<E>, KeccakTranscript> {
//             instruction_lookups: instruction_lookups_preprocessing,
//             generators: setup,
//             bytecode: Default::default(),
//             read_write_memory: Default::default(),
//             memory_layout: Default::default(),
//         }
//     };

//     let ops = lookups
//         .into_iter()
//         .map(|op| JoltTraceStep::<F, Instructions>::from_instruction_lookup(op))
//         .collect_vec();

//     if !args.solve_witness {
//         let mut rng = test_rng();
//         let instruction_lookups = LassoProof::<C, M, _, Subtables>::generate_witness(
//             &preprocessing.instruction_lookups,
//             &ops,
//         );
//         let polynomials = JoltPolynomials {
//             instruction_lookups,
//             ..Default::default()
//         };
//         let polynomials_shares =
//             Rep3JoltPolynomials::generate_secret_shares(&preprocessing, &polynomials, &mut rng);
//         rep3_net.send_requests(polynomials_shares)?;
//     }

//     let commitments = Rep3JoltPolynomials::receive_commitments::<C, PST13<E>, KeccakTranscript, _>(
//         &preprocessing,
//         &mut rep3_net,
//     )?;
//     if args.debug {
//         let polynomials_check = LassoProof::<C, M, _, Subtables>::generate_witness(
//             &preprocessing.instruction_lookups,
//             &ops,
//         );

//         if args.solve_witness {
//             let polynomials_shares = rep3_net.receive_responses(Default::default())?;
//             let polynomials =
//                 Rep3JoltPolynomials::combine_polynomials(&preprocessing, polynomials_shares);
//             let polynomials = polynomials.instruction_lookups;

//             assert_eq!(polynomials.dim, polynomials_check.dim);
//             assert_eq!(polynomials.read_cts, polynomials_check.read_cts);
//             assert_eq!(polynomials.final_cts, polynomials_check.final_cts);
//             assert_eq!(polynomials.E_polys, polynomials_check.E_polys);
//             assert_eq!(polynomials.lookup_outputs, polynomials_check.lookup_outputs);
//         }

//         // let commitment_check =
//         //     polynomials_check.commit::<C, PST13<E>, KeccakTranscript>(&preprocessing, &setup);
//         // assert_eq!(
//         //     commitment_check.trace_commitment,
//         //     commitments.trace_commitment
//         // );
//         // assert_eq!(
//         //     commitment_check.final_commitment,
//         //     commitments.final_commitment
//         // );

//         let mut transcript = KeccakTranscript::new(b"Lasso");
//         let mut opening_accumulator = ProverOpeningAccumulator::<F, KeccakTranscript>::new();
//         let mut jolt_polys = JoltPolynomials {
//             bytecode: Default::default(),
//             read_write_memory: Default::default(),
//             instruction_lookups: polynomials_check,
//             timestamp_range_check: Default::default(),
//             r1cs: Default::default(),
//         };
//         let proof = LassoProof::<C, M, Instructions, Subtables>::prove(
//             &preprocessing.generators,
//             &mut jolt_polys,
//             &preprocessing.instruction_lookups,
//             &mut opening_accumulator,
//             &mut transcript,
//         );

//         let mut verifier_transcript = KeccakTranscript::new(b"Lasso");
//         let mut opening_accumulator =
//             VerifierOpeningAccumulator::<F, PST13<E>, KeccakTranscript>::new();
//         LassoProof::<C, M, Instructions, Subtables>::verify(
//             &preprocessing.instruction_lookups,
//             &preprocessing.generators,
//             proof,
//             &commitments,
//             &mut opening_accumulator,
//             &mut verifier_transcript,
//         )
//         .context("while verifying Lasso (check) proof")?;
//     }

//     if args.debug {
//         return Ok(());
//     }

//     let mut transcript = KeccakTranscript::new(b"Lasso");

//     let proof = LassoProof::<C, M, Instructions, Subtables>::prove_rep3(
//         num_inputs,
//         &preprocessing.instruction_lookups,
//         &mut rep3_net,
//         &mut transcript,
//     )?;
//     let mut opening_accumulator =
//         VerifierOpeningAccumulator::<F, PST13<E>, KeccakTranscript>::new();
//     let mut verifier_transcript = KeccakTranscript::new(b"Lasso");
//     LassoProof::verify(
//         &preprocessing.instruction_lookups,
//         &preprocessing.generators,
//         proof,
//         &commitments,
//         &mut opening_accumulator,
//         &mut verifier_transcript,
//     )
//     .context("while verifying Lasso (rep3) proof")?;

//     Ok(())
// }

// pub fn init_tracing() {
//     let env_filter = EnvFilter::builder()
//         .with_default_directive(tracing::Level::INFO.into())
//         .from_env_lossy();
//     // .add_directive("jolt_core=trace".parse().unwrap());

//     let subscriber = Registry::default()
//         .with(env_filter)
//         .with(ForestLayer::default());

//     let _ = tracing::subscriber::set_global_default(subscriber);
// }
