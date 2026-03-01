//! PCG (Pseudorandom Correlation Generator) primitives for BarnOwl-compatible
//! daBit/edaBit generation.
//!
//! Implements the RDCF + PCF construction from Boyle et al. 2022 (EA-LPN codes),
//! adapted for 3-party Rep3 with P0 as trusted dealer.

pub mod dabit_gen;
pub mod edabits_pcg;
pub mod pcf_vole;
pub mod rdcf;
