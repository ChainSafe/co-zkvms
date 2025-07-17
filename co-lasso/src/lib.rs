#![feature(bool_to_result)]

pub mod memory_checking;
pub mod subprotocols;
pub mod poly;
pub mod utils;

pub use jolt_core::field;

pub trait Rep3Polynomials {
    fn num_lookups(&self) -> usize;
}
