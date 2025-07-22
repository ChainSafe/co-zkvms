use crate::poly::Rep3MultilinearPolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_common::rv_trace::MemoryLayout;
use jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryStuff;
use jolt_core::{
    field::JoltField, jolt::vm::timestamp_range_check::TimestampRangeCheckPolynomials,
};

use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};

pub type Rep3ReadWriteMemoryPolynomials<F: JoltField> =
    ReadWriteMemoryStuff<Rep3MultilinearPolynomial<F>>;

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3JoltDevice<F: JoltField> {
    pub input_words: Vec<Rep3PrimeFieldShare<F>>,
    pub output_words: Vec<Rep3PrimeFieldShare<F>>,
    pub panic: Rep3PrimeFieldShare<F>, // 0 if not panicked, 1 if panicked
    pub memory_layout: MemoryLayout,
}
