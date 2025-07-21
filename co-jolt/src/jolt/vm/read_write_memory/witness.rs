use crate::lasso::memory_checking::worker::Rep3ExogenousOpenings;
use crate::poly::Rep3MultilinearPolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryStuff;
use jolt_core::lasso::memory_checking::VerifierComputedOpening;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{self, PartyID};

pub type Rep3ReadWriteMemoryPolynomials<F: JoltField> =
    ReadWriteMemoryStuff<Rep3MultilinearPolynomial<F>>;
