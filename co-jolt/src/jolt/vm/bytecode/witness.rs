use crate::poly::Rep3MultilinearPolynomial;
use jolt_core::{field::JoltField, jolt::vm::bytecode::BytecodeStuff};

// #[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
// pub struct Rep3BytecodePolynomials<F: JoltField> {
//     pub a_read_write: MultilinearPolynomial<F>,
//     pub v_read_write: [Rep3DensePolynomial<F>; 6],
//     pub t_read: MultilinearPolynomial<F>,
//     pub t_final: MultilinearPolynomial<F>,
// }

pub type Rep3BytecodePolynomials<F: JoltField> = BytecodeStuff<Rep3MultilinearPolynomial<F>>;
