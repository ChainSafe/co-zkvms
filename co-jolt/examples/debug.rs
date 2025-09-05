use co_jolt::{
    field::JoltField,
    jolt::vm::{
        instruction_lookups::InstructionLookupPolynomials,
        read_write_memory::witness::Rep3ProgramIO,
    },
    poly::Rep3MultilinearPolynomial,
};
use itertools::izip;
use jolt_core::{
    jolt::vm::{
        bytecode::BytecodePolynomials,
        read_write_memory::{memory_address_to_witness_index, ReadWriteMemoryPolynomials},
    },
    poly::multilinear_polynomial::MultilinearPolynomial,
    r1cs::inputs::R1CSPolynomials,
};
use jolt_tracer::JoltDevice;

pub fn check_instruction_polys<F: JoltField>(
    polys: &InstructionLookupPolynomials<F>,
    check: &InstructionLookupPolynomials<F>,
) {
    check_poly(
        &polys.lookup_outputs,
        &check.lookup_outputs,
        "lookup_outputs",
    );
    check_polys(&polys.dim, &check.dim, "dim");
    check_polys(&polys.final_cts, &check.final_cts, "final_cts");
    check_polys(&polys.read_cts, &check.read_cts, "read_cts");
    check_polys(&polys.E_polys, &check.E_polys, "E_polys");
}

pub fn check_read_write_polys<F: JoltField>(
    polys: &ReadWriteMemoryPolynomials<F>,
    check: &ReadWriteMemoryPolynomials<F>,
) {
    check_poly(
        polys.v_init.as_ref().unwrap(),
        check.v_init.as_ref().unwrap(),
        "v_init",
    );
    check_poly(&polys.v_final, &check.v_final, "v_final");
    check_poly(&polys.v_read_rd, &check.v_read_rd, "read_rd");
    check_poly(&polys.v_read_rs1, &check.v_read_rs1, "read_rs1");
    check_poly(&polys.v_read_rs2, &check.v_read_rs2, "read_rs2");
    check_poly(&polys.v_read_ram, &check.v_read_ram, "read_ram");
    check_poly(&polys.v_write_rd, &check.v_write_rd, "write_rd");
    check_poly(&polys.v_write_ram, &check.v_write_ram, "write_ram");

    check_poly(&polys.a_ram, &check.a_ram, "a_ram");
    check_poly(&polys.t_read_rd, &check.t_read_rd, "t_read_rd");
    check_poly(&polys.t_read_rs1, &check.t_read_rs1, "t_read_rs1");
    check_poly(&polys.t_read_rs2, &check.t_read_rs2, "t_read_rs2");
    check_poly(&polys.t_read_ram, &check.t_read_ram, "t_read_ram");
    check_poly(&polys.t_final, &check.t_final, "t_final");
}

pub fn check_program_io<F: JoltField>(polys: Vec<Rep3ProgramIO<F>>, program_io: &JoltDevice) {
    let v_io: Vec<F> = Rep3MultilinearPolynomial::combine_shares(vec![
        polys[0].v_io.clone(),
        polys[1].v_io.clone(),
        polys[2].v_io.clone(),
    ])
    .coeffs_as_field_elements();

    let memory_size = v_io.len();

    let mut v_io_check: Vec<_> = vec![F::zero(); memory_size];
    let mut input_index = memory_address_to_witness_index(
        program_io.memory_layout.input_start,
        &program_io.memory_layout,
    );
    // Convert input bytes into words and populate `v_io`
    for chunk in program_io.inputs.chunks(4) {
        let mut word = [0u8; 4];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        let word = F::from_u32(u32::from_le_bytes(word));
        v_io_check[input_index] = word;
        input_index += 1;
    }
    let mut output_index = memory_address_to_witness_index(
        program_io.memory_layout.output_start,
        &program_io.memory_layout,
    );
    // Convert output bytes into words and populate `v_io`
    for chunk in program_io.outputs.chunks(4) {
        let mut word = [0u8; 4];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        let word = u32::from_le_bytes(word);
        v_io_check[output_index] = F::from_u32(word);
        output_index += 1;
    }

    // Copy panic bit
    v_io_check[memory_address_to_witness_index(
        program_io.memory_layout.panic,
        &program_io.memory_layout,
    )] = F::from_u32(program_io.panic as u32);
    if !program_io.panic {
        // Set termination bit
        v_io_check[memory_address_to_witness_index(
            program_io.memory_layout.termination,
            &program_io.memory_layout,
        )] = F::one();
    }

    assert_eq!(v_io, v_io_check);
}

pub fn check_bytecode<F: JoltField>(
    polys: &BytecodePolynomials<F>,
    check: &BytecodePolynomials<F>,
) {
    check_poly(&polys.a_read_write, &check.a_read_write, "a_read_write");
    check_polys(&polys.v_read_write, &check.v_read_write, "v_read_write");
    check_poly(&polys.t_read, &check.t_read, "t_read");
    check_poly(&polys.t_final, &check.t_final, "t_final");
}

pub fn check_r1cs<F: JoltField>(polys: &R1CSPolynomials<F>, check: &R1CSPolynomials<F>) {
    check_polys(&polys.chunks_x, &check.chunks_x, "chunks_x");
    check_polys(&polys.chunks_y, &check.chunks_y, "chunks_y");
    check_polys(&polys.circuit_flags, &check.circuit_flags, "circuit_flags");

    check_poly(
        &polys.aux.left_lookup_operand,
        &check.aux.left_lookup_operand,
        "left_lookup_operand",
    );
    check_poly(
        &polys.aux.right_lookup_operand,
        &check.aux.right_lookup_operand,
        "right_lookup_operand",
    );
    check_poly(&polys.aux.product, &check.aux.product, "product");
    check_polys(
        &polys.aux.relevant_y_chunks,
        &check.aux.relevant_y_chunks,
        "relevant_y_chunks",
    );
    check_poly(
        &polys.aux.write_lookup_output_to_rd,
        &check.aux.write_lookup_output_to_rd,
        "write_lookup_output_to_rd",
    );
    check_poly(
        &polys.aux.write_pc_to_rd,
        &check.aux.write_pc_to_rd,
        "write_pc_to_rd",
    );
    check_poly(
        &polys.aux.next_pc_jump,
        &check.aux.next_pc_jump,
        "next_pc_jump",
    );
    check_poly(
        &polys.aux.should_branch,
        &check.aux.should_branch,
        "should_branch",
    );
    check_poly(&polys.aux.next_pc, &check.aux.next_pc, "next_pc");
}

pub fn check_polys<F: JoltField>(
    polys: &[MultilinearPolynomial<F>],
    check: &[MultilinearPolynomial<F>],
    label: &str,
) {
    assert_eq!(polys.len(), check.len(), "len mismatch {}", label);
    for (i, (poly, check)) in izip!(polys, check).enumerate() {
        check_poly(poly, check, &(label.to_owned() + &format!("_{}", i)));
    }
}

pub fn check_poly<F: JoltField>(
    poly: &MultilinearPolynomial<F>,
    check: &MultilinearPolynomial<F>,
    label: &str,
) {
    assert_eq!(poly.len(), check.len(), "len mismatch {}", label);
    let poly = poly.coeffs_as_field_elements();
    let check = check.coeffs_as_field_elements();
    let p = izip!(&poly, &check).position(|(i, check)| *i != *check);
    if let Some(pos) = p {
        panic!(
            "{label} mismatch at position {} {:?} != {:?}",
            pos,
            &poly[pos..pos + 5],
            &check[pos..pos + 5]
        );
    }
}

fn main() {}
