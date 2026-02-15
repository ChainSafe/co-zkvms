# Co-Jolt2 MPC-ZK-VM

## Context

Building MPC version of Jolt zkVM by extending vanilla Jolt (`../../examples/jolt/tracer/` `../../examples/jolt/jolt-core/`) with MPC secret-share types. This plan uses a **hybrid approach** where vanilla Jolt generates public traces, and co-jolt2 converts them to MPC traces for witness generaion.

**Key Requirements:**
- Minimal modifications to vanilla Jolt codebase 
- No wrapper types (implement traits directly on vanilla types)

## References

This project is based on `../../examples/jolt` (refered to as "vanilla")
The previous version: `../co-jolt` (refered to as "v1"). It implements LEGACY Jolt VM (different from "vanilla" Jolt VM!)

- Summary of Jolt codebase wrt. Instructions `/Users/timofey/repos/examples/jolt/Instructions.md`
- Summary of Jolt codebase wrt. Witness `/Users/timofey/repos/examples/jolt/Instructions.md`
- Summary of `co-jolt` (v1) and `mpc-types` crates wrt. MPC types and core methods `../co-jolt/MpcTypes.md`


## Rules

## Usage of References

- Use vanilla Jolt (`../../examples/jolt/`) as the LOGIC reference w.r.t Jolt VM logic. 
- Co-jolt v1 (`../co-jolt/`) is LEGACY
  - use it as MPC arhitecture reference **but be extremely cautious not to use it as Jolt VM logic reference**
  - use it to copy proven MPC utility implementations (Rep3Operand, share operations) , never for Jolt related architectural patterns.


## Modifying Vanilla Jolt
When the vanilla codebase (`../../examples/jolt/`) has private methods or attributes that co-jolt2 needs, you are allowed to make them `pub` directly in the vanilla code. This is a prototype — don't waste time on workarounds.

## File Structure Must Mirror Vanilla
co-jolt2's `src/` directory structure must mirror the vanilla Jolt codebase. We combine:
- `../../examples/jolt/tracer/src/instruction/` (tracer instruction types)
- `../../examples/jolt/jolt-core/src/zkvm/instruction/` (jolt-core instruction logic)

into a single `src/zkvm/instruction/` directory.

**Mapping:**
| Vanilla | co-jolt2 |
|---------|----------|
| `jolt-core/src/zkvm/` | `src/zkvm/` |
| `jolt-core/src/zkvm/instruction/` | `src/zkvm/instruction/` |
| `tracer/src/instruction/format/` | `src/zkvm/instruction/format/` |

When adding new modules (e.g., `bytecode/`, `r1cs/`, `witness.rs`), place them at the same relative path as vanilla's `jolt-core/src/zkvm/`.

---

## Trace Data: Shared vs Public Classification

All data extracted from a `Cycle` during witness generation:

### Source Values from Cycle

| Value | Source | Type | Shared/Public | Notes |
|-------|--------|------|---------------|-------|
| `rs1` | `register_state.rs1` | `u64` | **SHARED** | Secret register value |
| `rs2` | `register_state.rs2` | `u64` | **SHARED** | Secret register value |
| `rd.0` (pre) | `register_state.rd_values().0` | `u64` | **SHARED** | Secret register pre-value |
| `rd.1` (post) | `register_state.rd_values().1` | `u64` | **SHARED** | Secret register post-value |
| `rd_write_flag` | `NormalizedOperands::from(..).rd` | `u8` | **PUBLIC** | Destination register index (0 = no write). Part of instruction encoding. |
| `imm` | `instruction.operands.imm` | `i64/u64` | **PUBLIC** | Immediate value, part of program |
| `address` (PC) | `instruction.address` | `u64` | **PUBLIC** | Program counter |
| `circuit_flags[..]` | `instruction.circuit_flags()` | `[bool; 18]` | **PUBLIC** | Determined by opcode |
| `lookup_output` | `to_lookup_output(cycle)` | `u64` | **SHARED** | Result of instruction computation on secret inputs |
| `advice` | `instruction.advice` | `u64` | **SHARED** | VirtualAdvice: secret value |
| `ram_access.pre_value` | `cycle.ram_access()` | `u64` | **SHARED** | Secret RAM value |
| `ram_access.post_value` | `cycle.ram_access()` | `u64` | **SHARED** | Secret RAM value |
| `ram_access.address` | `cycle.ram_access()` | `u64` | **PUBLIC** | Memory address (public in Jolt model) |

### Instruction Inputs (`to_instruction_inputs`)

| Format | `left` (u64) | `right` (i128) |
|--------|-------------|----------------|
| R-type (ADD, SUB, AND, ...) | `rs1` **SHARED** | `rs2 as i128` **SHARED** |
| I-type (ADDI, ANDI, ...) | `rs1` **SHARED** | `imm as i128` **PUBLIC** |
| B-type (BEQ, BNE, ...) | `rs1` **SHARED** | `rs2 as i128` **SHARED** |
| J-type (JAL) | `address` (PC) **PUBLIC** | `imm as i128` **PUBLIC** |
| U-type (LUI) | `0` **PUBLIC** | `imm as i128` **PUBLIC** |
| AUIPC | `address` (PC) **PUBLIC** | `imm as i128` **PUBLIC** |
| VirtualAdvice | `0` **PUBLIC** | `0` **PUBLIC** |

### Lookup Index (`to_lookup_index`)

For most instructions: `interleave_bits(left, right as u64)` — bit-interleaving of two u64 values.
For ADD-type (AddOperands flag): `left + right` (single combined operand).
For VirtualAdvice: `advice` value directly.

This is a **nonlinear** operation on shared values when both operands are shared (R-type, B-type).

### Witness Field Computations in `generate_witness_batch`

| Field | Computation | Operation Type |
|-------|-------------|----------------|
| `left_instruction_input[i]` | `left` | Direct assignment of shared or public value |
| `right_instruction_input[i]` | `right` | Direct assignment of shared or public value |
| `write_lookup_output_to_rd[i]` | `rd_write_flag * circuit_flags[WriteLookupOutputToRD]` | **PUBLIC * PUBLIC** (both from instruction encoding) |
| `write_pc_to_rd[i]` | `rd_write_flag * circuit_flags[Jump]` | **PUBLIC * PUBLIC** |
| `should_branch[i]` | `lookup_output * circuit_flags[Branch]` | **SHARED * PUBLIC** (public scalar mult) |
| `should_jump[i]` | `is_jump * (1 - is_next_noop)` | **PUBLIC * PUBLIC** |
| `rd_inc[i]` | `post_rd - pre_rd` (as i128) | **SHARED - SHARED** (but we use Option A: store separately) |
| `instruction_ra[j][i]` | `(lookup_index >> shift) % K_CHUNK` | Derived from **lookup_index** (see below) |
