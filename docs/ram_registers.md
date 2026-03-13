# RAM / Registers

## Intro

### Purpose

Describe why register and RAM checking use mixed public/shared state.

### Motivation

The Jolt memory design already separates public execution structure from witness values. The MPC port keeps that split so it can reuse the vanilla argument shape while only protecting the values that are actually witness-sensitive.

## Background

### Key definitions

- Public memory structure: addresses, order, timestamps, layout.
- Shared memory values: register contents, RAM payloads, advice-backed words.
- `val_init` / `val_final`: initial and final memory polynomials used by the RAM argument.

### References

- `papers/co-zkvms.md`, RW memory and distributed proving sections
- Jolt book: `src/how/architecture/ram.md`
- Useful code anchors:
  - `co-jolt2/src/zkvm/ram/mod.rs`
  - `co-jolt2/src/zkvm/registers/`

## Design

### Role

Registers and RAM prove read/write consistency while preserving the public/shared split already assumed by the Jolt memory argument.

### Assumptions

- Address structure and memory layout are public.
- Memory values and register values are witness data.
- The active RAM domain `K` is computed consistently across witness generation and RAM checks.

### Security invariants

- Advice-backed memory regions stay shared.
- Address remapping must be identical everywhere it is used.
- Mixed memory polynomials must not silently promote shared entries to public ones.

### Design choices

- Keep public regions public:
  - bytecode
  - inputs/outputs
  - address/order metadata
- Keep secret regions shared:
  - register values
  - RAM values
  - trusted and untrusted advice regions
- Use mixed `val_final` because the final memory state genuinely contains both kinds of data.

### Tradeoffs

- Public address structure avoids a more expensive oblivious memory model.
- The result is not access-pattern hiding.

### Notable limitations

- The current design proves consistency of public-address/shared-value memory, not fully oblivious RAM.
- Any inconsistency in `K` or address remapping breaks both correctness and security assumptions.
