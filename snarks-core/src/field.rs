pub trait FieldExt {
    const TWO_INV: Self;
}

impl FieldExt for ark_bn254::Fr {
    const TWO_INV: ark_bn254::Fr = ark_ff::MontFp!("0x183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000001");
}

// Vanilla Jolt re-exports its own `ark_bn254` dependency as `jolt_core::ark_bn254`.
// Implement `FieldExt` for that exact field type to keep trait bounds satisfiable in co-jolt2.
impl FieldExt for jolt_core::ark_bn254::Fr {
    const TWO_INV: jolt_core::ark_bn254::Fr =
        ark_ff::MontFp!("0x183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000001");
}
