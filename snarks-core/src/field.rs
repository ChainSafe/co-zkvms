pub trait FieldExt {
    const TWO_INV: Self;
}

impl FieldExt for ark_bn254::Fr {
    const TWO_INV: ark_bn254::Fr = ark_ff::MontFp!("0x183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000001");
}
