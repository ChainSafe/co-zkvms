pub trait JoltField: jolt_core::field::JoltField + snarks_core::field::FieldExt {}
impl<T: jolt_core::field::JoltField + snarks_core::field::FieldExt> JoltField for T {}
