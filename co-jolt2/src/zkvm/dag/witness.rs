use jolt_common::constants::{LookupIndexInt, XlenInt, XLEN};
use jolt_core::zkvm::instruction::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use jolt_core::zkvm::lookup_table::LookupTables;
use mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::Rep3RingShare;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::utils::types::Either;
use crate::utils::types::Rep3Value;

#[derive(Clone, Copy, Debug, Default)]
pub struct CycleMeta {
    /// Per-cycle program counter as a bytecode table index.
    pub pc_index: u64,
    /// Raw public RAM address for this cycle (remapping happens in RAM modules).
    pub ram_addr: u64,
    pub rd_addr: u8,
    pub rs1_addr: u8,
    pub rs2_addr: u8,
}

#[derive(Clone, Debug, Default)]
struct PcInputs {
    unexpanded_pc: Vec<u64>,
    /// Bit i corresponds to `CircuitFlags as usize == i`.
    flags_bits: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
struct Stage1Witness<F: JoltField> {
    imm: Vec<i128>,
    advice: Vec<Rep3PrimeFieldShare<F>>,

    /// Cached lookup output per cycle (field shares).
    lookup_output: Vec<Rep3PrimeFieldShare<F>>,

    rs1_value: Vec<Rep3PrimeFieldShare<F>>,
    rs2_value: Vec<Rep3PrimeFieldShare<F>>,
    rd_write_value: Vec<Rep3PrimeFieldShare<F>>,
    ram_read_value: Vec<Rep3PrimeFieldShare<F>>,
    ram_write_value: Vec<Rep3PrimeFieldShare<F>>,
}

#[derive(Clone, Debug, Default)]
pub struct ReadRafWitness {
    pub lookup_indices: Vec<Either<LookupIndexInt, Rep3RingShare<LookupIndexInt>>>,
    pub lookup_tables: Vec<Option<LookupTables<XLEN>>>,
    pub is_interleaved_operands: Vec<bool>,
    pub right_operand_public_mask: Vec<Option<u64>>,
}

#[derive(Clone, Debug, Default)]
struct IncWitness<F: JoltField> {
    rd_inc: Option<Rep3DensePolynomial<F>>,
    ram_inc: Option<Rep3DensePolynomial<F>>,
}

#[derive(Clone, Debug, Default)]
pub struct ProductInputs<F: JoltField> {
    pub left: Vec<Rep3PrimeFieldShare<F>>,
    pub right: Vec<Rep3PrimeFieldShare<F>>,
}

#[derive(Clone, Debug, Default)]
struct Stage3Witness<F: JoltField> {
    pc_sumcheck: Option<PcInputs>,
    read_raf: Option<ReadRafWitness>,
    product_inputs: Option<ProductInputs<F>>,
}

#[derive(Clone, Debug, Default)]
pub struct Stage3Update<F: JoltField> {
    pub pc_sumcheck: Option<(Vec<u64>, Vec<u32>)>,
    pub read_raf_tables_and_masks:
        Option<(Vec<Option<LookupTables<XLEN>>>, Vec<bool>, Vec<Option<u64>>)>,
    pub read_raf_lookup_indices: Option<Vec<Either<LookupIndexInt, Rep3RingShare<LookupIndexInt>>>>,
    pub product_inputs: Option<ProductInputs<F>>,
}

/// Field-domain per-cycle witness cache (mixed AoS/SoA with explicit drop points).
#[derive(Clone, Debug, Default)]
pub struct Rep3CycleWitnesses<F: JoltField> {
    len: usize,
    meta: Vec<CycleMeta>,
    /// Stage 1: data used for Spartan outer sumcheck and claimed witness evals.
    stage1: Option<Stage1Witness<F>>,
    /// Stage 2: increment polynomials consumed by registers/RAM subsystems.
    stage2: IncWitness<F>,
    /// Stage 3: PCSumcheck, Product virtualization inputs, and ReadRaf witness.
    stage3: Stage3Witness<F>,
}

impl<F: JoltField> Rep3CycleWitnesses<F> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn meta(&self) -> &[CycleMeta] {
        &self.meta
    }

    pub fn row_meta(&self, t: usize) -> CycleMeta {
        self.meta[t]
    }

    pub fn row_stage1(&self, t: usize) -> Stage1RowRef<'_, F> {
        Stage1RowRef { w: self, t }
    }

    pub fn stage1_lookup_output(&self) -> &[Rep3PrimeFieldShare<F>] {
        &self
            .stage1
            .as_ref()
            .expect("stage1 witness already dropped")
            .lookup_output
    }

    pub fn pc_sumcheck_unexpanded_pc(&self) -> &[u64] {
        &self.pc().unexpanded_pc
    }

    pub fn pc_sumcheck_flags_bits(&self) -> &[u32] {
        &self.pc().flags_bits
    }

    pub fn drop_stage1(&mut self) {
        self.stage1 = None;
    }

    pub fn drop_pc_sumcheck_inputs(&mut self) {
        self.stage3.pc_sumcheck = None;
    }

    pub fn take_read_raf(&mut self) -> ReadRafWitness {
        let rr = self
            .stage3
            .read_raf
            .take()
            .expect("read_raf witness already taken/dropped");
        assert!(
            rr.lookup_indices.len() == self.len,
            "read_raf.lookup_indices not populated"
        );
        rr
    }

    pub fn update_stage3(&mut self, update: Stage3Update<F>) {
        if let Some((unexpanded_pc, flags_bits)) = update.pc_sumcheck {
            self.stage3.pc_sumcheck = Some(PcInputs {
                unexpanded_pc,
                flags_bits,
            });
        }

        if let Some((lookup_tables, is_interleaved_operands, right_operand_public_mask)) =
            update.read_raf_tables_and_masks
        {
            let lookup_indices = self
                .stage3
                .read_raf
                .as_mut()
                .map(|rr| std::mem::take(&mut rr.lookup_indices))
                .unwrap_or_default();
            self.stage3.read_raf = Some(ReadRafWitness {
                lookup_indices,
                lookup_tables,
                is_interleaved_operands,
                right_operand_public_mask,
            });
        }

        if let Some(lookup_indices) = update.read_raf_lookup_indices {
            let rr = self
                .stage3
                .read_raf
                .as_mut()
                .expect("read_raf must be initialized before setting lookup_indices");
            rr.lookup_indices = lookup_indices;
        }

        if let Some(inputs) = update.product_inputs {
            self.stage3.product_inputs = Some(inputs);
        }
    }

    pub fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    pub fn set_meta(&mut self, meta: Vec<CycleMeta>) {
        self.meta = meta;
    }

    pub fn set_stage1(
        &mut self,
        imm: Vec<i128>,
        advice: Vec<Rep3PrimeFieldShare<F>>,
        lookup_output: Vec<Rep3PrimeFieldShare<F>>,
        rs1_value: Vec<Rep3PrimeFieldShare<F>>,
        rs2_value: Vec<Rep3PrimeFieldShare<F>>,
        rd_write_value: Vec<Rep3PrimeFieldShare<F>>,
        ram_read_value: Vec<Rep3PrimeFieldShare<F>>,
        ram_write_value: Vec<Rep3PrimeFieldShare<F>>,
    ) {
        self.stage1 = Some(Stage1Witness {
            imm,
            advice,
            lookup_output,
            rs1_value,
            rs2_value,
            rd_write_value,
            ram_read_value,
            ram_write_value,
        });
    }

    pub fn take_product_inputs(&mut self) -> ProductInputs<F> {
        self.stage3
            .product_inputs
            .take()
            .expect("product inputs already taken/dropped")
    }

    pub fn set_stage2_incs(
        &mut self,
        rd_inc: Option<Rep3DensePolynomial<F>>,
        ram_inc: Option<Rep3DensePolynomial<F>>,
    ) {
        if let Some(rd) = rd_inc {
            self.stage2.rd_inc = Some(rd);
        }
        if let Some(ram) = ram_inc {
            self.stage2.ram_inc = Some(ram);
        }
    }

    pub fn take_rd_inc(&mut self) -> Rep3DensePolynomial<F> {
        self.stage2.rd_inc.take().expect("rd_inc not populated")
    }

    pub fn rd_inc_ref(&self) -> &Rep3DensePolynomial<F> {
        self.stage2.rd_inc.as_ref().expect("rd_inc not populated")
    }

    pub fn take_ram_inc(&mut self) -> Rep3DensePolynomial<F> {
        self.stage2.ram_inc.take().expect("ram_inc not populated")
    }

    pub fn ram_inc_ref(&self) -> &Rep3DensePolynomial<F> {
        self.stage2.ram_inc.as_ref().expect("ram_inc not populated")
    }

    #[cfg(debug_assertions)]
    pub fn sanity_check_lengths(&self) {
        debug_assert_eq!(self.meta.len(), self.len);
        if let Some(pc) = &self.stage3.pc_sumcheck {
            debug_assert_eq!(pc.unexpanded_pc.len(), self.len);
            debug_assert_eq!(pc.flags_bits.len(), self.len);
        }
        if let Some(s1) = &self.stage1 {
            debug_assert_eq!(s1.imm.len(), self.len);
            debug_assert_eq!(s1.advice.len(), self.len);
            debug_assert_eq!(s1.lookup_output.len(), self.len);
            debug_assert_eq!(s1.rs1_value.len(), self.len);
            debug_assert_eq!(s1.rs2_value.len(), self.len);
            debug_assert_eq!(s1.rd_write_value.len(), self.len);
            debug_assert_eq!(s1.ram_read_value.len(), self.len);
            debug_assert_eq!(s1.ram_write_value.len(), self.len);
        }
        if let Some(rr) = &self.stage3.read_raf {
            debug_assert!(rr.lookup_indices.is_empty() || rr.lookup_indices.len() == self.len);
            debug_assert_eq!(rr.lookup_tables.len(), self.len);
            debug_assert_eq!(rr.is_interleaved_operands.len(), self.len);
            debug_assert_eq!(rr.right_operand_public_mask.len(), self.len);
        }
        if let Some(pi) = &self.stage3.product_inputs {
            debug_assert_eq!(pi.left.len(), self.len);
            debug_assert_eq!(pi.right.len(), self.len);
        }
    }

    fn pc(&self) -> &PcInputs {
        self.stage3
            .pc_sumcheck
            .as_ref()
            .expect("pc_sumcheck inputs already dropped")
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Stage1RowRef<'a, F: JoltField> {
    w: &'a Rep3CycleWitnesses<F>,
    t: usize,
}

impl<'a, F: JoltField> Stage1RowRef<'a, F> {
    fn pc_sumcheck(&self) -> &'a PcInputs {
        self.w.pc()
    }

    fn stage1(&self) -> &'a Stage1Witness<F> {
        self.w
            .stage1
            .as_ref()
            .expect("stage1 witness already dropped")
    }

    pub fn flags_bits(&self) -> u32 {
        self.pc_sumcheck().flags_bits[self.t]
    }

    pub fn pc_index(&self) -> u64 {
        self.w.meta[self.t].pc_index
    }

    pub fn unexpanded_pc(&self) -> u64 {
        self.pc_sumcheck().unexpanded_pc[self.t]
    }

    pub fn imm(&self) -> i128 {
        self.stage1().imm[self.t]
    }

    pub fn rd_addr(&self) -> u8 {
        self.w.meta[self.t].rd_addr
    }

    pub fn rs1_addr(&self) -> u8 {
        self.w.meta[self.t].rs1_addr
    }

    pub fn rs2_addr(&self) -> u8 {
        self.w.meta[self.t].rs2_addr
    }

    pub fn ram_addr(&self) -> u64 {
        self.w.meta[self.t].ram_addr
    }

    pub fn advice(&self) -> Rep3PrimeFieldShare<F> {
        self.stage1().advice[self.t]
    }

    pub fn lookup_output(&self) -> Rep3PrimeFieldShare<F> {
        self.stage1().lookup_output[self.t]
    }

    pub fn rs1_value(&self) -> Rep3PrimeFieldShare<F> {
        self.stage1().rs1_value[self.t]
    }

    pub fn rs2_value(&self) -> Rep3PrimeFieldShare<F> {
        self.stage1().rs2_value[self.t]
    }

    pub fn rd_write_value(&self) -> Rep3PrimeFieldShare<F> {
        self.stage1().rd_write_value[self.t]
    }

    pub fn ram_read_value(&self) -> Rep3PrimeFieldShare<F> {
        self.stage1().ram_read_value[self.t]
    }

    pub fn ram_write_value(&self) -> Rep3PrimeFieldShare<F> {
        self.stage1().ram_write_value[self.t]
    }

    pub fn flag(&self, flag: CircuitFlags) -> bool {
        debug_assert!(NUM_CIRCUIT_FLAGS <= 32);
        let bit = 1u32 << (flag as usize);
        (self.flags_bits() & bit) != 0
    }

    pub fn next_is_noop(&self) -> bool {
        if self.t + 1 >= self.w.len() {
            false
        } else {
            let bit = 1u32 << (CircuitFlags::IsNoop as usize);
            (self.pc_sumcheck().flags_bits[self.t + 1] & bit) != 0
        }
    }

    pub fn next_pc_index(&self) -> u64 {
        if self.t + 1 >= self.w.len() {
            0
        } else {
            self.w.meta[self.t + 1].pc_index
        }
    }

    pub fn next_unexpanded_pc(&self) -> u64 {
        if self.t + 1 >= self.w.len() {
            0
        } else {
            self.pc_sumcheck().unexpanded_pc[self.t + 1]
        }
    }

    pub fn should_jump(&self) -> bool {
        self.flag(CircuitFlags::Jump) && !self.next_is_noop()
    }

    pub fn to_left_public_input(&self) -> F {
        if self.flag(CircuitFlags::LeftOperandIsPC) {
            F::from_u64(self.unexpanded_pc())
        } else {
            F::zero()
        }
    }

    pub fn to_right_public_input(&self) -> F {
        if self.flag(CircuitFlags::RightOperandIsImm) {
            F::from_i128(self.imm() as XlenInt as i128)
        } else {
            F::zero()
        }
    }

    pub fn to_instruction_inputs_value(&self, _party_id: PartyID) -> (Rep3Value<F>, Rep3Value<F>) {
        let left = if self.flag(CircuitFlags::LeftOperandIsRs1Value) {
            Rep3Value::Shared(self.rs1_value())
        } else if self.flag(CircuitFlags::LeftOperandIsPC) {
            Rep3Value::Public(F::from_u64(self.unexpanded_pc()))
        } else {
            Rep3Value::Public(F::zero())
        };
        let right = if self.flag(CircuitFlags::RightOperandIsRs2Value) {
            Rep3Value::Shared(self.rs2_value())
        } else if self.flag(CircuitFlags::RightOperandIsImm) {
            Rep3Value::Public(F::from_i128(self.imm() as XlenInt as i128))
        } else {
            Rep3Value::Public(F::zero())
        };
        (left, right)
    }

    pub fn to_lookup_operands_value(
        &self,
        party_id: PartyID,
        product: Rep3Value<F>,
    ) -> (Rep3Value<F>, Rep3Value<F>) {
        let left_u64 = if self.flag(CircuitFlags::LeftOperandIsRs1Value) {
            Rep3Value::Shared(self.rs1_value())
        } else if self.flag(CircuitFlags::LeftOperandIsPC) {
            Rep3Value::Public(F::from_u64(self.unexpanded_pc()))
        } else {
            Rep3Value::Public(F::zero())
        };

        let right_u64 = if self.flag(CircuitFlags::RightOperandIsRs2Value) {
            Rep3Value::Shared(self.rs2_value())
        } else if self.flag(CircuitFlags::RightOperandIsImm) {
            Rep3Value::Public(F::from_u64(self.imm() as XlenInt as u64))
        } else {
            Rep3Value::Public(F::zero())
        };

        let zero = Rep3Value::Public(F::zero());

        if self.flag(CircuitFlags::AddOperands) {
            (zero, left_u64.add(&right_u64, party_id))
        } else if self.flag(CircuitFlags::SubtractOperands) {
            let two_pow_xlen = F::from_u128(1u128 << XLEN);
            (
                zero,
                left_u64
                    .sub(&right_u64, party_id)
                    .add_public(two_pow_xlen, party_id),
            )
        } else if self.flag(CircuitFlags::MultiplyOperands) {
            (zero, product)
        } else if self.flag(CircuitFlags::Advice) {
            (zero, Rep3Value::Shared(self.advice()))
        } else {
            (left_u64, right_u64)
        }
    }

    pub fn to_instruction_inputs(
        &self,
        party_id: PartyID,
    ) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>) {
        let left = if self.flag(CircuitFlags::LeftOperandIsRs1Value) {
            self.rs1_value()
        } else if self.flag(CircuitFlags::LeftOperandIsPC) {
            promote_to_trivial_share(party_id, F::from_u64(self.unexpanded_pc()))
        } else {
            Rep3PrimeFieldShare::zero_share()
        };
        let right = if self.flag(CircuitFlags::RightOperandIsRs2Value) {
            self.rs2_value()
        } else if self.flag(CircuitFlags::RightOperandIsImm) {
            promote_to_trivial_share(party_id, F::from_i128(self.imm() as XlenInt as i128))
        } else {
            Rep3PrimeFieldShare::zero_share()
        };
        (left, right)
    }

    pub fn to_lookup_operands(
        &self,
        party_id: PartyID,
        product: Rep3PrimeFieldShare<F>,
    ) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>) {
        let left_u64 = if self.flag(CircuitFlags::LeftOperandIsRs1Value) {
            self.rs1_value()
        } else if self.flag(CircuitFlags::LeftOperandIsPC) {
            promote_to_trivial_share(party_id, F::from_u64(self.unexpanded_pc()))
        } else {
            Rep3PrimeFieldShare::zero_share()
        };

        let right_u64 = if self.flag(CircuitFlags::RightOperandIsRs2Value) {
            self.rs2_value()
        } else if self.flag(CircuitFlags::RightOperandIsImm) {
            promote_to_trivial_share(party_id, F::from_u64(self.imm() as XlenInt as u64))
        } else {
            Rep3PrimeFieldShare::zero_share()
        };

        let zero = Rep3PrimeFieldShare::zero_share();

        if self.flag(CircuitFlags::AddOperands) {
            (zero, left_u64 + right_u64)
        } else if self.flag(CircuitFlags::SubtractOperands) {
            let two_pow_xlen = promote_to_trivial_share(party_id, F::from_u128(1u128 << XLEN));
            (zero, left_u64 - right_u64 + two_pow_xlen)
        } else if self.flag(CircuitFlags::MultiplyOperands) {
            (zero, product)
        } else if self.flag(CircuitFlags::Advice) {
            let _ = party_id;
            (zero, self.advice())
        } else {
            (left_u64, right_u64)
        }
    }

    pub fn to_lookup_output(&self) -> Rep3PrimeFieldShare<F> {
        self.lookup_output()
    }
}
