pub mod range_check;

use co_spartan::mpc::rep3::Rep3PrimeFieldShare;
use fixedbitset::FixedBitSet;
use jolt_core::poly::field::JoltField;
use mpc_core::protocols::rep3::Rep3BigUintShare;
use std::fmt::Debug;
use std::ops::Range;

// for some reason #[enum_dispatch] needs this

pub type SubtableId = String;
pub type LookupId = String;

pub trait LassoSubtable<F: JoltField>: 'static + Sync + Debug + SubtableClone<F>
{
    /// Returns the TypeId of this subtable.
    /// The `Jolt` trait has associated enum types `InstructionSet` and `Subtables`.
    /// This function is used to resolve the many-to-many mapping between `InstructionSet` variants
    /// and `Subtables` variants,
    fn subtable_id(&self) -> SubtableId;

    /// Fully materializes a subtable of size `M`, reprensented as a Vec of length `M`.
    fn materialize(&self, M: usize) -> Vec<F>;

    fn evaluate_mle(&self, point: &[F], M: usize) -> F;

    // /// Expression to evaluate the multilinear extension polynomial for this subtable at the given `point`,
    // /// interpreted to be of size log_2(M), where M is the size of the subtable.
    // fn evaluate_mle_expr(&self, log2_M: usize) -> MultilinearPolyTerms<F>;
}

pub trait LookupType<F: JoltField>: 'static + Send + Sync + Debug + LookupClone<F>
{
    /// Returns the identifier of this lookup type.
    fn lookup_id(&self) -> LookupId;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, operands: &[F], C: usize, M: usize) -> F;

    // fn combine_lookup_expressions(
    //     &self,
    //     expressions: Vec<Expression<E, usize>>,
    //     C: usize,
    //     M: usize,
    // ) -> Expression<E, usize>;

    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables(&self, C: usize, M: usize)
        -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)>;

    // fn to_indices<F: PrimeField>(&self, value: &F) -> Vec<usize>;

    fn output(&self, index: &F) -> F;

    fn output_rep3(&self, index: &Rep3PrimeFieldShare<F>) -> Rep3PrimeFieldShare<F>;

    fn chunk_bits(&self, M: usize) -> Vec<usize>;

    /// Returns the indices of each subtable lookups
    /// The length of `index_bits` is same as actual bit length of table index
    fn subtable_indices(&self, index_bits: Vec<bool>, log_M: usize) -> Vec<Vec<bool>>;

    fn subtable_indices_rep3(&self, index_bits: Vec<Rep3BigUintShare<F>>, log_M: usize) -> Vec<Vec<Rep3BigUintShare<F>>>;


    // fn num_memories(&self) -> usize;
}

pub trait LookupClone<F: JoltField> {
    fn clone_box(&self) -> Box<dyn LookupType<F>>;
}

impl<T, F: JoltField> LookupClone<F> for T
where
    T: LookupType<F> + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn LookupType<F>> {
        Box::new(self.clone())
    }
}

impl<F> Clone for Box<dyn LookupType<F>> {
    fn clone(&self) -> Self {
        unimplemented!()
        // self.clone_box()
    }
}

pub trait SubtableClone<F: JoltField> {
    fn clone_box(&self) -> Box<dyn LassoSubtable<F>>;
}

impl<T, F: JoltField> SubtableClone<F> for T
where
    T: LassoSubtable<F> + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn LassoSubtable<F>> {
        Box::new(self.clone())
    }
}

impl<F> Clone for Box<dyn LassoSubtable<F>> {
    fn clone(&self) -> Self {
        unimplemented!()
        // self.clone_box()
    }
}

#[derive(Clone)]
pub struct SubtableIndices {
    bitset: FixedBitSet,
}

impl SubtableIndices {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bitset: FixedBitSet::with_capacity(capacity),
        }
    }

    pub fn union_with(&mut self, other: &Self) {
        self.bitset.union_with(&other.bitset);
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bitset.ones()
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bitset.count_ones(..)
    }

    pub fn contains(&self, index: usize) -> bool {
        self.bitset.contains(index)
    }
}

impl From<usize> for SubtableIndices {
    fn from(index: usize) -> Self {
        let mut bitset = FixedBitSet::new();
        bitset.grow_and_insert(index);
        Self { bitset }
    }
}

impl From<Range<usize>> for SubtableIndices {
    fn from(range: Range<usize>) -> Self {
        let bitset = FixedBitSet::from_iter(range);
        Self { bitset }
    }
}
