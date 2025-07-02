use ark_ff::Field;
use serde::{Deserialize, Serialize};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{utils::serde_ark, FieldElement};


#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, CanonicalSerialize, CanonicalDeserialize)]
pub struct Interner<F: Field> {
    values: Vec<F>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InternedFieldElement(usize);

impl<F: Field> Interner<F> {
    pub fn new(values: Vec<F>) -> Self {
        Self { values }
    }

    /// Interning is slow in favour of faster lookups.
    pub fn intern(&mut self, value: F) -> InternedFieldElement {
        // Deduplicate
        if let Some(index) = self.values.iter().position(|v| *v == value) {
            return InternedFieldElement(index);
        }

        // Insert
        let index = self.values.len();
        self.values.push(value);
        InternedFieldElement(index)
    }

    pub fn get(&self, el: InternedFieldElement) -> Option<F> {
        self.values.get(el.0).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> + Clone {
        self.values.iter()
    }
}
