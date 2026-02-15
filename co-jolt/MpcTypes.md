# MPC Types Design and Implementation Primer

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core MPC Secret Share Types](#core-mpc-secret-share-types)
4. [Wrapper Types and Value Enums](#wrapper-types-and-value-enums)
5. [Polynomial Types for MPC](#polynomial-types-for-mpc)
6. [MPC Protocols and Operations](#mpc-protocols-and-operations)
7. [Constructor Methods and Type Conversions](#constructor-methods-and-type-conversions)
8. [Usage Patterns and Best Practices](#usage-patterns-and-best-practices)

---

## Overview

The co-jolt crate implements MPC (Multi-Party Computation) logic for the Jolt VM, a zkVM based on the Jolt proving system. The implementation uses replicated 3-party secret sharing (Rep3) as the primary MPC protocol, enabling three parties to jointly compute over secret-shared data without revealing the underlying values.

**Key Dependencies:**
- `mpc-types`: Core MPC type definitions and arithmetic operations
- `mpc-core`: Protocol implementations (Rep3, additive sharing, conversions)
- `mpc-net`: Networking layer for party communication
- `jolt-core`: Base Jolt VM implementation (non-MPC)

**Primary Goal:** Enable collaborative proof generation for Jolt VM programs where:
- Input data remains secret-shared among 3 parties
- Computation is performed on shares without reconstruction
- Final proof validates correctness without revealing secrets

---

## Architecture

### Crate Structure

```
co-zkvms/
├── mpc-types/          # Core type definitions
│   ├── protocols/
│   │   ├── rep3/       # Replicated 3-party sharing types
│   │   ├── additive/   # Additive sharing types
│   │   └── ...
│   └── serde_compat    # Serialization utilities
│
├── mpc-core/           # Protocol implementations
│   ├── protocols/
│   │   ├── rep3/       # Rep3 arithmetic, binary, conversions
│   │   ├── additive/   # Additive protocols
│   │   └── ...
│   └── lut/            # Lookup table implementations
│
└── co-jolt/            # MPC Jolt VM implementation
    ├── poly/           # MPC polynomial types
    ├── jolt/           # VM components
    ├── subprotocols/   # Grand product, sumcheck
    └── utils/          # Utility types
```

### Type System Layers

1. **Base Secret Share Types** (`mpc-types`)
   - `Rep3PrimeFieldShare<F>` - Replicated field element share
   - `Rep3BigUintShare<F>` - Replicated binary share
   - `AdditivePrimeFieldShare<F>` - Additive field element share
   - `Rep3PointShare<C>` - Replicated elliptic curve point share

2. **Wrapper/Enum Types** (`co-jolt/utils/types`)
   - `Rep3Value<F>` - Union of Public/Shared/Additive
   - `Either<Pub, Share>` - Public or Shared variant
   - `MaybeShared<U>` - Optional sharing wrapper

3. **Polynomial Types** (`co-jolt/poly`)
   - `Rep3DensePolynomial<F>` - Shared dense polynomial
   - `Rep3MultilinearPolynomial<F>` - Public or shared multilinear polynomial
   - `MixedPolynomial<F>` - Polynomial with mixed coefficient types

---

## Core MPC Secret Share Types

### Rep3PrimeFieldShare<F>

**Location:** `mpc-types/src/protocols/rep3/arithmetic/types.rs`

The fundamental replicated secret sharing type for field elements.

```rust
pub struct Rep3PrimeFieldShare<F: PrimeField> {
    pub a: F,  // Share of this party
    pub b: F,  // Share of the previous party
}
```

**Key Properties:**
- **Replicated Secret Sharing:** Each party holds 2 of 3 additive shares
- **Reconstruction:** `secret = share0.a + share1.a + share2.a`
- **Party Assignment:**
  - Party 0: holds shares (a₀, a₂)
  - Party 1: holds shares (a₁, a₀)
  - Party 2: holds shares (a₂, a₁)

**Methods:**
```rust
// Construction
Rep3PrimeFieldShare::new(a: F, b: F) -> Self
Rep3PrimeFieldShare::zero_share() -> Self

// Conversion
share.into_additive() -> AdditivePrimeFieldShare<F>  // (a + b) / 2
share.ab() -> (F, F)  // Unwrap to components

// Promotion from public
Rep3PrimeFieldShare::promote_from_trivial(val: &F, id: PartyID) -> Self
```

**Arithmetic Operations:**
- **Local Addition:** `share1 + share2` = `Rep3PrimeFieldShare::new(a1+a2, b1+b2)`
- **Local Scalar Mult:** `share * public` = `Rep3PrimeFieldShare::new(a*public, b*public)`
- **Multiplication:** `share1 * share2` → `AdditivePrimeFieldShare` (requires resharing for Rep3)

### AdditivePrimeFieldShare<F>

**Location:** `mpc-types/src/protocols/additive/types.rs`

Simpler additive secret sharing (each party holds one share).

```rust
#[repr(transparent)]
pub struct AdditivePrimeFieldShare<F: PrimeField>(pub(crate) F);
```

**Key Properties:**
- **Single Share:** Each party holds exactly one additive component
- **Reconstruction:** `secret = share0 + share1 + share2`
- **Zero-cost abstraction:** Same size/alignment as `F`

**Methods:**
```rust
// Construction
AdditivePrimeFieldShare::zero() -> Self
AdditivePrimeFieldShare::from_fe(value: F) -> Self
AdditivePrimeFieldShare::promote_from_trivial(public: F, id: PartyID) -> Self

// Conversion (unsafe but zero-cost)
AdditivePrimeFieldShare::into_fe(self) -> F
AdditivePrimeFieldShare::into_fe_vec(Vec<Self>) -> Vec<F>
AdditivePrimeFieldShare::from_fe_vec(Vec<F>) -> Vec<Self>

// Optimized public multiplication
share.mul_public_01_optimized(other: F) -> Self  // Checks for 0/1
```

**Why Additive Shares?**
- Result of Rep3 multiplication: `(a₁*a₂ + a₁*b₂ + b₁*a₂ + r)`
- Cannot be directly represented as Rep3 share
- Used as intermediate form before resharing

### Rep3BigUintShare<F>

**Location:** `mpc-types/src/protocols/rep3/binary/types.rs`

Binary (XOR) secret sharing for bit operations.

```rust
pub struct Rep3BigUintShare<F: PrimeField> {
    pub a: BigUint,
    pub b: BigUint,
    _phantom: PhantomData<F>,
}
```

**Key Properties:**
- **XOR-based:** `secret = share0.a ⊕ share1.a ⊕ share2.a`
- **Binary Operations:** AND, OR, XOR, shifts
- **Bit Size:** Constrained to `F::MODULUS_BIT_SIZE`

**Use Cases:**
- Bit decomposition of field elements
- Comparison operations (less-than, equality)
- Range checks

---

## Wrapper Types and Value Enums

### Rep3Value<F>

**Location:** `co-jolt/src/utils/types/rep3_value.rs`

Unified type representing public, replicated shared, or additive shared values.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rep3Value<F: JoltField> {
    Public(F),
    Shared(Rep3PrimeFieldShare<F>),
    Additive(AdditiveShare<F>),
}
```

**Design Rationale:**
- **Flexibility:** Single type handles all value states
- **Optimization:** Public values avoid network communication
- **Protocol Transitions:** Tracks state through computation stages

**Key Methods:**

```rust
// Construction
Rep3Value::zero_public() -> Self
Rep3Value::zero_share() -> Self
Rep3Value::zero_additive() -> Self

// Type checking and extraction
value.as_public() -> F                          // Panics if not public
value.as_shared() -> Rep3PrimeFieldShare<F>     // Panics if not shared
value.as_additive() -> AdditiveShare<F>         // Panics if not additive
value.try_into_public() -> Result<F>            // Safe extraction

// Conversions (with party_id context)
value.into_shared_rep3(party_id) -> Rep3PrimeFieldShare<F>
value.into_additive(party_id) -> AdditiveShare<F>
value.into_rep3_local(party_id) -> Rep3PrimeFieldShare<F>
```

**Arithmetic Operations:**

```rust
// Addition (party_id needed for public value promotion)
value1.add(&value2, party_id) -> Rep3Value<F>
value.add_public(public, party_id) -> Rep3Value<F>
value.add_shared(share, party_id) -> Rep3Value<F>

// Multiplication
value1.mul(&value2) -> Rep3Value<F>              // Local only, may return Additive
value1.mul_reshare(&value2, io_ctx) -> Rep3Value<F>  // With network resharing

// Subtraction
value1.sub(&value2, party_id) -> Rep3Value<F>
```

**Important Rules:**
1. **Addition:**
   - `Shared + Shared` → `Shared`
   - `Shared + Public` → `Shared` (public added to appropriate share)
   - `Additive + Additive` → `Additive`
   - `Shared + Additive` → `Additive` (converts Shared→Additive first)

2. **Multiplication:**
   - `Shared × Shared` → `Additive` (local mult, needs reshare for Rep3)
   - `Shared × Public` → `Shared`
   - `Additive × Public` → `Additive`
   - `Additive × Additive` → PANIC (not allowed)

3. **Public Value Promotion:**
   - Party 0 adds to share `a`
   - Party 1 adds to share `b`
   - Party 2 does nothing
   - Result: `a₀ + b₁ + 0 = public`

### Either<Pub, Share>

**Location:** `co-jolt/src/utils/types/either.rs`

Simple two-variant enum for public or shared values.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Either<Pub, Share> {
    Public(Pub),
    Shared(Share),
}
```

**Usage:**
- Simpler than `Rep3Value` when only two states needed
- Type-safe access: `either.as_public()`, `either.as_shared()`

### MaybeShared<U>

**Location:** `co-jolt/src/utils/types.rs`

Optional sharing with lazy initialization.

```rust
pub enum MaybeShared<U> {
    Public(Option<U>),
    Shared(U),
}
```

**Use Cases:**
- Deferred computation: `Public(None)` until value needed
- Coordinator pattern: Not all parties hold all data

---

## Polynomial Types for MPC

Polynomials are central to Jolt's sumcheck-based proving system. MPC versions maintain secret sharing throughout polynomial evaluations and bindings.

### Rep3DensePolynomial<F>

**Location:** `co-jolt/src/poly/dense_mlpoly.rs`

Dense multilinear polynomial with replicated shared coefficients.

```rust
pub struct Rep3DensePolynomial<F: JoltField> {
    num_vars: usize,
    pub(crate) coeffs: Arc<Vec<Rep3PrimeFieldShare<F>>>,
    bound_coeffs: Vec<Rep3PrimeFieldShare<F>>,
    binding_scratch_space: Option<Vec<Rep3PrimeFieldShare<F>>>,
    len: usize,
    chunk_range: (usize, usize),        // Local shard view
    global_chunk_range: Option<(usize, usize)>,  // Global position
    full_len: usize,
}
```

**Key Features:**

1. **Coefficient Sharing:** All coefficients are `Rep3PrimeFieldShare<F>`

2. **Binding Mechanism:**
   - `coeffs`: Original polynomial coefficients
   - `bound_coeffs`: Scratch space for progressive variable binding
   - `is_bound()`: Tracks whether binding has started

3. **Sharding Support:**
   - `chunk_range`: Local view of coefficient range
   - `global_chunk_range`: Global coordinate system
   - Enables distributed parallel evaluation

**Construction Methods:**

```rust
// Basic construction
Rep3DensePolynomial::new(coeffs: Vec<Rep3PrimeFieldShare<F>>) -> Self
Rep3DensePolynomial::new_padded(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self

// From separate share vectors
Rep3DensePolynomial::from_vec_shares(a: Vec<F>, b: Vec<F>) -> Self
Rep3DensePolynomial::from_poly_shares(
    a: DensePolynomial<F>,
    b: DensePolynomial<F>
) -> Self

// Sharded construction for distributed workers
Rep3DensePolynomial::new_shard(
    coeffs: Vec<Rep3PrimeFieldShare<F>>,
    full_len: usize,
    log_num_workers: usize,
    worker_idx: usize,
) -> Self
```

**Core Operations:**

```rust
// Evaluation at point r (returns additive share)
poly.evaluate(r: &[F]) -> AdditiveShare<F>
poly.evaluate_at_chi(chis: &[F]) -> AdditiveShare<F>

// Optimized evaluation (exploits 0/1 values in chi)
poly.evaluate_at_chi_optimized(chis: &[F]) -> AdditiveShare<F>

// Batch evaluation
Rep3DensePolynomial::batch_evaluate(
    polys: &[&Self],
    r: &[F]
) -> (Vec<AdditiveShare<F>>, Vec<F>)

// Linear combination: Σᵢ cᵢ·polyᵢ
Rep3DensePolynomial::linear_combination(
    polynomials: &[&Self],
    coefficients: &[F],
) -> Self

// Dot product with public vector
poly.dot_product_with_public(other: &[F]) -> Rep3PrimeFieldShare<F>
```

**Binding (Sumcheck Protocol):**

```rust
// Bind polynomial to value r for variable j
// P(x₀,...,xⱼ₋₁, r, xⱼ₊₁,...) = P_new(x₀,...,xⱼ₋₁, xⱼ₊₁,...)
poly.bind(r: F, order: BindingOrder)

// BindingOrder::LowToHigh: bind from x₀
//   P_new[i] = P[2i] + r·(P[2i+1] - P[2i])

// BindingOrder::HighToLow: bind from x_{n-1}
//   P_new[i] = P[i] + r·(P[i + len/2] - P[i])

// Parallel version
poly.bind_parallel(r: F, order: BindingOrder)
```

**Sharding Operations:**

```rust
// Split polynomial for distributed workers
Rep3DensePolynomial::split_poly(
    poly: Rep3DensePolynomial<F>,
    log_workers: usize,
) -> Vec<Rep3MultilinearPolynomial<F>>

// Get shard for specific worker
Rep3DensePolynomial::poly_shard_for_worker(
    poly: &Rep3DensePolynomial<F>,
    shard_nv: usize,
    worker_idx: usize,
) -> Rep3MultilinearPolynomial<F>
```

### Rep3MultilinearPolynomial<F>

**Location:** `co-jolt/src/poly/multilinear_polynomial.rs`

Enum wrapping either public or shared multilinear polynomials.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Rep3MultilinearPolynomial<F: JoltField> {
    Public(MultilinearPolynomial<F>),    // From jolt-core
    Shared(Rep3DensePolynomial<F>),      // MPC version
}
```

**Design Benefits:**
- **Unified Interface:** Same API for public and shared polynomials
- **Optimization:** Public polynomials avoid MPC overhead
- **Gradual Sharing:** Can convert public→shared as needed

**Construction:**

```rust
// Public polynomial
Rep3MultilinearPolynomial::public(poly: MultilinearPolynomial<F>) -> Self

// Shared polynomial
Rep3MultilinearPolynomial::shared(poly: Rep3DensePolynomial<F>) -> Self
Rep3MultilinearPolynomial::from_shared_coeffs(
    coeffs: Vec<Rep3PrimeFieldShare<F>>
) -> Self

// Sharded shared polynomial
Rep3MultilinearPolynomial::new_shard_shared(
    coeffs: Vec<Rep3PrimeFieldShare<F>>,
    full_len: usize,
    log_num_workers: usize,
    worker_idx: usize,
) -> Self

// Sharded public polynomials (compact representations)
Rep3MultilinearPolynomial::new_shard_public_u8(...)
Rep3MultilinearPolynomial::new_shard_public_u32(...)
Rep3MultilinearPolynomial::new_shard_public_u64(...)
```

**Unified Operations:**

```rust
// Get coefficient (returns Rep3Value)
poly.get_coeff(index: usize) -> Rep3Value<F>
poly.get_bound_coeff(index: usize) -> Rep3Value<F>

// Metadata
poly.len() -> usize
poly.get_num_vars() -> usize
poly.shard_range() -> (usize, usize)

// Type-specific access
poly.as_shared() -> &Rep3DensePolynomial<F>
poly.as_public() -> &MultilinearPolynomial<F>
```

**Advanced Operations:**

```rust
// Linear combination (handles mixed public/shared)
Rep3MultilinearPolynomial::linear_combination(
    polynomials: &[&Self],
    coefficients: &[F],
    party_id: PartyID,
) -> Self

// Batch evaluation
Rep3MultilinearPolynomial::batch_evaluate(
    polys: &[&Self],
    r: &[F]
) -> (Vec<Rep3Value<F>>, Vec<F>)

// Worker-specific batch evaluation
Rep3MultilinearPolynomial::batch_evaluate_worker(
    polys: &[&Self],
    r: &[F],
    log_num_workers: usize,
    worker_idx: usize,
) -> (Vec<Rep3Value<F>>, Vec<F>)
```

**Sumcheck Integration:**

```rust
// Get evaluations for sumcheck round
poly.sumcheck_evals(
    index: usize,
    degree: usize,
    order: BindingOrder,
) -> Vec<Rep3Value<F>>

// Convert to shares for combining
poly.sumcheck_evals_into_share(
    index: usize,
    degree: usize,
    order: BindingOrder,
    party_id: PartyID,
) -> Vec<Rep3PrimeFieldShare<F>>
```

### MixedPolynomial<F>

**Location:** `co-jolt/src/poly/mixed_polynomial.rs`

Polynomial with coefficients that may be public, shared, or additive.

```rust
pub struct MixedPolynomial<F: JoltField> {
    pub coeffs: Vec<Rep3Value<F>>,
    num_vars: usize,
    len: usize,
    party_id: PartyID,
}
```

**Use Cases:**
- **Intermediate Computations:** When polynomial operations mix public and shared values
- **Optimization:** Avoid unnecessary sharing of known-public coefficients
- **Protocol Transitions:** Bridge between different sharing types

**Methods:**

```rust
// Construction
MixedPolynomial::new(evals: Vec<Rep3Value<F>>, party_id: PartyID) -> Self
MixedPolynomial::from_public_evals(evals: Vec<F>, party_id: PartyID) -> Self

// Sumcheck evaluations (returns mixed values)
mixed_poly.sumcheck_evals(
    index: usize,
    degree: usize,
    order: BindingOrder,
    party_id: PartyID,
) -> Vec<Rep3Value<F>>

// Variable binding
mixed_poly.bound_poly_var_top(r: &F)   // Bind from high variables
mixed_poly.bound_poly_var_bot(r: &F)   // Bind from low variables
```

**Binding Implementation:**

```rust
// High-to-low binding: P(x₀,...,xⱼ₋₁, r)
// coeffs[i] += (coeffs[i + len/2] - coeffs[i]) * r
pub fn bound_poly_var_top(&mut self, r: &F) {
    let n = self.len() / 2;
    let (left, right) = self.coeffs.split_at_mut(n);
    left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
        a.add_assign(
            &b.sub(&a, self.party_id).mul_public(*r),
            self.party_id
        );
    });
    self.num_vars -= 1;
    self.len = n;
}
```

---

## MPC Protocols and Operations

### Replicated Secret Sharing (Rep3)

**Core Protocol:**

```rust
// Share a single field element
pub fn share_field_element<F: PrimeField, R: Rng>(
    val: F,
    rng: &mut R,
) -> [Rep3PrimeFieldShare<F>; 3] {
    let a = F::rand(rng);
    let b = F::rand(rng);
    let c = val - a - b;

    [
        Rep3PrimeFieldShare::new(a, c),  // Party 0: (a, c)
        Rep3PrimeFieldShare::new(b, a),  // Party 1: (b, a)
        Rep3PrimeFieldShare::new(c, b),  // Party 2: (c, b)
    ]
}

// Reconstruct from shares
pub fn combine_field_element<F: PrimeField>(
    share1: Rep3PrimeFieldShare<F>,
    share2: Rep3PrimeFieldShare<F>,
    share3: Rep3PrimeFieldShare<F>,
) -> F {
    share1.a + share2.a + share3.a
}
```

### Local Arithmetic Operations

**Location:** `mpc-core/src/protocols/rep3/arithmetic.rs`

**Addition (Local):**
```rust
// Shared + Shared: component-wise addition
share1 + share2 = Rep3PrimeFieldShare::new(
    share1.a + share2.a,
    share1.b + share2.b
)

// Shared + Public: only specific party adds
pub fn add_public<F: PrimeField>(
    shared: FieldShare<F>,
    public: F,
    id: PartyID
) -> FieldShare<F> {
    let mut res = shared;
    match id {
        PartyID::ID0 => res.a += public,  // Add to first share
        PartyID::ID1 => res.b += public,  // Add to second share
        PartyID::ID2 => {}                // Do nothing
    }
    res
}
```

**Multiplication (Requires Network):**
```rust
pub fn mul<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // 1. Local multiplication (produces additive share)
    let additive_result = (a * b).into_additive();

    // 2. Add random mask
    let local_a = additive_result.0
                + io_context.rngs.rand.masking_field_element::<F>();

    // 3. Reshare: receive share from previous party
    let local_b = io_context.network.reshare(local_a)?;

    // 4. Return new replicated share
    Ok(FieldShare::new(local_a, local_b))
}
```

**Why Resharing?**

When multiplying Rep3 shares:
```
(a₀, a₂) × (b₀, b₂) = a₀·b₀ + a₀·b₂ + a₂·b₀ + a₂·b₂
```

This produces an additive share, not a replicated share. Resharing converts back:
1. Each party computes local product + random mask
2. Parties exchange masked values in a ring
3. Result is new replicated sharing of `a·b`

### Promotion and Conversion

**Promote Public to Rep3:**
```rust
pub fn promote_to_trivial_share<F: PrimeField>(
    party_id: PartyID,
    val: F
) -> Rep3PrimeFieldShare<F> {
    match party_id {
        PartyID::ID0 => Rep3PrimeFieldShare::new(val, F::zero()),
        PartyID::ID1 => Rep3PrimeFieldShare::new(F::zero(), val),
        PartyID::ID2 => Rep3PrimeFieldShare::zero_share(),
    }
}
```

Result: `share0.a + share1.a + share2.a = val + 0 + 0 = val`

**Rep3 to Additive:**
```rust
impl<F: PrimeField> Rep3PrimeFieldShare<F> {
    pub fn into_additive(self) -> AdditivePrimeFieldShare<F> {
        // Average of two shares
        AdditivePrimeFieldShare((self.a + self.b) * F::TWO_INV)
    }
}
```

**Why TWO_INV?**

Each Rep3 share holds 2 of 3 additive components:
- Party 0: `(a₀, a₂)` → additive: `(a₀ + a₂)/2`
- Party 1: `(a₁, a₀)` → additive: `(a₁ + a₀)/2`
- Party 2: `(a₂, a₁)` → additive: `(a₂ + a₁)/2`

Reconstruction: `(a₀+a₂)/2 + (a₁+a₀)/2 + (a₂+a₁)/2 = (2a₀+2a₁+2a₂)/2 = a₀+a₁+a₂` ✓

### Vector Operations

**Batched Multiplication:**
```rust
pub fn mul_vec<F: PrimeField, N: Rep3Network>(
    lhs: &[FieldShare<F>],
    rhs: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    // 1. Local multiplication + masking
    let local_a: Vec<F> = lhs.iter().zip(rhs.iter())
        .map(|(l, r)| {
            (l * r).into_fe()
          + io_context.rngs.rand.masking_field_element::<F>()
        })
        .collect();

    // 2. Batch reshare (single network round)
    reshare_vec(local_a, io_context)
}
```

**Parallel Version:**
```rust
pub fn mul_vec_par<F: PrimeField, N: Rep3Network>(
    lhs: &[FieldShare<F>],
    rhs: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    // Generate all masks first (sequential RNG)
    let rngs: Vec<F> = (0..lhs.len())
        .map(|_| io_context.rngs.rand.masking_field_element::<F>())
        .collect();

    // Parallel local computation
    let local_a: Vec<F> = lhs.par_iter()
        .zip(rhs.par_iter())
        .zip(rngs.par_iter())
        .map(|((l, r), mask)| (l * r).into_fe() + *mask)
        .collect();

    // Batch reshare
    reshare_vec(local_a, io_context)
}
```

### Seeded Sharing (Compression)

**Location:** `mpc-types/src/protocols/rep3.rs`

For large data, avoid sending full random shares:

```rust
pub enum SeededType<T, U: Rng + SeedableRng> {
    Shares(T),              // Actual share data
    Seed(U::Seed, usize),   // RNG seed + length
}

pub struct ReplicatedSeedType<T, U: Rng + SeedableRng> {
    pub a: SeededType<T, U>,
    pub b: SeededType<T, U>,
}
```

**Usage:**
```rust
// Party 0 sends: [seed_b, seed_c] (compressed)
// Party 0 computes: a = val - expand(seed_b) - expand(seed_c)
// Party 0 stores: (a, expand(seed_c))

let shares = share_field_elements_seeded::<F, _, ChaCha12Rng>(
    &vals,
    &mut rng
);

// Later: expand when needed
let full_shares = shares[0].expand_vec()?;
```

---

## Constructor Methods and Type Conversions

### Rep3PrimeFieldShare

```rust
// Zero initialization
Rep3PrimeFieldShare::zero_share() -> Self
Rep3PrimeFieldShare::default() -> Self

// From components
Rep3PrimeFieldShare::new(a: F, b: F) -> Self

// From public value
Rep3PrimeFieldShare::promote_from_trivial(val: &F, id: PartyID) -> Self

// Conversion
share.into_additive() -> AdditivePrimeFieldShare<F>
share.ab() -> (F, F)
```

### Rep3Value

```rust
// Zero constructors
Rep3Value::zero_public() -> Self
Rep3Value::zero_share() -> Self
Rep3Value::zero_additive() -> Self

// From primitives (using From trait)
let value: Rep3Value<F> = field_element.into();
let value: Rep3Value<F> = rep3_share.into();
let value: Rep3Value<F> = additive_share.into();

// Conversions (require party_id)
value.into_shared_rep3(party_id) -> Rep3PrimeFieldShare<F>
value.into_additive(party_id) -> AdditiveShare<F>
value.into_rep3_local(party_id) -> Rep3PrimeFieldShare<F>

// Safe extraction
value.try_into_public() -> Result<F>
```

### Rep3DensePolynomial

```rust
// From shared coefficients
Rep3DensePolynomial::new(coeffs: Vec<Rep3PrimeFieldShare<F>>) -> Self
Rep3DensePolynomial::new_padded(evals: Vec<Rep3PrimeFieldShare<F>>) -> Self

// From separate share components
Rep3DensePolynomial::from_vec_shares(a: Vec<F>, b: Vec<F>) -> Self
Rep3DensePolynomial::from_poly_shares(
    a: DensePolynomial<F>,
    b: DensePolynomial<F>
) -> Self

// Sharded construction
Rep3DensePolynomial::new_shard(
    coeffs: Vec<Rep3PrimeFieldShare<F>>,
    full_len: usize,
    log_num_workers: usize,
    worker_idx: usize,
) -> Self

// To/from DensePolynomial (for commitment)
poly.into_distributed_commit_form() -> DensePolynomial<F>
poly.copy_share_a() -> DensePolynomial<F>

// Conversion
poly.into_poly_shares() -> (DensePolynomial<F>, DensePolynomial<F>)
```

### Rep3MultilinearPolynomial

```rust
// Construction
Rep3MultilinearPolynomial::public(poly: MultilinearPolynomial<F>) -> Self
Rep3MultilinearPolynomial::shared(poly: Rep3DensePolynomial<F>) -> Self
Rep3MultilinearPolynomial::from_shared_coeffs(
    coeffs: Vec<Rep3PrimeFieldShare<F>>
) -> Self

// From vectors (using From trait)
let poly: Rep3MultilinearPolynomial<F> = vec_rep3_shares.into();
let poly: Rep3MultilinearPolynomial<F> = vec_u8.into();
let poly: Rep3MultilinearPolynomial<F> = vec_u32.into();

// Sharded construction
Rep3MultilinearPolynomial::new_shard_shared(
    coeffs: Vec<Rep3PrimeFieldShare<F>>,
    full_len: usize,
    log_num_workers: usize,
    worker_idx: usize,
) -> Self

Rep3MultilinearPolynomial::new_shard_public_u8/u32/u64(...)

// Conversions (using TryInto)
let shared: &Rep3DensePolynomial<F> = (&poly).try_into()?;
let public: &MultilinearPolynomial<F> = (&poly).try_into()?;
```

### Sharing and Combining

```rust
// Generate shares from public polynomial
pub fn generate_poly_shares_rep3<F: JoltField, R: Rng>(
    poly: &MultilinearPolynomial<F>,
    rng: &mut R,
) -> Vec<Rep3MultilinearPolynomial<F>>

// For vector of polynomials (parallel)
pub fn generate_poly_shares_rep3_vec<F: JoltField, R: Rng>(
    polys: &[MultilinearPolynomial<F>],
    rng: &mut R,
) -> Vec<Vec<Rep3MultilinearPolynomial<F>>>

// Combine shares back to public
Rep3MultilinearPolynomial::combine_shares(
    polys: Vec<Self>
) -> MultilinearPolynomial<F>

pub fn combine_poly_shares_rep3<F: JoltField>(
    poly_shares: Vec<Rep3DensePolynomial<F>>
) -> DensePolynomial<F>
```

---

## Usage Patterns and Best Practices

### Pattern 1: Public-to-Shared Promotion

**When:** Converting public data to shares for MPC computation

```rust
// Single value
let public_val = F::from(42u64);
let share = Rep3PrimeFieldShare::promote_from_trivial(&public_val, party_id);

// Vector (using protocol function)
let public_vals = vec![F::from(1), F::from(2), F::from(3)];
let shares = rep3::arithmetic::promote_to_trivial_shares(public_vals, party_id);

// As Rep3Value (for mixed computation)
let rep3_value = Rep3Value::Public(public_val).into_shared_rep3(party_id);
```

### Pattern 2: Batched Network Operations

**Avoid:** Multiple sequential reshares
```rust
// ❌ BAD: N network rounds
let mut results = Vec::new();
for (a, b) in lhs.iter().zip(rhs.iter()) {
    results.push(rep3::arithmetic::mul(*a, *b, io_ctx)?);
}
```

**Prefer:** Single batched reshare
```rust
// ✅ GOOD: 1 network round
let results = rep3::arithmetic::mul_vec(lhs, rhs, io_ctx)?;
```

### Pattern 3: Mixed Value Arithmetic

**Using Rep3Value for flexible computation:**

```rust
fn compute_linear_combination(
    values: &[Rep3Value<F>],
    coeffs: &[F],
    party_id: PartyID,
) -> Rep3Value<F> {
    values.iter()
        .zip(coeffs.iter())
        .map(|(v, c)| v.mul_public(*c))
        .fold(Rep3Value::zero_public(), |acc, v|
            acc.add(&v, party_id)
        )
}
```

**Benefits:**
- Handles public, shared, and additive values uniformly
- Automatically promotes when mixing types
- Result type determined by input combination

### Pattern 4: Polynomial Evaluation Pipeline

**Typical flow in sumcheck protocol:**

```rust
// 1. Setup: polynomial can be public or shared
let poly: Rep3MultilinearPolynomial<F> = ...;

// 2. Batch evaluate at random point
let (evals, eq_poly) = Rep3MultilinearPolynomial::batch_evaluate(
    &polys_to_eval,
    &random_point,
);
// evals: Vec<Rep3Value<F>>

// 3. For sumcheck rounds
for round in 0..num_vars {
    // Get evaluations at 0, 1, (2,...) for sumcheck
    let round_evals: Vec<Rep3Value<F>> = poly.sumcheck_evals(
        index,
        degree,
        BindingOrder::LowToHigh,
    );

    // Verifier sends challenge
    let challenge = verifier_challenge(round_evals);

    // Bind polynomial to challenge
    poly.bind(challenge, BindingOrder::LowToHigh);
}

// 4. Final claim (should match expected value)
let final_claim: Rep3Value<F> = poly.final_sumcheck_claim();
```

### Pattern 5: Parallel Polynomial Operations

**Sharding for distributed workers:**

```rust
// Coordinator: split polynomial
let num_workers = 4;
let log_workers = 2;
let poly_shards = Rep3DensePolynomial::split_poly(poly, log_workers);

// Each worker receives their shard
// Worker i operates on shard i
let worker_result = poly_shards[worker_idx].evaluate_at_chi(&eq_poly);

// Coordinator: aggregate results
let final_result: AdditiveShare<F> = worker_results
    .into_iter()
    .sum();
```

### Pattern 6: Type Conversions

**Safe extraction with error handling:**

```rust
// ✅ GOOD: Handle potential public/shared mismatch
match poly {
    Rep3MultilinearPolynomial::Public(p) => {
        // Work with public polynomial
    }
    Rep3MultilinearPolynomial::Shared(p) => {
        // Work with shared polynomial
    }
}

// Or use TryInto
if let Ok(shared_poly) = (&poly).try_into() {
    let dense: &Rep3DensePolynomial<F> = shared_poly;
    // ...
}
```

**Unsafe extraction (when type guaranteed):**
```rust
// ❌ Use sparingly: panics if wrong type
let shared_poly = poly.as_shared();
```

### Pattern 7: Iterators with Rep3Value

**Using custom iterator traits:**

```rust
use crate::utils::types::{SharedOrPublicIter, SharedOrPublicParIter};

// Sequential sum
let sum: Rep3Value<F> = values.iter()
    .copied()
    .sum_for(party_id);

// Parallel sum (using Rayon)
let sum: Rep3Value<F> = values.par_iter()
    .copied()
    .sum_for(party_id);
```

### Pattern 8: Coefficient Access

**Unified access pattern:**

```rust
// Works for both Public and Shared variants
let coeff: Rep3Value<F> = poly.get_coeff(index);

// Pattern match for type-specific handling
match coeff {
    Rep3Value::Public(c) => {
        // Fast path: no network needed
        local_computation(c)
    }
    Rep3Value::Shared(c) => {
        // MPC path: may require communication
        mpc_computation(c, io_ctx)
    }
    Rep3Value::Additive(c) => {
        // Already in additive form
        additive_computation(c)
    }
}
```

### Pattern 9: Error Handling

**Network operations return IoResult:**

```rust
use eyre::Context;

// Multiplication (requires network)
let result = rep3::arithmetic::mul(a, b, io_ctx)
    .context("Failed to multiply shares")?;

// Batched operations
let products = rep3::arithmetic::mul_vec(lhs, rhs, io_ctx)
    .context("Failed to multiply share vectors")?;
```

### Pattern 10: Optimized Public Multiplication

**Exploit common cases (0, 1):**

```rust
// Direct multiplication
let result = share * public;  // Always works

// Optimized versions
let result = share.into_additive()
    .mul_public_01_optimized(public);  // Checks for 0/1

// In polynomial context
let scaled = poly.coeffs_ref()
    .par_iter()
    .map(|coeff| coeff.into_additive().mul_public_01_optimized(public))
    .collect();
```

**Optimization impact:**
- `mul_public_01_optimized`: Checks both 0 and 1
- `mul_public_0_optimized`: Only checks 0
- `mul_public_1_optimized`: Only checks 1
- Chi polynomials often contain many 0s and 1s → significant speedup

---

## Common Pitfalls and Solutions

### Pitfall 1: Additive × Additive Multiplication

**Problem:**
```rust
let a: AdditiveShare<F> = ...;
let b: AdditiveShare<F> = ...;
let result = a * b;  // ❌ PANIC: not allowed
```

**Solution:** Convert one operand to Rep3 first (requires coordination)
```rust
// Must reconstruct → reshare, or use different protocol
```

### Pitfall 2: Forgetting Party ID for Public Promotion

**Problem:**
```rust
let shared_val = Rep3Value::Shared(share);
let public_val = Rep3Value::Public(F::from(10));
let sum = shared_val.add(&public_val, ???);  // Which party adds?
```

**Solution:** Always pass party ID for operations mixing public/shared
```rust
let sum = shared_val.add(&public_val, party_id);  // ✅ Correct
```

### Pitfall 3: Ignoring Resharing After Multiplication

**Problem:**
```rust
let product = share1 * share2;  // Returns AdditiveShare<F>
// Later code expects Rep3PrimeFieldShare<F> ❌
```

**Solution:** Use `mul()` with IoContext, not raw `*` operator
```rust
let product = rep3::arithmetic::mul(share1, share2, io_ctx)?;  // ✅ Rep3 share
```

### Pitfall 4: Binding Order Confusion

**Problem:** Binding variables in wrong order breaks sumcheck

**Solution:** Match binding order to sumcheck protocol:
- **Sumcheck from low variables:** Use `BindingOrder::LowToHigh`
- **Sumcheck from high variables:** Use `BindingOrder::HighToLow`

### Pitfall 5: Shared Polynomial Access Without Checking

**Problem:**
```rust
let poly: Rep3MultilinearPolynomial<F> = ...;
let dense = poly.as_shared();  // ❌ Panics if Public
```

**Solution:** Use pattern matching or TryInto
```rust
if let Rep3MultilinearPolynomial::Shared(dense) = poly {
    // Safe access
}
```

---

## Performance Considerations

### Network Communication

**Cost Hierarchy:**
1. **Local Operations:** Free (addition, public multiplication)
2. **Single Reshare:** 1 field element round-trip
3. **Vector Reshare:** Amortized overhead, use `reshare_vec()`
4. **Sequential Reshares:** Avoid! Use batching

**Optimization:**
```rust
// Collect all local results first
let local_products: Vec<F> = lhs.par_iter()
    .zip(rhs.par_iter())
    .map(|(a, b)| (a * b).into_fe() + mask)
    .collect();

// Single network round for all reshares
let final_shares = reshare_vec(local_products, io_ctx)?;
```

### Parallelism

**Polynomial Operations:**
- Use `_par` variants for large vectors: `mul_vec_par()`, `par_iter()`
- Linear combinations naturally parallelize
- Binding can be parallelized: `bind_parallel()`

**Caveats:**
- RNG state must be sequential (masks generated in order)
- Use `par_iter()` for CPU work, not RNG generation

### Memory

**Coefficient Storage:**
- `Rep3DensePolynomial` uses `Arc<Vec<>>` for cheap cloning
- Binding uses scratch space to avoid reallocation
- Sharding reduces memory footprint per worker

**Optimization:**
```rust
// Pre-allocate scratch space
if poly.binding_scratch_space.is_none() {
    poly.binding_scratch_space = Some(unsafe_allocate_zero_share_vec(n));
}
```

---

## Further Reading

**Related Files:**
- `co-jolt/src/jolt/vm/`: VM execution with MPC
- `co-jolt/src/subprotocols/`: Grand product, sumcheck implementations
- `mpc-core/src/protocols/rep3/binary.rs`: Binary share operations
- `mpc-core/src/protocols/rep3/conversion.rs`: Rep3 ↔ Additive conversions
- `mpc-net/`: Networking layer for party communication

**Key Protocols:**
- **Sumcheck:** `co-jolt/src/subprotocols/sumcheck.rs`
- **Grand Product:** `co-jolt/src/subprotocols/grand_product.rs`
- **Memory Checking:** `co-jolt/src/lasso/memory_checking/`

**Testing:**
- `co-jolt/src/simulations/`: Simulated 3-party execution
- Integration tests for end-to-end VM execution

---

## Appendix: Quick Reference

### Type Summary

| Type | Sharing Type | Size | Use Case |
|------|--------------|------|----------|
| `Rep3PrimeFieldShare<F>` | Replicated | 2×F | Primary MPC operations |
| `AdditivePrimeFieldShare<F>` | Additive | 1×F | Multiplication results |
| `Rep3BigUintShare<F>` | Replicated XOR | 2×BigUint | Bit operations |
| `Rep3Value<F>` | Mixed | Enum | Unified value handling |
| `Rep3DensePolynomial<F>` | Replicated | Vec+metadata | Shared polynomials |
| `Rep3MultilinearPolynomial<F>` | Public/Shared | Enum | Flexible polynomials |
| `MixedPolynomial<F>` | Mixed | Vec<Rep3Value> | Mixed computations |

### Operation Complexity

| Operation | Network Rounds | Notes |
|-----------|----------------|-------|
| Addition | 0 | Local |
| Scalar mult | 0 | Local |
| Multiplication | 1 | Reshare required |
| Vector mult (n) | 1 | Batched reshare |
| Binding | 0 | Local |
| Evaluation | 0 | Local (returns additive) |
| Reconstruction | 1 | All parties send to one |

### Common Imports

```rust
// Types
use mpc_core::protocols::rep3::{Rep3PrimeFieldShare, PartyID};
use mpc_core::protocols::additive::AdditiveShare;
use crate::utils::types::{Rep3Value, Either};
use crate::poly::{Rep3DensePolynomial, Rep3MultilinearPolynomial};

// Protocols
use mpc_core::protocols::rep3::arithmetic;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};

// Standard operations
use mpc_types::field::PrimeField;
use rayon::prelude::*;
```
