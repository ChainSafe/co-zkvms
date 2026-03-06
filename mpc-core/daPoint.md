### Part 1: Protocol Sketch

**Setting & Notation:**

* **Target:** Compute a Rep3 group share of $X_{\text{corr}} = \sum x_i Q_i$.
* **Public:** Points $Q_i \in \mathbb{G}$.
* **Secret Inputs:** Rep3 binary shares of carry bits $x_i$.
* $x_i = x_{i,0} \oplus x_{i,1} \oplus x_{i,2}$
* $P_0$ holds $(x_{i,0}, x_{i,1})$
* $P_1$ holds $(x_{i,1}, x_{i,2})$
* $P_2$ holds $(x_{i,2}, x_{i,0})$



#### 1. Offline Phase (daPoints Preprocessing)

The goal is to generate correlated tuples where $P_0$ holds a secret bit $\gamma_i \in \{0,1\}$, and $P_1, P_2$ hold a 2-of-2 additive group sharing of the point $\Gamma_i = \gamma_i Q_i$.

1. **Generate Private Mask:** $P_0$ samples a uniform random bit $\gamma_i \leftarrow \{0,1\}$.
2. **Generate Shared Random Point:** $P_0$ and $P_1$ use their shared PRG stream to generate a random curve point $A_{i,1} \in \mathbb{G}$.
* *Optimization:* Map PRG bytes to $\mathbb{G}$ using a constant-time `hash_to_curve`. This completely avoids expensive curve scalar multiplications.


3. **Compute Counterpart:** $P_0$ computes the missing 2-of-2 share for $P_2$:

$$A_{i,2} = (\gamma_i Q_i) - A_{i,1}$$



*(Note: $\gamma_i Q_i$ is computed via a simple conditional selection: if $\gamma_i=1$ then $Q_i$, else $\mathcal{O}$.)*
4. **Communicate:** $P_0$ sends the batch of points $\{A_{i,2}\}$ to $P_2$.

#### 2. Online Phase (Fused Dot Product)

We now evaluate the dot product locally, mapping the Boolean relations into the curve group.

1. **Broadcast Masked Values:** $P_0$ computes the masked bit $m_i = x_{i,0} \oplus x_{i,1} \oplus \gamma_i$ and sends it to $P_1$ and $P_2$.
2. **Synchronized Unmasking:** Both $P_1$ and $P_2$ need to reconstruct $\beta_i = x_i \oplus \gamma_i$. Because $x_i = x_{i,0} \oplus x_{i,1} \oplus x_{i,2}$, this is equivalent to $m_i \oplus x_{i,2}$.
* $P_1$ computes $\beta_i = m_i \oplus x_{i,2}$ (using its second share component).
* $P_2$ computes $\beta_i = m_i \oplus x_{i,2}$ (using its first share component).
Now, crucially, $\beta_1 = \beta_2 = \beta_i$.


3. **Local Point Accumulation:**
Applying the identity $x_i = (-1)^{\beta_i}\gamma_i + \beta_i$, we map it directly to the group points:

$$x_i Q_i = \beta_i Q_i + (-1)^{\beta_i} \Gamma_i$$



Substituting $\Gamma_i = A_{i,1} + A_{i,2}$, $P_1$ and $P_2$ accumulate their 2-of-2 shares:
* $P_1$ locally sums: $X_1 = \sum \left( \beta_i Q_i + (-1)^{\beta_i} A_{i,1} \right)$
* $P_2$ locally sums: $X_2 = \sum \left( (-1)^{\beta_i} A_{i,2} \right)$


4. **Final Reshare:**
The parties hold a 2-of-2 additive sharing of the total sum: $X_{\text{corr}} = X_1 + X_2$. They execute a single Rep3 network reshare on the tuple $(0, X_1, X_2)$ to upgrade it to a standard Rep3 group share $\langle X_{\text{corr}} \rangle$.

---

### Part 2: Implementation Sketches

#### 1. EdaPoints Preprocessing

```rust
/// Correlated random tuple for EdaPoints correction.
/// 
/// For each public point Q_i, stores:
/// - `gamma`: random bit known only to P0
/// - `a_self`: 2-of-2 additive group share of `gamma * Q_i`
///   (P1 holds A_1, P2 holds A_2, where A_1 + A_2 = gamma * Q_i)
#[derive(Debug, Clone)]
pub struct EdaPointsBatch<C: CurveGroup> {
    pub gammas: Vec<Bit>, // Populated meaningfully only for P0
    pub a_selfs: Vec<C>,  // P1 holds A_1, P2 holds A_2
}

/// Generate `num` random EdaPoints tuples.
///
/// **Communication:** P0 → P2: `num` curve elements (one round).
#[tracing::instrument(skip_all, name = "edapoints_preprocess")]
pub fn random_edapoints<C: CurveGroup, N: Rep3NetworkWorker>(
    qs: &[C], // The public points Q_i
    io: &mut IoContextPool<N>,
) -> eyre::Result<EdaPointsBatch<C>> {
    let num = qs.len();
    let mut gammas = Vec::with_capacity(num);
    let mut a_selfs = Vec::with_capacity(num);

    for q in qs {
        // P0 generates secret gamma bit from XOR of correlated RNGs
        let (g1, g2): (u8, u8) = io.main().random_elements();
        let gamma_bit = (g1 ^ g2) & 1 == 1;
        
        let gamma = if io.party_id() == PartyID::ID0 {
            Bit::from(gamma_bit)
        } else {
            Bit::from(false)
        };

        // P0 and P1 generate shared random point A_1.
        // P0.rng1 == P1.rng2, so we extract 32 bytes and map to curve.
        let (seed_from_next, seed_from_prev) = io.main().random_bytes::<32>();
        let a_1_bytes = if io.party_id() == PartyID::ID1 { 
            seed_from_prev 
        } else { 
            seed_from_next 
        };
        // Use a fast hash-to-curve or PRF-to-curve to avoid scalar mul
        let a_1 = C::hash_to_curve(&a_1_bytes); 

        let a_self = match io.party_id() {
            PartyID::ID0 => a_1, // Hold temporarily to compute A_2
            PartyID::ID1 => a_1,
            PartyID::ID2 => C::zero(), // Overwritten below via network
        };

        gammas.push(gamma);
        a_selfs.push(a_self);
    }

    // P0 -> P2: Send A_2 = (gamma * Q) - A_1
    if io.party_id() == PartyID::ID0 {
        let a_2_all: Vec<C> = gammas.iter().zip(qs).zip(&a_selfs)
            .map(|((gamma, q), a_1)| {
                let gamma_q = if gamma.convert() { q.clone() } else { C::zero() };
                gamma_q - a_1
            }).collect();
            
        io.par_chunks(a_2_all, None, |chunk, io| {
            io.network.send_many(PartyID::ID2, &chunk)?;
            eyre::Ok(vec![()])
        })?;
    }
    
    if io.party_id() == PartyID::ID2 {
        let a_2_all: Vec<C> = io.par_chunks(rayon::iter::repeat_n((), num), None, |_, io| {
            io.network.recv_many(PartyID::ID0)
        })?;
        a_selfs = a_2_all;
    }

    Ok(EdaPointsBatch { gammas, a_selfs })
}

```

#### 2. Fused Online Phase

```rust
/// Securely compute a Rep3 group share of `Σ_i x[i] · qs[i]`.
///
/// **Communication:** /// - Round 1: P0 broadcasts N bits to P1 and P2.
/// - Round 2: A single `reshare` of the accumulated group element.
pub fn dot_product_edapoints_many<C, N>(
    x_binary: &[Rep3RingShare<Bit>],
    qs: &[C],
    batch: &EdaPointsBatch<C>,
    io: &mut IoContext<N>,
) -> eyre::Result<Rep3GroupShare<C>>
where
    C: CurveGroup,
    N: Rep3Network,
{
    let n = x_binary.len();
    if n == 0 { return Ok(Rep3GroupShare::zero()); }

    // --- Round 1: P0 broadcasts masked values ---
    let ms: Vec<Bit> = if io.id == PartyID::ID0 {
        let ms: Vec<_> = x_binary.iter().zip(&batch.gammas)
            .map(|(x, gamma)| x.a ^ x.b ^ *gamma)
            .collect();
        io.network.send_many(PartyID::ID1, &ms)?;
        io.network.send_many(PartyID::ID2, &ms)?;
        ms
    } else {
        io.network.recv_many(PartyID::ID0)?
    };

    // --- Local Accumulation: Zero scalar muls ---
    let mut acc = C::zero();
    
    if io.id != PartyID::ID0 {
        for (idx, (m, x)) in ms.iter().zip(x_binary.iter()).enumerate() {
            // CRITICAL FIX: Extract the shared x_2 component properly
            let beta = match io.id {
                PartyID::ID0 => unreachable!(),
                // P1 holds (x_1, x_2). x.b is x_2.
                PartyID::ID1 => *m ^ x.b,
                // P2 holds (x_2, x_0). x.a is x_2.
                PartyID::ID2 => *m ^ x.a,
            };

            let a_self = &batch.a_selfs[idx];
            let q = &qs[idx];

            let term = if beta.convert() {
                // beta = 1: P1 adds (Q - A_1), P2 adds (-A_2)
                if io.id == PartyID::ID1 { q.clone() - a_self } else { -a_self.clone() }
            } else {
                // beta = 0: P1 adds A_1, P2 adds A_2
                a_self.clone()
            };
            
            acc += term;
        }
    }

    // --- Round 2: Reshare the final sum to get a Rep3GroupShare ---
    // P0 inputs 0, P1 inputs X_1, P2 inputs X_2. reshare() handles the masking.
    let input_acc = if io.id == PartyID::ID0 { C::zero() } else { acc };
    let prev_acc = io.network.reshare(input_acc.clone())?;
    
    Ok(Rep3GroupShare::new(input_acc, prev_acc))
}

```
