## MSM over ring shares with correction via masked bit×public-point

### Setting

* Parties: (P_0,P_1,P_2) running Rep3.
* Public: bases (G_0,\dots,G_{n-1}\in \mathbb G) (BN256 (G1)), group order (r).
* Secret scalars: (v_i \in [0,2^{32})) (or bounded similarly).
* Ring: (\mathbb Z_{2^{64}}) for arithmetic sharing.
* Goal: compute (\sum_i [v_i],G_i \in \mathbb G) without converting general scalars to (Fr), and without opening carry info.

### Shares and representations

* Arithmetic Rep3 share of (x\in \mathbb Z_{2^{k}}): each party holds a pair ((a,b)) s.t. globally (x \equiv x_0+x_1+x_2 \pmod{2^k}) with replicated storage.
* Binary Rep3 share of (x\in {0,1}^k): replicated XOR sharing per bit.
* Group Rep3 share of (X\in \mathbb G): replicated additive sharing of group elements (point shares add to (X)).

### Core identity (carry correction)

Let ((x_{0,i},x_{1,i},x_{2,i})) be the (implicit) u32 arithmetic limbs in ([0,2^{32})) underlying the Rep3 arithmetic sharing of (v_i \bmod 2^{32}). Define the integer lift
[
S_i := x_{0,i} + x_{1,i} + x_{2,i} \in [0,2^{33}-2].
]
Then there exists (m_i \in {0,1,2}) such that
[
S_i = v_i + m_i\cdot 2^{32}.
]
Hence
[
[v_i]G_i = [S_i]G_i - [m_i]\cdot (2^{32}G_i).
]
We compute MSM for (S_i) “locally” (using parties’ known limbs) and subtract a correction MSM over the public points (Q_i := 2^{32}G_i) weighted by secret (m_i).

---

## Protocol

### Inputs

Per coefficient (i):

* Arithmetic u32 share ([v_i]*{A,32}) (Rep3 in (\mathbb Z*{2^{32}})).
* Binary u32 share ([v_i]_{B,32}) (Rep3 XOR).

### Outputs

* Group share (\langle \textsf{MSM}\rangle) of (\sum_i [v_i]G_i).

---

## Phase 1: lift arithmetic limbs to (\mathbb Z_{2^{64}}) and compute naive MSM share

1. **Local lift (no comm):**
   [
   [S_i]*{A,64} \leftarrow \textsf{ZeroExtend}*{32\to 64}([v_i]*{A,32})
   ]
   implemented as per-party cast of both share components u32→u64 (no value change; just interpretation in (\mathbb Z*{2^{64}})).
   This yields an arithmetic share of the integer lift (S_i) in (\mathbb Z_{2^{64}}) (not modulo (2^{32})).

2. **Local MSM share (no comm):**
   Each party computes
   [
   X^{(j)}_{\textsf{naive}} := \sum_i [s^{(j)}_i],G_i
   ]
   where (s^{(j)}_i) is the party’s local u64 limb value used as an integer scalar in (Fr) via canonical embedding (safe because scalars are small u64, but note: this is *not yet* equal to (v_i) due to carry).
   Implementation: standard bucket/Pippenger on per-party scalars (s^{(j)}_i).

---

## Phase 2: compute carry (m_i) in MPC (no opens of (m_i))

We compute (m_i\in{0,1,2}) as a secret value derived from the mismatch between the u64-lifted arithmetic value and the true bounded value (v_i).

3. **Lift binary share to (\mathbb Z_{2^{64}}) (no comm):**
   [
   [v_i]*{B,64} \leftarrow \textsf{ZeroExtend}*{32\to 64}([v_i]_{B,32})
   ]
   (bitwise, just widen container; upper bits are 0).

4. **Binary→Arithmetic in (\mathbb Z_{2^{64}}) (comm):**
   [
   [v_i]*{A,64} \leftarrow \textsf{B2A}*{64}([v_i]_{B,64})
   ]
   using your existing `b2a_many::<u64>`.

5. **Compute diff (no comm):**
   [
   [d_i]*{A,64} \leftarrow [S_i]*{A,64} - [v_i]*{A,64}.
   ]
   By construction, (d_i = m_i\cdot 2^{32}) in (\mathbb Z*{2^{64}}).

6. **Extract 2-bit carry without comparison (comm + local):**

   * Convert (d_i) to binary shares:
     [
     [d_i]*{B,64} \leftarrow \textsf{A2B}*{64}([d_i]_{A,64})
     ]
   * Let (b_{i,0} := \text{bit}*{32}([d_i]*{B,64})), (b_{i,1}:=\text{bit}*{33}([d_i]*{B,64})).
     Then (m_i = b_{i,0} + 2 b_{i,1}) (since (d_i\in{0,2^{32},2\cdot 2^{32}})).

At this point we have secret bits ([b_{i,0}]*B), ([b*{i,1}]_B). We never open them.

---

## Phase 3: correction MSM via secure bit×public-point

Define public points:

* (Q_i := (2^{32}\bmod r),G_i) in (\mathbb G)
* (2Q_i) is also public.

We compute a group sharing of
[
X_{\textsf{corr}} = \sum_i \big( b_{i,0},Q_i + b_{i,1},(2Q_i)\big),
]
then subtract it from the naive sum.

### Primitive: `MulBitPublicPoint` with correlated randomness

**Input:** secret bit ([b]_B), public point (Q).
**Preprocessed:** correlated pair (\big([r]_B,\ \langle R\rangle\big)) with uniform (r\in{0,1}) and group share (\langle R\rangle) of (R=rQ).
**Online:**

1. Compute ([c]_B := [b]_B \oplus [r]_B).
2. Open (c\in{0,1}) (public). (Safe because (r) is uniform ⇒ (c) is one-time-padded.)
3. Output group share:

   * if (c=0): (\langle bQ\rangle := \langle R\rangle)
   * if (c=1): (\langle bQ\rangle := \langle Q\rangle - \langle R\rangle)
     where (\langle Q\rangle) is the trivial group share of public (Q).

**Batching:** open all (c)’s for all coefficients/bits in one vector open.

### Correction computation

7. For each (i):

   * (\langle Y_{i,0}\rangle \leftarrow \textsf{MulBitPublicPoint}([b_{i,0}]*B, Q_i; [r*{i,0}]*B,\langle R*{i,0}\rangle))
   * (\langle Y_{i,1}\rangle \leftarrow \textsf{MulBitPublicPoint}([b_{i,1}]*B, 2Q_i; [r*{i,1}]*B,\langle R*{i,1}\rangle))
8. Sum locally in the group-share domain:
   [
   \langle X_{\textsf{corr}}\rangle := \sum_i \big(\langle Y_{i,0}\rangle + \langle Y_{i,1}\rangle\big).
   ]

---

## Phase 4: finalize MSM share

9. Each party holds group shares (\langle X^{(j)}*{\textsf{naive}}\rangle) (from phase 1) and (\langle X*{\textsf{corr}}\rangle). Output:
   [
   \langle X\rangle := \langle X_{\textsf{naive}}\rangle - \langle X_{\textsf{corr}}\rangle,
   ]
   which reconstructs to (\sum_i [v_i]G_i).

---

## Preprocessing requirements

For each coefficient (i) you need **two** correlated pairs (for (b_{i,0}) with (Q_i), and for (b_{i,1}) with (2Q_i)):

* ([r_{i,t}]_B): fresh uniform random bit in Rep3 binary sharing.
* (\langle R_{i,t}\rangle): group share of (R_{i,t} = r_{i,t}\cdot Q_{i,t}) where (Q_{i,0}=Q_i), (Q_{i,1}=2Q_i).

**One-time use.** Reuse of any (r_{i,t}) breaks privacy.

---

## Security notes

* Opening (c=b\oplus r) leaks **0 info** about (b) if (r) is uniform and independent (one-time pad).
* No per-coefficient carry (m_i) or diff (d_i) is opened; leakage avoided.
* Overall security reduces to the security of your Rep3 binary/arithmetic conversions (B2A/A2B) and correctness of the correlated randomness ((r, rQ)).
* If you require malicious security, you need consistency checks/authentication for preprocessing; otherwise a bad (\langle R\rangle) can bias/corrupt the output.

---

## Complexity (online)

* One B2A on (n) 64-bit words.
* One A2B on (n) 64-bit words (or equivalent extraction route).
* One batched open of (2n) masked bits (c).
* Group ops: (O(n)) point adds/subs (no scalar muls in correction), plus the local per-party naive MSM.
