# Co-jolt2 RSS Blowups (Tier 1 + Tier 2): Root-Cause Report + Fixes

Scope: **Tier 1 + Tier 2 only** (lifetime/scoping + chunking + per-phase scratch reuse). Tier 3 architectural refactors are explicitly out of scope.

## Context

Symptoms (single proof, SHA-256 chain, `NUM_ITERS=1`, `trace_len=32768`):
- Worker RSS stair-steps upward during proving and often doesn’t visibly drop between phases.
- Peak RSS grows with Rayon threads.

Important nuance:
- RSS is not “live heap bytes”. It’s a mix of live allocations + allocator/OS retention + other mappings.
- The instrumentation in this repo separates those using jemalloc telemetry and Tracy allocation tracking.

## 0) Why RSS is “too big” for `trace_len=32768`

The working set is not `trace_len * rep3_size`. Several stages allocate **multiple full-length vectors at padded lengths**, and some stages scale by **`M = 2^LOG_M`**, not by trace length:

- `padded_trace_length = next_power_of_two(trace_len)` → `65536` (for `trace_len=32768`).
- A dense coefficient vector of `Rep3PrimeFieldShare<F>` at length `65536` is ~`65536 * 64B ≈ 4MiB` *per polynomial*.
  - `convert_to_polynomials{count=28}` is already ~`28 * 4MiB ≈ 112MiB` in coeff arrays alone (before scratch / clones / derived polys).
- ReadRaf allocates many buffers sized by `M=65536` (suffix MLEs, histograms, Q polynomials, FWHT scratch). Multiplicity (`tables × suffixes × phases`) can easily reach **hundreds of MiB** even for small traces.

## 1) Why “edabits ~67MiB” doesn’t show a sharp RSS drop

Absence of an RSS drop is not evidence that objects are still live:

- `jemalloc` + Rayon threads uses multiple arenas and per-thread caches; freed blocks often stay in the process and are reused later.
- macOS frequently does not immediately reduce RSS even when pages are “logically freed” (e.g. `MADV_FREE`-style semantics).

To separate “**live heap bytes**” from “**RSS retained by allocator/OS**” this repo emits Tracy plots (feature `jemalloc-stats`):
- `jemalloc.allocated`, `jemalloc.active`, `jemalloc.resident`, `jemalloc.mapped`, `jemalloc.retained`

Glossary (jemalloc stats; see `tikv-jemalloc-ctl` docs):
- `allocated`: bytes allocated by the application (logical live heap payload).
- `active`: bytes in active pages backing allocations (page-granularity; ≥ `allocated`).
- `resident`: bytes in physically resident pages mapped by the allocator (includes allocator metadata + active pages + unused dirty pages; often tracks RSS more closely than `active`).
- `mapped`: bytes in active extents mapped by the allocator (virtual address space; ordering vs `resident` is not strict).
- `retained`: bytes in virtual memory mappings retained instead of returned to the OS (excluded from `mapped`; often purged/decommitted, so it may be much larger than RSS).
- `metadata`: bytes dedicated to jemalloc metadata.

Interpretation:
- `active` drops but RSS doesn’t → allocator retention / fragmentation / arenas / OS accounting.
- `active` stays high → real liveness (retained objects / caches / lifetimes).

---

## A) Memory Pattern Report

### A.1 Historical baseline (older trace, user-provided RSS timeline)

| Time (s) | RSS (MB) | Phase (zone name) |
|---:|---:|---|
| 0.0 | 9 | Process start |
| 1.5 | 44 | Preprocessing load |
| 1.9 | 158 | `populate_operands_casts` (334ms) |
| 3.5 | 265→241 | `compute_lookup_outputs` / `fulfill_batched` (3.2s) |
| 5.4 | 241→600 | `fill_field_from_operands_sparse_u64` — B2A spike |
| 5.7 | 708→1037 | `convert_to_polynomials{count=28}` + `one_hot::from_indices` |
| 6.2 | 1052→1426 | `Dory::batch_commit` — transient MSM buffers |
| 10.9 | 668 | Post-commit drop (MSM temporaries freed) |
| 12.2 | 669→915 | `SpartanDag::stage1_prove` |
| 13.1 | 915→967 | `stage2_instances` + `stage2_prove` |
| 14.5 | 967→1490 | `ReadRaf::new` — suffix MLEs, histograms |
| 15.1 | 1490→1675 | `stage3_prove` — 8 ReadRaf phases, FWHT, reshare |
| 28.8 | 1703→1720 | `stage5_reduce_and_prove` — peak RSS |
| 33.3 | 1703→1425 | Proof done, `prove()` returns |

### A.2 Current single proof: Max RSS (from `gtime -v`)

Captured with:
`cd co-jolt2 && REUSE_PREPROC=1 NUM_ITERS=1 TRACY_ALLOC=1 TRACY_CAPTURE=1 bash examples/run_rep3_jolt.sh`

- `worker0`: `Max RSS = 1235808 kB`
- `worker1`: `Max RSS = 1148528 kB`
- `worker2`: `Max RSS = 1176160 kB`

Traces written to:
- `co-jolt2/.traces/worker0.tracy`
- `co-jolt2/.traces/worker1.tracy`
- `co-jolt2/.traces/worker2.tracy`

### A.3 Repeated proofs steady-state (no allocation tracking)

Tracy allocation tracking keeps allocation history and can dominate memory when you run multiple proofs in one process. For steady-state checks, run without it:

`cd co-jolt2 && REUSE_PREPROC=1 NUM_ITERS=1 REPEAT_PROOFS=3 TRACY_ALLOC=0 bash examples/run_rep3_jolt.sh`

- `worker0`: `Max RSS = 1280816 kB`
- `worker1`: `Max RSS = 1245376 kB`
- `worker2`: `Max RSS = 1263424 kB`

### A.4 Phase correlation (worker0; sampled at phase end)

These are “nearest `jemalloc.*` sample after zone end”, extracted from `co-jolt2/.traces/worker0.tracy` via `tracy-csvexport -u -p`:

- `populate_cycle_witness` → `active≈192MiB`, `resident≈258MiB` (`co-jolt2/src/zkvm/witness.rs:357`)
- `Dory::batch_commit` → `active≈174MiB`, `resident≈529MiB` (`co-jolt2/src/poly/commitment/dory.rs:113`)
- `SpartanDag::stage1_prove` → `active≈164MiB`, `resident≈729MiB` (`co-jolt2/src/zkvm/spartan/worker.rs:27`)
- `stage2_instances` → `active≈199MiB`, `resident≈733MiB` (`co-jolt2/src/zkvm/dag/stage.rs:151`)
- `ReadRaf::new` → `active≈303MiB`, `resident≈828MiB` (`co-jolt2/src/zkvm/instruction_lookups/read_raf_checking.rs:154`)
- `stage3_prove` → `active≈198MiB`, `resident≈1.0GiB` (`co-jolt2/src/zkvm/dag/worker.rs:117`)
- `drop_stages` → `active≈216MiB`, `resident≈1.0GiB` (`co-jolt2/src/zkvm/dag/worker.rs`)
- `stage5_reduce_and_prove` → `active≈180MiB`, `resident≈1.0GiB` (`co-jolt2/src/zkvm/dag/worker.rs`)

### A.5 Top 10 live allocations (Tracy “Allocations” view)

CSV export doesn’t include allocation callstacks; collect this from the Tracy GUI:

1. Open `co-jolt2/.traces/worker0.tracy` in `tracy-profiler`.
2. Go to **Memory → Allocations**.
3. For snapshots:
   - right after `fill_field_from_operands_sparse_u64{...}` spike
   - right after `ReadRaf::new`
   - at peak (`stage5_reduce_and_prove`)
   - right after `prove()` returns
4. Record top 10 **Live allocations** (size + callstack + first-seen time).

### A.6 Linux x86 (larger trace) key deltas (NUM_ITERS=5 capture; do not use as default bench)

From `co-jolt2/.traces/tracy/worker2.tracy` (Linux x86, NUM_ITERS=5 run; captured to explain scaling), correlating a zone’s end timestamp to the nearest following jemalloc sample:

- `fulfill_index_futures{count=131072}` (0.088s): `active +~531MiB`, `resident +~552MiB` (major peak live-heap spike).
- `convert_to_polynomials{count=28}` (1.502s): `active -~487MiB` while `retained +~1.3GiB` (allocate+free churn → retained VM mappings).
- `witness_batch_generate` overall (1.605s): `retained +~1.4GiB`, `active +~67MiB`, `resident +~101MiB`.

---

## B) Rust Memory Antipatterns / Bugs Found

### B.1 Lifetimes keeping large stage state alive too long

- Borrowed stage state leaked into later stages (RA worker borrowing `lookups_dag`), preventing early drops.
  - Fix: RA sumcheck worker now owns `Arc<[Rep3OneHotPolynomial<_>; D]>`, allowing `lookups_dag` to be `take()`n and dropped earlier.

### B.2 One-shot “batch everything” causing peak + fragmentation

- `fill_field_from_operands_sparse_u64` converted all jobs in one batch, creating very large transient `Vec`s.
  - Fix: chunking with `B2A_CHUNK` (default `8192`) and per-chunk drops.

### B.3 ReadRaf allocating full-zero buffers and holding them across phases

- Pre-allocating “zero polynomials” (length `M`) and keeping per-phase scratch alive across multiple phases.
  - Fix: represent absent polys as `None`, clear per-phase scratch aggressively at phase transitions, and free big coeff vectors on first bind.

### B.4 Profiler-induced memory growth (not a leak)

- `tracy-mem` grows with “number of allocations observed”, especially across repeated proofs.
  - For steady-state testing use `TRACY_ALLOC=0`.

---

## C) Fix / Optimization Plan (Tier 1 + Tier 2 only) — Implemented

### Tier 1 (lifetime tightening)

- Drop stage DAG state before stage5: `co-jolt2/src/zkvm/dag/worker.rs`
- Make RA stage own its one-hot polys (no borrow): `co-jolt2/src/zkvm/instruction_lookups/ra_virtual.rs`, `co-jolt2/src/zkvm/dag/stage.rs`
- Add explicit `drop_*` spans + optional jemalloc purge checkpoints: `co-jolt2/src/zkvm/dag/worker.rs`, `co-jolt2/src/utils/memory.rs` (`JEMALLOC_PURGE=1`)

### Tier 2 (peak reductions + per-phase scratch reuse)

- Chunk B2A casts: `co-jolt2/src/zkvm/witness.rs` (`B2A_CHUNK=...`)
- ReadRaf phase-scope large buffers: `co-jolt2/src/zkvm/instruction_lookups/read_raf_checking.rs`, `co-jolt2/src/poly/additive_dense_poly.rs`
- Batch Dory commits: `co-jolt2/src/poly/commitment/dory.rs` (`DORY_COMMIT_BATCH=...`)

---

## Verification Checklist

- Single proof + Tracy traces:
  - `cd co-jolt2 && REUSE_PREPROC=1 NUM_ITERS=1 TRACY_ALLOC=1 TRACY_CAPTURE=1 bash examples/run_rep3_jolt.sh`
- Repeated proofs steady-state (no alloc tracking):
  - `cd co-jolt2 && REUSE_PREPROC=1 NUM_ITERS=1 REPEAT_PROOFS=3 TRACY_ALLOC=0 bash examples/run_rep3_jolt.sh`
- Extract jemalloc plot CSV:
  - `cd co-jolt2 && tracy-csvexport -u -p .traces/worker0.tracy | rg '^jemalloc\\.' | head`
