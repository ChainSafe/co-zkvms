# QUIC Transport Review and Optimization Plan for `mpc-net`, `mpc-core`, and `co-jolt2`

## Summary

The current QUIC stack has three distinct issues:

1. **Transport architecture is more expensive than the protocols assume**
   - `IoContextPool::init(..., rayon_threads)` creates one network fork per Rayon thread.
   - `Rep3QuicMpcNetWorker::fork()` currently opens **new endpoints and new QUIC connections per fork**, not just new streams.
   - This makes network parallelism scale like **connections × streams × buffers**, not just logical concurrency.

2. **Bulk MPC traffic is still “whole-message, whole-Vec”**
   - `send_many` serializes entire `&[F]` into one `Vec<u8>`.
   - `recv_many` deserializes the full message back into one `Vec<F>`.
   - `par_chunks_preproc` / `par_chunks_dabits` split already-materialized batches, so memory is still front-loaded.
   - This is the core bottleneck in preprocessing and large B2A / daBit paths.

3. **The byte-aware QUIC manager is only partially deployed**
   - Main worker channels use `manage_bytes_quic`.
   - Worker forks still regress to generic `ChannelHandle::manage(...)`.
   - Coordinator channels and coordinator forks also still use the generic manager.
   - So the optimized path is not actually the common path once parallelism/forks are used.

The phase 1 work should fix the concrete transport bugs and cap the working set. The phase 2 work is a deliberate redesign to remove the remaining whole-message / whole-Vec architecture.

---

## Current Findings

## `mpc-net` findings

### 1. Worker forking is the highest-impact transport anti-pattern
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/worker.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3/network.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/examples/rep3_jolt.rs`

**What happens now**
- `IoContextPool::init(network, num_forks)` calls `IoContext::fork()`.
- `IoContext::fork()` calls `network.fork()`.
- `Rep3QuicMpcNetWorker::fork()` creates a new worker config with shifted ports and establishes **new QUIC connections**.

**Why it is bad**
- Network fork count is currently tied to Rayon thread count in `rep3_jolt.rs`.
- So “CPU parallelism” accidentally means “open more sockets/endpoints/connections”.
- This inflates handshake cost, per-connection flow-control state, buffered frames, and resident memory.

**Impact**
- High on RSS
- High on preprocessing runtime
- High on startup / fork latency
- High on system resource usage

---

### 2. The optimized byte manager is bypassed on important paths
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/worker.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/coordinator.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/channel.rs`

**What happens now**
- Main worker channels use `ChannelHandle::manage_bytes_quic(...)`.
- Worker forks use `ChannelHandle::manage(...)`.
- Coordinator `new()` and `fork()` also use `ChannelHandle::manage(...)`.

**Why it is bad**
- Fork-heavy workloads fall back to `FramedRead`/`FramedWrite` buffering and generic queue behavior.
- The byte-budgeted path is therefore not the dominant runtime path.

**Impact**
- Medium to high on memory
- Medium on latency stability
- Medium on frame-level fairness

---

### 3. `send_many` / `recv_many` are bulk-copy APIs disguised as generic RPC
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3/network.rs`

**What happens now**
- `send_many` serializes the entire `&[F]` into one contiguous `Vec<u8>`.
- `recv_many` receives one contiguous byte frame and deserializes one `Vec<F>`.

**Why it is bad**
- Large field/ring/share vectors always exist in at least these forms:
  - typed source `Vec<T>`
  - serialized `Vec<u8>`
  - typed output `Vec<T>` on the receiver
- This is unavoidable under the current API shape, even if call sites chunk around it.

**Impact**
- High on memory
- High on memory bandwidth
- High on preprocessing and B2A hot paths

---

### 4. Coordinator fanout does avoidable O(workers × payload) cloning
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/coordinator.rs`

**What happens now**
- `broadcast_request` serializes once into `Vec<u8>` but then clones that `Vec` per worker.
- Similar avoidable copies exist in `send_request`, `send_request_to_workers`, and related helpers.

**Why it is bad**
- For large coordinator broadcasts, the temporary heap can scale with worker count.

**Impact**
- Medium on memory
- Medium on coordinator runtime
- Higher if worker count grows

---

### 5. Send semantics are still queueing semantics, not delivery semantics
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/worker.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/coordinator.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/channel.rs`

**What happens now**
- Most call sites drop the oneshot returned by `blocking_send`.
- That means “send” usually means “successfully enqueued”, not “frame written”.

**Why it matters**
- With byte backpressure this is less dangerous than before, but the API is still misleading.
- Large fanout broadcasts can still accumulate overlap if the caller assumes “sent” means “flushed”.

**Impact**
- Medium on queue overlap
- Medium on predictability
- Low on correctness as long as ordering remains deterministic

---

### 6. QUIC configuration is duplicated and static
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/worker.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/coordinator.rs`

**What happens now**
- Worker and coordinator configure transport separately.
- Defaults are hard-coded and not tied to deployment or measured BDP.
- There are also duplicated / inconsistent codec helpers (`codec_cfg`, inline `LengthDelimitedCodec` builders).

**Impact**
- Medium on maintainability
- Low to medium on performance
- Not the first bottleneck to fix

---

### 7. MPI is isolated and removable
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/mpi.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/mod.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/Cargo.toml`

**Current status**
- It is feature-gated and not used by `mpc-core` / `co-jolt2`.

**Impact**
- Low runtime impact
- Low-risk cleanup
- Good to remove now to simplify the crate surface

---

## `mpc-core` and `co-jolt2` usage findings

### 8. Network concurrency is coupled to CPU concurrency
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/examples/rep3_jolt.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3/network.rs`

**What happens now**
- `IoContextPool::init(network, rayon::current_num_threads() as u32)` is used in the example.
- `IoContextPool::par_chunks*` then uses all available network forks unless call sites cap it manually.

**Why it is bad**
- Network parallelism should be tuned by wire throughput and memory budget, not CPU worker count.
- On current QUIC worker fork semantics, this is especially expensive.

**Impact**
- High on RSS
- High on preprocessing runtime
- Medium on proof hot paths

---

### 9. `par_chunks(None)` is too common on bulk network paths
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/witness.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/suffixes/future.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/instruction_lookups/read_raf_checking.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3_ring/preprocessing/edabits.rs`

**What happens now**
- Many bulk operations derive chunking from fork count or use `None`.
- That means chunk size is often “payload / num_forks”, not “payload bytes bounded by a transport budget”.

**Why it is bad**
- This causes large message buffers when the payload is large.
- It also produces inconsistent runtime behavior across different thread/fork counts.

**Impact**
- High on memory stability
- Medium to high on throughput stability

---

### 10. The current API forces protocol code to hand-roll bulk transport policy
**Files**
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3/network.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3_ring/preprocessing/edabits.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/witness.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/suffixes/future.rs`

**What happens now**
- Each hot path invents its own chunk heuristic.
- Transport backpressure, fork count, and message sizing are not centralized.

**Impact**
- Medium on maintainability
- Medium on correctness risk
- Medium on tuning quality

---

## Phase 1 — Immediate fixes, optimizations, and refactors

## Goal

Make QUIC transport memory-bounded and materially cheaper **without changing protocol logic**.

## Phase 1.1 — Finish the byte-aware QUIC path everywhere

### Changes
1. Replace generic `ChannelHandle::manage(...)` with `ChannelHandle::manage_bytes_quic(...)` for all byte channels:
   - worker main channels
   - worker fork channels
   - worker coordinator channels
   - coordinator main channels
   - coordinator fork channels

2. Unify byte-channel creation around one helper and delete dead / duplicate codec config:
   - remove `codec_cfg()` if it remains unused
   - keep one `LengthDelimitedCodec` builder for byte channels only where still needed before `into_inner()`

3. Make byte budgets configurable for all byte channels:
   - `MPC_QUIC_WRITE_BUF_MB` default `64`
   - `MPC_QUIC_READ_BUF_MB` default `8`
   - keep `MPC_WORKER_WRITE_BUF_MB` as a deprecated alias to `MPC_QUIC_WRITE_BUF_MB` for compatibility during migration

### Files
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/channel.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/worker.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/coordinator.rs`

### Expected impact
- Bounded queued payload on all QUIC byte paths
- Lower RSS on coordinator and on fork-heavy worker jobs
- Better fairness for large frames

### Tradeoffs
- Slightly more blocking when queues are saturated
- Slightly more tuning surface via env vars

---

## Phase 1.2 — Replace worker “new connection per fork” with “new stream per fork”

### Changes
1. Add a shared-channel opener on `MpcNetworkHandlerWorker`:
   - `open_party_byte_channels(&self) -> HashMap<usize, BytesChannel<...>>`
   - `open_coordinator_byte_channel(&self) -> Option<BytesChannel<...>>`

2. Change `Rep3QuicMpcNetWorker::fork()` to:
   - reuse the existing `Arc<MpcNetworkHandlerWrapper>`
   - open fresh bidirectional streams on existing peer connections
   - wrap them with `manage_bytes_quic`
   - allocate a new logical `fork_id`
   - **not** create new endpoints, new bind ports, or new QUIC connections

3. Change `fork_with_coordinator()` to use the same shared-connection + byte-aware path.

4. Keep `get_worker_subnets()` as the path that creates real separate worker networks; do **not** change its semantics.

### Files
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/worker.rs`

### Expected impact
- Largest immediate transport win
- Worker network memory becomes roughly O(peers + active streams), not O(peers × forks × connections)
- Lower fork/setup latency
- Lower connection-state overhead
- Better QUIC congestion behavior because traffic stays on a small set of connections

### Tradeoffs
- Moderate refactor complexity
- Requires careful stream opening order to keep deterministic behavior
- Must preserve shutdown behavior when shared `net_handler` is dropped

---

## Phase 1.3 — Decouple network forks from Rayon threads

### Changes
1. Add an explicit network fork count in the example harness:
   - CLI flag: `--network-forks`
   - env fallback: `NETWORK_FORKS`
   - default: `min(rayon_threads, 4)`

2. Update `rep3_jolt.rs` to initialize:
   - `IoContextPool::init(network, network_forks as u32)`

3. Apply the same default in test utilities / helper binaries that currently pass CPU-thread count directly.

4. Document the policy:
   - CPU threads are for compute
   - network forks are for transport concurrency
   - they are intentionally different knobs

### Files
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/examples/rep3_jolt.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/utils/test_utils.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3/test_utils.rs`

### Expected impact
- Immediate RSS reduction even before deeper refactors
- Lower transport overhead on large preprocessing and B2A workloads
- More predictable scaling

### Tradeoffs
- Some CPU-heavy networked kernels may lose peak throughput if they previously relied on excessive fork counts
- Override remains available for experiments

---

## Phase 1.4 — Remove avoidable serialization and fanout copies

### Changes
1. In coordinator fanout methods:
   - serialize once
   - convert once to `Bytes`
   - clone `Bytes` cheaply per worker instead of cloning `Vec<u8>`

2. Remove single-send useless clones:
   - e.g. `Bytes::from(ser_data.clone())` in one-recipient paths becomes `Bytes::from(ser_data)`

3. Add explicit blocking variants for large broadcast/fanout requests:
   - `broadcast_request_blocking`
   - `send_requests_to_workers_blocking`
   - `send_request_to_workers_blocking`

4. Use the blocking variants in the largest coordinator call sites:
   - Dory row commitment broadcast
   - opening proof challenge vectors
   - any other broadcast where serialized payload routinely exceeds `256 KiB`

### Files
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/coordinator.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/poly/commitment/dory.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/poly/opening_proof.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/subprotocols/sumcheck.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/dag/stage.rs`

### Expected impact
- Lower coordinator heap spikes
- Less temporary cloning
- Lower burstiness on large request fanout

### Tradeoffs
- Blocking broadcast paths may slightly increase latency on some phases
- But that is desirable for memory-bounded behavior

---

## Phase 1.5 — Centralize bulk chunk sizing in `IoContextPool`

### Changes
1. Add a chunk-planning helper in `IoContextPool`:
   - `bulk_chunk_len(total_items, elem_bytes, expansion, max_chunk_mb, min_items_per_fork) -> usize`
   - `expansion` is the multiplicative payload factor, e.g. `T::K` for `alphas_flat`

2. Add env defaults:
   - `MPC_BULK_CHUNK_MB` default `4`
   - `MPC_MIN_FORK_ELEMS` default `2048`

3. Use this helper instead of `None` or fork-derived chunk sizing in high-volume call sites:
   - preprocessing send/recv helpers in `edabits.rs`
   - witness B2A in `witness.rs`
   - suffix B2A in `suffixes/future.rs`
   - ReadRaf chunked reshare / B2A hot paths in `read_raf_checking.rs`

4. Keep small-latency paths on existing `send_many` / `reshare_many`; do not over-apply chunking to tiny messages.

### Files
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3/network.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-core/src/protocols/rep3_ring/preprocessing/edabits.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/witness.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/suffixes/future.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/co-jolt2/src/zkvm/instruction_lookups/read_raf_checking.rs`

### Expected impact
- Message sizes become stable across different thread counts
- Lower RSS spikes on bulk operations
- Better transport tuning hygiene

### Tradeoffs
- Slightly more messages on large jobs
- Much better memory predictability

---

## Phase 1.6 — Add minimal transport correlation spans and counters

### Changes
Add trace-level instrumentation only, no info-log spam:
- queue wait before write permit acquisition
- time spent waiting for `blocking_send`
- bytes sent / received per logical bulk call
- stream-open count for worker forks

### Files
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/channel.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/worker.rs`
- `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/quic/coordinator.rs`

### Expected impact
- Makes future traces explain queueing vs actual wire time
- Needed to validate the fork refactor

### Tradeoffs
- Minimal trace overhead
- No protocol behavior change

---

## Phase 1.7 — Remove MPI transport

### Changes
1. Delete:
   - `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/mpi.rs`

2. Remove `mpi` feature and dependency from:
   - `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/Cargo.toml`

3. Remove module export from:
   - `/Users/timofey/repos/co-jolt2-opt-mem/mpc-net/src/rep3/mod.rs`

### Expected impact
- Smaller crate surface
- Less maintenance burden
- No runtime effect on QUIC path

### Tradeoffs
- Drops MPI compatibility entirely, which is acceptable per scope

---

## Phase 1 priority order

1. **Worker fork refactor to shared connections**
2. **Byte-aware manager on all byte channels**
3. **Network fork cap / decoupling from Rayon**
4. **Coordinator fanout copy fixes + blocking large broadcasts**
5. **Centralized bulk chunk sizing in use sites**
6. **Trace correlation**
7. **MPI deletion**

This order matters: items 1–3 change the scaling shape; items 4–5 clean up the remaining overhead once the transport topology is sane.

---

## Expected Phase 1 outcomes

### Memory
- Worker RSS no longer scales with “Rayon threads == QUIC connections”
- Bulk queue memory is bounded by explicit byte budgets
- Coordinator fanout spikes shrink materially

### Runtime
- Preprocessing should approach linear growth in artifact size once sender queueing and receiver copies are bounded
- Fork-heavy proof stages should stop paying connection-setup and duplicated buffer costs

### Code health
- Fewer duplicate transport paths
- Clearer distinction between control RPC and bulk transport

---

## Phase 2 — Larger overhaul / high-risk, high-reward redesign

## Goal

Remove the remaining “serialize whole Vec / deserialize whole Vec” architecture and make QUIC transport genuinely streaming and bulk-aware.

## Phase 2.1 — Replace `send_many` / `recv_many` for hot paths with typed bulk transport

### Design
Introduce a bulk transport layer alongside the current generic RPC API.

### New interfaces
In `mpc-core`:
- `Rep3BulkTransport` trait with methods like:
  - `send_field_slice<F>(&mut self, target, data: &[F])`
  - `recv_field_vec<F>(&mut self, from, len_hint: Option<usize>)`
  - `recv_field_into_store<F>(&mut self, from, len, store)`
  - `exchange_field_slices<F>(&mut self, target, out: &[F]) -> Vec<F>`

In `mpc-net`:
- byte-stream helpers that can send/receive fixed-width element sequences directly over QUIC streams without first building one giant serialized blob

### Why this matters
This is the actual fix for the remaining whole-message memory amplification.

### Risk
High:
- touches many hot protocols
- requires exact wire-order preservation
- needs careful test coverage

---

## Phase 2.2 — Separate control traffic from bulk traffic

### Design
Per peer connection, maintain:
- one or two long-lived control streams
- a bounded pool of bulk streams
- stream-class-aware write budgets

### Why this matters
Control RPCs should not sit behind multi-megabyte bulk frames.
This improves transcript responsiveness and reduces head-of-line behavior within the application.

### Risk
Medium to high:
- larger connection manager refactor
- changes how channels are provisioned and tracked

---

## Phase 2.3 — Async pipeline for compute ↔ network ↔ disk bulk flows

### Design
For preprocessing and similar bulk kernels:
- generate chunk
- send chunk / receive chunk
- append chunk to disk
- overlap these stages across a bounded pipeline

### Why this matters
Right now the system is mostly “compute big buffer, then send, then recv, then append”.
A pipelined design reduces both peak memory and idle time.

### Risk
High:
- requires async-aware protocol structuring
- touches deterministic ordering assumptions

---

## Phase 2.4 — Custom fixed-width codecs for ark fields and shares

### Design
For bulk types such as:
- `F`
- `Rep3PrimeFieldShare<F>`
- `Rep3RingShare<T>`
encode directly as fixed-width byte slices instead of generic ark serialization of `Vec<T>`.

### Why this matters
- lower CPU cost
- smaller temporary buffers
- easier recv-into-store and recv-into-preallocated-vec

### Risk
High:
- requires exact canonical representation decisions
- must preserve cross-platform correctness

---

## Phase 2.5 — Transport scheduler / credits instead of ad hoc per-call chunking

### Design
Replace scattered call-site chunk heuristics with a scheduler that owns:
- inflight byte credits
- stream assignment
- fairness across peers and traffic classes
- per-peer outstanding bulk operations

### Why this matters
Once transport is streaming, the right place for chunk policy is the transport layer, not protocol call sites.

### Risk
High:
- this is close to a transport rewrite
- should only happen after phase 1 proves the near-term bottlenecks are resolved

---

## Public API / interface changes

## Phase 1 changes
1. `mpc-net`
   - remove `mpi` feature and module
   - add / standardize:
     - `MPC_QUIC_WRITE_BUF_MB`
     - `MPC_QUIC_READ_BUF_MB`
   - keep compatibility alias:
     - `MPC_WORKER_WRITE_BUF_MB` -> `MPC_QUIC_WRITE_BUF_MB`
   - add worker shared-channel opener methods
   - add blocking coordinator fanout variants

2. `co-jolt2`
   - add `--network-forks`
   - env fallback `NETWORK_FORKS`

3. `mpc-core`
   - add `IoContextPool` bulk chunk planner helper
   - standardize use-site chunking on explicit byte budgets

## Phase 2 additions
- new bulk transport trait and methods in `mpc-core`
- new streaming send/recv helpers in `mpc-net`

---

## Test plan

## Unit tests

### `mpc-net`
1. Byte-budget backpressure test
   - small `MPC_QUIC_WRITE_BUF_MB`
   - enqueue many large frames
   - assert sender blocks instead of unbounded queue growth

2. Worker fork transport test
   - create worker
   - call `fork()` multiple times
   - assert it opens new streams on existing connections rather than creating new endpoints/connections

3. Coordinator broadcast clone test
   - instrument or inspect that large broadcast uses one serialized `Bytes` payload cloned cheaply, not N `Vec<u8>` clones

4. Read-budget test
   - large frames on byte-managed channels
   - assert read buffering stays under budget

## Integration tests

### `mpc-core`
1. Small REP3 roundtrip tests with:
   - `NETWORK_FORKS=1`
   - `NETWORK_FORKS=2`
   - `NETWORK_FORKS=4`
   - verify protocol outputs are identical

2. Preprocessing roundtrip with tiny chunk budget:
   - `PREPROC_MAX_MSG_MB=1`
   - `MPC_QUIC_WRITE_BUF_MB=8`
   - confirm chunk boundaries and bounded transport still produce correct pool contents

3. daBit chunk-forward path regression
   - verify deterministic chunk order and stored layout

## `co-jolt2`
1. Correctness smoke:
   - `RUSTFLAGS="-A warnings" cargo test -p co-jolt2 --test dag_correct --features test-utils -- --nocapture`

2. Bench / profiling smoke:
   - `cd /Users/timofey/repos/co-jolt2-opt-mem/co-jolt2 && REUSE_PREPROC=1 NUM_ITERS=1 bash examples/run_rep3_jolt.sh`

3. Preprocessing scaling trace:
   - Linux x86
   - `PREPROC_ONLY=1`
   - compare `N=10/20/40`
   - acceptance:
     - worker0 RSS is bounded by byte budget, not unsent data volume
     - worker2 runtime grows near-linearly with artifact bytes

---

## Acceptance criteria

## Phase 1 acceptance
1. `Rep3QuicMpcNetWorker::fork()` no longer opens fresh QUIC connections.
2. All byte channels use the byte-aware manager.
3. `rep3_jolt` no longer ties network forks to Rayon threads by default.
4. Large coordinator broadcasts no longer clone large serialized `Vec<u8>` per worker.
5. Bulk use sites derive chunk sizes from byte budget, not fork count.
6. On Linux preprocessing traces:
   - worker0 RSS stays bounded
   - worker2 runtime no longer balloons superlinearly in the old way

## Phase 2 acceptance
1. Bulk hot paths stop materializing whole-message serialized buffers.
2. Transport traces clearly separate control and bulk streams.
3. Large preprocessing and B2A paths become throughput-bound rather than heap-bound.

---

## Assumptions and defaults

- QUIC remains the only supported transport.
- MPI is removed in phase 1.
- Protocol send/recv ordering must remain exactly deterministic.
- Phase 1 preserves current protocol logic and wire semantics.
- Default network forks: `min(rayon_threads, 4)`.
- Default QUIC queue budgets:
  - `MPC_QUIC_WRITE_BUF_MB=64`
  - `MPC_QUIC_READ_BUF_MB=8`
- Default generic bulk chunking:
  - `MPC_BULK_CHUNK_MB=4`
  - `MPC_MIN_FORK_ELEMS=2048`
- Phase 2 may introduce new bulk APIs, but phase 1 should remain backward-compatible at the protocol level.
