# Environment Variable Settings

All knobs that control preprocessing and network performance in Rep3 Jolt.

## Preprocessing / Concurrency

| Variable | Default | Description |
|----------|---------|-------------|
| `PREPROC_ONLY` | `0` | When `1`, run only preprocessing and exit (skip proof generation). |
| `REUSE_PREPROC` | `0` | When `1`, load saved preprocessing artifacts and extend if budget exceeds available, instead of regenerating from scratch. |
| `PREPROC_DIR` | `./.preprocessing` | Directory where per-party preprocessing artifacts are loaded/saved (`<dir>/party_<id>/`). |
| `NUM_ITERS` | `1` | Benchmark workload size; drives trace length and therefore preprocessing budget. |
| `PREPROC_LANES` | `8` | Protocol-level preprocessing lane count. Controls how many edaBit types are processed concurrently. |
| `PREPROC_SEGMENT_MB` | `64` | Target segment size per preprocessing lane, in MB of field payload. Smaller segments reduce peak memory and improve I/O locality. |
| `PREPROC_STORE_BATCH_MB` | `16` | Target batch size for grouped store writes to file-backed preprocessing artifacts. |
| `PREPROC_MAX_MSG_MB` | `2` | Target maximum preprocessing wire-message size. Smaller values reduce allocation pressure on localhost. |
| `PREPROC_WARN_GB` | `10` | Warn threshold (in GB) for very large preprocessing artifact files. |
| `EDABITS_PRECACHE_T_LOG2` | `0` | If non-zero, derive a one-proof P0/P1 edaBits precache budget from `trace_len = 2^value` and build sidecars for that derived preprocessing budget. `0` disables precache. |
| `PREPROC_MIN_FORK_ELEMS` | (internal) | Minimum work size before preprocessing/fork chunking is considered worthwhile. |
| `PREPROC_DABIT_WINDOW` | (internal) | daBit pipeline lead/window size; keeps parties from running too far ahead. |

## Network / QUIC

| Variable | Default | Description |
|----------|---------|-------------|
| `NETWORK_FORKS` | `8` | Number of preinitialized logical IoContext forks exposed to protocol code. |
| `MPC_FORK_BULK_CHANNELS` | `0` | When `1`, forks allocate bulk QUIC channels (used only by preprocessing). When `0`, forks skip bulk channels to save 2 streams + tokio tasks per fork. |
| `MPC_QUIC_TOPOLOGY` | `conn-pool` | QUIC transport topology. `conn-pool` (one connection per lane) or `stream-pool` (single connection, multiple streams). `conn-pool` avoids head-of-line blocking and is faster. |
| `MPC_QUIC_CONN_LANES` | `8` (falls back to `NETWORK_FORKS`) | Number of physical QUIC transport lanes provisioned per peer. |
| `MPC_QUIC_WRITE_BUF_MB` | `64` | Outbound byte-budget cap for QUIC byte channels (semaphore size). |
| `MPC_WORKER_WRITE_BUF_MB` | (legacy) | Legacy alias/fallback for `MPC_QUIC_WRITE_BUF_MB`. |
| `MPC_QUIC_READ_BUF_MB` | `64` | Inbound byte-budget cap for QUIC byte channels. |
| `MPC_QUIC_CONN_RX_WINDOW_MB` | `256` | QUIC per-connection receive window. |
| `MPC_QUIC_STREAM_RX_WINDOW_MB` | `64` | QUIC per-stream receive window. |
| `MPC_QUIC_MAX_BIDI_STREAMS` | `256` | Maximum number of concurrent bidirectional QUIC streams. |
| `MPC_QUIC_WRITE_CHUNK_KB` | `16` | Write chunk size (KB) for normal/control QUIC byte traffic. |
| `MPC_QUIC_BULK_WRITE_CHUNK_KB` | `1024` | Write chunk size (KB) for preprocessing bulk QUIC traffic. |

## General Bulk Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `MPC_BULK_CHUNK_MB` | (internal) | Generic target chunk size for bulk networked work. |
| `MPC_MIN_FORK_ELEMS` | (internal) | Generic minimum element count before splitting work across forks. |

## Runtime / Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `RAYON_THREADS` | `4` | Number of Rayon threads (passed as `--rayon-threads` CLI arg). |
| `TRACY_CAPTURE` | `0` | When `1`, runs `tracy-capture` and writes `.tracy` files into `TRACE_DIR/tracy/`. |
| `TRACY_ALLOC` | `0` | When `1`, enables Tracy allocation tracking and jemalloc plots. |
| `TRACY_BASE_PORT` | `8086` | Base port for Tracy profiler (worker p listens on `BASE_PORT + p`). |
| `JEMALLOC_PRESET` | `default` | jemalloc tuning preset: `default`, `return_os`, `aggressive`, `narenas1`. |
| `REPEAT_PROOFS` | `1` | Repeat full proof pipeline N times in same process (requires `reuse-preproc` feature). |

## Recommended Configurations

### Best throughput (localhost, NUM_ITERS=20)
```bash
PREPROC_LANES=8 MPC_QUIC_CONN_LANES=8 NETWORK_FORKS=8 \
PREPROC_MAX_MSG_MB=2 PREPROC_STORE_BATCH_MB=16 PREPROC_SEGMENT_MB=64
```
Result: 97s preprocessing, 522 MB worker2 RSS.

### Low memory
```bash
PREPROC_LANES=4 MPC_QUIC_CONN_LANES=4 NETWORK_FORKS=4 \
PREPROC_MAX_MSG_MB=2 PREPROC_STORE_BATCH_MB=16 PREPROC_SEGMENT_MB=32
```
Result: 99s preprocessing, 845 MB worker2 RSS.
