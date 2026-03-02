# co-jolt2 Deferred Optimizations

## Batch condensation + cache_phase `mul_vec` (saves 6 communication rounds)

**File**: `src/zkvm/instruction_lookups/read_raf_checking.rs`

In `cache_phase`, `condensation_mul_vec` and the `ra_acc * v_shifted` mul_vec are currently
two separate calls. Since they operate on independent data within the same phase, they could
be merged into a single `mul_vec` call, saving 1 communication round per phase × 6 phases
(phases 2-7) = 6 rounds total.

**Approach**: Concatenate inputs for both mul_vec calls, execute once, split the result.
`v_shifted` must be kept alive slightly longer (4 MB for 65536-element table).

**Complexity**: Moderate refactor — requires threading `v_shifted` through the condensation path.


---

Update daBits budget [@rep3_jolt.rs (290:291)](file:///home/ubuntu/co-zkvms/co-jolt2/examples/rep3_jolt.rs#L290:291) and tests (dag_correct)

Carefully examine current use of [@edabits.rs (1152:1157)](file:///home/ubuntu/co-zkvms/mpc-core/src/protocols/rep3_ring/preprocessing/edabits.rs#L1152:1157)  [@dabits.rs (497:501)](file:///home/ubuntu/co-zkvms/mpc-core/src/protocols/rep3_ring/preprocessing/dabits.rs#L497:501) 
Implement daBits budget calculation in [@preproc_budget.rs](file:///home/ubuntu/co-zkvms/co-jolt2/src/zkvm/dag/preproc_budget.rs) 

Verify with:
- cargo test -p co-jolt2 --test dag_correct --features test-utils -- --nocapture
- bash ./examples/run_rep3_jolt.sh
