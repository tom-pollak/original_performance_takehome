# Software Pipeline Implementation Plan

## Context

Sequential kernel: 512 batches × 25 cycles = ~12,800 cycles. Software pipelining overlaps
load/compute/store across batches to achieve ~6 cycles/batch throughput (load-limited),
targeting ~3,100 total cycles.

## Key Parameters

```
PERIOD        = 6     # new batch every 6 cycles (load-engine-limited)
LOAD_STEPS    = 6     # L0: 2 vloads + addr mgmt, L1: 8 gather addrs, L2-L5: gather pairs
COMPUTE_STEPS = 18    # C0: XOR, C1-C12: hash, C13: mod+mul, C14-C17: branch/wrap
STORE_STEPS   = 1     # S0: vstore idx + vstore val (merged, store:2)
N_BUFS        = 5     # ceil(25/6) - max in-flight batches
```

## Steady-State Engine Usage (verified, all within limits)

```
Offset | load(2) | alu(12) | valu(6) | flow(1) | store(2) | Active batches
-------|---------|---------|---------|---------|----------|------
  0    |    2    |    4    |    3    |    0    |    2     | L0, C0, C6, C12, S0
  1    |    0    |    8    |    6    |    0    |    0     | L1, C1, C7, C13
  2    |    2    |    0    |    3    |    0    |    0     | L2, C2, C8, C14
  3    |    2    |    0    |    5    |    0    |    0     | L3, C3, C9, C15
  4    |    2    |    0    |    3    |    0    |    0     | L4, C4, C10, C16
  5    |    2    |    0    |    4    |    1    |    0     | L5, C5, C11, C17
```

Max valu is 6/6 at offset 1 — no headroom. Offset 1 has three 2-valu compute steps
(C1=hash_p1, C7=hash_p1, C13=mod+mul). Moving mul out of C13 would recover 1 slot if needed.

## Considered & Rejected Optimizations

**Period < 6**: Impossible. Load steps L0,L2-L5 each need 2/2 load slots (L1 is ALU-only).
With period 5, at offset 0 both the current batch's L0 and previous batch's L5 need the
load engine → 4 load slots, limit is 2. Period 6 is the minimum.

**valu for gather addresses** (replace 8 ALU in L1 with 1 valu `vadd`): Would require
broadcasting `forest_values_p` to a vector. The problem: L1 falls at offset 1, which is
already at valu:6. Adding valu:1 → 7, exceeds limit. Would need to split C13's mul back
out to compensate, negating the benefit.

**Overlap compute tail with store**: S1 (v_val store) could share a cycle with C17
(flow:1), but S0 (v_idx store) can't because C17 writes v_idx via vselect. Net savings
equals just merging S0+S1, which we already do.

**N_BUFS=4**: Would need total batch ≤ 24 cycles. Current is 25 (6+18+1). Every remaining
step has a strict data dependency on its predecessor — no further merges possible.

## Done

- `alloc_buffer(buf_id)` — per-buffer scratch (v_idx, v_val, v_node_val, v_tmp1/2/3, st_idx/val_addr)
- `do_load(b, buf, step, global_step, idx_addr, val_addr)` — 6 steps, load + ALU
- `do_compute(b, buf, step, global_step)` — 18 steps, valu + flow
- `do_store(b, buf, global_step)` — 1 step, store:2 (merged)
- `(val%2)+1` trick — eliminates eq+vselect, saves 1 compute step
- Mul moved from C0 to C13 — rebalances valu across offsets
- Sequential loop uses all three emit methods with 1 buffer — verified correct at 12,835 cycles

## Remaining: Build the Pipelined Schedule

### 1. Allocate 5 buffers instead of 1

```python
bufs = [self.alloc_buffer(i) for i in range(N_BUFS)]
```

Scratch cost: 50 words/buffer × 5 = 250. Plus ~150 shared = ~400 of 1536.

### 2. Build schedule table (pure Python, runs at codegen time)

```python
LOAD_STEPS, COMPUTE_STEPS, STORE_STEPS = 6, 18, 1
PERIOD, N_BUFS = 6, 5
batches_per_round = batch_size // VLEN  # 32
total_batches = rounds * batches_per_round  # 512

cycle_ops = defaultdict(list)
t = 0
for batch in range(total_batches):
    if batch > 0 and batch % batches_per_round == 0:
        cycle_ops[t].append(("reset", 0, 0))
        t += 1  # 1-cycle bubble for addr reset
    buf_idx = batch % N_BUFS
    global_step = batch * VLEN
    for s in range(LOAD_STEPS):
        cycle_ops[t + s].append(("load", buf_idx, s, global_step))
    for s in range(COMPUTE_STEPS):
        cycle_ops[t + LOAD_STEPS + s].append(("compute", buf_idx, s, global_step))
    cycle_ops[t + LOAD_STEPS + COMPUTE_STEPS].append(("store", buf_idx, 0, global_step))
    t += PERIOD
```

### 3. Emit one bundle per cycle

```python
for cy in range(max(cycle_ops) + 1):
    with self.bundle() as b:
        for op in cycle_ops[cy]:
            phase, buf_idx, step, gs = op
            if phase == "reset":
                b.alu("+", idx_addr, self.scratch["inp_indices_p"], self.get_const(0))
                b.alu("+", val_addr, self.scratch["inp_values_p"], self.get_const(0))
            elif phase == "load":
                self.do_load(b, bufs[buf_idx], step, gs, idx_addr, val_addr)
            elif phase == "compute":
                self.do_compute(b, bufs[buf_idx], step, gs)
            elif phase == "store":
                self.do_store(b, bufs[buf_idx], gs)
```

### 4. Keep prologue and epilogue

- Prologue (constant loading, init vars, hash consts, pause) — unchanged
- Final `flow("pause")` — keep after the emission loop

## Address Management

- **Global `idx_addr`/`val_addr`**: Advance by VLEN during each L0 step (self-overwriting:
  vload reads old, ALU writes new). Only 1 batch loads at a time → no conflicts.
- **Per-buffer `st_idx_addr`/`st_val_addr`**: Saved copies of global addr during L0, used
  later by store phase. Decouples store from global pointer state.
- **Round boundaries**: 1-cycle reset event rewrites global pointers to base. 15 resets
  across 16 rounds = 15 cycles overhead.

## Cross-Round Safety

Batch B of round N stores at cycle `~(N*32+B)*6 + 25`. Batch B of round N+1 loads at
cycle `~((N+1)*32+B)*6`. Gap: `32*6 - 25 = 167 cycles`. Huge margin — no stalls needed.

## Cycle Count Estimate

### Prologue (constant loading, init, pause)

Counting bundles from build_kernel before the loop:
- 4 bundles: scalar consts (0-6, VLEN) — 2 loads each
- 3 bundles: vconsts for 0,1,2 — packed with scalar const bundles
- 4 bundles: init_vars loads (7 vars, 2 per bundle, ceil(7/2)=4)
- ~4 bundles: hash scalar consts (~8 unique values, 2 per bundle)
- ~2 bundles: hash vconsts (~8 values, 6 per bundle)
- 1 bundle: n_nodes scalar const
- 1 bundle: v_n_nodes vconst
- 1 bundle: pause + comment

**~17 cycles prologue**

### Pipeline

Parameters:
- batch_size=256, VLEN=8 → batches_per_round = 32
- rounds=16 → total_batches = 512
- PERIOD=6, batch_latency=25 (6 load + 18 compute + 1 store)

Schedule builder advances t by PERIOD(6) per batch, plus 1 for each round reset:
- Round 0: 32 batches, t goes 0 → 192 (32×6). No reset.
- Rounds 1-15: each starts with 1-cycle reset bubble, then 32 batches.
  Per round: 1 + 32×6 = 193 cycles of t-advancement.
- Total t-advancement: 192 + 15×193 = 192 + 2,895 = 3,087

Last batch (#511) starts at t = 3,087 - 6 = 3,081.
Its store is scheduled at cycle 3,081 + 6 + 18 = 3,105 (load_steps + compute_steps).

**Pipeline span: 3,106 cycles** (cycles 0 through 3,105 inclusive)

### Final pause

1 cycle.

### Total

```
Prologue:       ~17 cycles
Pipeline:     3,106 cycles
Pause:            1 cycle
─────────────────────────
Total:       ~3,124 cycles
```

Speedup: 147,734 / 3,124 ≈ **47x over baseline**
Speedup: 13,347 / 3,124 ≈ **4.3x over current sequential**

### Sanity check

Steady-state throughput: 1 batch per 6 cycles.
Ideal: 512 batches × 6 = 3,072 cycles (no overhead).
Overhead: 3,124 - 3,072 = 52 cycles (prologue 17 + fill 19 + 15 resets + pause 1 = 52). ✓

```bash
uv run python perf_takehome.py Tests.test_kernel_cycles
uv run python tests/submission_tests.py
```

---

# Beyond-3,124 Optimization Strategies

Target: ~1,400-1,800 cycles (~2x further improvement over period-6 pipeline).

The fundamental bottleneck: 512 batches × 8 gather loads each = 4,096 gathers total.
At 2 loads/cycle, that's a **2,048-cycle floor** just for gathers. Beating this requires
eliminating gathers for some rounds.

## Strategy 1: multiply_add for Hash Stages (save 3 compute cycles/batch)

The `multiply_add(dest, a, b, c)` valu instruction computes `a*b+c` in a single slot.
Three hash stages have the form `(a + const) + (a << k)` = `a * (2^k + 1) + const`:

| Stage | Expression                           | multiply_add form                           |
|-------|--------------------------------------|---------------------------------------------|
| 1     | `(a + 0x7ED55D16) + (a << 12)`      | `multiply_add(v, v, v_4097, v_0x7ED55D16)`  |
| 3     | `(a + 0x165667B1) + (a << 5)`       | `multiply_add(v, v, v_33, v_0x165667B1)`    |
| 5     | `(a + 0xFD7046C5) + (a << 3)`       | `multiply_add(v, v, v_9, v_0xFD7046C5)`     |

Stages 2, 4, 6 use XOR as the combining op → can't use multiply_add, stay at 2 cycles.

**Hash: 12 → 9 cycles. Total compute: 18 → 15 cycles. Batch latency: 25 → 22.**

New constants needed: v_4097, v_33, v_9 (3 scalar + 3 vconst allocations).
N_BUFS drops from 5 to 4 (ceil(22/6)). Period stays 6 (load-limited, unchanged).

**Impact on total: ~3,124 → ~3,106** (only saves drain cycles — modest alone).
Real value: enables tighter pipelining when combined with strategies below.

## Strategy 2: Scratch-Resident idx/val (eliminate vload/vstore per batch)

Keep all 32 batch vectors (v_idx and v_val) permanently in scratch across all 16 rounds.
No vload/vstore per batch — only a bulk load at program start and bulk store at the end.

**Scratch budget:**
```
32 batches × VLEN × 2 (idx+val) = 512 words  (persistent state)
Pipeline temps: 4 bufs × 48 words            = 192 words  (v_node_val, v_tmp1/2/3, v_tmp4/5)
Hash vconsts: ~12 unique values × VLEN        =  96 words
Scalar consts + misc                          =  50 words
────────────────────────────────────────────────────────
Total                                         ≈ 850 of 1536 ✓
```

**What this eliminates per batch:**
- L0 step (2 vloads + 4 ALU for address mgmt) → gone
- S0 step (2 vstores) → gone
- Per-buffer `st_idx_addr`/`st_val_addr` → gone
- Round reset overhead → gone (no address pointers)

**New load phase:** just address computation + 4 gather cycles = 4–5 cycles.
**New batch structure:** 5 load + 15 compute = 20 cycles (or 4+15=19 with addr overlap).

**Period drops from 6 to 4–5** (gather-limited: 8 loads @ 2/cycle = 4 cycles min).

**New overhead:**
- Prologue bulk load: 32 batches × 2 vloads = 64 vloads @ 2/cycle = 32 cycles
- Epilogue bulk store: 32 batches × 2 vstores = 64 vstores @ 2/cycle = 32 cycles

**Estimated total with period=4:** 512 × 4 + 15 (drain) + 32 (load) + 32 (store) + 17 (prologue) + 1 (pause) ≈ **2,145 cycles**

## Strategy 3: Merge Address Computation (period 5 → 4)

With scratch-resident data, the load phase becomes:
- L0: compute 8 gather addresses (8 ALU: `forest_values_p + v_idx[i]`) — load:0
- L1–L4: gather 2 scalars each — load:2

Period=5 is safe (only one batch in load phase at a time), but L0 uses 0 loads — wasted.

**Optimization:** Merge L0's address computation into the previous batch's last compute
cycle. The previous batch writes new `v_idx` values, and the *next* batch reads from
different scratch addresses → no conflict.

This eliminates L0 entirely. Load phase = 4 cycles (pure gather). **Period = 4.**

**Estimated total with period=4:** 512 × 4 + 15 + 64 + 17 + 1 ≈ **2,145 cycles**

## Strategy 4: VALU-Based Wrap (free the flow engine)

Current wrap uses `flow("vselect", ...)` — the only flow slot per cycle.
Replace with pure VALU arithmetic:

```
idx_new = idx * (idx < n_nodes)    # if idx >= n_nodes, zero it out
```

Two VALU ops:
1. `valu("<", v_tmp, v_idx, v_n_nodes)`  → produces 0 or 1
2. `valu("*", v_idx, v_idx, v_tmp)`      → zeroes out-of-bounds indices

**Benefit:** Frees the flow engine slot for loop control (`cond_jump`/`jump`).
Compute stays at 15 cycles (swaps 1 flow for 1 valu — same count, different engine).

## Strategy 5: Loop Instructions (compact code, enable branching)

Use `flow("cond_jump", ...)` and `flow("jump", ...)` instead of unrolling 512 batches.
Each loop iteration = one pipeline period (4–6 cycles depending on strategy).

**Does NOT reduce cycle count** — same number of cycles executed. But:
- Instruction count drops from ~12,800 to ~100 (fits in instruction cache)
- Enables **round-aware branching** (Strategy 6)
- Counter management: `flow("add_imm", counter, counter, 1)` per iteration

Loop structure:
```
prologue: load consts, bulk-load idx/val
loop_start:
    [one pipeline period of interleaved load/compute]
    add_imm loop_counter, loop_counter, 1
    cond_jump (counter < 512), loop_start
epilogue: bulk-store idx/val, pause
```

## Strategy 6: Round Specialization (avoid gathers for coherent rounds)

Tree structure with height=10 (2047 nodes): indices at level k span [2^k - 1, 2^(k+1) - 2].

All 256 elements start at index 0. After round k, elements are scattered across level k.
After round 10 (tree height), indices wrap back to 0 via the `idx >= n_nodes → idx = 0` check.

**Round-to-level mapping:**
```
Round  0: level  0 — 1 unique node (idx=0)        → broadcast
Round  1: level  1 — 2 unique nodes (idx 1,2)      → 2 loads + vselect
Round  2: level  2 — 4 nodes (idx 3–6)             → 4 loads + 2 vselects
Round  3: level  3 — 8 nodes (idx 7–14)            → 1 vload + vselect chain OR gather
...
Round  9: level  9 — 512 nodes                     → full gather
Round 10: level 10 — wraps to 0                    → broadcast (all back to root)
Round 11: level  1 — same as round 1               → 2 loads + vselect
...
Round 15: level  5 — 32 nodes                      → gather
```

**Gather-free rounds** (0, 1, 10, 11): 4 rounds × 32 batches = 128 batches.
At period ≈ 2–3 (compute-limited, no gather), saves ~128–256 cycles.

**Partially optimized** (2, 12): 2 rounds × 32 = 64 batches with 4 loads instead of 8.

**Hard rounds** (3–9, 13–15): 10 rounds × 32 = 320 batches at full period=4.

**Estimated total with round specialization:**
```
Gather-free batches:  128 × 3  =   384 cycles
Partial batches:       64 × 3  =   192 cycles
Full-gather batches:  320 × 4  = 1,280 cycles
Drain:                            +  15 cycles
Prologue + bulk load/store:       +  82 cycles
──────────────────────────────────────────────
Total:                           ≈ 1,953 cycles
```

With aggressive overlap and tighter scheduling: **~1,700–1,800 cycles**.

## Strategy 7: Scalar ALU Offloading (reduce valu pressure)

The 12-slot scalar ALU is underutilized during compute. XOR hash stages (2, 4, 6) each
need 3 vector ops across 2 cycles. Can move to 8 scalar ops per element:

```
Cycle 1: 8× alu("^", a_i, a_i, const) + 4× alu(">>", tmp_i, a_i, shift)  = 12 ALU
Cycle 2: 4× alu(">>", tmp_i, a_i, shift) + 8× alu("^", a_i, val_i, tmp_i) = 12 ALU
```

Same cycle count per stage, but frees VALU slots for tighter pipelining at lower periods.
**Useful when valu contention prevents period < 4.**

## Priority Order

| # | Strategy | Estimated Savings | Complexity | Dependencies |
|---|----------|-------------------|------------|-------------|
| 1 | multiply_add hash | 3 cycles/batch latency | Low | None |
| 2 | Scratch-resident idx/val | Period 6→5, eliminate vload/vstore | Medium | None |
| 3 | Merge addr computation | Period 5→4 | Low | Needs #2 |
| 4 | VALU wrap | Frees flow engine | Low | None |
| 5 | Loop instructions | Enables branching | Medium | Needs #4 |
| 6 | Round specialization | ~200–500 cycles | High | Needs #5 |
| 7 | Scalar ALU offload | Enables period 3 | Medium | Needs #1,#2 |

**Recommended implementation order:** 1 → 2 → 3 → 4 → 5 → 6

## Theoretical Floor

- 4,096 total gathers, minus ~640 for gather-free rounds = 3,456 gathers
- At 2/cycle: 1,728 cycles just for loads
- Plus compute overhead: ~200 cycles
- Plus prologue/epilogue: ~100 cycles
- **Theoretical minimum: ~2,028 cycles** (gather-limited)

To beat this, need to eliminate more gathers (deeper round specialization, caching top
tree levels in scratch) or find ways to reduce gather count per batch.
