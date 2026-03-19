# Optimization Roadmap

Current: **12,835 cycles** (sequential, 1 buffer, 25 cycles/batch)
Target: **~1,360 cycles** (R=2 fused pipeline, ~5 cycles/batch initiation)

## The Gather Wall

The fundamental bottleneck is gather loads (loading tree node values per vector lane):

```
512 batches × 8 gathers/batch = 4,096 gathers
Load engine: 2 slots/cycle
Minimum gather time: 2,048 cycles
```

Everything below is about getting as close to this floor as possible.

## Step 1: Period-6 Pipeline (~3,124 cycles)

Overlap load/compute/store across batches. Batches within a round are independent.

- Allocate 5 buffers (ceil(25/6) max in-flight batches)
- Build schedule table at codegen time: `cycle_ops[cycle] = [(phase, buf, step, global_step), ...]`
- Emit one bundle per cycle with interleaved ops from multiple batches
- 1-cycle reset bubble between rounds (rewrite global address pointers)

**Why period 6:** Load steps L0,L2-L5 each use 2/2 load slots. With period 5,
two batches' load phases would overlap → 4 load slots needed, only 2 available.

Engine usage at steady state (all within limits):
```
Offset | load(2) | alu(12) | valu(6) | flow(1) | store(2)
  0    |    2    |    4    |    3    |    0    |    2
  1    |    0    |    8    |    6    |    0    |    0
  2    |    2    |    0    |    3    |    0    |    0
  3    |    2    |    0    |    5    |    0    |    0
  4    |    2    |    0    |    3    |    0    |    0
  5    |    2    |    0    |    4    |    1    |    0
```

Schedule builder (pure Python, runs at codegen time):
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

Emit one bundle per cycle:
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

**Verification:** `uv run python perf_takehome.py Tests.test_kernel_cycles`

## Step 2: multiply_add for Hash (~3,106 cycles)

Three hash stages have the form `(a + const) + (a << k)` = `a * (2^k+1) + const`:

| Stage | multiply_add form                          |
|-------|--------------------------------------------|
| 1     | `multiply_add(v, v, v_4097, v_0x7ED55D16)` |
| 3     | `multiply_add(v, v, v_33, v_0x165667B1)`   |
| 5     | `multiply_add(v, v, v_9, v_0xFD7046C5)`    |

Hash: 12 → 9 cycles. Compute: 18 → 15. Batch latency: 25 → 22.
N_BUFS drops from 5 to 4. Small direct savings but enables tighter pipelining later.

## Step 3: Scratch-Resident idx/val (~2,145 cycles)

Keep all 32 batch vectors (v_idx + v_val) permanently in scratch across all 16 rounds.
Bulk load at start, bulk store at end. No per-batch vload/vstore.

**What this eliminates per batch:**
- L0 step (2 vloads + 4 ALU for address mgmt) → gone
- S0 step (2 vstores) → gone
- Round reset overhead → gone

**New load phase:** address computation + 4 gather cycles = 4-5 cycles.
**Period drops to 4-5** (gather-limited: 8 loads @ 2/cycle = 4 cycles min).

Scratch budget: 512 (persistent idx/val) + 192 (temps) + 96 (vconsts) + 50 (misc) ≈ 850 of 1536.

## Step 4: R=2 Fusing + Greedy Scheduler (~1,360 cycles)

Fuse 2 tree rounds into each pipeline batch: 16 rounds → 8 double-rounds → 256 batches.

Each fused batch:
```
R1 load:    6 cycles  (vload/addr + 4 gather)  ← load engine busy
R1 compute: 15 cycles                           ← load engine IDLE
R2 load:    4 cycles  (gathers only)            ← load engine busy
R2 compute: 15 cycles                           ← load engine IDLE
Store:      1 cycle
Total:      ~41 cycles latency
```

Key insight: load usage is bursty (10 of 41 cycles). Other batches use the load engine
during the 15-cycle compute gaps. A greedy resource-constrained scheduler (codegen time)
places each step at the earliest cycle with free engine slots → average period ~5.

```
256 batches × 5 cycles  = 1,280 (steady-state)
+ drain (~36) + prologue (~17) + resets (~8) + pause (1) ≈ 1,342 cycles
```

With scheduling imperfections: **~1,360 cycles** (matches README's 1,363).

## Why Not R=3/R=4?

Diminishing returns:
- R=2→R=3 saves one more vload/vstore pair, but batch latency grows → more buffers → more scratch
- More overlapping load phases compete for the same compute-gap slots
- Total gathers unchanged (4,096) — the real bottleneck doesn't shrink

## Reference

- `problem.py`: Simulator source, engine limits, instruction semantics
