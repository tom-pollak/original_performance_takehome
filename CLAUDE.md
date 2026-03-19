# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Process

Go slow. Explain one concept at a time. Make targeted, incremental changes. Do not dump large blocks of code or multiple optimizations at once. Walk the user through each step, making sure they understand before moving on. When the user asks a question, answer just that question concisely. Let the user drive the pace.

## Project Overview

Anthropic's original performance take-home: optimize a kernel running on a custom VLIW SIMD machine simulator. The goal is to minimize the cycle count of a tree-traversal computation. The baseline is 147,734 cycles; competitive solutions reach ~1,400 cycles.

## Commands

```bash
# Run the main performance test (reports cycle count)
uv run python perf_takehome.py Tests.test_kernel_cycles

# Run with execution trace output (generates trace.json)
uv run python perf_takehome.py Tests.test_kernel_trace

# Run official submission tests (correctness + performance tiers)
uv run python tests/submission_tests.py

# Visualize execution trace in Perfetto UI (run after generating trace.json)
uv run python watch_trace.py

# Validate tests haven't been modified
git diff origin/main tests/
```

## Architecture

### Simulator (`problem.py`)
A VLIW SIMD machine simulator with these key properties:
- **Engines**: alu (12 slots), valu (6 slots), load (2 slots), store (2 slots), flow (1 slot), debug (64 slots)
- **VLEN=8**: Vector instructions operate on 8 elements
- **N_CORES=1**: Multicore is intentionally disabled; do not change this
- **SCRATCH_SIZE=1536**: Fast scratch memory (register file), 32-bit words
- **Memory**: Flat address space containing tree data and input/output arrays
- Instructions execute in parallel across engines within each cycle (VLIW)
- Effects of instructions don't take effect until end of cycle (writes visible next cycle)

### Kernel (`perf_takehome.py`)
- **Bundle**: Helper class for building VLIW instruction bundles. Each method (`alu`, `valu`, `load`, `store`, `flow`, `debug`) adds a slot to the current bundle.
- **KernelBuilder**: Generates instruction sequences for the simulator
  - `bundle()`: Context manager that creates a Bundle, yields it, and appends the resulting instruction dict to `self.instrs` on exit. Each `with self.bundle() as b:` block = one cycle.
  - `add(engine, slot)`: Shorthand for emitting a single-slot bundle (used during init/setup)
  - `build_kernel()`: Main entry point that constructs the computation loop
  - `alloc_scratch()`: Manually allocates scratch memory; vector vars need `length=VLEN`
  - `alloc_const(b, val)` / `alloc_vconst(b, val)`: Allocate and deduplicate scalar/vector constants

### Computation
For each round, for each batch element: load tree node at current index, XOR with value, hash the result, traverse left/right child based on hash parity, wrap at tree bounds. The hash function (`myhash`) is a 6-stage 32-bit operation using add, XOR, and shift.

### Memory Layout (`build_mem_image`)
Header (7 words): rounds, n_nodes, batch_size, forest_height, forest_values_p, inp_indices_p, inp_values_p. Followed by tree values, then input indices, then input values.

## Current State

Fully vectorized sequential kernel: **12,835 cycles** (11.5x over baseline).

### What's implemented
- `alloc_buffer(buf_id)` — per-buffer scratch: v_idx, v_val, v_node_val, v_tmp1/2/3, st_idx/val_addr
- `do_load(b, buf, step, global_step, idx_addr, val_addr)` — 6 steps (vload + 8 ALU addr + 4×2 gather)
- `do_compute(b, buf, step, global_step)` — 18 steps (XOR, 6-stage hash, mod/mul/add branching, vselect wrap)
- `do_store(b, buf, step, global_step)` — 1 step, store:2 (merged idx+val vstore)
- `(val%2)+1` trick — eliminates eq+vselect, saves 1 compute step
- Self-overwriting gather: `load v_node_val[i], v_node_val[i]` (address in = value out)
- Sequential loop: 1 buffer, 16 rounds × 32 batches, 25 cycles/batch

### Next: Software pipelining → R=2 fusing
See `ROADMAP.md` for the 4-step implementation plan targeting ~1,360 cycles.

## Critical Rules

- **Never modify files in `tests/`** — the submission tests use a frozen copy of the simulator and verify test integrity
- Only edit `perf_takehome.py` for optimizations
- Correctness is validated against reference kernels in `problem.py`; output must match exactly
- Performance is measured in simulator cycles, not wall-clock time
