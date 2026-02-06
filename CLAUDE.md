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
  - `build_hash(b, ...)`: Adds 6-stage hash instructions to a given bundle `b`
  - `alloc_scratch()`: Manually allocates scratch memory; vector vars need `length=VLEN`
  - `scratch_const()`: Allocates and deduplicates constants via `const_map`

### Computation
For each round, for each batch element: load tree node at current index, XOR with value, hash the result, traverse left/right child based on hash parity, wrap at tree bounds. The hash function (`myhash`) is a 6-stage 32-bit operation using add, XOR, and shift.

### Memory Layout (`build_mem_image`)
Header (7 words): rounds, n_nodes, batch_size, forest_height, forest_values_p, inp_indices_p, inp_values_p. Followed by tree values, then input indices, then input values.

## Current State

Work in progress: migrating from scalar to SIMD (VLEN=8) vectorized kernel.

### Completed (lines 181-227)
- **vload for indices and values** — 8 contiguous elements per load
- **Gather for node_val** — self-overwriting scalar loads (see technique below)
- **XOR** — `valu("^", v_val, v_val, v_node_val)`
- **Hash** — `build_hash` now uses `valu` ops (but still passes scalar `tmp1`, `tmp2` — needs vector temps)

### Helper functions defined
- `gather_node_val(b, v_node_val, pair)` — adds 2 scalar loads per call (load engine limit)
- `load_and_compute_next_addr(b, load_dest, addr_reg, next_base, offset)` — overlaps vload with next address computation

### Self-overwriting trick (gather)
Uses same scratch slots for address and result: `load v_node_val[j], v_node_val[j]`
- Reads use `core.scratch` (old value = address)
- Writes go to `scratch_write` (applied at end of cycle = loaded value)
- 8 ALU address ops merged with v_val vload (free cycle)
- 4 bundles × 2 loads = 4 cycles for gather

### Next: Vectorize remaining code (lines 229-260)
Remaining scalar code references undefined `tmp_val`, `tmp_idx`, `tmp_addr`. Need to convert:
- **Branch decision** (`% 2`, select): use `valu` + `flow("vselect", ...)`
- **Wrap check** (`idx >= n_nodes`): use `valu` + `flow("vselect", ...)`
- **Store back**: use `store("vstore", addr, data)` — addr is scalar, data is vector
- **Constants for valu**: need vector versions via `valu("vbroadcast", v_const, scalar_const)`

### Future: Software pipelining
Batch iterations within a round are independent — can overlap:
- **Load phase** (vloads + gather) uses load engine, ALU mostly idle
- **Compute phase** (hash + branch + wrap) uses valu, load engine idle
- Pipeline: load batch `i+1` while computing batch `i`
- Requires double-buffered scratch vectors (`v_idx_A`/`v_idx_B`, etc.)

## Critical Rules

- **Never modify files in `tests/`** — the submission tests use a frozen copy of the simulator and verify test integrity
- Only edit `perf_takehome.py` for optimizations
- Correctness is validated against reference kernels in `problem.py`; output must match exactly
- Performance is measured in simulator cycles, not wall-clock time
