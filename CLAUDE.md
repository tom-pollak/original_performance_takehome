# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### Kernel (`perf_takehome.py`)
- **KernelBuilder**: Generates instruction sequences for the simulator
- `build_kernel()`: Main entry point that constructs the computation loop
- `build_hash()`: Generates the 6-stage hash function instructions
- Scratch memory is manually allocated via `KernelBuilder.alloc()`
- Constants are deduplicated via `const_map`

### Computation
For each round, for each batch element: load tree node at current index, XOR with value, hash the result, traverse left/right child based on hash parity, wrap at tree bounds. The hash function (`myhash`) is a 6-stage 32-bit operation using multiply, XOR, and shift.

### Memory Layout (`build_mem_image`)
Header (6 words): n_rounds, batch_size, tree_depth, tree_offset, idx_offset, val_offset, followed by tree values, then input indices, then input values.

## Critical Rules

- **Never modify files in `tests/`** — the submission tests use a frozen copy of the simulator and verify test integrity
- Only edit `perf_takehome.py` for optimizations
- Correctness is validated against reference kernels in `problem.py`; output must match exactly
- Performance is measured in simulator cycles, not wall-clock time
