"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from contextlib import contextmanager
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class Bundle:
    def __init__(self):
        self._slots = defaultdict(list)

    def alu(self, *slot):
        self._slots["alu"].append(slot)

    def valu(self, *slot):
        self._slots["valu"].append(slot)

    def load(self, *slot):
        self._slots["load"].append(slot)

    def store(self, *slot):
        self._slots["store"].append(slot)

    def flow(self, *slot):
        self._slots["flow"].append(slot)

    def debug(self, *slot):
        self._slots["debug"].append(slot)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    @contextmanager
    def bundle(self):
        b = Bundle()
        yield b
        if b._slots:
            self.instrs.append(dict(b._slots))

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, b, val_hash_addr, tmp1, tmp2, round, i):
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            b.alu(op1, tmp1, val_hash_addr, self.scratch_const(val1))
            b.alu(op3, tmp2, val_hash_addr, self.scratch_const(val3))
            b.alu(op2, val_hash_addr, tmp1, tmp2)
            b.debug("compare", val_hash_addr, (round, i, "hash_stage", hi))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # Scalar scratch registers
        v_addr = self.alloc_scratch("v_addr")

        v_idx = self.alloc_scratch("v_idx", VLEN)  # reserves 8 consecutive scratch slots
        v_val = self.alloc_scratch("v_val", VLEN)

        v_node_val = self.alloc_scratch("v_node_val", VLEN)

        for round in range(rounds):
            for i in range(0, batch_size, VLEN): # block=8
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                with self.bundle() as b:
                    b.alu("+", v_addr, self.scratch["inp_indices_p"], i_const)
                with self.bundle() as b:
                    b.load("vload", v_idx, v_addr)
                with self.bundle() as b:
                    b.debug("vcompare", v_idx, [(round, j, "idx") for j in range(i, i + VLEN)])

                # val = mem[inp_values_p + i]
                with self.bundle() as b:
                    b.alu("+", v_addr, self.scratch["inp_values_p"], i_const)

                with self.bundle() as b:
                    # node_val = mem[forest_values_p + idx]
                    for j in range(VLEN):   # ALU engine (8 of 12 slots)
                        b.alu("+", v_node_val + j, self.scratch["forest_values_p"], v_idx + j)

                    b.load("vload", v_val, v_addr)  # load engine

                with self.bundle() as b:
                    b.debug("vcompare", v_val, [(round, j, "val") for j in range(i, i + VLEN)])


                # 2 loads per cycle, self-overwriting address with loaded value
                for j in range(0, VLEN, 2):
                    with self.bundle() as b:
                        b.load("load", v_node_val + j, v_node_val + j)
                        b.load("load", v_node_val + j + 1, v_node_val + j + 1)


                with self.bundle() as b:
                    b.debug("vcompare", v_node_val, [(round, j, "node_val") for j in range(i, i + VLEN)])


                # val = myhash(val ^ node_val)
                with self.bundle() as b:
                    b.valu("^", v_val, v_val, v_node_val)



                with self.bundle() as b:
                    self.build_hash(b, tmp_val, tmp1, tmp2, round, i)


                with self.bundle() as b:
                    b.debug("compare", tmp_val, (round, i, "hashed_val"))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                with self.bundle() as b:
                    b.alu("%", tmp1, tmp_val, two_const)
                with self.bundle() as b:
                    b.alu("==", tmp1, tmp1, zero_const)
                with self.bundle() as b:
                    b.flow("select", tmp3, tmp1, one_const, two_const)
                with self.bundle() as b:
                    b.alu("*", tmp_idx, tmp_idx, two_const)
                with self.bundle() as b:
                    b.alu("+", tmp_idx, tmp_idx, tmp3)
                with self.bundle() as b:
                    b.debug("compare", tmp_idx, (round, i, "next_idx"))
                # idx = 0 if idx >= n_nodes else idx
                with self.bundle() as b:
                    b.alu("<", tmp1, tmp_idx, self.scratch["n_nodes"])
                with self.bundle() as b:
                    b.flow("select", tmp_idx, tmp1, tmp_idx, zero_const)
                with self.bundle() as b:
                    b.debug("compare", tmp_idx, (round, i, "wrapped_idx"))
                # mem[inp_indices_p + i] = idx
                with self.bundle() as b:
                    b.alu("+", tmp_addr, self.scratch["inp_indices_p"], i_const)
                with self.bundle() as b:
                    b.store("store", tmp_addr, tmp_idx)
                # mem[inp_values_p + i] = val
                with self.bundle() as b:
                    b.alu("+", tmp_addr, self.scratch["inp_values_p"], i_const)
                with self.bundle() as b:
                    b.store("store", tmp_addr, tmp_val)

        # Required to match with the yield in reference_kernel2
        with self.bundle() as b:
            b.flow("pause")

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
