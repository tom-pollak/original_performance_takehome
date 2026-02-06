# fmt: off
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

    def alloc_const(self, b, val, name=None):
        """Allocate and load a scalar constant into the given bundle."""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            b.load("const", addr, val)
            self.const_map[val] = addr
        return self.const_map[val]

    def alloc_vconst(self, b, val, name=None):
        """Allocate and broadcast a vector constant into the given bundle."""
        key = ("v", val)
        if key not in self.const_map:
            scalar_addr = self.get_const(val)  # must already exist
            v_addr = self.alloc_scratch(name, VLEN)
            b.valu("vbroadcast", v_addr, scalar_addr)
            self.const_map[key] = v_addr
        return self.const_map[key]

    def get_const(self, val):
        return self.const_map[val]

    def get_vconst(self, val):
        return self.const_map[("v", val)]


    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.

        alu: 12
        valu: 6
        load: 2
        store: 2
        flow: 1
        debug: 64

        """
        def gather_node_val(b, v_node_val, pair):
            """
            2 loads per cycle, self-overwriting address with loaded value
            Load addr at v_node_val and store it at same location
            """
            j = pair * 2
            b.load("load", v_node_val + j, v_node_val + j)
            b.load("load", v_node_val + j + 1, v_node_val + j + 1)

        def load_and_compute_next_addr(b, load_dest, addr_reg, next_base, offset):
            """
            Load from current addr_reg into load_dest (vload),
            then overwrite addr_reg with next_base + offset for next cycle.
            """
            # val = mem[inp_values_p + i]
            # v_addr is written to at the end of the cycle
            b.load("vload", load_dest, addr_reg)
            b.alu("+", addr_reg, next_base, offset)

        def hash_p1(b, stage, v_val, v_tmp1, v_tmp2):
            """
            Hash stage first half: compute temps from val. (2 valu slots)
            Can run in parallel with loads or other work.
            """
            op1, val1, op2, op3, val3 = HASH_STAGES[stage]
            b.valu(op1, v_tmp1, v_val, self.get_vconst(val1))
            b.valu(op3, v_tmp2, v_val, self.get_vconst(val3))

        def hash_p2(b, stage, v_val, v_tmp1, v_tmp2):
            """
            Hash stage second half: combine temps into val. (1 valu slot)
            Depends on p1 completing in previous cycle.
            """
            op1, val1, op2, op3, val3 = HASH_STAGES[stage]
            b.valu(op2, v_val, v_tmp1, v_tmp2)


        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)

        # Constants for init_vars (indices 0-6)
        with self.bundle() as b:
            self.alloc_const(b, 0)
            self.alloc_const(b, 1)
        with self.bundle() as b:
            self.alloc_const(b, 2)
            self.alloc_const(b, 3)
            self.alloc_vconst(b, 0) # alloc vconst for 0,1,2
            self.alloc_vconst(b, 1)
        with self.bundle() as b:
            self.alloc_const(b, 4)
            self.alloc_const(b, 5)
            self.alloc_vconst(b, 2)
        with self.bundle() as b:
            self.alloc_const(b, 6)
            self.alloc_const(b, VLEN)  # for address incrementing

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

        # Constants 0-6 are already allocated from earlier
        # Now pack memory loads, 2 per cycle
        for chunk in range(0, len(init_vars), 2):
            with self.bundle() as b:
                for i, v in enumerate(init_vars[chunk:chunk+2], start=chunk):
                    b.load("load", self.scratch[v], self.get_const(i))

        ### HASH CONSTS
        hash_consts = set()
        for _, val1, _, _, val3 in HASH_STAGES:
            hash_consts.add(val1)
            hash_consts.add(val3)
        hash_consts = list(hash_consts)

        ## To make vconsts, we must first load them as consts
        for chunk in range(0, len(hash_consts), 2): # 2 loads per cycle
            with self.bundle() as b:
                for val in hash_consts[chunk:chunk+2]:
                    self.alloc_const(b, val)

        for chunk in range(0, len(hash_consts), 6): # 6 valu per cycle
            with self.bundle() as b:
                for val in hash_consts[chunk:chunk+6]:
                    self.alloc_vconst(b, val)

        ### n_nodes as vector const for wrap comparison
        with self.bundle() as b:
            self.alloc_const(b, n_nodes, "n_nodes_const")
        with self.bundle() as b:
            self.alloc_vconst(b, n_nodes, "v_n_nodes")

        with self.bundle() as b:
            # Pause instructions are matched up with yield statements in the reference
            # kernel to let you debug at intermediate steps. The testing harness in this
            # file requires these match up to the reference kernel's yields, but the
            # submission harness ignores them.
            b.flow("pause")
            # Any debug engine instruction is ignored by the submission simulator
            b.debug("comment", "Starting loop")

        # Scratch registers for maintained addresses
        idx_addr = self.alloc_scratch("idx_addr")
        val_addr = self.alloc_scratch("val_addr")
        v_idx = self.alloc_scratch("v_idx", VLEN)  # reserves 8 consecutive scratch slots
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        vlen_const = self.get_const(VLEN)

        for round in range(rounds):
            # Reset base addresses for this round
            with self.bundle() as b:
                b.alu("+", idx_addr, self.scratch["inp_indices_p"], self.get_const(0))
                b.alu("+", val_addr, self.scratch["inp_values_p"], self.get_const(0))

            for i in range(0, batch_size, VLEN): # block=8
                # Load indices from current idx_addr
                with self.bundle() as b:
                    b.load("vload", v_idx, idx_addr)

                with self.bundle() as b:
                    b.debug("vcompare", v_idx, [(round, j, "idx") for j in range(i, i + VLEN)])

                # Compute gather addrs + load values from val_addr
                with self.bundle() as b:
                    for j in range(VLEN):   # ALU engine (8 of 12 slots)
                        b.alu("+", v_node_val + j, self.scratch["forest_values_p"], v_idx + j)
                    b.load("vload", v_val, val_addr)  # load engine

                    b.valu("*", v_idx, v_idx, self.get_vconst(2)) # later used in the hashing

                with self.bundle() as b:
                    b.debug("vcompare", v_val, [(round, j, "val") for j in range(i, i + VLEN)])

                for j in range(VLEN//2):
                    with self.bundle() as b:
                        gather_node_val(b, v_node_val, j)

                with self.bundle() as b:
                    b.debug("vcompare", v_node_val, [(round, j, "node_val") for j in range(i, i + VLEN)])

                # val = myhash(val ^ node_val)
                with self.bundle() as b:
                    b.valu("^", v_val, v_val, v_node_val)

                # Sequential 6-stage hash. 12 cycles total.
                # TODO: Pipeline with loads or interleave batches to use spare slots.
                for stage in range(len(HASH_STAGES)):
                    with self.bundle() as b:
                        hash_p1(b, stage, v_val, v_tmp1, v_tmp2)
                    with self.bundle() as b:
                        hash_p2(b, stage, v_val, v_tmp1, v_tmp2)

                with self.bundle() as b:
                    b.debug("vcompare", v_val, [(round, j, "hashed_val") for j in range(i, i + VLEN)])

                ### UP TO HERE ###

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                with self.bundle() as b:
                    b.valu("%", v_tmp1, v_val, self.get_vconst(2))
                with self.bundle() as b:
                    b.valu("==", v_tmp1, v_tmp1, self.get_vconst(0))
                with self.bundle() as b:
                    b.flow("vselect", v_tmp3, v_tmp1, self.get_vconst(1), self.get_vconst(2))
                with self.bundle() as b:
                    b.valu("+", v_idx, v_idx, v_tmp3)
                with self.bundle() as b:
                    b.debug("vcompare", v_idx, [(round, j, "next_idx") for j in range(i, i + VLEN)])
                # idx = 0 if idx >= n_nodes else idx
                with self.bundle() as b:
                    b.valu("<", v_tmp1, v_idx, self.get_vconst(n_nodes))
                with self.bundle() as b:
                    b.flow("vselect", v_idx, v_tmp1, v_idx, self.get_vconst(0))
                with self.bundle() as b:
                    b.debug("vcompare", v_idx, [(round, j, "wrapped_idx") for j in range(i, i + VLEN)])

                # Store idx + advance idx_addr (self-overwriting: store reads old addr, ALU writes new)
                with self.bundle() as b:
                    b.store("vstore", idx_addr, v_idx)
                    b.alu("+", idx_addr, idx_addr, vlen_const)
                # Store val + advance val_addr
                with self.bundle() as b:
                    b.store("vstore", val_addr, v_val)
                    b.alu("+", val_addr, val_addr, vlen_const)

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
