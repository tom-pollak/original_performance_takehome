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

    def debug_vcompare(self, b, v, label, global_step):
        round = global_step // self.batch_size
        i = global_step % self.batch_size
        b.debug("vcompare", v, [(round, j, label) for j in range(i, i + VLEN)])

    def alloc_buffer(self, buf_id):
        p = f"b{buf_id}_"
        return {
            "v_idx": self.alloc_scratch(p + "v_idx", VLEN),
            "v_val": self.alloc_scratch(p + "v_val", VLEN),
            "v_node_val": self.alloc_scratch(p + "v_node_val", VLEN),
            "v_tmp1": self.alloc_scratch(p + "v_tmp1", VLEN),
            "v_tmp2": self.alloc_scratch(p + "v_tmp2", VLEN),
            "v_tmp3": self.alloc_scratch(p + "v_tmp3", VLEN),
            "st_idx_addr": self.alloc_scratch(p + "st_idx_addr"),
            "st_val_addr": self.alloc_scratch(p + "st_val_addr"),
        }


    def do_load(self, b, buf, step, global_step, idx_addr, val_addr):
        """
        LOAD stage - 6 cycles
        ---
        vload v_idx, v_val + save addresses & advance global pointers for next cycle
        alu compute node_val gather addresses (8/12 slots)
        4x load cycles gathering node_val
        ---
        """
        if step == 0:
            # load v_idx, v_val
            b.load("vload", buf["v_idx"], idx_addr)
            b.load("vload", buf["v_val"], val_addr)
            ## operates on old data
            # save store addresses
            b.alu("+", buf["st_idx_addr"], idx_addr, self.get_const(0))
            b.alu("+", buf["st_val_addr"], val_addr, self.get_const(0))
            # advance global pointers
            b.alu("+", idx_addr, idx_addr, self.get_const(VLEN))
            b.alu("+", val_addr, val_addr, self.get_const(VLEN))

        elif step == 1:
            self.debug_vcompare(b, buf["v_idx"], "idx", global_step)
            self.debug_vcompare(b, buf["v_val"], "val", global_step)

            # compute node_val addr
            for i in range(VLEN):
                b.alu("+", buf["v_node_val"] + i, self.scratch["forest_values_p"], buf["v_idx"] + i)

        else:
            i = (step - 2) * 2 # compute our current gather pair
            # 2 loads per cycle, self-overwriting address with loaded value
            # Load addr at v_node_val and store it at same location
            b.load("load", buf["v_node_val"] + i, buf["v_node_val"] + i)
            b.load("load", buf["v_node_val"] + i + 1, buf["v_node_val"] + i + 1)


    def do_compute(self, b, buf, step, global_step):
        if step == 0:
            # val = myhash(val ^ node_val)
            self.debug_vcompare(b, buf["v_node_val"], "node_val", global_step)
            b.valu("*", buf["v_idx"], buf["v_idx"], self.get_vconst(2))
            b.valu("^", buf["v_val"], buf["v_val"], buf["v_node_val"])

        elif step < 13:
            i = (step - 1)
            stage = i // 2
            part = i % 2
            if part == 0:
                op1, val1, op2, op3, val3 = HASH_STAGES[stage]
                b.valu(op1, buf["v_tmp1"], buf["v_val"], self.get_vconst(val1))
                b.valu(op3, buf["v_tmp2"], buf["v_val"], self.get_vconst(val3))
            else:
                op1, val1, op2, op3, val3 = HASH_STAGES[stage]
                b.valu(op2, buf["v_val"], buf["v_tmp1"], buf["v_tmp2"])

        # idx = 2*idx + (1 if val % 2 == 0 else 2)
        elif step == 13:
            self.debug_vcompare(b, buf["v_val"], "hashed_val", global_step)
            b.valu("%", buf["v_tmp1"], buf["v_val"], self.get_vconst(2))
        elif step == 14:
            b.valu("==", buf["v_tmp1"], buf["v_tmp1"], self.get_vconst(0))
        elif step == 15:
            b.flow("vselect", buf["v_tmp3"], buf["v_tmp1"], self.get_vconst(1), self.get_vconst(2))
        elif step == 16:
            b.valu("+", buf["v_idx"], buf["v_idx"], buf["v_tmp3"])
        # idx = 0 if idx >= n_nodes else idx
        elif step == 17:
            self.debug_vcompare(b, buf["v_idx"], "next_idx", global_step)
            b.valu("<", buf["v_tmp1"], buf["v_idx"], self.get_vconst(self.n_nodes))
        elif step == 18:
            b.flow("vselect", buf["v_idx"], buf["v_tmp1"], buf["v_idx"], self.get_vconst(0))

    def do_store(self, b, buf, step, global_step):
        if step == 0:
            b.debug(b, buf["v_idx"], "wrapped_idx", global_step)
            b.store("vstore", buf["st_idx_addr"], buf["v_idx"])
        else:
            b.store("vstore", buf["st_val_addr"], buf["v_val"])

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

        # Single pipeline buffer - scale to N_BUFS
        buf = self.alloc_buffer(0)

        self.batch_size = batch_size
        self.n_nodes = n_nodes

        for round in range(rounds):
            # Reset base addresses for this round
            with self.bundle() as b:
                b.alu("+", idx_addr, self.scratch["inp_indices_p"], self.get_const(0))
                b.alu("+", val_addr, self.scratch["inp_values_p"], self.get_const(0))

            for i in range(0, batch_size, VLEN):
                global_step = round * batch_size + i

                # LOAD
                for step in range(6):
                    # vload, 8alu, 4x [2load]
                    with self.bundle() as b:
                        self.do_load(b, buf, step, global_step, idx_addr, val_addr)

                # COMPUTE
                for step in range(19):
                    # 2valu
                    # 6x [2valu, valu]
                    # valu, valu, flow, valu, valu, flow
                    with self.bundle() as b:
                        self.do_compute(b, buf, step, global_step)

                # STORE
                for step in range(2):
                    # store, store
                    with self.bundle() as b:
                        self.do_store(b, buf, step, global_step)

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
