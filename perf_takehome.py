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
        """Pipeline buffer: working scratch only. v_idx/v_val are set dynamically
        per batch to point at the permanent scratch-resident arrays."""
        p = f"b{buf_id}_"
        return {
            "v_node_val": self.alloc_scratch(p + "v_node_val", VLEN),
            "v_tmp1": self.alloc_scratch(p + "v_tmp1", VLEN),
            "v_tmp2": self.alloc_scratch(p + "v_tmp2", VLEN),
            "v_tmp3": self.alloc_scratch(p + "v_tmp3", VLEN),
        }


    def do_load(self, b, buf, step, global_step):
        """
        LOAD stage - 5 cycles (scratch-resident: no vload/vstore needed)

        Step 0: compute gather addresses (8 alu) + debug
        Steps 1-4: gather node values (2 loads/cycle)
        """
        if step == 0:
            self.debug_vcompare(b, buf["v_idx"], "idx", global_step)
            self.debug_vcompare(b, buf["v_val"], "val", global_step)
            # compute node_val gather addresses: forest_values_p + idx[i]
            for i in range(VLEN):
                b.alu("+", buf["v_node_val"] + i, self.scratch["forest_values_p"], buf["v_idx"] + i)
        else:
            # gather 2 node values per cycle
            i = (step - 1) * 2
            b.load("load", buf["v_node_val"] + i, buf["v_node_val"] + i)
            b.load("load", buf["v_node_val"] + i + 1, buf["v_node_val"] + i + 1)


    def do_compute(self, b, buf, step, global_step):
        if step == 0:
            # val = myhash(val ^ node_val)
            self.debug_vcompare(b, buf["v_node_val"], "node_val", global_step)
            b.valu("^", buf["v_val"], buf["v_val"], buf["v_node_val"])

        elif step < 10:
            # 9 hash steps: fuseable stages use multiply_add (1 step),
            # non-fuseable use two parallel ops then combine (2 steps)
            action, stage_idx = self.hash_schedule[step - 1]
            op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
            if action == "fused":
                mul_const = (1 << val3) + 1
                b.valu("multiply_add", buf["v_val"], buf["v_val"], self.get_vconst(mul_const), self.get_vconst(val1))
            elif action == "part0":
                b.valu(op1, buf["v_tmp1"], buf["v_val"], self.get_vconst(val1))
                b.valu(op3, buf["v_tmp2"], buf["v_val"], self.get_vconst(val3))
            elif action == "part1":
                b.valu(op2, buf["v_val"], buf["v_tmp1"], buf["v_tmp2"])

        # idx = 2*idx + ((val % 2) + 1)  -- note: 0→1, 1→2 replaces eq+vselect
        elif step == 10:
            self.debug_vcompare(b, buf["v_val"], "hashed_val", global_step)
            b.valu("%", buf["v_tmp1"], buf["v_val"], self.get_vconst(2))
            b.valu("*", buf["v_idx"], buf["v_idx"], self.get_vconst(2))
        elif step == 11:
            b.valu("+", buf["v_tmp1"], buf["v_tmp1"], self.get_vconst(1))
        elif step == 12:
            b.valu("+", buf["v_idx"], buf["v_idx"], buf["v_tmp1"])
        # idx = 0 if idx >= n_nodes else idx
        elif step == 13:
            self.debug_vcompare(b, buf["v_idx"], "next_idx", global_step)
            b.valu("<", buf["v_tmp1"], buf["v_idx"], self.get_vconst(self.n_nodes))
        elif step == 14:
            b.flow("vselect", buf["v_idx"], buf["v_tmp1"], buf["v_idx"], self.get_vconst(0))

    def build_schedule(self, batch_size: int, rounds: int):
        LOAD_STEPS = 5   # addr compute (8 alu) + 4 gather cycles (2 load each)
        COMPUTE_STEPS = 15  # XOR + 9 hash + 5 branch/wrap
        TOTAL_STEPS = LOAD_STEPS + COMPUTE_STEPS  # 20 cycles per batch (no store)
        PERIOD = 4   # gather-limited: 4 load steps use 2/2 slots each
        N_BUFS = 5   # ceil(20/4) = 5 buffers in flight

        bufs = [self.alloc_buffer(i) for i in range(N_BUFS)]

        batches_per_round = batch_size // VLEN
        total_batches = rounds * batches_per_round

        # Each batch occupies 20 cycles, staggered by PERIOD=4.
        # No resets needed: idx/val live in scratch across all rounds.
        #
        #  cycle:  0  1  2  3  4  5  6  7  8  9 ...
        #  bat 0: L0 L1 L2 L3 L4 C0 C1 C2 C3 C4 ...
        #  bat 1:             L0 L1 L2 L3 L4 C0 C1 ...
        #
        # Steady state (5 in-flight), resources per offset:
        # off | steps                     | load(2) | alu(12) | valu(6) | flow(1)
        #  0  | L0, L4, C3, C7, C11       |    2    |    8    |    3    |    0
        #  1  | L1, C0, C4, C8, C12       |    2    |    0    |    5    |    0
        #  2  | L2, C1, C5, C9, C13       |    2    |    0    |    5    |    0
        #  3  | L3, C2, C6, C10, C14      |    2    |    0    |    5    |    1
        schedule = defaultdict(list)
        t = 0

        for batch in range(total_batches):
            buf_idx = batch % N_BUFS
            round_num = batch // batches_per_round
            batch_in_round = batch % batches_per_round
            gs = round_num * batch_size + batch_in_round * VLEN

            for s in range(LOAD_STEPS):
                schedule[t + s].append(("load", buf_idx, s, gs, batch_in_round))
            for s in range(COMPUTE_STEPS):
                schedule[t + LOAD_STEPS + s].append(("compute", buf_idx, s, gs, batch_in_round))

            t += PERIOD
        return bufs, schedule


    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized pipelined kernel with scratch-resident idx/val.

        All 32 batch vectors live permanently in scratch across rounds.
        Bulk vload at start, pipelined compute (period-4), bulk vstore at end.

        Per-batch pipeline: 5 load + 15 compute = 20 cycles latency.
        Period 4, 5 buffers in flight. No per-batch store or round resets.

        Engine limits: alu(12), valu(6), load(2), store(2), flow(1), debug(64)
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
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_consts.add(val1)
            if op2 == "+" and op3 == "<<":
                # Fuseable: (a + val1) + (a << val3) = a * (2^val3 + 1) + val1
                hash_consts.add((1 << val3) + 1)
            else:
                hash_consts.add(val3)
        hash_consts = list(hash_consts)

        # Build hash schedule: fuseable stages take 1 step, others take 2
        self.hash_schedule = []
        for stage_idx, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op2 == "+" and op3 == "<<":
                self.hash_schedule.append(("fused", stage_idx))
            else:
                self.hash_schedule.append(("part0", stage_idx))
                self.hash_schedule.append(("part1", stage_idx))

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

        self.batch_size = batch_size
        self.n_nodes = n_nodes
        batches_per_round = batch_size // VLEN  # 32

        # Permanent scratch-resident arrays for all batch elements.
        # These persist across all 16 rounds — no per-batch vload/vstore needed.
        perm_idx = [self.alloc_scratch(f"perm_idx_{i}", VLEN) for i in range(batches_per_round)]
        perm_val = [self.alloc_scratch(f"perm_val_{i}", VLEN) for i in range(batches_per_round)]

        # Bulk load all idx/val from memory into scratch (32 cycles, 2 vloads/cycle)
        bulk_idx_addr = self.alloc_scratch("bulk_idx_addr")
        bulk_val_addr = self.alloc_scratch("bulk_val_addr")
        with self.bundle() as b:
            b.alu("+", bulk_idx_addr, self.scratch["inp_indices_p"], self.get_const(0))
            b.alu("+", bulk_val_addr, self.scratch["inp_values_p"], self.get_const(0))
        for i in range(batches_per_round):
            with self.bundle() as b:
                b.load("vload", perm_idx[i], bulk_idx_addr)
                b.load("vload", perm_val[i], bulk_val_addr)
                b.alu("+", bulk_idx_addr, bulk_idx_addr, self.get_const(VLEN))
                b.alu("+", bulk_val_addr, bulk_val_addr, self.get_const(VLEN))

        # Build and emit pipeline schedule
        bufs, schedule = self.build_schedule(batch_size, rounds)

        for cy in sorted(schedule.keys()):
            with self.bundle() as b:
                for op in schedule[cy]:
                    match op:
                        case ("load", buf_idx, step, gs, bir):
                            bufs[buf_idx]["v_idx"] = perm_idx[bir]
                            bufs[buf_idx]["v_val"] = perm_val[bir]
                            self.do_load(b, bufs[buf_idx], step, gs)
                        case ("compute", buf_idx, step, gs, bir):
                            bufs[buf_idx]["v_idx"] = perm_idx[bir]
                            bufs[buf_idx]["v_val"] = perm_val[bir]
                            self.do_compute(b, bufs[buf_idx], step, gs)

        # Bulk store val (and idx) back to memory (32 cycles, 2 vstores/cycle)
        with self.bundle() as b:
            b.alu("+", bulk_idx_addr, self.scratch["inp_indices_p"], self.get_const(0))
            b.alu("+", bulk_val_addr, self.scratch["inp_values_p"], self.get_const(0))
        for i in range(batches_per_round):
            with self.bundle() as b:
                b.store("vstore", bulk_idx_addr, perm_idx[i])
                b.store("vstore", bulk_val_addr, perm_val[i])
                b.alu("+", bulk_idx_addr, bulk_idx_addr, self.get_const(VLEN))
                b.alu("+", bulk_val_addr, bulk_val_addr, self.get_const(VLEN))

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
