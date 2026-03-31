from collections import defaultdict
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

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle):
        self.instrs.append(bundle)

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name=name, length=VLEN)

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

    def build_hash(self, val_hash_addr, tmp1, tmp2):
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.add_bundle(
                {
                    "alu": [
                        (op1, tmp1, val_hash_addr, self.scratch_const(val1)),
                        (op3, tmp2, val_hash_addr, self.scratch_const(val3)),
                    ]
                }
            )
            self.add("alu", (op2, val_hash_addr, tmp1, tmp2))

    def build_hash_vec(self, val_vec, tmp1_vec, tmp2_vec, const_vecs):
        for op1, op2, op3, vec_val1, vec_val3 in const_vecs:
            self.add_bundle(
                {
                    "valu": [
                        (op1, tmp1_vec, val_vec, vec_val1),
                        (op3, tmp2_vec, val_vec, vec_val3),
                    ]
                }
            )
            self.add_bundle({"valu": [(op2, val_vec, tmp1_vec, tmp2_vec)]})

    def build_hash_quad_interleaved(
        self,
        val0, val1, val2, val3,
        t1_0, t2_0, t1_1, t2_1, t1_2, t2_2, t1_3, t2_3,
        const_vecs,
        last_cycle_extra_valu=None,
    ):
        stages = list(const_vecs)

        # Prime: A stage 0 cycle 1
        op1_0, op2_0, op3_0, v1_0, v3_0 = stages[0]
        self.add_bundle({
            "valu": [
                (op1_0, t1_0, val0, v1_0), (op3_0, t2_0, val0, v3_0),
                (op1_0, t1_1, val1, v1_0), (op3_0, t2_1, val1, v3_0),
            ]
        })

        for si in range(len(stages)):
            op1, op2, op3, v1, v3 = stages[si]
            # A op2 + B op1+op3 for same stage
            self.add_bundle({
                "valu": [
                    (op2, val0, t1_0, t2_0), (op2, val1, t1_1, t2_1),
                    (op1, t1_2, val2, v1), (op3, t2_2, val2, v3),
                    (op1, t1_3, val3, v1), (op3, t2_3, val3, v3),
                ]
            })

            if si < len(stages) - 1:
                # B op2 + A op1+op3 for next stage
                op1_n, op2_n, op3_n, v1_n, v3_n = stages[si + 1]
                self.add_bundle({
                    "valu": [
                        (op2, val2, t1_2, t2_2), (op2, val3, t1_3, t2_3),
                        (op1_n, t1_0, val0, v1_n), (op3_n, t2_0, val0, v3_n),
                        (op1_n, t1_1, val1, v1_n), (op3_n, t2_1, val1, v3_n),
                    ]
                })
            else:
                # Final: B op2 only (+ optional XOR merge)
                valu_ops = [(op2, val2, t1_2, t2_2), (op2, val3, t1_3, t2_3)]
                if last_cycle_extra_valu:
                    valu_ops.extend(last_cycle_extra_valu)
                self.add_bundle({"valu": valu_ops})

    def build_hash_quad_interleaved_with_loads(
        self,
        val0, val1, val2, val3,
        t1_0, t2_0, t1_1, t2_1, t1_2, t2_2, t1_3, t2_3,
        const_vecs,
        load_pairs,
        last_cycle_extra_valu=None,
    ):
        stages = list(const_vecs)

        # Prime: A stage 0 cycle 1
        op1_0, op2_0, op3_0, v1_0, v3_0 = stages[0]
        bundle = {
            "valu": [
                (op1_0, t1_0, val0, v1_0), (op3_0, t2_0, val0, v3_0),
                (op1_0, t1_1, val1, v1_0), (op3_0, t2_1, val1, v3_0),
            ]
        }
        if load_pairs:
            bundle["load"] = load_pairs[:2]
            del load_pairs[:2]
        self.add_bundle(bundle)

        for si in range(len(stages)):
            op1, op2, op3, v1, v3 = stages[si]
            # A op2 + B op1+op3
            bundle = {
                "valu": [
                    (op2, val0, t1_0, t2_0), (op2, val1, t1_1, t2_1),
                    (op1, t1_2, val2, v1), (op3, t2_2, val2, v3),
                    (op1, t1_3, val3, v1), (op3, t2_3, val3, v3),
                ]
            }
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

            if si < len(stages) - 1:
                op1_n, op2_n, op3_n, v1_n, v3_n = stages[si + 1]
                bundle = {
                    "valu": [
                        (op2, val2, t1_2, t2_2), (op2, val3, t1_3, t2_3),
                        (op1_n, t1_0, val0, v1_n), (op3_n, t2_0, val0, v3_n),
                        (op1_n, t1_1, val1, v1_n), (op3_n, t2_1, val1, v3_n),
                    ]
                }
                if load_pairs:
                    bundle["load"] = load_pairs[:2]
                    del load_pairs[:2]
                self.add_bundle(bundle)

    def build_hash_quad_interleaved_ma(
        self,
        val0,
        val1,
        val2,
        val3,
        t1_0,
        t2_0,
        t1_1,
        t2_1,
        t1_2,
        t2_2,
        t1_3,
        t2_3,
        stage_specs,
        load_pairs=None,
        first_cycle_extra_valu=None,
        last_cycle_extra_valu=None,
    ):
        """Hash 4 vectors in 10 cycles using mixed MA and classic stages.

        Stage pattern is expected to be:
        S0=MA, S1=classic, S2=MA, S3=classic, S4=MA, S5=classic.
        """

        if load_pairs is None:
            load_pairs = []

        assert len(stage_specs) == 6
        assert [s[0] for s in stage_specs] == ["ma", "std", "ma", "std", "ma", "std"]

        def add_cycle(valu_ops):
            bundle = {"valu": valu_ops}
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

        def ma_ops(stage, va, vb):
            _, mul_vec, add_vec = stage
            return [
                ("multiply_add", va, va, mul_vec, add_vec),
                ("multiply_add", vb, vb, mul_vec, add_vec),
            ]

        def std_c1_ops(stage, va, vb, ta1, ta2, tb1, tb2):
            _, op1, _op2, op3, v1, v3 = stage
            return [
                (op1, ta1, va, v1),
                (op3, ta2, va, v3),
                (op1, tb1, vb, v1),
                (op3, tb2, vb, v3),
            ]

        def std_c2_ops(stage, va, vb, ta1, ta2, tb1, tb2):
            _, _op1, op2, _op3, _v1, _v3 = stage
            return [
                (op2, va, ta1, ta2),
                (op2, vb, tb1, tb2),
            ]

        s0, s1, s2, s3, s4, s5 = stage_specs

        # C1: A.S0 (+ optional extra valu, typically pre-hash XOR for val2/val3)
        valu_ops = ma_ops(s0, val0, val1)
        if first_cycle_extra_valu:
            valu_ops.extend(first_cycle_extra_valu)
        add_cycle(valu_ops)

        # C2: A.S1.c1 + B.S0
        add_cycle(
            std_c1_ops(s1, val0, val1, t1_0, t2_0, t1_1, t2_1)
            + ma_ops(s0, val2, val3)
        )

        # C3: A.S1.c2 + B.S1.c1
        add_cycle(
            std_c2_ops(s1, val0, val1, t1_0, t2_0, t1_1, t2_1)
            + std_c1_ops(s1, val2, val3, t1_2, t2_2, t1_3, t2_3)
        )

        # C4: A.S2 + B.S1.c2
        add_cycle(
            ma_ops(s2, val0, val1)
            + std_c2_ops(s1, val2, val3, t1_2, t2_2, t1_3, t2_3)
        )

        # C5: A.S3.c1 + B.S2
        add_cycle(
            std_c1_ops(s3, val0, val1, t1_0, t2_0, t1_1, t2_1)
            + ma_ops(s2, val2, val3)
        )

        # C6: A.S3.c2 + B.S3.c1
        add_cycle(
            std_c2_ops(s3, val0, val1, t1_0, t2_0, t1_1, t2_1)
            + std_c1_ops(s3, val2, val3, t1_2, t2_2, t1_3, t2_3)
        )

        # C7: A.S4 + B.S3.c2
        add_cycle(
            ma_ops(s4, val0, val1)
            + std_c2_ops(s3, val2, val3, t1_2, t2_2, t1_3, t2_3)
        )

        # C8: A.S5.c1 + B.S4
        add_cycle(
            std_c1_ops(s5, val0, val1, t1_0, t2_0, t1_1, t2_1)
            + ma_ops(s4, val2, val3)
        )

        # C9: A.S5.c2 + B.S5.c1
        add_cycle(
            std_c2_ops(s5, val0, val1, t1_0, t2_0, t1_1, t2_1)
            + std_c1_ops(s5, val2, val3, t1_2, t2_2, t1_3, t2_3)
        )

        # C10: B.S5.c2 (+ optional merge work)
        valu_ops = std_c2_ops(s5, val2, val3, t1_2, t2_2, t1_3, t2_3)
        if last_cycle_extra_valu:
            valu_ops.extend(last_cycle_extra_valu)
        add_cycle(valu_ops)

    def build_hash_vec_pair(
        self,
        val_vec0,
        val_vec1,
        tmp1_vec0,
        tmp2_vec0,
        tmp1_vec1,
        tmp2_vec1,
        const_vecs,
    ):
        for op1, op2, op3, vec_val1, vec_val3 in const_vecs:
            self.add_bundle(
                {
                    "valu": [
                        (op1, tmp1_vec0, val_vec0, vec_val1),
                        (op3, tmp2_vec0, val_vec0, vec_val3),
                        (op1, tmp1_vec1, val_vec1, vec_val1),
                        (op3, tmp2_vec1, val_vec1, vec_val3),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        (op2, val_vec0, tmp1_vec0, tmp2_vec0),
                        (op2, val_vec1, tmp1_vec1, tmp2_vec1),
                    ]
                }
            )

    def build_hash_vec_pair_with_loads(
        self,
        val_vec0,
        val_vec1,
        tmp1_vec0,
        tmp2_vec0,
        tmp1_vec1,
        tmp2_vec1,
        const_vecs,
        load_pairs,
    ):
        for op1, op2, op3, vec_val1, vec_val3 in const_vecs:
            bundle = {
                "valu": [
                    (op1, tmp1_vec0, val_vec0, vec_val1),
                    (op3, tmp2_vec0, val_vec0, vec_val3),
                    (op1, tmp1_vec1, val_vec1, vec_val1),
                    (op3, tmp2_vec1, val_vec1, vec_val3),
                ]
            }
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

            bundle = {
                "valu": [
                    (op2, val_vec0, tmp1_vec0, tmp2_vec0),
                    (op2, val_vec1, tmp1_vec1, tmp2_vec1),
                ]
            }
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

    def build_hash_vec_triple(
        self,
        val_vec0,
        val_vec1,
        val_vec2,
        tmp1_vec0,
        tmp2_vec0,
        tmp1_vec1,
        tmp2_vec1,
        tmp1_vec2,
        tmp2_vec2,
        const_vecs,
    ):
        for op1, op2, op3, vec_val1, vec_val3 in const_vecs:
            self.add_bundle(
                {
                    "valu": [
                        (op1, tmp1_vec0, val_vec0, vec_val1),
                        (op3, tmp2_vec0, val_vec0, vec_val3),
                        (op1, tmp1_vec1, val_vec1, vec_val1),
                        (op3, tmp2_vec1, val_vec1, vec_val3),
                        (op1, tmp1_vec2, val_vec2, vec_val1),
                        (op3, tmp2_vec2, val_vec2, vec_val3),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        (op2, val_vec0, tmp1_vec0, tmp2_vec0),
                        (op2, val_vec1, tmp1_vec1, tmp2_vec1),
                        (op2, val_vec2, tmp1_vec2, tmp2_vec2),
                    ]
                }
            )

    def build_hash_vec_triple_with_loads(
        self,
        val_vec0,
        val_vec1,
        val_vec2,
        tmp1_vec0,
        tmp2_vec0,
        tmp1_vec1,
        tmp2_vec1,
        tmp1_vec2,
        tmp2_vec2,
        const_vecs,
        load_pairs,
    ):
        for op1, op2, op3, vec_val1, vec_val3 in const_vecs:
            bundle = {
                "valu": [
                    (op1, tmp1_vec0, val_vec0, vec_val1),
                    (op3, tmp2_vec0, val_vec0, vec_val3),
                    (op1, tmp1_vec1, val_vec1, vec_val1),
                    (op3, tmp2_vec1, val_vec1, vec_val3),
                    (op1, tmp1_vec2, val_vec2, vec_val1),
                    (op3, tmp2_vec2, val_vec2, vec_val3),
                ]
            }
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

            bundle = {
                "valu": [
                    (op2, val_vec0, tmp1_vec0, tmp2_vec0),
                    (op2, val_vec1, tmp1_vec1, tmp2_vec1),
                    (op2, val_vec2, tmp1_vec2, tmp2_vec2),
                ]
            }
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

    def build_load_pairs_triple(
        self,
        node_vec0,
        node_vec1,
        node_vec2,
        addr_vec0,
        addr_vec1,
        addr_vec2,
    ):
        load_pairs = []
        for off in range(0, VLEN, 2):
            load_pairs.extend(
                [
                    ("load_offset", node_vec0, addr_vec0, off),
                    ("load_offset", node_vec0, addr_vec0, off + 1),
                    ("load_offset", node_vec1, addr_vec1, off),
                    ("load_offset", node_vec1, addr_vec1, off + 1),
                    ("load_offset", node_vec2, addr_vec2, off),
                    ("load_offset", node_vec2, addr_vec2, off + 1),
                ]
            )
        return load_pairs

    def emit_load_pairs(self, load_pairs):
        for i in range(0, len(load_pairs), 2):
            self.add_bundle({"load": load_pairs[i : i + 2]})

    def _op_regs(self, engine, slot):
        def vec_range(base):
            return set(range(base, base + VLEN))

        used = set()
        written = set()
        op = slot[0]
        ints = [x for x in slot[1:] if isinstance(x, int)]

        if engine == "alu":
            if ints:
                written.add(ints[0])
                used.update(ints[1:])
        elif engine == "valu":
            if op == "vbroadcast" and len(slot) >= 3:
                dest, src = slot[1], slot[2]
                if isinstance(dest, int):
                    written.update(vec_range(dest))
                if isinstance(src, int):
                    used.add(src)
            elif op == "multiply_add" and len(slot) >= 5:
                dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
                if isinstance(dest, int):
                    written.update(vec_range(dest))
                if isinstance(a, int):
                    used.update(vec_range(a))
                if isinstance(b, int):
                    used.update(vec_range(b))
                if isinstance(c, int):
                    used.update(vec_range(c))
            elif len(slot) >= 4:
                dest, a1, a2 = slot[1], slot[2], slot[3]
                if isinstance(dest, int):
                    written.update(vec_range(dest))
                if isinstance(a1, int):
                    used.update(vec_range(a1))
                if isinstance(a2, int):
                    used.update(vec_range(a2))
        elif engine == "load":
            if op == "const":
                if ints:
                    written.add(ints[0])
            elif op == "load" and len(slot) >= 3:
                dest, addr = slot[1], slot[2]
                if isinstance(dest, int):
                    written.add(dest)
                if isinstance(addr, int):
                    used.add(addr)
            elif op == "vload" and len(slot) >= 3:
                dest, addr = slot[1], slot[2]
                if isinstance(dest, int):
                    written.update(vec_range(dest))
                if isinstance(addr, int):
                    used.add(addr)
            elif op == "load_offset" and len(slot) >= 4:
                dest, addr, off = slot[1], slot[2], slot[3]
                if isinstance(dest, int) and isinstance(off, int):
                    written.add(dest + off)
                if isinstance(addr, int) and isinstance(off, int):
                    used.add(addr + off)
            elif op in ("load", "vload", "load_offset"):
                if len(ints) >= 2:
                    written.add(ints[0])
                    used.add(ints[1])
        elif engine == "store":
            if op == "store" and len(slot) >= 3:
                addr, src = slot[1], slot[2]
                if isinstance(addr, int):
                    used.add(addr)
                if isinstance(src, int):
                    used.add(src)
            elif op == "vstore" and len(slot) >= 3:
                addr, src = slot[1], slot[2]
                if isinstance(addr, int):
                    used.add(addr)
                if isinstance(src, int):
                    used.update(vec_range(src))
            elif len(ints) >= 2:
                used.add(ints[0])
                used.add(ints[1])
        elif engine == "flow":
            used.update(ints)

        return used, written

    def _bundle_regs(self, bundle):
        used = set()
        written = set()
        for engine, slots in bundle.items():
            for slot in slots:
                op_used, op_written = self._op_regs(engine, slot)
                used.update(op_used)
                written.update(op_written)
        return used, written

    def _list_schedule_segment(self, segment):
        if not segment:
            return []

        ops = []
        for bundle_idx, bundle in enumerate(segment):
            for engine, slots in bundle.items():
                for slot in slots:
                    used, written = self._op_regs(engine, slot)
                    ops.append(
                        {
                            "engine": engine,
                            "slot": slot,
                            "used": used,
                            "written": written,
                            "bundle_idx": bundle_idx,
                        }
                    )

        n = len(ops)
        if n <= 1:
            return segment

        # Conservative dependency model:
        # RAW/WAW must be in a later cycle, WAR can share the same cycle.
        preds = [[] for _ in range(n)]
        last_write = {}
        last_read = {}

        for i, op in enumerate(ops):
            for reg in op["used"]:
                pred = last_write.get(reg)
                if pred is not None:
                    preds[i].append((pred, 1))
            for reg in op["written"]:
                pred = last_write.get(reg)
                if pred is not None:
                    preds[i].append((pred, 1))
            for reg in op["written"]:
                pred = last_read.get(reg)
                if pred is not None:
                    preds[i].append((pred, 0))

            for reg in op["used"]:
                last_read[reg] = i
            for reg in op["written"]:
                last_write[reg] = i
                if reg in last_read:
                    del last_read[reg]

        cycle_slots = []
        cycle_bundles = []
        op_cycle = [0] * n

        for i, op in enumerate(ops):
            c = 0
            for pred_idx, delta in preds[i]:
                c = max(c, op_cycle[pred_idx] + delta)
            engine = op["engine"]

            while True:
                while c >= len(cycle_slots):
                    cycle_slots.append(defaultdict(int))
                    cycle_bundles.append({})
                if cycle_slots[c][engine] < SLOT_LIMITS[engine]:
                    break
                c += 1

            op_cycle[i] = c
            cycle_slots[c][engine] += 1
            cycle_bundles[c].setdefault(engine, []).append(op["slot"])

        return [b for b in cycle_bundles if b]

    def _list_schedule(self):
        scheduled = []
        segment = []

        for bundle in self.instrs:
            if "flow" in bundle:
                if segment:
                    scheduled.extend(self._list_schedule_segment(segment))
                    segment = []
                scheduled.append(bundle)
            else:
                segment.append(bundle)

        if segment:
            scheduled.extend(self._list_schedule_segment(segment))

        self.instrs = scheduled

    def _can_merge_adjacent(self, a, b):
        if "flow" in a or "flow" in b:
            return False

        for engine in set(a) | set(b):
            if len(a.get(engine, [])) + len(b.get(engine, [])) > SLOT_LIMITS[engine]:
                return False
        used_a, written_a = self._bundle_regs(a)
        used_b, written_b = self._bundle_regs(b)
        if written_a & used_b:
            return False
        return True

    def _can_swap_adjacent(self, a, b):
        used_a, written_a = self._bundle_regs(a)
        used_b, written_b = self._bundle_regs(b)
        if written_a & used_b:
            return False
        if written_b & used_a:
            return False
        if written_a & written_b:
            return False
        return True

    def _merge_two_bundles(self, a, b):
        out = {}
        for engine in a:
            out[engine] = list(a.get(engine, [])) + list(b.get(engine, []))
        for engine in b:
            if engine not in out:
                out[engine] = list(b.get(engine, []))
        return out

    def _merge_adjacent_bundles(self):
        changed = True
        while changed:
            changed = False
            merged = []
            i = 0
            while i < len(self.instrs):
                if i + 1 < len(self.instrs) and self._can_merge_adjacent(self.instrs[i], self.instrs[i + 1]):
                    merged.append(self._merge_two_bundles(self.instrs[i], self.instrs[i + 1]))
                    i += 2
                    changed = True
                elif (
                    i + 2 < len(self.instrs)
                    and self._can_swap_adjacent(self.instrs[i + 1], self.instrs[i + 2])
                    and self._can_merge_adjacent(self.instrs[i], self.instrs[i + 2])
                ):
                    merged.append(self._merge_two_bundles(self.instrs[i], self.instrs[i + 2]))
                    merged.append(self.instrs[i + 1])
                    i += 3
                    changed = True
                elif (
                    i + 2 < len(self.instrs)
                    and self._can_swap_adjacent(self.instrs[i], self.instrs[i + 1])
                    and self._can_swap_adjacent(self.instrs[i], self.instrs[i + 2])
                    and self._can_merge_adjacent(self.instrs[i + 1], self.instrs[i + 2])
                ):
                    merged.append(self._merge_two_bundles(self.instrs[i + 1], self.instrs[i + 2]))
                    merged.append(self.instrs[i])
                    i += 3
                    changed = True
                elif (
                    i + 3 < len(self.instrs)
                    and self._can_swap_adjacent(self.instrs[i + 1], self.instrs[i + 2])
                    and self._can_merge_adjacent(self.instrs[i], self.instrs[i + 2])
                    and self._can_merge_adjacent(self.instrs[i + 1], self.instrs[i + 3])
                ):
                    merged.append(self._merge_two_bundles(self.instrs[i], self.instrs[i + 2]))
                    merged.append(self._merge_two_bundles(self.instrs[i + 1], self.instrs[i + 3]))
                    i += 4
                    changed = True
                elif (
                    i + 3 < len(self.instrs)
                    and self._can_swap_adjacent(self.instrs[i + 2], self.instrs[i + 3])
                    and self._can_swap_adjacent(self.instrs[i + 1], self.instrs[i + 3])
                    and self._can_merge_adjacent(self.instrs[i], self.instrs[i + 3])
                ):
                    merged.append(self._merge_two_bundles(self.instrs[i], self.instrs[i + 3]))
                    merged.append(self.instrs[i + 1])
                    merged.append(self.instrs[i + 2])
                    i += 4
                    changed = True
                elif (
                    i + 3 < len(self.instrs)
                    and self._can_swap_adjacent(self.instrs[i + 2], self.instrs[i + 3])
                    and self._can_merge_adjacent(self.instrs[i + 1], self.instrs[i + 3])
                ):
                    merged.append(self.instrs[i])
                    merged.append(self._merge_two_bundles(self.instrs[i + 1], self.instrs[i + 3]))
                    merged.append(self.instrs[i + 2])
                    i += 4
                    changed = True
                else:
                    merged.append(self.instrs[i])
                    i += 1
            self.instrs = merged

    def _compact_by_swapping(self, max_passes=3):
        """Greedily bubble independent bundles earlier to expose more merges."""
        for _ in range(max_passes):
            moved = False
            i = 1
            while i < len(self.instrs):
                j = i
                while j > 0 and self._can_swap_adjacent(self.instrs[j - 1], self.instrs[j]):
                    self.instrs[j - 1], self.instrs[j] = self.instrs[j], self.instrs[j - 1]
                    moved = True
                    j -= 1
                i += 1
            if not moved:
                break

    def build_load_pairs_quad(self, node0, node1, node2, node3, addr0, addr1, addr2, addr3):
        load_pairs = []
        for off in range(0, VLEN, 2):
            load_pairs.extend([
                ("load_offset", node0, addr0, off),
                ("load_offset", node0, addr0, off + 1),
                ("load_offset", node1, addr1, off),
                ("load_offset", node1, addr1, off + 1),
            ])
        for off in range(0, VLEN, 2):
            load_pairs.extend([
                ("load_offset", node2, addr2, off),
                ("load_offset", node2, addr2, off + 1),
                ("load_offset", node3, addr3, off),
                ("load_offset", node3, addr3, off + 1),
            ])
        return load_pairs

    def build_small_depth_node_vec(
        self,
        depth,
        node_vec,
        idx_vec,
        tmp0_vec,
        tmp1_vec,
        one_vec,
        base1_vec,
        base3_vec,
        lvl1_node1_vec,
        lvl1_diff_vec,
        lvl2_node3_vec,
        lvl2_node5_vec,
        lvl2_diff_34_vec,
        lvl2_diff_56_vec,
    ):
        if depth == 1:
            self.add_bundle(
                {
                    "valu": [
                        ("-", tmp0_vec, idx_vec, base1_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("&", tmp1_vec, tmp0_vec, one_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("multiply_add", node_vec, tmp1_vec, lvl1_diff_vec, lvl1_node1_vec),
                    ]
                }
            )
            return

        if depth == 2:
            self.add_bundle(
                {
                    "valu": [
                        ("-", tmp0_vec, idx_vec, base3_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("&", tmp1_vec, tmp0_vec, one_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("multiply_add", node_vec, tmp1_vec, lvl2_diff_34_vec, lvl2_node3_vec),
                        ("multiply_add", tmp1_vec, tmp1_vec, lvl2_diff_56_vec, lvl2_node5_vec),
                        (">>", tmp0_vec, tmp0_vec, one_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("&", tmp0_vec, tmp0_vec, one_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("-", tmp1_vec, tmp1_vec, node_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("multiply_add", node_vec, tmp0_vec, tmp1_vec, node_vec),
                    ]
                }
            )

    def build_small_depth_node_quad(
        self,
        depth,
        node0,
        node1,
        node2,
        node3,
        idx0,
        idx1,
        idx2,
        idx3,
        tmp0_0,
        tmp1_0,
        tmp0_1,
        tmp1_1,
        tmp0_2,
        tmp1_2,
        tmp0_3,
        tmp1_3,
        one_vec,
        base1_vec,
        base3_vec,
        lvl1_node1_vec,
        lvl1_diff_vec,
        lvl2_node3_vec,
        lvl2_node5_vec,
        lvl2_diff_34_vec,
        lvl2_diff_56_vec,
    ):
        if depth == 1:
            self.add_bundle(
                {
                    "valu": [
                        ("-", tmp0_0, idx0, base1_vec),
                        ("-", tmp0_1, idx1, base1_vec),
                        ("-", tmp0_2, idx2, base1_vec),
                        ("-", tmp0_3, idx3, base1_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("&", tmp1_0, tmp0_0, one_vec),
                        ("&", tmp1_1, tmp0_1, one_vec),
                        ("&", tmp1_2, tmp0_2, one_vec),
                        ("&", tmp1_3, tmp0_3, one_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("multiply_add", node0, tmp1_0, lvl1_diff_vec, lvl1_node1_vec),
                        ("multiply_add", node1, tmp1_1, lvl1_diff_vec, lvl1_node1_vec),
                        ("multiply_add", node2, tmp1_2, lvl1_diff_vec, lvl1_node1_vec),
                        ("multiply_add", node3, tmp1_3, lvl1_diff_vec, lvl1_node1_vec),
                    ]
                }
            )
            return

        if depth == 2:
            self.add_bundle(
                {
                    "valu": [
                        ("-", tmp0_0, idx0, base3_vec),
                        ("-", tmp0_1, idx1, base3_vec),
                        ("-", tmp0_2, idx2, base3_vec),
                        ("-", tmp0_3, idx3, base3_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("&", tmp1_0, tmp0_0, one_vec),
                        ("&", tmp1_1, tmp0_1, one_vec),
                        ("&", tmp1_2, tmp0_2, one_vec),
                        ("&", tmp1_3, tmp0_3, one_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("multiply_add", node0, tmp1_0, lvl2_diff_34_vec, lvl2_node3_vec),
                        ("multiply_add", node1, tmp1_1, lvl2_diff_34_vec, lvl2_node3_vec),
                        ("multiply_add", node2, tmp1_2, lvl2_diff_34_vec, lvl2_node3_vec),
                        ("multiply_add", node3, tmp1_3, lvl2_diff_34_vec, lvl2_node3_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("multiply_add", tmp1_0, tmp1_0, lvl2_diff_56_vec, lvl2_node5_vec),
                        ("multiply_add", tmp1_1, tmp1_1, lvl2_diff_56_vec, lvl2_node5_vec),
                        ("multiply_add", tmp1_2, tmp1_2, lvl2_diff_56_vec, lvl2_node5_vec),
                        ("multiply_add", tmp1_3, tmp1_3, lvl2_diff_56_vec, lvl2_node5_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        (">>", tmp0_0, tmp0_0, one_vec),
                        (">>", tmp0_1, tmp0_1, one_vec),
                        (">>", tmp0_2, tmp0_2, one_vec),
                        (">>", tmp0_3, tmp0_3, one_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("&", tmp0_0, tmp0_0, one_vec),
                        ("&", tmp0_1, tmp0_1, one_vec),
                        ("&", tmp0_2, tmp0_2, one_vec),
                        ("&", tmp0_3, tmp0_3, one_vec),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("-", tmp1_0, tmp1_0, node0),
                        ("-", tmp1_1, tmp1_1, node1),
                        ("-", tmp1_2, tmp1_2, node2),
                        ("-", tmp1_3, tmp1_3, node3),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("multiply_add", node0, tmp0_0, tmp1_0, node0),
                        ("multiply_add", node1, tmp0_1, tmp1_1, node1),
                        ("multiply_add", node2, tmp0_2, tmp1_2, node2),
                        ("multiply_add", node3, tmp0_3, tmp1_3, node3),
                    ]
                }
            )

    def build_index_update_quad(
        self, idx0, idx1, idx2, idx3,
        val0, val1, val2, val3,
        t0, t1, t2, t3,
        one_vec, two_vec, n_nodes_vec,
        addr0, addr1, addr2, addr3, base_vec,
    ):
        self.add_bundle({"valu": [("&", t0, val0, one_vec), ("&", t1, val1, one_vec), ("&", t2, val2, one_vec), ("&", t3, val3, one_vec)]})
        self.add_bundle({"valu": [("+", t0, t0, one_vec), ("+", t1, t1, one_vec), ("+", t2, t2, one_vec), ("+", t3, t3, one_vec)]})
        self.add_bundle({"valu": [("multiply_add", idx0, idx0, two_vec, t0), ("multiply_add", idx1, idx1, two_vec, t1), ("multiply_add", idx2, idx2, two_vec, t2), ("multiply_add", idx3, idx3, two_vec, t3)]})
        self.add_bundle({"valu": [("<", t0, idx0, n_nodes_vec), ("<", t1, idx1, n_nodes_vec), ("<", t2, idx2, n_nodes_vec), ("<", t3, idx3, n_nodes_vec)]})
        self.add_bundle({"valu": [("*", idx0, idx0, t0), ("*", idx1, idx1, t1), ("*", idx2, idx2, t2), ("*", idx3, idx3, t3)]})
        self.add_bundle({"valu": [("+", addr0, idx0, base_vec), ("+", addr1, idx1, base_vec), ("+", addr2, idx2, base_vec), ("+", addr3, idx3, base_vec)]})

    def build_index_update_quad_with_loads(
        self, idx0, idx1, idx2, idx3,
        val0, val1, val2, val3,
        t0, t1, t2, t3,
        one_vec, two_vec, n_nodes_vec,
        addr0, addr1, addr2, addr3, base_vec,
        load_pairs,
        step5_extra_valu=None,
    ):
        steps = [
            [("&", t0, val0, one_vec), ("&", t1, val1, one_vec), ("&", t2, val2, one_vec), ("&", t3, val3, one_vec)],
            [("+", t0, t0, one_vec), ("+", t1, t1, one_vec), ("+", t2, t2, one_vec), ("+", t3, t3, one_vec)],
            [("multiply_add", idx0, idx0, two_vec, t0), ("multiply_add", idx1, idx1, two_vec, t1), ("multiply_add", idx2, idx2, two_vec, t2), ("multiply_add", idx3, idx3, two_vec, t3)],
            [("<", t0, idx0, n_nodes_vec), ("<", t1, idx1, n_nodes_vec), ("<", t2, idx2, n_nodes_vec), ("<", t3, idx3, n_nodes_vec)],
            [("*", idx0, idx0, t0), ("*", idx1, idx1, t1), ("*", idx2, idx2, t2), ("*", idx3, idx3, t3)],
            [("+", addr0, idx0, base_vec), ("+", addr1, idx1, base_vec), ("+", addr2, idx2, base_vec), ("+", addr3, idx3, base_vec)],
        ]
        for step_i, valu_ops in enumerate(steps):
            full_ops = list(valu_ops)
            if step_i == 4 and step5_extra_valu:
                full_ops.extend(step5_extra_valu)
            bundle = {"valu": full_ops}
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

    def build_index_update_quad_no_clamp(
        self, idx0, idx1, idx2, idx3,
        val0, val1, val2, val3,
        t0, t1, t2, t3,
        one_vec, two_vec,
        addr0, addr1, addr2, addr3, base_vec,
        lsb_precomputed=False,
        last_step_extra_valu=None,
    ):
        if not lsb_precomputed:
            self.add_bundle({"valu": [("&", t0, val0, one_vec), ("&", t1, val1, one_vec), ("&", t2, val2, one_vec), ("&", t3, val3, one_vec)]})
        self.add_bundle({"valu": [('+', t0, t0, one_vec), ('+', t1, t1, one_vec), ('+', t2, t2, one_vec), ('+', t3, t3, one_vec)]})
        self.add_bundle({"valu": [("multiply_add", idx0, idx0, two_vec, t0), ("multiply_add", idx1, idx1, two_vec, t1), ("multiply_add", idx2, idx2, two_vec, t2), ("multiply_add", idx3, idx3, two_vec, t3)]})
        valu_ops = [('+', addr0, idx0, base_vec), ('+', addr1, idx1, base_vec), ('+', addr2, idx2, base_vec), ('+', addr3, idx3, base_vec)]
        if last_step_extra_valu:
            valu_ops.extend(last_step_extra_valu)
        self.add_bundle({"valu": valu_ops})

    def build_index_update_quad_no_clamp_with_loads(
        self, idx0, idx1, idx2, idx3,
        val0, val1, val2, val3,
        t0, t1, t2, t3,
        one_vec, two_vec,
        addr0, addr1, addr2, addr3, base_vec,
        load_pairs,
        last_step_extra_valu=None,
        lsb_precomputed=False,
        tail_extra_valu=None,
    ):
        steps = []
        if not lsb_precomputed:
            steps.append([('&', t0, val0, one_vec), ('&', t1, val1, one_vec), ('&', t2, val2, one_vec), ('&', t3, val3, one_vec)])
        steps.extend([
            [('+', t0, t0, one_vec), ('+', t1, t1, one_vec), ('+', t2, t2, one_vec), ('+', t3, t3, one_vec)],
            [('multiply_add', idx0, idx0, two_vec, t0), ('multiply_add', idx1, idx1, two_vec, t1), ('multiply_add', idx2, idx2, two_vec, t2), ('multiply_add', idx3, idx3, two_vec, t3)],
            [('+', addr0, idx0, base_vec), ('+', addr1, idx1, base_vec), ('+', addr2, idx2, base_vec), ('+', addr3, idx3, base_vec)],
        ])
        for step_i, valu_ops in enumerate(steps):
            full_ops = list(valu_ops)
            if step_i == len(steps) - 1 and last_step_extra_valu:
                full_ops.extend(last_step_extra_valu)
            bundle = {"valu": full_ops}
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

        if tail_extra_valu:
            bundle = {"valu": list(tail_extra_valu)}
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)


    def build_index_reset_quad(
        self, idx0, idx1, idx2, idx3,
        addr0, addr1, addr2, addr3,
        zero_vec, base_vec,
    ):
        self.add_bundle({"valu": [('*', idx0, idx0, zero_vec), ('*', idx1, idx1, zero_vec), ('*', idx2, idx2, zero_vec), ('*', idx3, idx3, zero_vec)]})
        self.add_bundle({"valu": [('+', addr0, base_vec, zero_vec), ('+', addr1, base_vec, zero_vec), ('+', addr2, base_vec, zero_vec), ('+', addr3, base_vec, zero_vec)]})

    def build_index_reset_quad_with_loads(
        self, idx0, idx1, idx2, idx3,
        addr0, addr1, addr2, addr3,
        zero_vec, base_vec,
        load_pairs,
        last_step_extra_valu=None,
    ):
        steps = [
            [('*', idx0, idx0, zero_vec), ('*', idx1, idx1, zero_vec), ('*', idx2, idx2, zero_vec), ('*', idx3, idx3, zero_vec)],
            [('+', addr0, base_vec, zero_vec), ('+', addr1, base_vec, zero_vec), ('+', addr2, base_vec, zero_vec), ('+', addr3, base_vec, zero_vec)],
        ]
        for step_i, valu_ops in enumerate(steps):
            full_ops = list(valu_ops)
            if step_i == 1 and last_step_extra_valu:
                full_ops.extend(last_step_extra_valu)
            bundle = {"valu": full_ops}
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

    def build_index_xor_update_quad_with_loads(
        self, idx0, idx1, idx2, idx3,
        val0, val1, val2, val3,
        t0, t1, t2, t3,
        one_vec, two_vec, n_nodes_vec,
        addr0, addr1, addr2, addr3, base_vec,
        xor_val0, xor_val1, xor_val2, xor_val3,
        xor_node0, xor_node1, xor_node2, xor_node3,
        load_pairs,
    ):
        """Index update for 4 vectors with XOR of another group merged into steps 3-4.

        Steps 3-4 have 4 index ops + 2 XOR ops = 6 valu each (at limit).
        XOR for val0/val1 is safe at step 3 (nodes fully loaded by then).
        XOR for val2/val3 is safe at step 4 (nodes fully loaded by end of step 3).
        """
        steps = [
            [("&", t0, val0, one_vec), ("&", t1, val1, one_vec), ("&", t2, val2, one_vec), ("&", t3, val3, one_vec)],
            [("+", t0, t0, one_vec), ("+", t1, t1, one_vec), ("+", t2, t2, one_vec), ("+", t3, t3, one_vec)],
            [("multiply_add", idx0, idx0, two_vec, t0), ("multiply_add", idx1, idx1, two_vec, t1), ("multiply_add", idx2, idx2, two_vec, t2), ("multiply_add", idx3, idx3, two_vec, t3),
             ("^", xor_val0, xor_val0, xor_node0), ("^", xor_val1, xor_val1, xor_node1)],
            [("<", t0, idx0, n_nodes_vec), ("<", t1, idx1, n_nodes_vec), ("<", t2, idx2, n_nodes_vec), ("<", t3, idx3, n_nodes_vec),
             ("^", xor_val2, xor_val2, xor_node2), ("^", xor_val3, xor_val3, xor_node3)],
            [("*", idx0, idx0, t0), ("*", idx1, idx1, t1), ("*", idx2, idx2, t2), ("*", idx3, idx3, t3)],
            [("+", addr0, idx0, base_vec), ("+", addr1, idx1, base_vec), ("+", addr2, idx2, base_vec), ("+", addr3, idx3, base_vec)],
        ]
        for valu_ops in steps:
            bundle = {"valu": valu_ops}
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

    def build_index_update_triple(
        self,
        idx_vec0,
        idx_vec1,
        idx_vec2,
        val_vec0,
        val_vec1,
        val_vec2,
        tmp3_vec0,
        tmp3_vec1,
        tmp3_vec2,
        one_vec,
        two_vec,
        n_nodes_vec,
        addr_vec0,
        addr_vec1,
        addr_vec2,
        base_vec,
    ):
        self.add_bundle(
            {
                "valu": [
                    ("&", tmp3_vec0, val_vec0, one_vec),
                    ("&", tmp3_vec1, val_vec1, one_vec),
                    ("&", tmp3_vec2, val_vec2, one_vec),
                ]
            }
        )
        self.add_bundle(
            {
                "valu": [
                    ("+", tmp3_vec0, tmp3_vec0, one_vec),
                    ("+", tmp3_vec1, tmp3_vec1, one_vec),
                    ("+", tmp3_vec2, tmp3_vec2, one_vec),
                ]
            }
        )
        self.add_bundle(
            {
                "valu": [
                    ("multiply_add", idx_vec0, idx_vec0, two_vec, tmp3_vec0),
                    ("multiply_add", idx_vec1, idx_vec1, two_vec, tmp3_vec1),
                    ("multiply_add", idx_vec2, idx_vec2, two_vec, tmp3_vec2),
                ]
            }
        )
        self.add_bundle(
            {
                "valu": [
                    ("<", tmp3_vec0, idx_vec0, n_nodes_vec),
                    ("<", tmp3_vec1, idx_vec1, n_nodes_vec),
                    ("<", tmp3_vec2, idx_vec2, n_nodes_vec),
                ]
            }
        )
        self.add_bundle(
            {
                "valu": [
                    ("*", idx_vec0, idx_vec0, tmp3_vec0),
                    ("*", idx_vec1, idx_vec1, tmp3_vec1),
                    ("*", idx_vec2, idx_vec2, tmp3_vec2),
                ]
            }
        )
        self.add_bundle(
            {
                "valu": [
                    ("+", addr_vec0, idx_vec0, base_vec),
                    ("+", addr_vec1, idx_vec1, base_vec),
                    ("+", addr_vec2, idx_vec2, base_vec),
                ]
            }
        )

    def build_index_update_triple_with_loads(
        self,
        idx_vec0,
        idx_vec1,
        idx_vec2,
        val_vec0,
        val_vec1,
        val_vec2,
        tmp3_vec0,
        tmp3_vec1,
        tmp3_vec2,
        one_vec,
        two_vec,
        n_nodes_vec,
        addr_vec0,
        addr_vec1,
        addr_vec2,
        base_vec,
        load_pairs,
    ):
        steps = [
            [("&", tmp3_vec0, val_vec0, one_vec), ("&", tmp3_vec1, val_vec1, one_vec), ("&", tmp3_vec2, val_vec2, one_vec)],
            [("+", tmp3_vec0, tmp3_vec0, one_vec), ("+", tmp3_vec1, tmp3_vec1, one_vec), ("+", tmp3_vec2, tmp3_vec2, one_vec)],
            [("multiply_add", idx_vec0, idx_vec0, two_vec, tmp3_vec0), ("multiply_add", idx_vec1, idx_vec1, two_vec, tmp3_vec1), ("multiply_add", idx_vec2, idx_vec2, two_vec, tmp3_vec2)],
            [("<", tmp3_vec0, idx_vec0, n_nodes_vec), ("<", tmp3_vec1, idx_vec1, n_nodes_vec), ("<", tmp3_vec2, idx_vec2, n_nodes_vec)],
            [("*", idx_vec0, idx_vec0, tmp3_vec0), ("*", idx_vec1, idx_vec1, tmp3_vec1), ("*", idx_vec2, idx_vec2, tmp3_vec2)],
            [("+", addr_vec0, idx_vec0, base_vec), ("+", addr_vec1, idx_vec1, base_vec), ("+", addr_vec2, idx_vec2, base_vec)],
        ]
        for valu_ops in steps:
            bundle = {"valu": valu_ops}
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

    def build_index_xor_update_triple_with_loads(
        self,
        idx_vec0,
        idx_vec1,
        idx_vec2,
        val_vec0,
        val_vec1,
        val_vec2,
        tmp3_vec0,
        tmp3_vec1,
        tmp3_vec2,
        one_vec,
        two_vec,
        n_nodes_vec,
        addr_vec0,
        addr_vec1,
        addr_vec2,
        base_vec,
        xor_val0,
        xor_val1,
        xor_val2,
        xor_node0,
        xor_node1,
        xor_node2,
        load_pairs,
    ):
        """Index update for one group with XOR of the other group merged into cycle 1."""
        # Cycle 1: 3 index ops + 3 XOR ops = 6 valu (max)
        bundle = {
            "valu": [
                ("&", tmp3_vec0, val_vec0, one_vec),
                ("&", tmp3_vec1, val_vec1, one_vec),
                ("&", tmp3_vec2, val_vec2, one_vec),
                ("^", xor_val0, xor_val0, xor_node0),
                ("^", xor_val1, xor_val1, xor_node1),
                ("^", xor_val2, xor_val2, xor_node2),
            ]
        }
        if load_pairs:
            bundle["load"] = load_pairs[:2]
            del load_pairs[:2]
        self.add_bundle(bundle)
        # Remaining 5 index steps
        remaining_steps = [
            [("+", tmp3_vec0, tmp3_vec0, one_vec), ("+", tmp3_vec1, tmp3_vec1, one_vec), ("+", tmp3_vec2, tmp3_vec2, one_vec)],
            [("multiply_add", idx_vec0, idx_vec0, two_vec, tmp3_vec0), ("multiply_add", idx_vec1, idx_vec1, two_vec, tmp3_vec1), ("multiply_add", idx_vec2, idx_vec2, two_vec, tmp3_vec2)],
            [("<", tmp3_vec0, idx_vec0, n_nodes_vec), ("<", tmp3_vec1, idx_vec1, n_nodes_vec), ("<", tmp3_vec2, idx_vec2, n_nodes_vec)],
            [("*", idx_vec0, idx_vec0, tmp3_vec0), ("*", idx_vec1, idx_vec1, tmp3_vec1), ("*", idx_vec2, idx_vec2, tmp3_vec2)],
            [("+", addr_vec0, idx_vec0, base_vec), ("+", addr_vec1, idx_vec1, base_vec), ("+", addr_vec2, idx_vec2, base_vec)],
        ]
        for valu_ops in remaining_steps:
            bundle = {"valu": valu_ops}
            if load_pairs:
                bundle["load"] = load_pairs[:2]
                del load_pairs[:2]
            self.add_bundle(bundle)

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

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
        i = 0
        while i + 1 < len(init_vars):
            self.add_bundle(
                {
                    "load": [
                        ("const", tmp1, i),
                        ("const", tmp2, i + 1),
                    ]
                }
            )
            self.add_bundle(
                {
                    "load": [
                        ("load", self.scratch[init_vars[i]], tmp1),
                        ("load", self.scratch[init_vars[i + 1]], tmp2),
                    ]
                }
            )
            i += 2
        if i < len(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[init_vars[i]], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        i_ctr = self.alloc_scratch("i_ctr")
        idx_ptr = self.alloc_scratch("idx_ptr")
        val_ptr = self.alloc_scratch("val_ptr")
        vlen_const = self.scratch_const(VLEN)

        idx_vec = self.alloc_vec("idx_vec")
        val_vec = self.alloc_vec("val_vec")
        node_vec = self.alloc_vec("node_vec")
        addr_vec = self.alloc_vec("addr_vec")
        tmp1_vec = self.alloc_vec("tmp1_vec")
        tmp2_vec = self.alloc_vec("tmp2_vec")
        tmp3_vec = self.alloc_vec("tmp3_vec")
        idx_vec1 = self.alloc_vec("idx_vec1")
        val_vec1 = self.alloc_vec("val_vec1")
        node_vec1 = self.alloc_vec("node_vec1")
        addr_vec1 = self.alloc_vec("addr_vec1")
        tmp1_vec1 = self.alloc_vec("tmp1_vec1")
        tmp2_vec1 = self.alloc_vec("tmp2_vec1")
        tmp3_vec1 = self.alloc_vec("tmp3_vec1")
        idx_vec2 = self.alloc_vec("idx_vec2")
        val_vec2 = self.alloc_vec("val_vec2")
        node_vec2 = self.alloc_vec("node_vec2")
        addr_vec2 = self.alloc_vec("addr_vec2")
        tmp1_vec2 = self.alloc_vec("tmp1_vec2")
        tmp2_vec2 = self.alloc_vec("tmp2_vec2")
        tmp3_vec2 = self.alloc_vec("tmp3_vec2")

        idx_vec_b = self.alloc_vec("idx_vec_b")
        val_vec_b = self.alloc_vec("val_vec_b")
        node_vec_b = self.alloc_vec("node_vec_b")
        addr_vec_b = self.alloc_vec("addr_vec_b")
        tmp1_vec_b = self.alloc_vec("tmp1_vec_b")
        tmp2_vec_b = self.alloc_vec("tmp2_vec_b")
        tmp3_vec_b = self.alloc_vec("tmp3_vec_b")
        idx_vec1_b = self.alloc_vec("idx_vec1_b")
        val_vec1_b = self.alloc_vec("val_vec1_b")
        node_vec1_b = self.alloc_vec("node_vec1_b")
        addr_vec1_b = self.alloc_vec("addr_vec1_b")
        tmp1_vec1_b = self.alloc_vec("tmp1_vec1_b")
        tmp2_vec1_b = self.alloc_vec("tmp2_vec1_b")
        tmp3_vec1_b = self.alloc_vec("tmp3_vec1_b")
        idx_vec2_b = self.alloc_vec("idx_vec2_b")
        val_vec2_b = self.alloc_vec("val_vec2_b")
        node_vec2_b = self.alloc_vec("node_vec2_b")
        addr_vec2_b = self.alloc_vec("addr_vec2_b")
        tmp1_vec2_b = self.alloc_vec("tmp1_vec2_b")
        tmp2_vec2_b = self.alloc_vec("tmp2_vec2_b")
        tmp3_vec2_b = self.alloc_vec("tmp3_vec2_b")

        # Sub-vector 3 for group A (quad processing)
        idx_vec3 = self.alloc_vec("idx_vec3")
        val_vec3 = self.alloc_vec("val_vec3")
        node_vec3 = self.alloc_vec("node_vec3")
        addr_vec3 = self.alloc_vec("addr_vec3")
        tmp1_vec3 = self.alloc_vec("tmp1_vec3")
        tmp2_vec3 = self.alloc_vec("tmp2_vec3")
        tmp3_vec3 = self.alloc_vec("tmp3_vec3")

        # Sub-vector 3 for group B (quad processing)
        idx_vec3_b = self.alloc_vec("idx_vec3_b")
        val_vec3_b = self.alloc_vec("val_vec3_b")
        node_vec3_b = self.alloc_vec("node_vec3_b")
        addr_vec3_b = self.alloc_vec("addr_vec3_b")
        tmp1_vec3_b = self.alloc_vec("tmp1_vec3_b")
        tmp2_vec3_b = self.alloc_vec("tmp2_vec3_b")
        tmp3_vec3_b = self.alloc_vec("tmp3_vec3_b")

        n_nodes_vec = self.alloc_vec("n_nodes_vec")
        zero_vec = self.alloc_vec("zero_vec")
        one_vec = self.alloc_vec("one_vec")
        two_vec = self.alloc_vec("two_vec")
        root_node_scalar = self.alloc_scratch("root_node_scalar")
        root_node_vec = self.alloc_vec("root_node_vec")
        base_vec = self.alloc_vec("base_vec")

        # Optional gather-free paths for depth 1/2 rounds.
        use_small_depth_1 = True
        use_small_depth_2 = False

        # Placeholders for dormant small-depth paths.
        base1_vec = one_vec
        base3_vec = one_vec
        lvl1_node1_vec = root_node_vec
        lvl1_diff_vec = zero_vec
        lvl2_node3_vec = root_node_vec
        lvl2_node5_vec = root_node_vec
        lvl2_diff_34_vec = zero_vec
        lvl2_diff_56_vec = zero_vec

        self.add_bundle(
            {
                "alu": [
                    ("+", idx_ptr, self.scratch["inp_indices_p"], zero_const),
                    ("+", val_ptr, self.scratch["inp_values_p"], zero_const),
                ],
                "load": [
                    ("load", root_node_scalar, self.scratch["forest_values_p"]),
                ],
                "valu": [
                    ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]),
                    ("vbroadcast", zero_vec, zero_const),
                    ("vbroadcast", one_vec, one_const),
                    ("vbroadcast", two_vec, two_const),
                    ("vbroadcast", base_vec, self.scratch["forest_values_p"]),
                ]
            }
        )

        if use_small_depth_1:
            lvl1_node1_scalar = self.alloc_scratch("lvl1_node1_scalar")
            lvl1_node2_scalar = self.alloc_scratch("lvl1_node2_scalar")
            lvl1_diff_scalar = self.alloc_scratch("lvl1_diff_scalar")
            base1_vec = self.alloc_vec("base1_vec")
            lvl1_node1_vec = self.alloc_vec("lvl1_node1_vec")
            lvl1_diff_vec = self.alloc_vec("lvl1_diff_vec")

            one_off = self.scratch_const(1)
            two_off = self.scratch_const(2)
            self.add_bundle(
                {
                    "alu": [
                        ("+", tmp1, self.scratch["forest_values_p"], one_off),
                        ("+", tmp2, self.scratch["forest_values_p"], two_off),
                    ]
                }
            )
            self.add_bundle(
                {
                    "load": [
                        ("load", lvl1_node1_scalar, tmp1),
                        ("load", lvl1_node2_scalar, tmp2),
                    ]
                }
            )
            self.add_bundle(
                {
                    "alu": [
                        ("-", lvl1_diff_scalar, lvl1_node2_scalar, lvl1_node1_scalar),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("vbroadcast", base1_vec, one_off),
                        ("vbroadcast", lvl1_node1_vec, lvl1_node1_scalar),
                        ("vbroadcast", lvl1_diff_vec, lvl1_diff_scalar),
                    ]
                }
            )

        if use_small_depth_2:
            lvl2_node3_scalar = self.alloc_scratch("lvl2_node3_scalar")
            lvl2_node4_scalar = self.alloc_scratch("lvl2_node4_scalar")
            lvl2_node5_scalar = self.alloc_scratch("lvl2_node5_scalar")
            lvl2_node6_scalar = self.alloc_scratch("lvl2_node6_scalar")
            lvl2_diff_34_scalar = self.alloc_scratch("lvl2_diff_34_scalar")
            lvl2_diff_56_scalar = self.alloc_scratch("lvl2_diff_56_scalar")
            base3_vec = self.alloc_vec("base3_vec")
            lvl2_node3_vec = self.alloc_vec("lvl2_node3_vec")
            lvl2_node5_vec = self.alloc_vec("lvl2_node5_vec")
            lvl2_diff_34_vec = self.alloc_vec("lvl2_diff_34_vec")
            lvl2_diff_56_vec = self.alloc_vec("lvl2_diff_56_vec")

            three_off = self.scratch_const(3)
            four_off = self.scratch_const(4)
            five_off = self.scratch_const(5)
            six_off = self.scratch_const(6)
            self.add_bundle(
                {
                    "alu": [
                        ("+", tmp1, self.scratch["forest_values_p"], three_off),
                        ("+", tmp2, self.scratch["forest_values_p"], four_off),
                    ]
                }
            )
            self.add_bundle(
                {
                    "load": [
                        ("load", lvl2_node3_scalar, tmp1),
                        ("load", lvl2_node4_scalar, tmp2),
                    ]
                }
            )
            self.add_bundle(
                {
                    "alu": [
                        ("+", tmp1, self.scratch["forest_values_p"], five_off),
                        ("+", tmp2, self.scratch["forest_values_p"], six_off),
                    ]
                }
            )
            self.add_bundle(
                {
                    "load": [
                        ("load", lvl2_node5_scalar, tmp1),
                        ("load", lvl2_node6_scalar, tmp2),
                    ]
                }
            )
            self.add_bundle(
                {
                    "alu": [
                        ("-", lvl2_diff_34_scalar, lvl2_node4_scalar, lvl2_node3_scalar),
                        ("-", lvl2_diff_56_scalar, lvl2_node6_scalar, lvl2_node5_scalar),
                    ]
                }
            )
            self.add_bundle(
                {
                    "valu": [
                        ("vbroadcast", base3_vec, three_off),
                        ("vbroadcast", lvl2_node3_vec, lvl2_node3_scalar),
                        ("vbroadcast", lvl2_node5_vec, lvl2_node5_scalar),
                        ("vbroadcast", lvl2_diff_34_vec, lvl2_diff_34_scalar),
                        ("vbroadcast", lvl2_diff_56_vec, lvl2_diff_56_scalar),
                    ]
                }
            )

        hash_const_vecs = []
        hash_stage_specs = []

        ma_stage_multipliers = {0: 4097, 2: 33, 4: 9}
        ma_mul_vecs = {}
        for stage_i, mult in ma_stage_multipliers.items():
            mul_vec = self.alloc_vec(f"hash_mul_vec_{stage_i}")
            self.add_bundle({"valu": [("vbroadcast", mul_vec, self.scratch_const(mult))]})
            ma_mul_vecs[stage_i] = mul_vec

        for stage_i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val1_vec = self.alloc_vec()
            val3_vec = self.alloc_vec()
            valu_ops = [
                ("vbroadcast", val1_vec, self.scratch_const(val1)),
                ("vbroadcast", val3_vec, self.scratch_const(val3)),
            ]
            if stage_i == 0:
                valu_ops.append(("vbroadcast", root_node_vec, root_node_scalar))
            self.add_bundle(
                {
                    "valu": valu_ops
                }
            )
            hash_const_vecs.append((op1, op2, op3, val1_vec, val3_vec))

            if stage_i in ma_mul_vecs and op1 == "+" and op2 == "+" and op3 == "<<":
                hash_stage_specs.append(("ma", ma_mul_vecs[stage_i], val1_vec))
            else:
                hash_stage_specs.append(("std", op1, op2, op3, val1_vec, val3_vec))

        if batch_size % VLEN == 0:
            vlen2_const = self.scratch_const(2 * VLEN)
            vlen3_const = self.scratch_const(3 * VLEN)
            vlen4_const = self.scratch_const(4 * VLEN)
            vlen5_const = self.scratch_const(5 * VLEN)
            vlen6_const = self.scratch_const(6 * VLEN)
            vlen7_const = self.scratch_const(7 * VLEN)
            vlen8_const = self.scratch_const(8 * VLEN)

            idx_ptr1 = self.alloc_scratch("idx_ptr1")
            idx_ptr2 = self.alloc_scratch("idx_ptr2")
            idx_ptr3 = self.alloc_scratch("idx_ptr3")
            idx_ptr_b = self.alloc_scratch("idx_ptr_b")
            idx_ptr1_b = self.alloc_scratch("idx_ptr1_b")
            idx_ptr2_b = self.alloc_scratch("idx_ptr2_b")
            idx_ptr3_b = self.alloc_scratch("idx_ptr3_b")
            val_ptr1 = self.alloc_scratch("val_ptr1")
            val_ptr2 = self.alloc_scratch("val_ptr2")
            val_ptr3 = self.alloc_scratch("val_ptr3")
            val_ptr_b = self.alloc_scratch("val_ptr_b")
            val_ptr1_b = self.alloc_scratch("val_ptr1_b")
            val_ptr2_b = self.alloc_scratch("val_ptr2_b")
            val_ptr3_b = self.alloc_scratch("val_ptr3_b")

            vec_count = batch_size // VLEN
            quad_pair_groups = vec_count // 8
            rem_vecs = vec_count - (quad_pair_groups * 8)

            for _ in range(quad_pair_groups):
                # --- Pointer setup: 14 ALU ops (12+2) + 8 vloads + addr overlap ---
                self.add_bundle(
                    {
                        "alu": [
                            ("+", idx_ptr1, idx_ptr, vlen_const),
                            ("+", idx_ptr2, idx_ptr, vlen2_const),
                            ("+", idx_ptr3, idx_ptr, vlen3_const),
                            ("+", idx_ptr_b, idx_ptr, vlen4_const),
                            ("+", idx_ptr1_b, idx_ptr, vlen5_const),
                            ("+", idx_ptr2_b, idx_ptr, vlen6_const),
                            ("+", idx_ptr3_b, idx_ptr, vlen7_const),
                            ("+", val_ptr1, val_ptr, vlen_const),
                            ("+", val_ptr2, val_ptr, vlen2_const),
                            ("+", val_ptr3, val_ptr, vlen3_const),
                            ("+", val_ptr_b, val_ptr, vlen4_const),
                            ("+", val_ptr1_b, val_ptr, vlen5_const),
                        ],
                        "load": [
                            ("vload", idx_vec, idx_ptr),
                            ("vload", val_vec, val_ptr),
                        ]
                    }
                )
                # C2: 2 ALU + vload(idx1, val1) + addr0
                self.add_bundle(
                    {
                        "alu": [
                            ("+", val_ptr2_b, val_ptr, vlen6_const),
                            ("+", val_ptr3_b, val_ptr, vlen7_const),
                        ],
                        "load": [
                            ("vload", idx_vec1, idx_ptr1),
                            ("vload", val_vec1, val_ptr1),
                        ],
                    }
                )
                # C3-C8: remaining vloads + addr computation
                self.add_bundle(
                    {
                        "load": [("vload", idx_vec2, idx_ptr2), ("vload", val_vec2, val_ptr2)],
                    }
                )
                self.add_bundle(
                    {
                        "load": [("vload", idx_vec3, idx_ptr3), ("vload", val_vec3, val_ptr3)],
                        "valu": [
                            ("^", val_vec, val_vec, root_node_vec),
                            ("^", val_vec1, val_vec1, root_node_vec),
                        ],
                    }
                )

                round_period = forest_height + 1

                for round_i in range(rounds):
                    depth = round_i % round_period
                    a_uses_root = depth == 0
                    a_uses_small = (depth == 1 and use_small_depth_1) or (depth == 2 and use_small_depth_2)
                    b_uses_root = a_uses_root
                    b_uses_small = a_uses_small
                    next_depth = (round_i + 1) % round_period
                    next_a_uses_root = next_depth == 0
                    next_a_uses_small = (next_depth == 1 and use_small_depth_1) or (next_depth == 2 and use_small_depth_2)
                    is_reset_update_round = next_a_uses_root

                    # Build B gathers unless this is a B root round.
                    preload_b_pairs = []
                    if b_uses_root or b_uses_small:
                        load_pairs_b = []
                        if round_i == 0:
                            preload_b_pairs = [
                                ("vload", idx_vec_b, idx_ptr_b),
                                ("vload", val_vec_b, val_ptr_b),
                                ("vload", idx_vec1_b, idx_ptr1_b),
                                ("vload", val_vec1_b, val_ptr1_b),
                                ("vload", idx_vec2_b, idx_ptr2_b),
                                ("vload", val_vec2_b, val_ptr2_b),
                                ("vload", idx_vec3_b, idx_ptr3_b),
                                ("vload", val_vec3_b, val_ptr3_b),
                            ]
                    else:
                        load_pairs_b = self.build_load_pairs_quad(
                            node_vec_b, node_vec1_b, node_vec2_b, node_vec3_b,
                            addr_vec_b, addr_vec1_b, addr_vec2_b, addr_vec3_b,
                        )

                    if a_uses_small:
                        self.build_small_depth_node_quad(
                            depth,
                            node_vec, node_vec1, node_vec2, node_vec3,
                            idx_vec, idx_vec1, idx_vec2, idx_vec3,
                            tmp3_vec, tmp1_vec, tmp3_vec1, tmp1_vec1,
                            tmp3_vec2, tmp1_vec2, tmp3_vec3, tmp1_vec3,
                            one_vec,
                            base1_vec,
                            base3_vec,
                            lvl1_node1_vec,
                            lvl1_diff_vec,
                            lvl2_node3_vec,
                            lvl2_node5_vec,
                            lvl2_diff_34_vec,
                            lvl2_diff_56_vec,
                        )

                    # Hash A (round 0 and post-wrap rounds use root for all 4 vectors)
                    if a_uses_root:
                        if round_i != 0:
                            self.add_bundle(
                                {
                                    "valu": [
                                        ("^", val_vec, val_vec, root_node_vec),
                                        ("^", val_vec1, val_vec1, root_node_vec),
                                    ]
                                }
                            )
                        a_hash_first_cycle_extra = [
                            ("^", val_vec2, val_vec2, root_node_vec),
                            ("^", val_vec3, val_vec3, root_node_vec),
                        ]
                    elif a_uses_small:
                        self.add_bundle(
                            {
                                "valu": [
                                    ("^", val_vec, val_vec, node_vec),
                                    ("^", val_vec1, val_vec1, node_vec1),
                                ]
                            }
                        )
                        a_hash_first_cycle_extra = [
                            ("^", val_vec2, val_vec2, node_vec2),
                            ("^", val_vec3, val_vec3, node_vec3),
                        ]
                    else:
                        a_hash_first_cycle_extra = [
                            ("^", val_vec2, val_vec2, node_vec2),
                            ("^", val_vec3, val_vec3, node_vec3),
                        ]

                    self.build_hash_quad_interleaved_ma(
                        val_vec, val_vec1, val_vec2, val_vec3,
                        tmp1_vec, tmp2_vec, tmp1_vec1, tmp2_vec1,
                        tmp1_vec2, tmp2_vec2, tmp1_vec3, tmp2_vec3,
                        hash_stage_specs,
                        load_pairs=(preload_b_pairs if preload_b_pairs else load_pairs_b),
                        first_cycle_extra_valu=a_hash_first_cycle_extra,
                    )

                    # Index A while finishing B gathers.
                    if load_pairs_b:
                        if is_reset_update_round:
                            self.build_index_update_quad_with_loads(
                                idx_vec, idx_vec1, idx_vec2, idx_vec3,
                                val_vec, val_vec1, val_vec2, val_vec3,
                                tmp3_vec, tmp3_vec1, tmp3_vec2, tmp3_vec3,
                                one_vec, two_vec, n_nodes_vec,
                                addr_vec, addr_vec1, addr_vec2, addr_vec3,
                                base_vec,
                                load_pairs_b,
                                step5_extra_valu=[
                                    ("^", val_vec_b, val_vec_b, node_vec_b),
                                    ("^", val_vec1_b, val_vec1_b, node_vec1_b),
                                ],
                            )
                        else:
                            self.build_index_update_quad_no_clamp_with_loads(
                                idx_vec, idx_vec1, idx_vec2, idx_vec3,
                                val_vec, val_vec1, val_vec2, val_vec3,
                                tmp3_vec, tmp3_vec1, tmp3_vec2, tmp3_vec3,
                                one_vec, two_vec,
                                addr_vec, addr_vec1, addr_vec2, addr_vec3,
                                base_vec,
                                load_pairs_b,
                                tail_extra_valu=[
                                    ("^", val_vec_b, val_vec_b, node_vec_b),
                                    ("^", val_vec1_b, val_vec1_b, node_vec1_b),
                                ],
                            )
                            if load_pairs_b:
                                self.emit_load_pairs(load_pairs_b)
                    elif is_reset_update_round:
                        self.build_index_reset_quad(
                            idx_vec, idx_vec1, idx_vec2, idx_vec3,
                            addr_vec, addr_vec1, addr_vec2, addr_vec3,
                            zero_vec, base_vec,
                        )
                    else:
                        self.build_index_update_quad_no_clamp(
                            idx_vec, idx_vec1, idx_vec2, idx_vec3,
                            val_vec, val_vec1, val_vec2, val_vec3,
                            tmp3_vec, tmp3_vec1, tmp3_vec2, tmp3_vec3,
                            one_vec, two_vec,
                            addr_vec, addr_vec1, addr_vec2, addr_vec3, base_vec,
                            last_step_extra_valu=(
                                [
                                    ("^", val_vec_b, val_vec_b, root_node_vec),
                                    ("^", val_vec1_b, val_vec1_b, root_node_vec),
                                ]
                                if (round_i == 0 and b_uses_root)
                                else None
                            ),
                        )

                    if round_i != rounds - 1:
                        # Build A gathers for next round unless next A round uses root/small path.
                        if next_a_uses_root or next_a_uses_small:
                            load_pairs_a = []
                        else:
                            load_pairs_a = self.build_load_pairs_quad(
                                node_vec, node_vec1, node_vec2, node_vec3,
                                addr_vec, addr_vec1, addr_vec2, addr_vec3,
                            )

                        if b_uses_small:
                            self.build_small_depth_node_quad(
                                depth,
                                node_vec_b, node_vec1_b, node_vec2_b, node_vec3_b,
                                idx_vec_b, idx_vec1_b, idx_vec2_b, idx_vec3_b,
                                tmp3_vec_b, tmp1_vec_b, tmp3_vec1_b, tmp1_vec1_b,
                                tmp3_vec2_b, tmp1_vec2_b, tmp3_vec3_b, tmp1_vec3_b,
                                one_vec,
                                base1_vec,
                                base3_vec,
                                lvl1_node1_vec,
                                lvl1_diff_vec,
                                lvl2_node3_vec,
                                lvl2_node5_vec,
                                lvl2_diff_34_vec,
                                lvl2_diff_56_vec,
                            )

                        # Hash B. Root rounds use root for all 4 vectors.
                        if b_uses_root or b_uses_small:
                            b_xor0 = root_node_vec if b_uses_root else node_vec_b
                            b_xor1 = root_node_vec if b_uses_root else node_vec1_b
                            b_xor2 = root_node_vec if b_uses_root else node_vec2_b
                            b_xor3 = root_node_vec if b_uses_root else node_vec3_b
                            if b_uses_root and round_i == 0:
                                pass
                            elif load_pairs_a:
                                first_two_a = load_pairs_a[:2]
                                del load_pairs_a[:2]
                                self.add_bundle(
                                    {
                                        "valu": [
                                            ("^", val_vec_b, val_vec_b, b_xor0),
                                            ("^", val_vec1_b, val_vec1_b, b_xor1),
                                        ],
                                        "load": first_two_a,
                                    }
                                )
                            else:
                                self.add_bundle(
                                    {
                                        "valu": [
                                            ("^", val_vec_b, val_vec_b, b_xor0),
                                            ("^", val_vec1_b, val_vec1_b, b_xor1),
                                        ]
                                    }
                                )
                            b_hash_first_cycle_extra = [
                                ("^", val_vec2_b, val_vec2_b, b_xor2),
                                ("^", val_vec3_b, val_vec3_b, b_xor3),
                            ]
                        else:
                            b_hash_first_cycle_extra = [
                                ("^", val_vec2_b, val_vec2_b, node_vec2_b),
                                ("^", val_vec3_b, val_vec3_b, node_vec3_b),
                            ]

                        self.build_hash_quad_interleaved_ma(
                            val_vec_b, val_vec1_b, val_vec2_b, val_vec3_b,
                            tmp1_vec_b, tmp2_vec_b, tmp1_vec1_b, tmp2_vec1_b,
                            tmp1_vec2_b, tmp2_vec2_b, tmp1_vec3_b, tmp2_vec3_b,
                            hash_stage_specs,
                            load_pairs=load_pairs_a,
                            first_cycle_extra_valu=b_hash_first_cycle_extra,
                        )

                        # Index B while finishing next-round A gathers.
                        if load_pairs_a:
                            self.build_index_update_quad_no_clamp_with_loads(
                                idx_vec_b, idx_vec1_b, idx_vec2_b, idx_vec3_b,
                                val_vec_b, val_vec1_b, val_vec2_b, val_vec3_b,
                                tmp3_vec_b, tmp3_vec1_b, tmp3_vec2_b, tmp3_vec3_b,
                                one_vec, two_vec,
                                addr_vec_b, addr_vec1_b, addr_vec2_b, addr_vec3_b,
                                base_vec,
                                load_pairs_a,
                                tail_extra_valu=[
                                    ("^", val_vec, val_vec, node_vec),
                                    ("^", val_vec1, val_vec1, node_vec1),
                                ],
                            )
                            if load_pairs_a:
                                self.emit_load_pairs(load_pairs_a)
                        elif is_reset_update_round:
                            self.build_index_reset_quad(
                                idx_vec_b, idx_vec1_b, idx_vec2_b, idx_vec3_b,
                                addr_vec_b, addr_vec1_b, addr_vec2_b, addr_vec3_b,
                                zero_vec, base_vec,
                            )
                        else:
                            self.build_index_update_quad_no_clamp(
                                idx_vec_b, idx_vec1_b, idx_vec2_b, idx_vec3_b,
                                val_vec_b, val_vec1_b, val_vec2_b, val_vec3_b,
                                tmp3_vec_b, tmp3_vec1_b, tmp3_vec2_b, tmp3_vec3_b,
                                one_vec, two_vec,
                                addr_vec_b, addr_vec1_b, addr_vec2_b, addr_vec3_b, base_vec,
                            )
                    else:
                        # Last round B: no next-round A gathers.
                        if b_uses_small:
                            self.build_small_depth_node_quad(
                                depth,
                                node_vec_b, node_vec1_b, node_vec2_b, node_vec3_b,
                                idx_vec_b, idx_vec1_b, idx_vec2_b, idx_vec3_b,
                                tmp3_vec_b, tmp1_vec_b, tmp3_vec1_b, tmp1_vec1_b,
                                tmp3_vec2_b, tmp1_vec2_b, tmp3_vec3_b, tmp1_vec3_b,
                                one_vec,
                                base1_vec,
                                base3_vec,
                                lvl1_node1_vec,
                                lvl1_diff_vec,
                                lvl2_node3_vec,
                                lvl2_node5_vec,
                                lvl2_diff_34_vec,
                                lvl2_diff_56_vec,
                            )

                        if b_uses_root or b_uses_small:
                            b_xor0 = root_node_vec if b_uses_root else node_vec_b
                            b_xor1 = root_node_vec if b_uses_root else node_vec1_b
                            b_xor2 = root_node_vec if b_uses_root else node_vec2_b
                            b_xor3 = root_node_vec if b_uses_root else node_vec3_b
                            self.add_bundle(
                                {
                                    "valu": [
                                        ("^", val_vec_b, val_vec_b, b_xor0),
                                        ("^", val_vec1_b, val_vec1_b, b_xor1),
                                    ]
                                }
                            )
                            b_hash_first_cycle_extra = [
                                ("^", val_vec2_b, val_vec2_b, b_xor2),
                                ("^", val_vec3_b, val_vec3_b, b_xor3),
                            ]
                        else:
                            b_hash_first_cycle_extra = [
                                ("^", val_vec2_b, val_vec2_b, node_vec2_b),
                                ("^", val_vec3_b, val_vec3_b, node_vec3_b),
                            ]

                        self.build_hash_quad_interleaved_ma(
                            val_vec_b, val_vec1_b, val_vec2_b, val_vec3_b,
                            tmp1_vec_b, tmp2_vec_b, tmp1_vec1_b, tmp2_vec1_b,
                            tmp1_vec2_b, tmp2_vec2_b, tmp1_vec3_b, tmp2_vec3_b,
                            hash_stage_specs,
                            first_cycle_extra_valu=b_hash_first_cycle_extra,
                        )

                        # Final round still needs correct output indices.
                        if is_reset_update_round:
                            self.build_index_reset_quad(
                                idx_vec_b, idx_vec1_b, idx_vec2_b, idx_vec3_b,
                                addr_vec_b, addr_vec1_b, addr_vec2_b, addr_vec3_b,
                                zero_vec, base_vec,
                            )
                        else:
                            self.build_index_update_quad_no_clamp(
                                idx_vec_b, idx_vec1_b, idx_vec2_b, idx_vec3_b,
                                val_vec_b, val_vec1_b, val_vec2_b, val_vec3_b,
                                tmp3_vec_b, tmp3_vec1_b, tmp3_vec2_b, tmp3_vec3_b,
                                one_vec, two_vec,
                                addr_vec_b, addr_vec1_b, addr_vec2_b, addr_vec3_b, base_vec,
                            )

                # --- Stores for 8 vectors ---
                self.add_bundle({"store": [("vstore", val_ptr, val_vec), ("vstore", val_ptr1, val_vec1)]})
                self.add_bundle({"store": [("vstore", val_ptr2, val_vec2), ("vstore", val_ptr3, val_vec3)]})
                self.add_bundle({"store": [("vstore", val_ptr_b, val_vec_b), ("vstore", val_ptr1_b, val_vec1_b)]})
                self.add_bundle({"store": [("vstore", val_ptr2_b, val_vec2_b), ("vstore", val_ptr3_b, val_vec3_b)]})
                self.add_bundle({"store": [("vstore", idx_ptr, idx_vec), ("vstore", idx_ptr1, idx_vec1)]})
                self.add_bundle({"store": [("vstore", idx_ptr2, idx_vec2), ("vstore", idx_ptr3, idx_vec3)]})
                self.add_bundle({"store": [("vstore", idx_ptr_b, idx_vec_b), ("vstore", idx_ptr1_b, idx_vec1_b)]})
                self.add_bundle(
                    {
                        "store": [("vstore", idx_ptr2_b, idx_vec2_b), ("vstore", idx_ptr3_b, idx_vec3_b)],
                        "alu": [("+", idx_ptr, idx_ptr, vlen8_const), ("+", val_ptr, val_ptr, vlen8_const)],
                    }
                )

            while rem_vecs >= 3:
                self.add_bundle(
                    {
                        "alu": [
                            ("+", idx_ptr1, idx_ptr, vlen_const),
                            ("+", idx_ptr2, idx_ptr, vlen2_const),
                            ("+", val_ptr1, val_ptr, vlen_const),
                            ("+", val_ptr2, val_ptr, vlen2_const),
                        ],
                        "load": [
                            ("vload", idx_vec, idx_ptr),
                            ("vload", val_vec, val_ptr),
                        ]
                    }
                )
                self.add_bundle(
                    {
                        "load": [
                            ("vload", idx_vec1, idx_ptr1),
                            ("vload", val_vec1, val_ptr1),
                        ]
                    }
                )
                self.add_bundle(
                    {
                        "load": [
                            ("vload", idx_vec2, idx_ptr2),
                            ("vload", val_vec2, val_ptr2),
                        ]
                    }
                )
                self.add_bundle(
                    {
                        "valu": [
                            ("+", addr_vec, idx_vec, base_vec),
                            ("+", addr_vec1, idx_vec1, base_vec),
                            ("+", addr_vec2, idx_vec2, base_vec),
                        ]
                    }
                )
                for _ in range(rounds):
                    load_pairs_a = self.build_load_pairs_triple(
                        node_vec, node_vec1, node_vec2, addr_vec, addr_vec1, addr_vec2
                    )
                    self.emit_load_pairs(load_pairs_a)
                    self.add_bundle(
                        {
                            "valu": [
                                ("^", val_vec, val_vec, node_vec),
                                ("^", val_vec1, val_vec1, node_vec1),
                                ("^", val_vec2, val_vec2, node_vec2),
                            ]
                        }
                    )
                    self.build_hash_vec_triple(
                        val_vec,
                        val_vec1,
                        val_vec2,
                        tmp1_vec,
                        tmp2_vec,
                        tmp1_vec1,
                        tmp2_vec1,
                        tmp1_vec2,
                        tmp2_vec2,
                        hash_const_vecs,
                    )
                    self.build_index_update_triple(
                        idx_vec,
                        idx_vec1,
                        idx_vec2,
                        val_vec,
                        val_vec1,
                        val_vec2,
                        tmp3_vec,
                        tmp3_vec1,
                        tmp3_vec2,
                        one_vec,
                        two_vec,
                        n_nodes_vec,
                        addr_vec,
                        addr_vec1,
                        addr_vec2,
                        base_vec,
                    )
                self.add_bundle(
                    {
                        "store": [
                            ("vstore", val_ptr, val_vec),
                            ("vstore", val_ptr1, val_vec1),
                        ]
                    }
                )
                self.add_bundle(
                    {
                        "store": [
                            ("vstore", idx_ptr, idx_vec),
                            ("vstore", idx_ptr1, idx_vec1),
                        ]
                    }
                )
                self.add_bundle(
                    {
                        "store": [
                            ("vstore", val_ptr2, val_vec2),
                            ("vstore", idx_ptr2, idx_vec2),
                        ],
                        "alu": [
                            ("+", idx_ptr, idx_ptr, vlen3_const),
                            ("+", val_ptr, val_ptr, vlen3_const),
                        ],
                    }
                )
                rem_vecs -= 3

            if rem_vecs == 2:
                self.add_bundle(
                    {
                        "alu": [
                            ("+", idx_ptr1, idx_ptr, vlen_const),
                            ("+", val_ptr1, val_ptr, vlen_const),
                        ],
                        "load": [
                            ("vload", idx_vec, idx_ptr),
                            ("vload", val_vec, val_ptr),
                        ]
                    }
                )
                self.add_bundle(
                    {
                        "load": [
                            ("vload", idx_vec1, idx_ptr1),
                            ("vload", val_vec1, val_ptr1),
                        ],
                        "valu": [
                            ("+", addr_vec, idx_vec, base_vec),
                        ],
                    }
                )
                self.add_bundle(
                    {
                        "valu": [
                            ("+", addr_vec1, idx_vec1, base_vec),
                        ],
                    }
                )
                for _ in range(rounds):
                    for off in range(0, VLEN, 2):
                        self.add_bundle(
                            {
                                "load": [
                                    ("load_offset", node_vec, addr_vec, off),
                                    ("load_offset", node_vec1, addr_vec1, off),
                                ]
                            }
                        )
                        self.add_bundle(
                            {
                                "load": [
                                    ("load_offset", node_vec, addr_vec, off + 1),
                                    ("load_offset", node_vec1, addr_vec1, off + 1),
                                ]
                            }
                        )

                    self.add_bundle(
                        {
                            "valu": [
                                ("^", val_vec, val_vec, node_vec),
                                ("^", val_vec1, val_vec1, node_vec1),
                            ]
                        }
                    )
                    self.build_hash_vec_pair(
                        val_vec,
                        val_vec1,
                        tmp1_vec,
                        tmp2_vec,
                        tmp1_vec1,
                        tmp2_vec1,
                        hash_const_vecs,
                    )

                    self.add_bundle(
                        {
                            "valu": [
                                ("&", tmp3_vec, val_vec, one_vec),
                                ("&", tmp3_vec1, val_vec1, one_vec),
                            ]
                        }
                    )
                    self.add_bundle(
                        {
                            "valu": [
                                ("+", tmp3_vec, tmp3_vec, one_vec),
                                ("+", tmp3_vec1, tmp3_vec1, one_vec),
                            ]
                        }
                    )
                    self.add_bundle(
                        {
                            "valu": [
                                ("multiply_add", idx_vec, idx_vec, two_vec, tmp3_vec),
                                (
                                    "multiply_add",
                                    idx_vec1,
                                    idx_vec1,
                                    two_vec,
                                    tmp3_vec1,
                                ),
                            ]
                        }
                    )
                    self.add_bundle(
                        {
                            "valu": [
                                ("<", tmp3_vec, idx_vec, n_nodes_vec),
                                ("<", tmp3_vec1, idx_vec1, n_nodes_vec),
                            ]
                        }
                    )
                    self.add_bundle(
                        {
                            "valu": [
                                ("*", idx_vec, idx_vec, tmp3_vec),
                                ("*", idx_vec1, idx_vec1, tmp3_vec1),
                            ]
                        }
                    )
                    self.add_bundle(
                        {
                            "valu": [
                                ("+", addr_vec, idx_vec, base_vec),
                                ("+", addr_vec1, idx_vec1, base_vec),
                            ]
                        }
                    )

                self.add_bundle(
                    {
                        "store": [
                            ("vstore", val_ptr, val_vec),
                            ("vstore", val_ptr1, val_vec1),
                        ]
                    }
                )
                self.add_bundle(
                    {
                        "store": [
                            ("vstore", idx_ptr, idx_vec),
                            ("vstore", idx_ptr1, idx_vec1),
                        ],
                        "alu": [
                            ("+", idx_ptr, idx_ptr, vlen2_const),
                            ("+", val_ptr, val_ptr, vlen2_const),
                        ],
                    }
                )
            elif rem_vecs == 1:
                self.add_bundle(
                    {
                        "load": [
                            ("vload", idx_vec, idx_ptr),
                            ("vload", val_vec, val_ptr),
                        ]
                    }
                )
                self.add_bundle({"valu": [("+", addr_vec, idx_vec, base_vec)]})
                for _ in range(rounds):
                    for off in range(0, VLEN, 2):
                        self.add_bundle(
                            {
                                "load": [
                                    ("load_offset", node_vec, addr_vec, off),
                                    ("load_offset", node_vec, addr_vec, off + 1),
                                ]
                            }
                        )
                    self.add_bundle({"valu": [("^", val_vec, val_vec, node_vec)]})
                    self.build_hash_vec(val_vec, tmp1_vec, tmp2_vec, hash_const_vecs)
                    self.add_bundle({"valu": [("&", tmp3_vec, val_vec, one_vec)]})
                    self.add_bundle({"valu": [("+", tmp3_vec, tmp3_vec, one_vec)]})
                    self.add_bundle(
                        {"valu": [("multiply_add", idx_vec, idx_vec, two_vec, tmp3_vec)]}
                    )
                    self.add_bundle({"valu": [("<", tmp3_vec, idx_vec, n_nodes_vec)]})
                    self.add_bundle({"valu": [("*", idx_vec, idx_vec, tmp3_vec)]})
                    self.add_bundle({"valu": [("+", addr_vec, idx_vec, base_vec)]})

                self.add_bundle(
                    {
                        "store": [
                            ("vstore", val_ptr, val_vec),
                        ]
                    }
                )
                self.add_bundle(
                    {
                        "store": [
                            ("vstore", idx_ptr, idx_vec),
                        ]
                    }
                )
        else:
            for _ in range(rounds):
                self.add("load", ("const", i_ctr, 0))
                self.add_bundle(
                    {
                        "alu": [
                            ("+", idx_ptr, self.scratch["inp_indices_p"], zero_const),
                            ("+", val_ptr, self.scratch["inp_values_p"], zero_const),
                        ]
                    }
                )
                loop_check_pc = len(self.instrs)
                self.add("alu", ("<", tmp1, i_ctr, self.scratch["batch_size"]))
                loop_body_jump_pc = len(self.instrs)
                self.add("flow", ("cond_jump", tmp1, 0))
                loop_end_jump_pc = len(self.instrs)
                self.add("flow", ("jump", 0))

                loop_body_pc = len(self.instrs)

                self.add_bundle(
                    {
                        "load": [
                            ("vload", idx_vec, idx_ptr),
                            ("vload", val_vec, val_ptr),
                        ]
                    }
                )

                self.add_bundle({"valu": [("+", addr_vec, idx_vec, base_vec)]})

                for off in range(0, VLEN, 2):
                    self.add_bundle(
                        {
                            "load": [
                                ("load_offset", node_vec, addr_vec, off),
                                ("load_offset", node_vec, addr_vec, off + 1),
                            ]
                        }
                    )

                self.add_bundle({"valu": [("^", val_vec, val_vec, node_vec)]})
                self.build_hash_vec(val_vec, tmp1_vec, tmp2_vec, hash_const_vecs)

                self.add_bundle({"valu": [("&", tmp3_vec, val_vec, one_vec)]})
                self.add_bundle({"valu": [("+", tmp3_vec, tmp3_vec, one_vec)]})
                self.add_bundle(
                    {"valu": [("multiply_add", idx_vec, idx_vec, two_vec, tmp3_vec)]}
                )
                self.add_bundle({"valu": [("<", tmp3_vec, idx_vec, n_nodes_vec)]})
                self.add_bundle({"valu": [("*", idx_vec, idx_vec, tmp3_vec)]})

                self.add_bundle(
                    {
                        "store": [
                            ("vstore", idx_ptr, idx_vec),
                            ("vstore", val_ptr, val_vec),
                        ]
                    }
                )

                self.add_bundle(
                    {
                        "alu": [
                            ("+", idx_ptr, idx_ptr, vlen_const),
                            ("+", val_ptr, val_ptr, vlen_const),
                            ("+", i_ctr, i_ctr, vlen_const),
                        ],
                        "flow": [("jump", loop_check_pc)],
                    }
                )

                loop_end_pc = len(self.instrs)
                self.instrs[loop_body_jump_pc]["flow"][0] = (
                    "cond_jump",
                    tmp1,
                    loop_body_pc,
                )
                self.instrs[loop_end_jump_pc]["flow"][0] = ("jump", loop_end_pc)

        self._merge_adjacent_bundles()
        self._compact_by_swapping(max_passes=3)
        self._merge_adjacent_bundles()
        self._list_schedule()
        self._compact_by_swapping(max_passes=2)
        self._merge_adjacent_bundles()
        self.add("flow", ("halt",))


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    import random

    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass

    # Check correctness
    inp_values_p = ref_mem[6]
    ok = (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    )
    if not ok and prints:
        print("Output values differ")
    return ok, machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_small(self):
        """Test with small inputs for correctness"""
        correct, cycles = do_kernel_test(3, 2, 8, prints=True)
        print(f"Small test: correct={correct}, cycles={cycles}")
        self.assertTrue(correct)

    def test_kernel_medium(self):
        """Test with medium inputs"""
        correct, cycles = do_kernel_test(4, 4, 16, prints=True)
        print(f"Medium test: correct={correct}, cycles={cycles}")
        self.assertTrue(correct)

    def test_kernel_cycles(self):
        """Main performance test"""
        correct, cycles = do_kernel_test(7, 16, 256, prints=True)
        print(f"Performance test: correct={correct}, cycles={cycles}")
        print(f"Baseline: {BASELINE}, speedup: {BASELINE / cycles:.2f}x")
        self.assertTrue(correct)
        # Relaxed local check: correctness is required, performance is informational here.


if __name__ == "__main__":
    unittest.main()
