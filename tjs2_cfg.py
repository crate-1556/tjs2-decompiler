#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from tjs2_decompiler import VM, Instruction


@dataclass
class BasicBlock:
    id: int
    start_idx: int
    end_idx: int
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    terminator: Optional[str] = None
    cond_true: Optional[int] = None
    cond_false: Optional[int] = None

    idom: Optional[int] = None
    dom_children: List[int] = field(default_factory=list)

    ipdom: Optional[int] = None
    pdom_children: List[int] = field(default_factory=list)


VIRTUAL_ENTRY_ID = -1
VIRTUAL_EXIT_ID = -2


@dataclass
class CFG:
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_id: int = VIRTUAL_ENTRY_ID
    exit_id: int = VIRTUAL_EXIT_ID
    addr_to_block: Dict[int, int] = field(default_factory=dict)
    idx_to_block: Dict[int, int] = field(default_factory=dict)

    def get_block(self, block_id: int) -> Optional[BasicBlock]:
        return self.blocks.get(block_id)

    def real_blocks(self) -> List[BasicBlock]:
        return sorted(
            [b for b in self.blocks.values() if b.id >= 0],
            key=lambda b: b.start_idx
        )

    def block_instructions(self, block: BasicBlock, instructions: List[Instruction]) -> List[Instruction]:
        return instructions[block.start_idx:block.end_idx]


def build_cfg(instructions: List[Instruction]) -> CFG:
    if not instructions:
        cfg = CFG()
        _add_virtual_nodes(cfg)
        return cfg

    addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}
    n = len(instructions)

    leaders = set()
    leaders.add(0)

    for i, instr in enumerate(instructions):
        if instr.op in (VM.JF, VM.JNF, VM.JMP):
            target_addr = instr.addr + instr.operands[0]
            target_idx = addr_to_idx.get(target_addr)
            if target_idx is not None:
                leaders.add(target_idx)
            if i + 1 < n:
                leaders.add(i + 1)

        elif instr.op in (VM.RET, VM.THROW):
            if i + 1 < n:
                leaders.add(i + 1)

        elif instr.op == VM.ENTRY:
            catch_addr = instr.addr + instr.operands[0]
            catch_idx = addr_to_idx.get(catch_addr)
            if catch_idx is not None:
                leaders.add(catch_idx)
            if i + 1 < n:
                leaders.add(i + 1)

        elif instr.op in (VM.SETF, VM.SETNF):
            if i + 1 < n:
                leaders.add(i + 1)

    sorted_leaders = sorted(leaders)
    cfg = CFG()

    for li, leader_idx in enumerate(sorted_leaders):
        if li + 1 < len(sorted_leaders):
            end_idx = sorted_leaders[li + 1]
        else:
            end_idx = n

        block_id = leader_idx
        block = BasicBlock(id=block_id, start_idx=leader_idx, end_idx=end_idx)

        if end_idx > leader_idx:
            last_instr = instructions[end_idx - 1]
            if last_instr.op == VM.JMP:
                block.terminator = 'jmp'
            elif last_instr.op == VM.JF:
                block.terminator = 'jf'
            elif last_instr.op == VM.JNF:
                block.terminator = 'jnf'
            elif last_instr.op == VM.RET:
                block.terminator = 'ret'
            elif last_instr.op == VM.THROW:
                block.terminator = 'throw'
            elif last_instr.op == VM.ENTRY:
                block.terminator = 'entry'
            else:
                block.terminator = 'fall'

        cfg.blocks[block_id] = block

        for idx in range(leader_idx, end_idx):
            cfg.idx_to_block[idx] = block_id
            cfg.addr_to_block[instructions[idx].addr] = block_id

    for block in list(cfg.blocks.values()):
        if block.end_idx <= block.start_idx:
            continue

        last_instr = instructions[block.end_idx - 1]

        if block.terminator == 'jmp':
            target_addr = last_instr.addr + last_instr.operands[0]
            target_idx = addr_to_idx.get(target_addr)
            if target_idx is not None and target_idx in cfg.blocks:
                _add_edge(cfg, block.id, target_idx)

        elif block.terminator in ('jf', 'jnf'):
            if block.end_idx < n and block.end_idx in cfg.blocks:
                fall_through_id = block.end_idx
                _add_edge(cfg, block.id, fall_through_id)
                if block.terminator == 'jnf':
                    block.cond_true = fall_through_id
                else:
                    block.cond_false = fall_through_id

            target_addr = last_instr.addr + last_instr.operands[0]
            target_idx = addr_to_idx.get(target_addr)
            if target_idx is not None and target_idx in cfg.blocks:
                _add_edge(cfg, block.id, target_idx)
                if block.terminator == 'jnf':
                    block.cond_false = target_idx
                else:
                    block.cond_true = target_idx

        elif block.terminator == 'entry':
            if block.end_idx < n and block.end_idx in cfg.blocks:
                _add_edge(cfg, block.id, block.end_idx)

            catch_addr = last_instr.addr + last_instr.operands[0]
            catch_idx = addr_to_idx.get(catch_addr)
            if catch_idx is not None and catch_idx in cfg.blocks:
                _add_edge(cfg, block.id, catch_idx)

        elif block.terminator in ('ret', 'throw'):
            pass

        elif block.terminator == 'fall':
            if block.end_idx < n and block.end_idx in cfg.blocks:
                _add_edge(cfg, block.id, block.end_idx)

    _add_virtual_nodes(cfg)

    return cfg


def _add_edge(cfg: CFG, from_id: int, to_id: int):
    from_block = cfg.blocks.get(from_id)
    to_block = cfg.blocks.get(to_id)
    if from_block is None or to_block is None:
        return
    if to_id not in from_block.successors:
        from_block.successors.append(to_id)
    if from_id not in to_block.predecessors:
        to_block.predecessors.append(from_id)


def _add_virtual_nodes(cfg: CFG):
    entry_block = BasicBlock(id=VIRTUAL_ENTRY_ID, start_idx=-1, end_idx=-1)
    cfg.blocks[VIRTUAL_ENTRY_ID] = entry_block
    cfg.entry_id = VIRTUAL_ENTRY_ID

    if 0 in cfg.blocks:
        _add_edge(cfg, VIRTUAL_ENTRY_ID, 0)

    exit_block = BasicBlock(id=VIRTUAL_EXIT_ID, start_idx=-1, end_idx=-1)
    cfg.blocks[VIRTUAL_EXIT_ID] = exit_block
    cfg.exit_id = VIRTUAL_EXIT_ID

    for block in cfg.blocks.values():
        if block.terminator in ('ret', 'throw'):
            _add_edge(cfg, block.id, VIRTUAL_EXIT_ID)

    for block in cfg.blocks.values():
        if block.id >= 0 and not block.successors:
            _add_edge(cfg, block.id, VIRTUAL_EXIT_ID)


def _compute_rpo(cfg: CFG, entry_id: int, get_successors) -> List[int]:
    visited = set()
    post_order = []

    def dfs(block_id):
        if block_id in visited:
            return
        visited.add(block_id)
        for succ_id in get_successors(block_id):
            if succ_id in cfg.blocks:
                dfs(succ_id)
        post_order.append(block_id)

    dfs(entry_id)
    return list(reversed(post_order))


def _intersect(idom: Dict[int, int], rpo_number: Dict[int, int], b1: int, b2: int) -> int:
    finger1 = b1
    finger2 = b2
    while finger1 != finger2:
        while rpo_number.get(finger1, float('inf')) > rpo_number.get(finger2, float('inf')):
            finger1 = idom.get(finger1, finger1)
            if finger1 == idom.get(finger1):
                break
        while rpo_number.get(finger2, float('inf')) > rpo_number.get(finger1, float('inf')):
            finger2 = idom.get(finger2, finger2)
            if finger2 == idom.get(finger2):
                break
    return finger1


def compute_dominators(cfg: CFG):
    entry_id = cfg.entry_id

    def get_successors(block_id):
        block = cfg.blocks.get(block_id)
        return block.successors if block else []

    rpo = _compute_rpo(cfg, entry_id, get_successors)
    rpo_number = {block_id: i for i, block_id in enumerate(rpo)}

    idom = {}
    idom[entry_id] = entry_id

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == entry_id:
                continue

            block = cfg.blocks.get(b)
            if block is None:
                continue

            new_idom = None
            for p in block.predecessors:
                if p in idom:
                    new_idom = p
                    break

            if new_idom is None:
                continue

            for p in block.predecessors:
                if p == new_idom:
                    continue
                if p in idom:
                    new_idom = _intersect(idom, rpo_number, new_idom, p)

            if idom.get(b) != new_idom:
                idom[b] = new_idom
                changed = True

    for block_id, dom_id in idom.items():
        block = cfg.blocks.get(block_id)
        if block:
            block.idom = dom_id
            block.dom_children = []

    for block_id, dom_id in idom.items():
        if block_id != dom_id:
            parent = cfg.blocks.get(dom_id)
            if parent:
                parent.dom_children.append(block_id)


def compute_postdominators(cfg: CFG):
    exit_id = cfg.exit_id

    def get_predecessors(block_id):
        block = cfg.blocks.get(block_id)
        return block.predecessors if block else []

    rpo = _compute_rpo(cfg, exit_id, get_predecessors)
    rpo_number = {block_id: i for i, block_id in enumerate(rpo)}

    ipdom = {}
    ipdom[exit_id] = exit_id

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == exit_id:
                continue

            block = cfg.blocks.get(b)
            if block is None:
                continue

            reverse_preds = block.successors

            new_ipdom = None
            for s in reverse_preds:
                if s in ipdom:
                    new_ipdom = s
                    break

            if new_ipdom is None:
                continue

            for s in reverse_preds:
                if s == new_ipdom:
                    continue
                if s in ipdom:
                    new_ipdom = _intersect(ipdom, rpo_number, new_ipdom, s)

            if ipdom.get(b) != new_ipdom:
                ipdom[b] = new_ipdom
                changed = True

    for block_id, pdom_id in ipdom.items():
        block = cfg.blocks.get(block_id)
        if block:
            block.ipdom = pdom_id
            block.pdom_children = []

    for block_id, pdom_id in ipdom.items():
        if block_id != pdom_id:
            parent = cfg.blocks.get(pdom_id)
            if parent:
                parent.pdom_children.append(block_id)


def dominates(cfg: CFG, a: int, b: int) -> bool:
    if a == b:
        return True
    current = b
    visited = set()
    while current is not None and current not in visited:
        visited.add(current)
        block = cfg.blocks.get(current)
        if block is None:
            return False
        if block.idom == a:
            return True
        if block.idom == current:
            return False
        current = block.idom
    return False


def postdominates(cfg: CFG, a: int, b: int) -> bool:
    if a == b:
        return True
    current = b
    visited = set()
    while current is not None and current not in visited:
        visited.add(current)
        block = cfg.blocks.get(current)
        if block is None:
            return False
        if block.ipdom == a:
            return True
        if block.ipdom == current:
            return False
        current = block.ipdom
    return False


def get_merge_point(cfg: CFG, block_id: int) -> Optional[int]:
    block = cfg.blocks.get(block_id)
    if block is None:
        return None
    return block.ipdom


def get_back_edges(cfg: CFG) -> List[Tuple[int, int]]:
    back_edges = []
    for block in cfg.blocks.values():
        if block.id < 0:
            continue
        for succ_id in block.successors:
            if dominates(cfg, succ_id, block.id):
                back_edges.append((block.id, succ_id))
    return back_edges


def get_natural_loop(cfg: CFG, back_edge: Tuple[int, int]) -> Set[int]:
    tail, header = back_edge
    loop_blocks = {header, tail}

    if tail == header:
        return loop_blocks

    worklist = [tail]
    while worklist:
        block_id = worklist.pop()
        block = cfg.blocks.get(block_id)
        if block is None:
            continue
        for pred_id in block.predecessors:
            if pred_id not in loop_blocks and pred_id >= 0:
                loop_blocks.add(pred_id)
                worklist.append(pred_id)

    return loop_blocks
