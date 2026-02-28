import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Set, Tuple, Any

from tjs2_decompiler import (
    VM, Instruction, CodeObject, Expr, Stmt, ConstExpr, VarExpr, VoidExpr,
    BinaryExpr, UnaryExpr, TernaryExpr, AssignExpr, PropertyExpr,
    CallExpr, MethodCallExpr, CommaExpr, TypeofExpr, IsValidExpr, TypeCastExpr,
    ArrayExpr, DictExpr,
    ExprStmt, VarDeclStmt, IfStmt, WhileStmt, DoWhileStmt, ForStmt, TryStmt,
    BreakStmt, ContinueStmt, ReturnStmt, SwapExpr, SwitchStmt,
    BINARY_OP_SYMBOLS
)
from tjs2_cfg import (
    CFG, BasicBlock, VIRTUAL_ENTRY_ID, VIRTUAL_EXIT_ID,
    get_back_edges, get_natural_loop, get_merge_point,
    dominates, postdominates
)

class RegionType(Enum):
    BLOCK = auto()
    SEQUENCE = auto()
    IF_THEN = auto()
    IF_THEN_ELSE = auto()
    WHILE = auto()
    DO_WHILE = auto()
    INFINITE = auto()
    SWITCH = auto()
    TRY_CATCH = auto()
    SC_EXPR = auto()

@dataclass
class LoopInfo:
    header: int
    back_edge_source: int
    body_blocks: Set[int]
    exit_blocks: Set[int]
    loop_type: str
    cond_block: Optional[int] = None
    all_back_edge_sources: List[int] = field(default_factory=list)

@dataclass
class SwitchCase:
    value_expr: Optional[Expr]
    body_blocks: List[int]
    body_region: Optional['Region'] = None
    has_break: bool = True
    fall_through: bool = False
    cond_block_id: Optional[int] = None
    body_is_continue: bool = False
    has_continue: bool = False

@dataclass
class Region:
    type: RegionType
    header_block: int
    blocks: Set[int]
    children: List['Region'] = field(default_factory=list)
    exit_block: Optional[int] = None

    then_region: Optional['Region'] = None
    else_region: Optional['Region'] = None
    body_region: Optional['Region'] = None
    cond_block: Optional[int] = None

    loop_info: Optional[LoopInfo] = None

    switch_cases: List[SwitchCase] = field(default_factory=list)
    switch_ref_reg: Optional[int] = None
    switch_break_target: Optional[int] = None

    try_region: Optional['Region'] = None
    catch_region: Optional['Region'] = None
    catch_block: Optional[int] = None
    exception_reg: Optional[int] = None

_DOWHILE_COND_PREAMBLE_OPS = frozenset({
    VM.CONST, VM.CP, VM.CL, VM.CCL,
    VM.GPD, VM.GPI, VM.GPDS, VM.GPIS, VM.GLOBAL,
    VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.MOD, VM.IDIV,
    VM.BAND, VM.BOR, VM.BXOR, VM.SAR, VM.SAL, VM.SR,
    VM.TYPEOF, VM.TYPEOFD, VM.TYPEOFI,
    VM.CHKINV, VM.CHKINS,
    VM.INT, VM.REAL, VM.STR,
    VM.LNOT, VM.CHS, VM.NUM, VM.BNOT, VM.ASC, VM.CHR,
    VM.CALL, VM.CALLD, VM.CALLI, VM.NEW,
    VM.CHGTHIS,
})

def _block_dominates(cfg: CFG, a: int, b: int) -> bool:
    if a == b:
        return True
    current = b
    visited = set()
    while current is not None and current not in visited:
        visited.add(current)
        block = cfg.get_block(current)
        if block is None or block.idom is None:
            return False
        if block.idom == a:
            return True
        current = block.idom
    return False

def detect_loops(cfg: CFG, instructions: List[Instruction]) -> List[LoopInfo]:
    back_edges = get_back_edges(cfg)
    loops = []

    for tail, header in back_edges:
        body = get_natural_loop(cfg, (tail, header))

        header_block = cfg.get_block(header)
        tail_block = cfg.get_block(tail)
        if header_block and tail_block:
            loop_start_idx = header_block.start_idx
            loop_end_idx = tail_block.end_idx
            header_addr = instructions[header_block.start_idx].addr
            dead_jmp_end = loop_end_idx
            for block in cfg.blocks.values():
                if (block.id >= 0
                        and not block.predecessors
                        and block.terminator == 'jmp'
                        and block.start_idx >= loop_start_idx):
                    jmp_instr = instructions[block.end_idx - 1]
                    target_addr = jmp_instr.addr + jmp_instr.operands[0]
                    if target_addr == header_addr:
                        dead_jmp_end = max(dead_jmp_end, block.end_idx)
            if tail_block.terminator == 'jmp':
                loop_end_idx = dead_jmp_end
            for block_id in list(body):
                blk = cfg.get_block(block_id)
                if blk and blk.terminator == 'entry':
                    entry_instr = instructions[blk.end_idx - 1]
                    if entry_instr.op == VM.ENTRY:
                        catch_addr = entry_instr.addr + entry_instr.operands[0]
                        cb_id = cfg.addr_to_block.get(catch_addr)
                        if cb_id is not None and cb_id >= 0:
                            cb = cfg.get_block(cb_id)
                            if cb:
                                loop_end_idx = max(loop_end_idx, cb.end_idx)
            changed = True
            while changed:
                changed = False
                for block in cfg.blocks.values():
                    if block.id < 0 or block.id in body:
                        continue
                    if block.start_idx < loop_start_idx or block.end_idx > loop_end_idx:
                        continue
                    has_reachable_pred = any(
                        p >= 0 and cfg.get_block(p) and cfg.get_block(p).predecessors
                        for p in block.predecessors
                    )
                    if not has_reachable_pred:
                        if any(s in body for s in block.successors if s >= 0):
                            body.add(block.id)
                            changed = True
                        continue
                    all_preds_in_body = all(
                        p in body
                        for p in block.predecessors if p >= 0
                        and cfg.get_block(p) and cfg.get_block(p).predecessors
                        and cfg.get_block(p).start_idx < block.start_idx
                    )
                    if not all_preds_in_body:
                        continue
                    if _block_dominates(cfg, header, block.id):
                        body.add(block.id)
                        changed = True

            if (dead_jmp_end > loop_end_idx
                    and tail_block.terminator in ('jf', 'jnf')):
                header_has_exit = False
                if header_block.terminator in ('jf', 'jnf'):
                    for s in header_block.successors:
                        if s not in body and s >= 0:
                            header_has_exit = True
                            break
                if header_has_exit:
                    loop_end_idx = dead_jmp_end
                    changed = True
                    while changed:
                        changed = False
                        for block in cfg.blocks.values():
                            if block.id < 0 or block.id in body:
                                continue
                            if (block.start_idx < loop_start_idx
                                    or block.end_idx > loop_end_idx):
                                continue
                            has_reachable_pred = any(
                                p >= 0 and cfg.get_block(p)
                                and cfg.get_block(p).predecessors
                                for p in block.predecessors
                            )
                            if not has_reachable_pred:
                                if any(s in body for s in block.successors
                                       if s >= 0):
                                    body.add(block.id)
                                    changed = True
                                continue
                            all_preds_in_body = all(
                                p in body
                                for p in block.predecessors if p >= 0
                                and cfg.get_block(p)
                                and cfg.get_block(p).predecessors
                                and cfg.get_block(p).start_idx < block.start_idx
                            )
                            if not all_preds_in_body:
                                continue
                            if _block_dominates(cfg, header, block.id):
                                body.add(block.id)
                                changed = True

        exit_blocks = set()
        for block_id in body:
            block = cfg.get_block(block_id)
            if block:
                for succ_id in block.successors:
                    if succ_id not in body and succ_id >= 0:
                        exit_blocks.add(succ_id)

        loop_type, cond_block = _classify_loop(cfg, instructions, header, tail, body, exit_blocks)

        if _is_switch_back_jump(cfg, instructions, header, tail, body):
            continue

        loops.append(LoopInfo(
            header=header,
            back_edge_source=tail,
            body_blocks=body,
            exit_blocks=exit_blocks,
            loop_type=loop_type,
            cond_block=cond_block
        ))

    merged = {}
    header_tails = {}
    for loop in loops:
        h = loop.header
        header_tails.setdefault(h, []).append(loop.back_edge_source)
        if h in merged:
            existing = merged[h]
            existing.body_blocks = existing.body_blocks | loop.body_blocks
            existing.exit_blocks = set()
        else:
            merged[h] = loop

    for loop in merged.values():
        exit_blocks = set()
        for block_id in loop.body_blocks:
            block = cfg.get_block(block_id)
            if block:
                for succ_id in block.successors:
                    if succ_id not in loop.body_blocks and succ_id >= 0:
                        exit_blocks.add(succ_id)
        loop.exit_blocks = exit_blocks

    for loop in merged.values():
        tails = header_tails.get(loop.header, [])
        loop.all_back_edge_sources = sorted(tails,
            key=lambda t: (cfg.get_block(t).end_idx if cfg.get_block(t) else -1))
        if len(tails) <= 1:
            continue
        jmp_tails = []
        for t in tails:
            tb = cfg.get_block(t)
            if tb and tb.terminator == 'jmp':
                jmp_tails.append(t)
        if jmp_tails:
            best = max(jmp_tails,
                       key=lambda t: cfg.get_block(t).end_idx)
            loop.back_edge_source = best
        else:
            best = max(tails,
                       key=lambda t: (cfg.get_block(t).end_idx
                                      if cfg.get_block(t) else -1))
            loop.back_edge_source = best
        loop.loop_type, loop.cond_block = _classify_loop(
            cfg, instructions, loop.header, loop.back_edge_source,
            loop.body_blocks, loop.exit_blocks)

    result = sorted(merged.values(), key=lambda l: l.header)
    return result

def _classify_loop(cfg: CFG, instructions: List[Instruction],
                   header: int, tail: int, body: Set[int],
                   exit_blocks: Set[int]) -> Tuple[str, Optional[int]]:
    header_block = cfg.get_block(header)
    tail_block = cfg.get_block(tail)

    if header_block is None or tail_block is None:
        return 'infinite', None

    if header == tail and header_block.terminator in ('jf', 'jnf'):
        return 'do_while', tail

    if header_block.terminator in ('jf', 'jnf'):
        for succ_id in header_block.successors:
            if succ_id not in body:
                return 'while', header

    if tail_block.terminator in ('jf', 'jnf'):
        if header in tail_block.successors:
            return 'do_while', tail

    if tail_block.terminator == 'jmp':
        for block_id in body:
            block = cfg.get_block(block_id)
            if block and block.terminator in ('jf', 'jnf'):
                for succ_id in block.successors:
                    if succ_id not in body and succ_id >= 0:
                        if block_id == header:
                            return 'while', header

        if header_block.terminator in ('jf', 'jnf'):
            _NO_SIDE_EFFECT_OPS = frozenset({
                VM.CONST, VM.CP, VM.CL, VM.CCL,
                VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT, VM.SETF, VM.SETNF,
                VM.TT, VM.TF, VM.NF,
                VM.JF, VM.JNF, VM.JMP,
                VM.CALLD, VM.CALLI, VM.CALL, VM.NEW,
                VM.GPD, VM.GPDS, VM.GPI, VM.GPIS, VM.GLOBAL,
                VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.MOD, VM.IDIV,
                VM.BAND, VM.BOR, VM.BXOR, VM.SAR, VM.SAL, VM.SR,
                VM.TYPEOF, VM.TYPEOFD, VM.TYPEOFI,
                VM.CHKINS, VM.CHKINV,
                VM.INT, VM.REAL, VM.STR,
                VM.LNOT, VM.CHS, VM.NUM, VM.BNOT,
                VM.CHGTHIS,
            })
            visited_chain = set()
            chain_queue = [header]
            has_exit = False
            all_pure = True
            while chain_queue and all_pure:
                bid = chain_queue.pop(0)
                if bid in visited_chain:
                    continue
                visited_chain.add(bid)
                blk = cfg.get_block(bid)
                if blk is None or bid not in body:
                    continue
                if blk.terminator not in ('jf', 'jnf'):
                    continue
                for idx in range(blk.start_idx, blk.end_idx):
                    if idx >= len(instructions):
                        break
                    instr = instructions[idx]
                    if instr.op not in _NO_SIDE_EFFECT_OPS:
                        all_pure = False
                        break
                    if instr.op in (VM.CALLD, VM.CALLI, VM.CALL):
                        res_reg = instr.operands[0] if instr.operands else 0
                        if res_reg == 0 or res_reg < -2:
                            all_pure = False
                            break
                if not all_pure:
                    break
                for succ_id in blk.successors:
                    if succ_id not in body and succ_id >= 0:
                        has_exit = True
                    elif succ_id in body and succ_id not in visited_chain:
                        chain_queue.append(succ_id)
            if all_pure and has_exit:
                return 'while', header

    return 'infinite', None

def _is_switch_back_jump(cfg: CFG, instructions: List[Instruction],
                          header: int, tail: int, body: Set[int]) -> bool:
    header_block = cfg.get_block(header)
    if header_block is None:
        return False

    header_has_ceq = False
    header_ceq_reg = None
    for idx in range(header_block.start_idx, header_block.end_idx):
        if idx < len(instructions) and instructions[idx].op == VM.CEQ:
            header_has_ceq = True
            header_ceq_reg = instructions[idx].operands[0]
            break

    if not header_has_ceq:
        return False

    ceq_count = 1
    for block_id in sorted(body):
        if block_id == header:
            continue
        block = cfg.get_block(block_id)
        if block is None:
            continue
        for idx in range(block.start_idx, block.end_idx):
            if idx < len(instructions) and instructions[idx].op == VM.CEQ:
                if instructions[idx].operands[0] == header_ceq_reg:
                    ceq_count += 1
                break

    return ceq_count >= 3

def _is_short_circuit_expr(cfg: CFG, instructions: List[Instruction],
                            block_id: int) -> Optional[int]:
    block = cfg.get_block(block_id)
    if block is None or block.terminator not in ('jf', 'jnf'):
        return None

    cond_jump_idx = block.end_idx - 1
    cond_instr = instructions[cond_jump_idx]
    if cond_instr.op not in (VM.JF, VM.JNF):
        return None

    target_addr = cond_instr.addr + cond_instr.operands[0]

    target_idx = None
    for i, instr in enumerate(instructions):
        if instr.addr == target_addr:
            target_idx = i
            break

    if target_idx is None or target_idx >= len(instructions):
        return None

    target_instr = instructions[target_idx]
    if target_instr.op in (VM.SETF, VM.SETNF):
        fall_through_idx = cond_jump_idx + 1
        simple_ok = True
        for j in range(fall_through_idx, target_idx):
            instr = instructions[j]
            if instr.op in (VM.JF, VM.JNF, VM.JMP, VM.RET, VM.SRV, VM.ENTRY):
                if instr.op in (VM.JF, VM.JNF):
                    inner_target = instr.addr + instr.operands[0]
                    if inner_target == target_addr:
                        continue
                simple_ok = False
                break
        if simple_ok:
            return target_idx + 1

    fall_through_idx = cond_jump_idx + 1

    if target_instr.op in (VM.SETF, VM.SETNF):
        setf_idx = target_idx
        setf_addr = target_addr
    else:
        setf_idx = None
        setf_addr = None
        scan_limit = min(fall_through_idx + 60, len(instructions))
        for j in range(fall_through_idx, scan_limit):
            if instructions[j].op in (VM.SETF, VM.SETNF):
                setf_idx = j
                setf_addr = instructions[j].addr
                break

    if setf_idx is None:
        return None

    if target_addr != setf_addr:
        _STMT_ARITH_OPS = frozenset({
            VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.IDIV, VM.MOD,
            VM.BOR, VM.BAND, VM.BXOR, VM.LOR, VM.LAND,
            VM.SAR, VM.SAL, VM.SR,
        })
        _STMT_BREAKING_OPS = frozenset({
            VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS, VM.SPI, VM.SPIE, VM.SPIS,
            VM.DELD, VM.DELI, VM.NEW,
            VM.THROW, VM.SRV, VM.INV, VM.EVAL,
        })
        for j in range(fall_through_idx, target_idx):
            instr = instructions[j]
            if instr.op in _STMT_BREAKING_OPS:
                return None
            if instr.op in _STMT_ARITH_OPS and instr.operands and instr.operands[0] < 0:
                return None
            if instr.op in (VM.CALL, VM.CALLD, VM.CALLI):
                result_reg = instr.operands[0] if instr.operands else 0
                if result_reg == 0 or result_reg < -2:
                    return None

    SC_BREAKING_OPS = frozenset({
        VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS, VM.SPI, VM.SPIE, VM.SPIS,
        VM.DELD, VM.DELI, VM.NEW,
        VM.THROW, VM.SRV, VM.INV, VM.EVAL,
    })

    start_addr = instructions[fall_through_idx].addr if fall_through_idx < len(instructions) else 0
    has_direct_setf_target = (target_addr == setf_addr)
    for j in range(fall_through_idx, setf_idx):
        instr = instructions[j]
        if instr.op in SC_BREAKING_OPS:
            return None
        if instr.op in (VM.CALL, VM.CALLD, VM.CALLI):
            result_reg = instr.operands[0] if instr.operands else 0
            if result_reg == 0 or result_reg < -2:
                return None
        if instr.op in (VM.JF, VM.JNF):
            inner_target = instr.addr + instr.operands[0]
            if inner_target == setf_addr:
                has_direct_setf_target = True
                continue
            if start_addr <= inner_target <= setf_addr:
                continue
            return None
        if instr.op == VM.JMP:
            jmp_target = instr.addr + instr.operands[0]
            if jmp_target == setf_addr:
                continue
            if start_addr <= jmp_target <= setf_addr:
                continue
            return None
        if instr.op in (VM.RET, VM.ENTRY):
            return None

    if not has_direct_setf_target:
        return None

    if not (start_addr <= target_addr <= setf_addr):
        return None

    return setf_idx + 1

def _get_short_circuit_end_idx(cfg: CFG, instructions: List[Instruction],
                                 block_id: int) -> Optional[int]:
    return _is_short_circuit_expr(cfg, instructions, block_id)

def _detect_condition_chain(cfg: CFG, start_block_id: int, instructions: List[Instruction] = None) -> Optional[Tuple[List[int], int, int, Optional[int]]]:
    CONDITION_BREAKING_OPS = frozenset({
        VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS, VM.SPI, VM.SPIE, VM.SPIS,
        VM.DELD, VM.DELI, VM.NEW,
        VM.THROW, VM.SRV, VM.INV, VM.EVAL,
        VM.INCPD, VM.DECPD, VM.ADDPD, VM.SUBPD, VM.MULPD, VM.DIVPD,
        VM.IDIVPD, VM.MODPD, VM.BORPD, VM.BANDPD, VM.BXORPD,
        VM.LORPD, VM.LANDPD, VM.SARPD, VM.SALPD, VM.SRPD,
        VM.INCPI, VM.DECPI, VM.ADDPI, VM.SUBPI, VM.MULPI, VM.DIVPI,
        VM.IDIVPI, VM.MODPI, VM.BORPI, VM.BANDPI, VM.BXORPI,
        VM.LORPI, VM.LANDPI, VM.SARPI, VM.SALPI, VM.SRPI,
    })

    chain_blocks = []
    current = start_block_id
    nf_block_ids = set()

    def _is_nf_block(block_id):
        blk = cfg.get_block(block_id)
        if blk is None or instructions is None:
            return False
        if blk.start_idx >= len(instructions):
            return False
        return instructions[blk.start_idx].op == VM.NF

    def _is_pure_condition_block(block):
        if instructions is None:
            return True
        for idx in range(block.start_idx, block.end_idx):
            instr = instructions[idx]
            if instr.op in CONDITION_BREAKING_OPS:
                return False
            if instr.op in (VM.CALL, VM.CALLD, VM.CALLI):
                result_reg = instr.operands[0] if instr.operands else 0
                if result_reg == 0 or result_reg < -2:
                    return False
            if (instr.op in (VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.IDIV, VM.MOD,
                             VM.BOR, VM.BAND, VM.BXOR, VM.SAR, VM.SAL, VM.SR,
                             VM.LOR, VM.LAND)
                    and len(instr.operands) >= 1 and instr.operands[0] < -2):
                return False
            if instr.op == VM.CP and len(instr.operands) >= 2 and instr.operands[0] < -2:
                src_reg = instr.operands[1]
                is_assign_in_cond = False
                dest_local = instr.operands[0]
                for j in range(idx + 1, block.end_idx):
                    later = instructions[j]
                    if later.op in (VM.CDEQ, VM.CEQ, VM.CLT, VM.CGT):
                        if later.operands[0] == src_reg:
                            is_assign_in_cond = True
                        break
                    if later.op in (VM.JF, VM.JNF):
                        break
                    if (later.operands and later.operands[0] == src_reg
                            and later.op not in (VM.CDEQ, VM.CEQ, VM.CLT, VM.CGT,
                                                  VM.TT, VM.TF, VM.NF,
                                                  VM.JF, VM.JNF, VM.JMP)):
                        if (later.op == VM.CP and len(later.operands) >= 2
                                and later.operands[1] == dest_local):
                            continue
                        if later.op == VM.TYPEOF:
                            continue
                        break
                if not is_assign_in_cond:
                    for j2 in range(idx + 1, block.end_idx):
                        i2 = instructions[j2]
                        if (i2.op in (VM.CALLD, VM.CALLI)
                                and len(i2.operands) >= 2
                                and i2.operands[1] == src_reg):
                            derived = i2.operands[0]
                            for k2 in range(j2 + 1, block.end_idx):
                                i3 = instructions[k2]
                                if i3.op in (VM.CDEQ, VM.CEQ, VM.CLT, VM.CGT):
                                    if i3.operands[0] == derived:
                                        is_assign_in_cond = True
                                    break
                                if i3.op in (VM.JF, VM.JNF):
                                    break
                            break
                        if (i2.operands and i2.operands[0] == src_reg
                                and i2.op not in (VM.CDEQ, VM.CEQ, VM.CLT, VM.CGT,
                                                  VM.TT, VM.TF, VM.NF,
                                                  VM.JF, VM.JNF, VM.JMP)):
                            break
                        if i2.op in (VM.JF, VM.JNF):
                            break
                if not is_assign_in_cond:
                    return False
            if instr.op in (VM.INC, VM.DEC) and len(instr.operands) >= 1:
                target_reg = instr.operands[0]
                if target_reg < -2:
                    is_postfix = (idx > block.start_idx
                                  and instructions[idx - 1].op == VM.CP
                                  and len(instructions[idx - 1].operands) >= 2
                                  and instructions[idx - 1].operands[1] == target_reg
                                  and instructions[idx - 1].operands[0] >= 0)
                    if not is_postfix:
                        used_in_cmp = False
                        for j in range(idx + 1, block.end_idx):
                            later = instructions[j]
                            if later.op in (VM.CDEQ, VM.CEQ, VM.CLT, VM.CGT):
                                if target_reg in later.operands:
                                    used_in_cmp = True
                                break
                            if later.op in (VM.JF, VM.JNF, VM.JMP):
                                break
                        if not used_in_cmp:
                            return False
        return True

    def _try_collect_nf_and_continue(block):
        if block.terminator != 'fall' or not block.successors:
            return None
        nf_candidate_id = block.successors[0]
        nf_candidate = cfg.get_block(nf_candidate_id)
        if nf_candidate is None:
            return None
        if not _is_nf_block(nf_candidate_id):
            return None

        if _is_pure_condition_block(block):
            chain_blocks.append(block.id)

        while True:
            chain_blocks.append(nf_candidate_id)
            nf_block_ids.add(nf_candidate_id)
            if nf_candidate.terminator in ('jf', 'jnf'):
                break
            if nf_candidate.terminator == 'fall' and nf_candidate.successors:
                next_id = nf_candidate.successors[0]
                next_block = cfg.get_block(next_id)
                if next_block is not None and _is_nf_block(next_id):
                    nf_candidate_id = next_id
                    nf_candidate = next_block
                    continue
            return None

        if nf_candidate.terminator == 'jf':
            nf_fall = nf_candidate.cond_false
        else:
            nf_fall = nf_candidate.cond_true
        if nf_fall is not None:
            nf_fall_block = cfg.get_block(nf_fall)
            if (nf_fall_block is not None and
                    nf_fall_block.terminator in ('jf', 'jnf') and
                    _is_pure_condition_block(nf_fall_block)):
                return nf_fall
        return None

    while current is not None:
        block = cfg.get_block(current)
        if block is None:
            break

        if block.terminator in ('jf', 'jnf'):
            if block.cond_true == block.cond_false:
                break
            if chain_blocks:
                if not _is_pure_condition_block(block):
                    break
                for pred_id in block.predecessors:
                    pred = cfg.get_block(pred_id)
                    if pred is not None and pred.start_idx > block.start_idx:
                        break
                else:
                    chain_blocks.append(current)
                    if block.terminator == 'jf':
                        current = block.cond_false
                    else:
                        current = block.cond_true
                    continue
                break
            chain_blocks.append(current)
            if block.terminator == 'jf':
                current = block.cond_false
            else:
                current = block.cond_true
            continue

        if block.terminator == 'fall' and chain_blocks:
            next_after_nf = _try_collect_nf_and_continue(block)
            if next_after_nf is not None:
                current = next_after_nf
                continue
            break

        break

    if len(chain_blocks) < 2:
        return None

    last_block = cfg.get_block(chain_blocks[-1])
    if last_block.terminator == 'jnf':
        body_block = last_block.cond_true
        else_block = last_block.cond_false
    elif last_block.terminator == 'jf':
        body_block = last_block.cond_false
        else_block = last_block.cond_true
    else:
        return None

    if body_block is None or else_block is None:
        return None
    if body_block < 0 or else_block < 0:
        return None

    chain_set = set(chain_blocks)
    if body_block in chain_set or else_block in chain_set:
        return None

    def _resolve_jmp_target(bid):
        blk = cfg.get_block(bid)
        if blk is None:
            return bid
        if blk.terminator == 'jmp' and len(blk.successors) == 1:
            if blk.end_idx - blk.start_idx == 1:
                return blk.successors[0]
        return bid

    resolved_body = _resolve_jmp_target(body_block)
    resolved_else = _resolve_jmp_target(else_block)

    valid_external = {body_block, else_block}
    if resolved_body != body_block:
        valid_external.add(resolved_body)
    if resolved_else != else_block:
        valid_external.add(resolved_else)
    for nf_id in nf_block_ids:
        valid_external.add(nf_id)

    while len(chain_blocks) >= 2:
        chain_set = set(chain_blocks)
        all_valid = True
        for bid in chain_blocks:
            blk = cfg.get_block(bid)
            for succ in blk.successors:
                if succ not in chain_set and succ not in valid_external:
                    all_valid = False
                    break
            if not all_valid:
                break
        if all_valid:
            break
        removed = chain_blocks.pop()
        if removed in nf_block_ids:
            nf_block_ids.discard(removed)
        if not chain_blocks:
            break
        last_block = cfg.get_block(chain_blocks[-1])
        if last_block.terminator == 'jnf':
            body_block = last_block.cond_true
            else_block = last_block.cond_false
        elif last_block.terminator == 'jf':
            body_block = last_block.cond_false
            else_block = last_block.cond_true
        else:
            break
        if body_block is None or else_block is None or body_block < 0 or else_block < 0:
            break
        resolved_body = _resolve_jmp_target(body_block)
        resolved_else = _resolve_jmp_target(else_block)
        valid_external = {body_block, else_block}
        if resolved_body != body_block:
            valid_external.add(resolved_body)
        if resolved_else != else_block:
            valid_external.add(resolved_else)
        for nf_id in nf_block_ids:
            valid_external.add(nf_id)

    if len(chain_blocks) < 2:
        return None

    chain_set = set(chain_blocks)
    for bid in chain_blocks:
        blk = cfg.get_block(bid)
        for succ in blk.successors:
            if succ not in chain_set and succ not in valid_external:
                return None

    return chain_blocks, body_block, else_block, nf_block_ids

def detect_switch_at(cfg: CFG, instructions: List[Instruction],
                     block_id: int) -> Optional[Dict]:
    block = cfg.get_block(block_id)
    if block is None or block.terminator not in ('jf', 'jnf'):
        return None

    ceq_instr = None
    ref_reg = None
    for idx in range(block.start_idx, block.end_idx):
        instr = instructions[idx]
        if instr.op == VM.CEQ:
            ceq_instr = instr
            ref_reg = instr.operands[0]
            break

    if ceq_instr is None or ref_reg is None:
        return None

    if ref_reg < 0:
        return None

    if block.terminator != 'jnf':
        return None

    non_writing_ops = {VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT, VM.CHKINS,
                       VM.TT, VM.TF, VM.NF, VM.JMP, VM.JF, VM.JNF,
                       VM.RET, VM.SRV, VM.THROW, VM.EXTRY, VM.ENTRY,
                       VM.SETF, VM.SETNF, VM.CHKINV}

    def _has_compound_condition(case_bid, next_case_bid):
        cb = cfg.get_block(case_bid)
        if cb is None:
            return False
        fall_through_bid = cb.cond_true
        ft = cfg.get_block(fall_through_bid) if fall_through_bid is not None else None
        if ft is None:
            return False
        if ft.terminator == 'jnf':
            if ft.cond_false == next_case_bid:
                return True
        return False

    case_blocks = [block_id]
    current = block.cond_false

    visited = set()
    while current is not None and current not in visited:
        visited.add(current)
        next_block = cfg.get_block(current)
        if next_block is None:
            break

        has_ceq = False
        ref_modified = False
        for idx in range(next_block.start_idx, next_block.end_idx):
            instr = instructions[idx]
            if instr.op == VM.CEQ and instr.operands[0] == ref_reg:
                has_ceq = True
                break
            if (instr.op not in non_writing_ops and
                    len(instr.operands) > 0 and instr.operands[0] == ref_reg):
                ref_modified = True
                break

        if has_ceq and not ref_modified and next_block.terminator in ('jnf',):
            case_blocks.append(current)
            current = next_block.cond_false
        else:
            break

    if len(case_blocks) < 2:
        return None

    for i in range(len(case_blocks) - 1):
        next_case_bid = case_blocks[i + 1]
        if _has_compound_condition(case_blocks[i], next_case_bid):
            return None
    if current is not None and len(case_blocks) < 3:
        if _has_compound_condition(case_blocks[-1], current):
            return None

    return {
        'case_blocks': case_blocks,
        'ref_reg': ref_reg,
        'default_or_end': current
    }

def detect_try_at(cfg: CFG, instructions: List[Instruction],
                  block_id: int) -> Optional[Dict]:
    block = cfg.get_block(block_id)
    if block is None or block.terminator != 'entry':
        return None

    last_instr = instructions[block.end_idx - 1]
    if last_instr.op != VM.ENTRY:
        return None

    catch_offset = last_instr.operands[0]
    exception_reg = last_instr.operands[1]
    catch_addr = last_instr.addr + catch_offset

    catch_block_id = cfg.addr_to_block.get(catch_addr)
    if catch_block_id is None:
        return None

    try_body_start = block.end_idx
    try_body_start_id = try_body_start if try_body_start in cfg.blocks else None

    return {
        'entry_block': block_id,
        'try_body_start': try_body_start_id,
        'catch_block': catch_block_id,
        'exception_reg': exception_reg,
    }

def build_region_tree(cfg: CFG, instructions: List[Instruction],
                      loops: List[LoopInfo]) -> Region:
    loop_by_header = {loop.header: loop for loop in loops}

    processed = set()
    processed.add(VIRTUAL_ENTRY_ID)
    processed.add(VIRTUAL_EXIT_ID)

    root = _build_region_recursive(
        cfg, instructions, 0, loop_by_header, processed, None, set()
    )

    return root

def _build_region_recursive(cfg: CFG, instructions: List[Instruction],
                             entry_block_id: int,
                             loop_by_header: Dict[int, LoopInfo],
                             processed: Set[int],
                             containing_loop: Optional[LoopInfo],
                             loop_blocks: Set[int],
                             switch_exit_addr: Optional[int] = None) -> Region:
    children = []
    all_blocks = set()
    current = entry_block_id

    valid_blocks = loop_blocks if loop_blocks else None

    visited_in_sequence = set()

    while current is not None and current >= 0 and current not in processed:
        if valid_blocks is not None and current not in valid_blocks:
            break

        if current in visited_in_sequence:
            break
        visited_in_sequence.add(current)

        block = cfg.get_block(current)
        if block is None:
            break

        if current in loop_by_header and current not in processed:
            loop_info = loop_by_header[current]
            loop_region = _build_loop_region(
                cfg, instructions, loop_info, loop_by_header, processed,
                switch_exit_addr=switch_exit_addr
            )
            children.append(loop_region)
            all_blocks.update(loop_region.blocks)
            current = loop_region.exit_block
            continue

        if block.terminator == 'entry':
            try_info = detect_try_at(cfg, instructions, current)
            if try_info:
                try_region = _build_try_region(
                    cfg, instructions, try_info, loop_by_header, processed,
                    containing_loop, loop_blocks
                )
                children.append(try_region)
                all_blocks.update(try_region.blocks)
                current = try_region.exit_block
                continue

        if block.terminator in ('jf', 'jnf'):
            sc_end = _is_short_circuit_expr(cfg, instructions, current)
            if sc_end is not None:
                remaining_blocks = set()
                remaining_end = sc_end
                for bid in sorted(cfg.blocks.keys()):
                    if bid < 0:
                        continue
                    if bid in processed:
                        continue
                    if valid_blocks is not None and bid not in valid_blocks:
                        continue
                    b = cfg.get_block(bid)
                    if b and b.start_idx >= block.start_idx:
                        remaining_blocks.add(bid)
                        remaining_end = max(remaining_end, b.end_idx)

                candidate_blocks = sorted(
                    [(bid, cfg.get_block(bid)) for bid in remaining_blocks],
                    key=lambda x: x[1].start_idx
                )

                sc_blocks = set()
                next_block_id = None
                for bid, b in candidate_blocks:
                    if b.start_idx < sc_end:
                        sc_blocks.add(bid)
                    elif next_block_id is None:
                        next_block_id = bid

                for bid in sc_blocks:
                    processed.add(bid)
                    all_blocks.add(bid)

                sc_actual_end = sc_end
                for bid in sc_blocks:
                    b = cfg.get_block(bid)
                    if b is not None and b.end_idx > sc_actual_end:
                        sc_actual_end = b.end_idx

                sc_region = Region(
                    type=RegionType.SC_EXPR,
                    header_block=current,
                    blocks=sc_blocks
                )
                sc_region._sc_end_idx = sc_actual_end
                children.append(sc_region)
                current = next_block_id
                continue

            switch_info = detect_switch_at(cfg, instructions, current)
            if switch_info is not None:
                switch_region = _build_switch_region(
                    cfg, instructions, switch_info, loop_by_header, processed,
                    containing_loop, loop_blocks
                )
                children.append(switch_region)
                all_blocks.update(switch_region.blocks)
                current = switch_region.exit_block
                continue

            chain_info = _detect_condition_chain(cfg, current, instructions)
            if chain_info is not None:
                chain_blocks, body_blk, else_blk, nf_blk_ids = chain_info
                chain_valid = all(b not in processed for b in chain_blocks)
                if chain_valid:
                    chain_region = _build_condition_chain_if_region(
                        cfg, instructions, chain_blocks, body_blk, else_blk,
                        loop_by_header, processed, containing_loop, loop_blocks,
                        nf_block_ids=nf_blk_ids
                    )
                    children.append(chain_region)
                    all_blocks.update(chain_region.blocks)
                    current = chain_region.exit_block
                    continue

            if_region = _build_if_region(
                cfg, instructions, current, loop_by_header, processed,
                containing_loop, loop_blocks
            )
            children.append(if_region)
            all_blocks.update(if_region.blocks)
            current = if_region.exit_block
            continue

        processed.add(current)
        all_blocks.add(current)
        simple_region = Region(
            type=RegionType.BLOCK,
            header_block=current,
            blocks={current}
        )
        children.append(simple_region)

        if block.terminator == 'fall' and block.successors:
            current = block.successors[0]
        elif block.terminator == 'jmp' and block.successors:
            target = block.successors[0]
            if valid_blocks is not None and target not in valid_blocks:
                if switch_exit_addr is not None:
                    jmp_instr = instructions[block.end_idx - 1]
                    if jmp_instr.op == VM.JMP:
                        jmp_target_addr = jmp_instr.addr + jmp_instr.operands[0]
                        if jmp_target_addr >= switch_exit_addr:
                            simple_region._switch_break_exit = True
                current = None
            else:
                current = target
        elif block.terminator in ('ret', 'throw'):
            current = None
        else:
            current = None

    if len(children) == 1:
        return children[0]

    return Region(
        type=RegionType.SEQUENCE,
        header_block=entry_block_id,
        blocks=all_blocks,
        children=children
    )

def _build_loop_region(cfg: CFG, instructions: List[Instruction],
                        loop_info: LoopInfo,
                        loop_by_header: Dict[int, LoopInfo],
                        processed: Set[int],
                        switch_exit_addr: Optional[int] = None) -> Region:
    header = loop_info.header
    body_blocks = loop_info.body_blocks

    exit_block = None
    exit_candidates = sorted(loop_info.exit_blocks)
    if exit_candidates:
        exit_block = exit_candidates[0]

    header_block = cfg.get_block(header)
    if header_block and header_block.terminator in ('jf', 'jnf'):
        for succ_id in header_block.successors:
            if succ_id not in body_blocks and succ_id >= 0:
                exit_block = succ_id
                break

    for block_id in body_blocks:
        processed.add(block_id)

    body_entry = header
    body_blocks_for_recursive = set(body_blocks)

    if loop_info.loop_type == 'while':
        if header_block:
            for succ_id in header_block.successors:
                if succ_id in body_blocks and succ_id != header:
                    body_entry = succ_id
                    break
        body_blocks_for_recursive.discard(header)
    elif loop_info.loop_type == 'do_while':
        if loop_info.back_edge_source != header:
            body_blocks_for_recursive.discard(loop_info.back_edge_source)

    is_self_loop = (loop_info.back_edge_source == header and
                    loop_info.loop_type == 'do_while')

    if is_self_loop:
        body_region = Region(
            type=RegionType.BLOCK,
            header_block=header,
            blocks={header}
        )
    else:
        saved_loop_entry = loop_by_header.pop(header, None)

        body_processed = set(processed)
        for block_id in body_blocks_for_recursive:
            body_processed.discard(block_id)

        body_region = _build_region_recursive(
            cfg, instructions, body_entry, loop_by_header, body_processed,
            loop_info, body_blocks_for_recursive,
            switch_exit_addr=switch_exit_addr
        )

        if saved_loop_entry is not None:
            loop_by_header[header] = saved_loop_entry

    if loop_info.loop_type == 'while':
        region_type = RegionType.WHILE
    elif loop_info.loop_type == 'do_while':
        region_type = RegionType.DO_WHILE
    else:
        region_type = RegionType.INFINITE

    region = Region(
        type=region_type,
        header_block=header,
        blocks=body_blocks,
        exit_block=exit_block,
        body_region=body_region,
        loop_info=loop_info,
        cond_block=loop_info.cond_block
    )

    return region

def _has_skip_else_jmp(cfg: CFG, else_entry: Optional[int]) -> bool:
    if else_entry is None or else_entry < 0:
        return False
    else_block = cfg.get_block(else_entry)
    if else_block is None:
        return False

    prev_idx = else_block.start_idx - 1
    if prev_idx < 0:
        return False
    prev_block_id = cfg.idx_to_block.get(prev_idx)
    if prev_block_id is None:
        return False
    prev_block = cfg.get_block(prev_block_id)
    if prev_block is None:
        return False

    return (prev_block.terminator == 'jmp'
            and prev_block.end_idx == else_block.start_idx)

def _build_if_region(cfg: CFG, instructions: List[Instruction],
                      cond_block_id: int,
                      loop_by_header: Dict[int, LoopInfo],
                      processed: Set[int],
                      containing_loop: Optional[LoopInfo],
                      loop_blocks: Set[int],
                      switch_exit_addr: Optional[int] = None) -> Region:
    block = cfg.get_block(cond_block_id)
    processed.add(cond_block_id)

    merge_point = get_merge_point(cfg, cond_block_id)

    if block.terminator == 'jf':
        then_entry = block.cond_false
        else_entry = block.cond_true
    else:
        then_entry = block.cond_true
        else_entry = block.cond_false

    if merge_point is None or merge_point < 0:
        if not _has_skip_else_jmp(cfg, else_entry):
            merge_point = else_entry

    all_blocks = {cond_block_id}

    then_blocks = _collect_branch_blocks(cfg, then_entry, merge_point, processed, loop_blocks)
    else_blocks = _collect_branch_blocks(cfg, else_entry, merge_point, processed, loop_blocks)

    shared = then_blocks & else_blocks
    if shared and (merge_point is None or merge_point < 0):
        real_merge = min(shared, key=lambda b: cfg.get_block(b).start_idx if cfg.get_block(b) else float('inf'))
        else_entry_block = cfg.get_block(else_entry) if else_entry is not None else None
        is_switch_shared = False
        if else_entry_block and real_merge in else_entry_block.successors:
            real_merge_block = cfg.get_block(real_merge)
            shared_is_terminal = real_merge_block and all(
                s < 0 for s in real_merge_block.successors
            )
            if shared_is_terminal:
                then_remaining = then_blocks - shared
                is_switch_shared = bool(then_remaining) and all(
                    cfg.get_block(b) and cfg.get_block(b).terminator == 'jmp'
                    and any(s in shared for s in cfg.get_block(b).successors)
                    for b in then_remaining
                )
        if not is_switch_shared:
            merge_point = real_merge
            then_blocks -= shared
            else_blocks -= shared

    then_region = None
    if then_entry is not None and then_entry != merge_point and then_blocks:
        then_processed = set(processed)
        for b in then_blocks:
            then_processed.discard(b)
        then_region = _build_region_recursive(
            cfg, instructions, then_entry, loop_by_header, then_processed,
            containing_loop, then_blocks,
            switch_exit_addr=switch_exit_addr
        )
        all_blocks.update(then_blocks)
        for b in then_blocks:
            processed.add(b)

    else_region = None
    if else_entry is not None and else_entry != merge_point and else_blocks:
        else_processed = set(processed)
        for b in else_blocks:
            else_processed.discard(b)
        else_region = _build_region_recursive(
            cfg, instructions, else_entry, loop_by_header, else_processed,
            containing_loop, else_blocks,
            switch_exit_addr=switch_exit_addr
        )
        all_blocks.update(else_blocks)
        for b in else_blocks:
            processed.add(b)

    if then_region and else_region:
        region_type = RegionType.IF_THEN_ELSE
    else:
        region_type = RegionType.IF_THEN

    return Region(
        type=region_type,
        header_block=cond_block_id,
        blocks=all_blocks,
        then_region=then_region,
        else_region=else_region,
        cond_block=cond_block_id,
        exit_block=merge_point if (merge_point is not None and merge_point >= 0) else None
    )

def _build_condition_chain_if_region(cfg: CFG, instructions: List[Instruction],
                                      chain_blocks: List[int],
                                      body_block: int, else_block: int,
                                      loop_by_header: Dict[int, LoopInfo],
                                      processed: Set[int],
                                      containing_loop: Optional[LoopInfo],
                                      loop_blocks: Set[int],
                                      nf_block_ids: Optional[set] = None,
                                      switch_exit_addr: Optional[int] = None) -> Region:
    if nf_block_ids is None:
        nf_block_ids = set()

    all_blocks = set(chain_blocks)
    for bid in chain_blocks:
        processed.add(bid)

    merge_point = get_merge_point(cfg, chain_blocks[0])

    if merge_point is None or merge_point < 0:
        if not _has_skip_else_jmp(cfg, else_block):
            merge_point = else_block

    then_blocks = _collect_branch_blocks(cfg, body_block, merge_point, processed, loop_blocks)
    else_blocks = _collect_branch_blocks(cfg, else_block, merge_point, processed, loop_blocks)

    shared = then_blocks & else_blocks
    if shared and (merge_point is None or merge_point < 0):
        real_merge = min(shared, key=lambda b: cfg.get_block(b).start_idx if cfg.get_block(b) else float('inf'))
        else_entry_block = cfg.get_block(else_block) if else_block is not None else None
        is_switch_shared = False
        if else_entry_block and real_merge in else_entry_block.successors:
            real_merge_block = cfg.get_block(real_merge)
            shared_is_terminal = real_merge_block and all(
                s < 0 for s in real_merge_block.successors
            )
            if shared_is_terminal:
                then_remaining = then_blocks - shared
                is_switch_shared = bool(then_remaining) and all(
                    cfg.get_block(b) and cfg.get_block(b).terminator == 'jmp'
                    and any(s in shared for s in cfg.get_block(b).successors)
                    for b in then_remaining
                )
        if not is_switch_shared:
            merge_point = real_merge
            then_blocks -= shared
            else_blocks -= shared

    then_region = None
    if body_block is not None and body_block != merge_point and then_blocks:
        then_processed = set(processed)
        for b in then_blocks:
            then_processed.discard(b)
        then_region = _build_region_recursive(
            cfg, instructions, body_block, loop_by_header, then_processed,
            containing_loop, then_blocks,
            switch_exit_addr=switch_exit_addr
        )
        all_blocks.update(then_blocks)
        for b in then_blocks:
            processed.add(b)

    else_region = None
    if else_block is not None and else_block != merge_point and else_blocks:
        else_processed = set(processed)
        for b in else_blocks:
            else_processed.discard(b)
        else_region = _build_region_recursive(
            cfg, instructions, else_block, loop_by_header, else_processed,
            containing_loop, else_blocks,
            switch_exit_addr=switch_exit_addr
        )
        all_blocks.update(else_blocks)
        for b in else_blocks:
            processed.add(b)

    region_type = RegionType.IF_THEN_ELSE if (then_region and else_region) else RegionType.IF_THEN

    region = Region(
        type=region_type,
        header_block=chain_blocks[0],
        blocks=all_blocks,
        then_region=then_region,
        else_region=else_region,
        cond_block=chain_blocks[0],
        exit_block=merge_point if (merge_point is not None and merge_point >= 0) else None
    )

    region._condition_chain = chain_blocks
    region._chain_body_block = body_block
    region._chain_else_block = else_block
    region._chain_nf_block_ids = nf_block_ids
    return region

def _collect_branch_blocks(cfg: CFG, entry: Optional[int], merge_point: Optional[int],
                            processed: Set[int], loop_blocks: Set[int]) -> Set[int]:
    if entry is None or entry < 0:
        return set()
    if entry == merge_point:
        return set()

    blocks = set()
    worklist = [entry]
    visited = set()

    while worklist:
        block_id = worklist.pop()
        if block_id in visited or block_id < 0:
            continue
        if block_id == merge_point:
            continue
        if loop_blocks and block_id not in loop_blocks:
            continue
        if block_id in processed:
            continue
        visited.add(block_id)
        blocks.add(block_id)

        block = cfg.get_block(block_id)
        if block is None:
            continue

        for succ_id in block.successors:
            if succ_id not in visited and succ_id != merge_point:
                worklist.append(succ_id)

    return blocks

def _find_switch_end(cfg: CFG, instructions: List[Instruction],
                     case_blocks: List[int], default_or_end: Optional[int],
                     loop_update_addr: Optional[int] = None
                     ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    first_case_block = cfg.get_block(case_blocks[0])
    last_case_block = cfg.get_block(case_blocks[-1])
    if first_case_block is None or last_case_block is None:
        return None, None, None, None

    switch_start_idx = first_case_block.start_idx
    switch_start_addr = instructions[switch_start_idx].addr

    case_cond_block_ids = set(case_blocks)
    all_block_ids = sorted(bid for bid in cfg.blocks.keys() if bid >= 0)

    default_or_end_addr = None
    if default_or_end is not None:
        dor_block = cfg.get_block(default_or_end)
        if dor_block is not None:
            default_or_end_addr = instructions[dor_block.start_idx].addr

    last_case_end_addr = instructions[last_case_block.end_idx - 1].addr
    case_body_entry_addrs = set()
    for cb_id in case_blocks:
        cb = cfg.get_block(cb_id)
        if cb and cb.cond_true is not None:
            entry_block = cfg.get_block(cb.cond_true)
            if entry_block:
                case_body_entry_addrs.add(instructions[entry_block.start_idx].addr)

    scan_end_idx = None
    if default_or_end is not None:
        dor_block = cfg.get_block(default_or_end)
        if dor_block is not None and dor_block.start_idx > switch_start_idx:
            scan_end_idx = dor_block.start_idx

    break_targets = {}
    for bid in all_block_ids:
        b = cfg.get_block(bid)
        if b is None or b.start_idx < switch_start_idx:
            continue
        if scan_end_idx is not None and b.start_idx >= scan_end_idx:
            continue
        if bid in case_cond_block_ids:
            continue
        if b.terminator == 'jmp' and b.end_idx - b.start_idx >= 1:
            jmp_instr = instructions[b.end_idx - 1]
            target_addr = jmp_instr.addr + jmp_instr.operands[0]
            if target_addr > last_case_end_addr and target_addr not in case_body_entry_addrs:
                break_targets[target_addr] = break_targets.get(target_addr, 0) + 1
    exit_end_addr = None
    exit_switch_bid = None
    backward_jmp_bid = None

    if break_targets:
        if (default_or_end_addr is not None and
                default_or_end_addr in break_targets and
                any(t > default_or_end_addr for t in break_targets)):
            exit_code_count = 0
            if scan_end_idx is not None:
                idx = scan_end_idx - 1
                while idx >= 0:
                    instr = instructions[idx]
                    if instr.op != VM.JMP:
                        break
                    t = instr.addr + instr.operands[0]
                    if t != default_or_end_addr:
                        break
                    bid = cfg.idx_to_block.get(idx)
                    if bid is None:
                        break
                    b = cfg.get_block(bid)
                    if b is None or b.end_idx - b.start_idx != 1:
                        break
                    exit_code_count += 1
                    idx -= 1
            larger_targets = [t for t in break_targets if t > default_or_end_addr]
            is_tail_padding = (loop_update_addr is not None
                               and all(t == loop_update_addr for t in larger_targets))
            if not is_tail_padding:
                del break_targets[default_or_end_addr]
        exit_end_addr = max(break_targets, key=lambda t: (break_targets[t], t))

        if (exit_end_addr == loop_update_addr
                and loop_update_addr is not None
                and default_or_end_addr is not None
                and default_or_end_addr != loop_update_addr):
            bridge_candidates = {t: c for t, c in break_targets.items()
                                 if default_or_end_addr < t < loop_update_addr}
            if bridge_candidates:
                exit_end_addr = max(bridge_candidates,
                                    key=lambda t: (bridge_candidates[t], -t))
            else:
                if default_or_end_addr in break_targets:
                    has_backward_jmp_to_doa = False
                    for bid in all_block_ids:
                        b = cfg.get_block(bid)
                        if b is None:
                            continue
                        block_addr = instructions[b.start_idx].addr
                        if block_addr <= default_or_end_addr:
                            continue
                        if block_addr >= loop_update_addr:
                            break
                        if b.terminator == 'jmp':
                            jmp_instr = instructions[b.end_idx - 1]
                            target = jmp_instr.addr + jmp_instr.operands[0]
                            if target == default_or_end_addr and jmp_instr.operands[0] < 0:
                                has_backward_jmp_to_doa = True
                                break
                    if not has_backward_jmp_to_doa:
                        exit_end_addr = default_or_end_addr

        if exit_end_addr == default_or_end_addr:
            forward_count = break_targets.get(exit_end_addr, 0)
            if forward_count <= len(case_blocks):
                backward_targets = {}
                for bid in all_block_ids:
                    b = cfg.get_block(bid)
                    if b is None or b.start_idx < switch_start_idx:
                        continue
                    if scan_end_idx is not None and b.start_idx >= scan_end_idx:
                        continue
                    if bid in case_cond_block_ids:
                        continue
                    if b.terminator == 'jmp' and b.end_idx - b.start_idx >= 1:
                        jmp_instr = instructions[b.end_idx - 1]
                        target_addr = jmp_instr.addr + jmp_instr.operands[0]
                        if target_addr < switch_start_addr and target_addr not in case_body_entry_addrs:
                            backward_targets[target_addr] = backward_targets.get(target_addr, 0) + 1
                if backward_targets:
                    exit_end_addr = max(backward_targets, key=backward_targets.get)

        if default_or_end_addr is not None and default_or_end_addr != exit_end_addr:
            scan_start_idx = cfg.get_block(default_or_end).start_idx if cfg.get_block(default_or_end) else last_case_block.end_idx
            for cb_id in case_blocks:
                cb = cfg.get_block(cb_id)
                if cb and cb.cond_true is not None:
                    entry_block = cfg.get_block(cb.cond_true)
                    if entry_block:
                        scan_start_idx = max(scan_start_idx, entry_block.start_idx)
            for bid in all_block_ids:
                b = cfg.get_block(bid)
                if b is None or b.start_idx < scan_start_idx:
                    continue
                if bid in case_cond_block_ids:
                    continue
                if b.terminator == 'jmp':
                    jmp_instr = instructions[b.end_idx - 1]
                    target_addr = jmp_instr.addr + jmp_instr.operands[0]
                    next_idx = b.end_idx
                    if (next_idx < len(instructions) and instructions[next_idx].op == VM.JMP):
                        next_instr = instructions[next_idx]
                        next_target = next_instr.addr + next_instr.operands[0]
                        if (target_addr == exit_end_addr and
                                (next_target >= switch_start_addr or
                                 (default_or_end_addr is not None and next_target == default_or_end_addr)) and
                                (next_target < exit_end_addr or exit_end_addr < switch_start_addr)):
                            back_bid = cfg.idx_to_block.get(next_idx)
                            is_single = (b.end_idx - b.start_idx == 1)
                            if is_single:
                                exit_switch_bid = bid
                                backward_jmp_bid = back_bid
                            else:
                                exit_switch_bid = None
                                backward_jmp_bid = back_bid
                            break

        default_body_addr = None
        if default_or_end_addr is not None and default_or_end_addr != exit_end_addr:
            default_body_addr = default_or_end_addr

        return exit_end_addr, default_body_addr, exit_switch_bid, backward_jmp_bid

    scan_start_idx = last_case_block.end_idx
    if default_or_end is not None:
        dor_block = cfg.get_block(default_or_end)
        if dor_block is not None:
            scan_start_idx = dor_block.start_idx

    for bid in all_block_ids:
        b = cfg.get_block(bid)
        if b is None or b.start_idx < scan_start_idx:
            continue
        if bid in case_cond_block_ids:
            continue

        if b.terminator == 'jmp':
            jmp_instr = instructions[b.end_idx - 1]
            target_addr = jmp_instr.addr + jmp_instr.operands[0]
            next_idx = b.end_idx
            if (next_idx < len(instructions) and instructions[next_idx].op == VM.JMP):
                next_instr = instructions[next_idx]
                next_target = next_instr.addr + next_instr.operands[0]
                if (target_addr > jmp_instr.addr and
                        next_target < target_addr and
                        (next_target >= switch_start_addr or
                         (default_or_end_addr is not None and next_target == default_or_end_addr)) and
                        next_target <= (default_or_end_addr if default_or_end_addr else target_addr)):
                    exit_end_addr = target_addr
                    back_bid = cfg.idx_to_block.get(next_idx)
                    is_single = (b.end_idx - b.start_idx == 1)
                    if is_single:
                        exit_switch_bid = bid
                        backward_jmp_bid = back_bid
                    else:
                        exit_switch_bid = None
                        backward_jmp_bid = back_bid

                    default_body_addr = next_target if next_target != default_or_end_addr else None
                    if default_or_end_addr is not None and default_or_end_addr != exit_end_addr:
                        default_body_addr = default_or_end_addr
                    return exit_end_addr, default_body_addr, exit_switch_bid, backward_jmp_bid

    if default_or_end is not None:
        dor_block = cfg.get_block(default_or_end)
        if dor_block:
            exit_end_addr = instructions[dor_block.start_idx].addr

    return exit_end_addr, None, None, None

def _build_switch_region(cfg: CFG, instructions: List[Instruction],
                          switch_info: Dict,
                          loop_by_header: Dict[int, LoopInfo],
                          processed: Set[int],
                          containing_loop: Optional[LoopInfo],
                          loop_blocks: Set[int],
                          switch_exit_addr: Optional[int] = None) -> Region:
    case_blocks = switch_info['case_blocks']
    ref_reg = switch_info['ref_reg']
    default_or_end = switch_info.get('default_or_end')
    first_case = case_blocks[0]

    all_blocks = set()

    for block_id in case_blocks:
        processed.add(block_id)
        all_blocks.add(block_id)

    loop_update_addr = None
    if containing_loop is not None:
        tail_block = cfg.get_block(containing_loop.back_edge_source)
        if tail_block:
            loop_update_addr = instructions[tail_block.start_idx].addr

    exit_end_addr, default_body_addr, exit_switch_bid, backward_jmp_bid = _find_switch_end(
        cfg, instructions, case_blocks, default_or_end, loop_update_addr
    )

    exit_end_block = None
    if exit_end_addr is not None:
        exit_end_block = cfg.addr_to_block.get(exit_end_addr)

    default_body_block = None
    if default_body_addr is not None:
        default_body_block = cfg.addr_to_block.get(default_body_addr)

    case_body_map = []

    for cb_id in case_blocks:
        cb = cfg.get_block(cb_id)
        if cb is None:
            continue
        body_entry = cb.cond_true
        ceq_const_idx = None
        for idx in range(cb.start_idx, cb.end_idx):
            instr = instructions[idx]
            if instr.op == VM.CEQ and instr.operands[0] == ref_reg:
                ceq_const_idx = instr.operands[1]
                break
        case_body_map.append((cb_id, body_entry, ceq_const_idx))

    body_groups = OrderedDict()
    for cb_id, body_entry, ceq_const_idx in case_body_map:
        if body_entry not in body_groups:
            body_groups[body_entry] = []
        body_groups[body_entry].append((cb_id, ceq_const_idx))

    body_entry_set = set(body_groups.keys())

    body_entry_addr_set = set()
    body_entry_addr_to_bid = {}
    for be_bid in body_entry_set:
        if be_bid is not None:
            be_block = cfg.get_block(be_bid)
            if be_block:
                be_addr = instructions[be_block.start_idx].addr
                body_entry_addr_set.add(be_addr)
                body_entry_addr_to_bid[be_addr] = be_bid

    if default_body_block is not None:
        body_entry_set.add(default_body_block)

    first_case_block = cfg.get_block(first_case)
    switch_start_addr = instructions[first_case_block.start_idx].addr if first_case_block else 0
    if default_body_addr is not None and default_body_addr >= switch_start_addr:
        body_entry_addr_set.add(default_body_addr)

    def _get_block_addr(bid):
        b = cfg.get_block(bid)
        return instructions[b.start_idx].addr if b else None

    def _get_jmp_target_addr(b):
        if b.terminator == 'jmp' and b.end_idx > b.start_idx:
            jmp_instr = instructions[b.end_idx - 1]
            return jmp_instr.addr + jmp_instr.operands[0]
        return None

    switch_cases = []
    body_entries_ordered = list(body_groups.keys())

    for gi, body_entry in enumerate(body_entries_ordered):
        group = body_groups[body_entry]
        if body_entry is None:
            continue

        body_entry_addr = _get_block_addr(body_entry)

        body_blocks = set()
        worklist = [body_entry]
        visited = set()
        has_break = False
        has_continue = False
        falls_through = False
        falls_through_target = None

        while worklist:
            bid = worklist.pop()
            if bid in visited or bid < 0:
                continue
            bid_addr = _get_block_addr(bid)
            if bid_addr is not None and bid_addr == exit_end_addr:
                has_break = True
                continue
            if bid in all_blocks:
                has_break = True
                continue
            if bid == exit_switch_bid:
                continue
            if bid == backward_jmp_bid:
                continue
            if bid_addr is not None and bid_addr < switch_start_addr:
                has_continue = True
                continue
            if bid in body_entry_set and bid != body_entry:
                falls_through = True
                continue
            if loop_blocks and bid not in loop_blocks:
                continue
            if bid in processed:
                continue

            visited.add(bid)
            body_blocks.add(bid)

            b = cfg.get_block(bid)
            if b is None:
                continue

            if b.terminator == 'jmp':
                target_addr = _get_jmp_target_addr(b)
                if target_addr is not None and target_addr == exit_end_addr:
                    has_break = True
                    continue
                elif (target_addr is not None and loop_update_addr is not None
                      and target_addr == loop_update_addr
                      and target_addr > exit_end_addr):
                    has_continue = True
                    continue
                elif target_addr is not None and target_addr in body_entry_addr_set and target_addr != body_entry_addr:
                    falls_through = True
                    falls_through_target = target_addr
                    continue
                elif b.successors:
                    worklist.append(b.successors[0])
            elif b.terminator in ('ret', 'throw'):
                continue
            else:
                for succ in b.successors:
                    succ_addr = _get_block_addr(succ)
                    if succ_addr is not None and succ_addr == exit_end_addr:
                        has_break = True
                    else:
                        worklist.append(succ)

        for bid in body_blocks:
            processed.add(bid)
            all_blocks.add(bid)

        body_region = None
        if body_blocks:
            body_processed = set(processed)
            for bid in body_blocks:
                body_processed.discard(bid)
            body_region = _build_region_recursive(
                cfg, instructions, body_entry, loop_by_header, body_processed,
                containing_loop, body_blocks,
                switch_exit_addr=exit_end_addr
            )

        body_only_jmp = True
        for bid in body_blocks:
            bb = cfg.get_block(bid)
            if bb is not None:
                for idx in range(bb.start_idx, bb.end_idx):
                    if instructions[idx].op != VM.JMP:
                        body_only_jmp = False
                        break
            if not body_only_jmp:
                break
        is_body_continue = (has_continue and not has_break and not falls_through
                            and len(body_blocks) <= 1 and body_only_jmp)

        for i, (cb_id, ceq_const_idx) in enumerate(group):
            is_last_in_group = (i == len(group) - 1)
            sc = SwitchCase(
                value_expr=ceq_const_idx,
                body_blocks=sorted(body_blocks),
                body_region=body_region if is_last_in_group else None,
                has_break=has_break if is_last_in_group else False,
                fall_through=falls_through if is_last_in_group else True,
                cond_block_id=cb_id,
                body_is_continue=is_body_continue if is_last_in_group else False,
                has_continue=has_continue if is_last_in_group else False
            )
            switch_cases.append(sc)

    default_shared = False
    if default_body_block is not None and default_body_addr != exit_end_addr:
        if default_body_addr in body_entry_addr_to_bid:
            shared_body_bid = body_entry_addr_to_bid[default_body_addr]
            for i, sc in enumerate(switch_cases):
                if shared_body_bid in sc.body_blocks:
                    switch_cases.insert(i, SwitchCase(
                        value_expr=None,
                        body_blocks=[],
                        body_region=None,
                        has_break=False,
                        fall_through=True
                    ))
                    default_shared = True
                    break

        if not default_shared:
            backward_default_continue = False
            if (containing_loop is not None and
                    default_body_addr is not None and
                    default_body_addr < switch_start_addr):
                loop_header_block = cfg.get_block(containing_loop.header)
                if loop_header_block:
                    loop_header_addr = instructions[loop_header_block.start_idx].addr
                    if default_body_addr == loop_header_addr:
                        backward_default_continue = True

            if backward_default_continue:
                switch_cases.append(SwitchCase(
                    value_expr=None,
                    body_blocks=[],
                    body_region=None,
                    has_break=False,
                    fall_through=False,
                    body_is_continue=True
                ))
            else:
                default_blocks = set()
                worklist = [default_body_block]
                visited = set()
                default_has_break = False

                while worklist:
                    bid = worklist.pop()
                    if bid in visited or bid < 0:
                        continue
                    bid_addr = _get_block_addr(bid)
                    if bid_addr is not None and bid_addr == exit_end_addr:
                        default_has_break = True
                        continue
                    if bid == exit_switch_bid:
                        continue
                    if bid == backward_jmp_bid:
                        continue
                    if bid in all_blocks:
                        continue
                    if loop_blocks and bid not in loop_blocks:
                        continue
                    if bid in processed:
                        continue
                    visited.add(bid)
                    default_blocks.add(bid)

                    b = cfg.get_block(bid)
                    if b is None:
                        continue

                    if b.terminator == 'jmp':
                        target_addr = _get_jmp_target_addr(b)
                        if target_addr is not None and target_addr == exit_end_addr:
                            default_has_break = True
                            continue
                        elif b.successors:
                            worklist.append(b.successors[0])
                    elif b.terminator in ('ret', 'throw'):
                        continue
                    else:
                        for succ in b.successors:
                            worklist.append(succ)

                for bid in default_blocks:
                    processed.add(bid)
                    all_blocks.add(bid)

                default_region = None
                if default_blocks:
                    body_processed = set(processed)
                    for bid in default_blocks:
                        body_processed.discard(bid)
                    default_region = _build_region_recursive(
                        cfg, instructions, default_body_block, loop_by_header, body_processed,
                        containing_loop, default_blocks,
                        switch_exit_addr=exit_end_addr
                    )

                switch_cases.append(SwitchCase(
                    value_expr=None,
                    body_blocks=sorted(default_blocks),
                    body_region=default_region,
                    has_break=default_has_break,
                    fall_through=False
                ))

    if exit_switch_bid is not None:
        processed.add(exit_switch_bid)
        all_blocks.add(exit_switch_bid)
    if backward_jmp_bid is not None:
        processed.add(backward_jmp_bid)
        all_blocks.add(backward_jmp_bid)

    return Region(
        type=RegionType.SWITCH,
        header_block=first_case,
        blocks=all_blocks,
        exit_block=exit_end_block,
        switch_ref_reg=ref_reg,
        switch_cases=switch_cases,
        switch_break_target=exit_end_addr
    )

def _build_try_region(cfg: CFG, instructions: List[Instruction],
                       try_info: Dict,
                       loop_by_header: Dict[int, LoopInfo],
                       processed: Set[int],
                       containing_loop: Optional[LoopInfo],
                       loop_blocks: Set[int],
                       switch_exit_addr: Optional[int] = None) -> Region:
    entry_block_id = try_info['entry_block']
    try_body_start = try_info['try_body_start']
    catch_block_id = try_info['catch_block']
    exception_reg = try_info['exception_reg']

    processed.add(entry_block_id)
    all_blocks = {entry_block_id}

    skip_catch_target = None

    try_start_block = cfg.get_block(try_body_start) if try_body_start is not None else None
    catch_start_block = cfg.get_block(catch_block_id)
    try_start_idx = try_start_block.start_idx if try_start_block else 0
    catch_start_idx = catch_start_block.start_idx if catch_start_block else len(instructions)

    extry_idx = None
    for idx in range(try_start_idx, catch_start_idx):
        if instructions[idx].op == VM.EXTRY:
            extry_idx = idx

    if extry_idx is not None:
        for idx in range(extry_idx + 1, catch_start_idx):
            if instructions[idx].op == VM.JMP:
                jmp_target_addr = instructions[idx].addr + instructions[idx].operands[0]
                if jmp_target_addr >= instructions[catch_start_idx].addr if catch_start_idx < len(instructions) else True:
                    skip_catch_target = cfg.addr_to_block.get(jmp_target_addr)
                    break

    merge_point = get_merge_point(cfg, entry_block_id)
    exit_block = skip_catch_target or merge_point
    if exit_block is not None and exit_block < 0:
        exit_block = None

    exit_guard = exit_block is not None and exit_block >= 0 and exit_block not in processed
    if exit_guard:
        processed.add(exit_block)
    try_blocks = set()
    if try_body_start is not None:
        try_blocks = _collect_branch_blocks(
            cfg, try_body_start, catch_block_id, processed, loop_blocks
        )
        all_blocks.update(try_blocks)
        for b in try_blocks:
            processed.add(b)
    if exit_guard:
        processed.discard(exit_block)

    catch_blocks = _collect_branch_blocks(
        cfg, catch_block_id, exit_block, processed, loop_blocks
    )
    all_blocks.update(catch_blocks)
    for b in catch_blocks:
        processed.add(b)

    try_region = None
    if try_body_start is not None and try_blocks:
        try_sub_processed = set(processed)
        for b in try_blocks:
            try_sub_processed.discard(b)
        try_region = _build_region_recursive(
            cfg, instructions, try_body_start, loop_by_header, try_sub_processed,
            containing_loop, try_blocks,
            switch_exit_addr=switch_exit_addr
        )

    catch_region = None
    if catch_blocks:
        catch_sub_processed = set(processed)
        for b in catch_blocks:
            catch_sub_processed.discard(b)
        catch_region = _build_region_recursive(
            cfg, instructions, catch_block_id, loop_by_header, catch_sub_processed,
            containing_loop, catch_blocks,
            switch_exit_addr=switch_exit_addr
        )

    return Region(
        type=RegionType.TRY_CATCH,
        header_block=entry_block_id,
        blocks=all_blocks,
        exit_block=exit_block,
        try_region=try_region,
        catch_region=catch_region,
        catch_block=catch_block_id,
        exception_reg=exception_reg
    )

def generate_code(region: Region, cfg: CFG, instructions: List[Instruction],
                  decompiler: 'Decompiler', obj: CodeObject,
                  loop_context: Optional[Tuple[int, int, int]] = None,
                  is_top_level: bool = False) -> List[Stmt]:
    if region.type == RegionType.BLOCK:
        if (decompiler._for_loop_skip_tail_bid is not None
                and region.header_block == decompiler._for_loop_skip_tail_bid):
            return []
        return _generate_block(region, cfg, instructions, decompiler, obj, loop_context)
    elif region.type == RegionType.SEQUENCE:
        return _generate_sequence(region, cfg, instructions, decompiler, obj, loop_context)
    elif region.type in (RegionType.IF_THEN, RegionType.IF_THEN_ELSE):
        return _generate_if(region, cfg, instructions, decompiler, obj, loop_context)
    elif region.type == RegionType.WHILE:
        return _generate_while(region, cfg, instructions, decompiler, obj)
    elif region.type == RegionType.DO_WHILE:
        return _generate_do_while(region, cfg, instructions, decompiler, obj)
    elif region.type == RegionType.INFINITE:
        return _generate_infinite(region, cfg, instructions, decompiler, obj)
    elif region.type == RegionType.SWITCH:
        return _generate_switch(region, cfg, instructions, decompiler, obj, loop_context)
    elif region.type == RegionType.TRY_CATCH:
        return _generate_try_catch(region, cfg, instructions, decompiler, obj, loop_context)
    elif region.type == RegionType.SC_EXPR:
        return _generate_sc_expr(region, cfg, instructions, decompiler, obj, loop_context)
    else:
        return []

def _generate_sc_expr(region: Region, cfg: CFG, instructions: List[Instruction],
                      decompiler: 'Decompiler', obj: CodeObject,
                      loop_context: Optional[Tuple[int, int, int]]) -> List[Stmt]:
    header_block = cfg.get_block(region.header_block)
    if header_block is None:
        return []
    sc_end_idx = getattr(region, '_sc_end_idx', None)
    if sc_end_idx is None:
        return []

    setf_block_id = None
    setf_instr = None
    for bid in sorted(region.blocks):
        block = cfg.get_block(bid)
        if block is None:
            continue
        for idx in range(block.start_idx, block.end_idx):
            if instructions[idx].op in (VM.SETF, VM.SETNF):
                setf_block_id = bid
                setf_instr = instructions[idx]
                break
        if setf_block_id is not None:
            break

    if setf_block_id is None:
        return decompiler._generate_structured_code(
            instructions, obj, header_block.start_idx, sc_end_idx,
            loop_context=loop_context)

    setf_reg = setf_instr.operands[0]
    is_setnf = (setf_instr.op == VM.SETNF)

    chain_entry_id = _find_sc_chain_entry(
        region.header_block, setf_block_id, cfg, instructions, region.blocks)

    preamble_stmts = []
    chain_entry_block = cfg.get_block(chain_entry_id)
    if chain_entry_block and chain_entry_block.start_idx > header_block.start_idx:
        preamble_stmts = decompiler._generate_structured_code(
            instructions, obj, header_block.start_idx, chain_entry_block.start_idx,
            loop_context=loop_context)

    chain_stmts = []
    expr = _build_sc_chain_expr(
        chain_entry_id, setf_block_id, setf_block_id,
        cfg, instructions, decompiler, obj, region.blocks, chain_stmts)

    if is_setnf:
        expr = decompiler._negate_expr(expr)

    decompiler.regs[setf_reg] = expr

    return preamble_stmts + chain_stmts

def _find_sc_chain_entry(header: int, setf_block_id: int, cfg: CFG,
                          instructions: List[Instruction],
                          region_blocks: Set[int]) -> int:
    current = header
    visited = set()

    while current is not None and current != setf_block_id and current not in visited:
        visited.add(current)
        block = cfg.get_block(current)
        if block is None:
            break

        if block.terminator not in ('jf', 'jnf'):
            next_id = None
            for s in (block.successors or []):
                if s in region_blocks:
                    next_id = s
                    break
            current = next_id
            continue

        if block.terminator == 'jnf':
            fall_through, jump_target = block.cond_true, block.cond_false
        else:
            fall_through, jump_target = block.cond_false, block.cond_true

        if jump_target == setf_block_id:
            return current

        ft_block = cfg.get_block(fall_through)
        if ft_block is not None and ft_block.terminator == 'jmp' and ft_block.successors:
            jmp_target = ft_block.successors[0]
            if jmp_target != setf_block_id and jmp_target in region_blocks:
                if _sc_reaches(jump_target, jmp_target, cfg, region_blocks):
                    merge_block = cfg.get_block(jmp_target)
                    if merge_block is not None:
                        merge_first_op = None
                        for mi in range(merge_block.start_idx, merge_block.end_idx):
                            if instructions[mi].op != VM.NOP:
                                merge_first_op = instructions[mi].op
                                break
                        if merge_first_op in (VM.TT, VM.TF, VM.JF, VM.JNF):
                            return current
                    current = jmp_target
                    continue

        if _sc_body_has_side_effects(fall_through, jump_target, cfg, instructions, region_blocks):
            current = jump_target
        else:
            return current

    return header

def _sc_body_has_side_effects(start_bid: int, end_bid: int, cfg: CFG,
                               instructions: List[Instruction],
                               region_blocks: Set[int]) -> bool:
    current = start_bid
    visited = set()
    while current is not None and current != end_bid and current not in visited:
        visited.add(current)
        block = cfg.get_block(current)
        if block is None:
            break
        for idx in range(block.start_idx, block.end_idx):
            instr = instructions[idx]
            if instr.op == VM.CP and len(instr.operands) >= 1 and instr.operands[0] < -2:
                return True
            if instr.op in (VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS,
                           VM.SPI, VM.SPIE, VM.SPIS):
                return True
            if instr.op in (VM.CALL, VM.CALLD, VM.CALLI):
                if len(instr.operands) >= 1 and (instr.operands[0] == 0 or instr.operands[0] < -2):
                    return True
        next_bid = None
        for s in (block.successors or []):
            if s in region_blocks and s != end_bid:
                next_bid = s
                break
        current = next_bid
    return False

def _build_sc_chain_expr(block_id: int, boundary_id: int, setf_block_id: int,
                          cfg: CFG, instructions: List[Instruction],
                          decompiler: 'Decompiler', obj: CodeObject,
                          region_blocks: Set[int],
                          chain_stmts: List[Stmt]) -> Expr:
    if block_id == boundary_id or block_id == setf_block_id:
        return decompiler._get_condition(False)

    block = cfg.get_block(block_id)
    if block is None:
        return decompiler._get_condition(False)

    if block.terminator in ('jf', 'jnf'):
        if block.terminator == 'jnf':
            fall_through, jump_target = block.cond_true, block.cond_false
        else:
            fall_through, jump_target = block.cond_false, block.cond_true

        preamble, cond, _, deferred = _process_condition_block_preamble(
            instructions, decompiler, obj, block.start_idx, block.end_idx)
        cond, merged = decompiler._apply_cond_side_effects(
            cond, instructions, block.start_idx, block.end_idx - 1)
        _emit_unmerged_side_effects(preamble, deferred, merged)
        cond = _detect_assignment_in_condition(preamble, cond)
        chain_stmts.extend(preamble)

        if jump_target == boundary_id:
            op = '&&' if block.terminator == 'jnf' else '||'
            right = _build_sc_chain_expr(
                fall_through, boundary_id, setf_block_id,
                cfg, instructions, decompiler, obj, region_blocks, chain_stmts)
            return BinaryExpr(cond, op, right)

        if jump_target == setf_block_id:
            if boundary_id != setf_block_id and fall_through == boundary_id:
                return cond
            if boundary_id != setf_block_id and fall_through != boundary_id:
                op = '&&' if block.terminator == 'jnf' else '||'
                right = _build_sc_chain_expr(
                    fall_through, boundary_id, setf_block_id,
                    cfg, instructions, decompiler, obj, region_blocks, chain_stmts)
                return BinaryExpr(cond, op, right)
            op = '&&' if block.terminator == 'jnf' else '||'
            right = _build_sc_chain_expr(
                fall_through, boundary_id, setf_block_id,
                cfg, instructions, decompiler, obj, region_blocks, chain_stmts)
            return BinaryExpr(cond, op, right)

        merge_point = _find_sc_ternary_merge(
            fall_through, jump_target, setf_block_id, boundary_id,
            cfg, region_blocks)

        if merge_point is not None:
            if block.terminator == 'jnf':
                true_entry, false_entry = fall_through, jump_target
                ternary_cond = cond
            else:
                true_entry, false_entry = jump_target, fall_through
                ternary_cond = decompiler._negate_expr(cond)

            true_expr, true_target_reg = _process_sc_ternary_branch(
                true_entry, merge_point, cfg, instructions, decompiler, obj)
            false_expr, false_target_reg = _process_sc_ternary_branch(
                false_entry, merge_point, cfg, instructions, decompiler, obj)

            ternary = TernaryExpr(ternary_cond, true_expr, false_expr)
            decompiler.flag = ternary
            decompiler.flag_negated = False

            ternary_reg = true_target_reg if true_target_reg is not None else false_target_reg
            if ternary_reg is not None:
                decompiler.regs[ternary_reg] = ternary

            return _build_sc_chain_expr(
                merge_point, boundary_id, setf_block_id,
                cfg, instructions, decompiler, obj, region_blocks, chain_stmts)

        inner_op = '&&' if block.terminator == 'jnf' else '||'
        inner = _build_sc_chain_expr(
            fall_through, jump_target, setf_block_id,
            cfg, instructions, decompiler, obj, region_blocks, chain_stmts)
        combined = BinaryExpr(cond, inner_op, inner)

        outer_op = _find_sc_subgroup_exit_op(
            fall_through, jump_target, setf_block_id, cfg, region_blocks)

        if outer_op is not None:
            continuation = _build_sc_chain_expr(
                jump_target, boundary_id, setf_block_id,
                cfg, instructions, decompiler, obj, region_blocks, chain_stmts)
            return BinaryExpr(combined, outer_op, continuation)

        decompiler.flag = combined
        decompiler.flag_negated = False

        return _build_sc_chain_expr(
            jump_target, boundary_id, setf_block_id,
            cfg, instructions, decompiler, obj, region_blocks, chain_stmts)

    elif block.terminator == 'jmp':
        for idx in range(block.start_idx, block.end_idx):
            instr = instructions[idx]
            if instr.op == VM.JMP:
                break
            stmt = decompiler._translate_instruction(instr, obj)
            decompiler._collect_pre_stmts(chain_stmts)
            if stmt:
                chain_stmts.append(stmt)
        return decompiler._get_condition(False)

    else:
        for idx in range(block.start_idx, block.end_idx):
            stmt = decompiler._translate_instruction(instructions[idx], obj)
            decompiler._collect_pre_stmts(chain_stmts)
            if stmt:
                chain_stmts.append(stmt)
        next_id = None
        for s in (block.successors or []):
            if s in region_blocks:
                next_id = s
                break
        if next_id is not None:
            return _build_sc_chain_expr(
                next_id, boundary_id, setf_block_id,
                cfg, instructions, decompiler, obj, region_blocks, chain_stmts)
        return decompiler._get_condition(False)

def _find_sc_ternary_merge(true_entry: int, false_entry: int,
                            setf_block_id: int, boundary_id: int,
                            cfg: CFG, region_blocks: Set[int]) -> Optional[int]:
    current = true_entry
    visited = set()
    while current is not None and current not in visited:
        if current == setf_block_id or current == boundary_id:
            break
        visited.add(current)
        b = cfg.get_block(current)
        if b is None:
            break
        if b.terminator == 'jmp' and b.successors:
            merge = b.successors[0]
            if _sc_reaches(false_entry, merge, cfg, region_blocks):
                return merge
            break
        if b.terminator in ('jf', 'jnf'):
            break
        next_id = None
        for s in (b.successors or []):
            if s in region_blocks:
                next_id = s
                break
        current = next_id
    return None

def _sc_reaches(start_bid: int, target_bid: int, cfg: CFG,
                 region_blocks: Set[int]) -> bool:
    current = start_bid
    visited = set()
    while current is not None and current not in visited:
        if current == target_bid:
            return True
        visited.add(current)
        b = cfg.get_block(current)
        if b is None:
            break
        if b.terminator in ('jf', 'jnf'):
            break
        next_id = None
        for s in (b.successors or []):
            if s in region_blocks:
                next_id = s
                break
        current = next_id
    return False

def _find_sc_subgroup_exit_op(start_bid: int, boundary_bid: int,
                               setf_block_id: int, cfg: CFG,
                               region_blocks: Set[int]) -> Optional[str]:
    current = start_bid
    visited = set()
    exit_op = None

    while current is not None and current != boundary_bid and current not in visited:
        visited.add(current)
        b = cfg.get_block(current)
        if b is None:
            break
        if b.terminator in ('jf', 'jnf'):
            if b.terminator == 'jnf':
                jt = b.cond_false
            else:
                jt = b.cond_true
            if jt == setf_block_id:
                exit_op = '&&' if b.terminator == 'jnf' else '||'
        next_id = None
        if b.terminator == 'jnf':
            next_id = b.cond_true
        elif b.terminator == 'jf':
            next_id = b.cond_false
        elif b.successors:
            for s in b.successors:
                if s in region_blocks:
                    next_id = s
                    break
        current = next_id

    return exit_op

def _process_sc_ternary_branch(entry_bid: int, merge_bid: int,
                                cfg: CFG, instructions: List[Instruction],
                                decompiler: 'Decompiler', obj: CodeObject) -> Tuple[Expr, Optional[int]]:
    saved_regs = dict(decompiler.regs)
    saved_flag = decompiler.flag
    saved_flag_negated = decompiler.flag_negated

    branch_blocks = []
    current = entry_bid
    visited = set()
    while current is not None and current != merge_bid and current not in visited:
        visited.add(current)
        branch_blocks.append(current)
        b = cfg.get_block(current)
        if b is None:
            break
        if b.terminator == 'jmp':
            break
        next_id = None
        for s in (b.successors or []):
            next_id = s
            break
        current = next_id

    target_info = _find_branch_target_reg(cfg, instructions, branch_blocks)

    for bid in branch_blocks:
        b = cfg.get_block(bid)
        if b is None:
            continue
        for idx in range(b.start_idx, b.end_idx):
            instr = instructions[idx]
            if instr.op in (VM.JMP, VM.JF, VM.JNF):
                continue
            decompiler._translate_instruction(instr, obj)

    target_reg = None
    if target_info is not None and target_info[0] > 0 and not target_info[1]:
        expr = decompiler.regs.get(target_info[0], decompiler._get_condition(False))
        target_reg = target_info[0]
    else:
        expr = decompiler._get_condition(False)

    decompiler.regs = dict(saved_regs)
    decompiler.flag = saved_flag
    decompiler.flag_negated = saved_flag_negated

    return expr, target_reg

def _generate_block(region: Region, cfg: CFG, instructions: List[Instruction],
                     decompiler: 'Decompiler', obj: CodeObject,
                     loop_context: Optional[Tuple[int, int, int]]) -> List[Stmt]:
    block = cfg.get_block(region.header_block)
    if block is None:
        return []

    sc_end_idx = getattr(region, '_sc_end_idx', None)
    if sc_end_idx is not None:
        return decompiler._generate_structured_code(
            instructions, obj, block.start_idx, sc_end_idx,
            loop_context=loop_context
        )

    stmts = []
    i = block.start_idx
    jmp_handled = False

    while i < block.end_idx:
        instr = instructions[i]

        if instr.op in (VM.JF, VM.JNF, VM.EXTRY):
            i += 1
            continue

        if instr.op == VM.JMP:
            target = instr.addr + instr.operands[0]
            if loop_context:
                loop_start_addr, loop_exit_addr, continue_target = loop_context
                if target == continue_target:
                    flushed = decompiler._flush_pending_spie()
                    if flushed:
                        stmts.append(flushed)
                    decompiler._collect_pre_stmts(stmts)
                    stmts.append(ContinueStmt())
                    jmp_handled = True
                    i += 1
                    continue
                elif target >= loop_exit_addr:
                    flushed = decompiler._flush_pending_spie()
                    if flushed:
                        stmts.append(flushed)
                    decompiler._collect_pre_stmts(stmts)
                    stmts.append(BreakStmt())
                    jmp_handled = True
                    i += 1
                    continue
            i += 1
            continue

        if instr.op == VM.ENTRY:
            i += 1
            continue

        swap_result = decompiler._try_detect_swap(instructions, obj, i, block.end_idx)
        if swap_result:
            stmts.append(swap_result['stmt'])
            i = swap_result['next_idx']
            continue

        stmt = decompiler._translate_instruction(instr, obj)
        decompiler._collect_pre_stmts(stmts)
        if stmt:
            stmts.append(stmt)
            if decompiler._deferred_cp_stmts:
                stmts.extend(decompiler._deferred_cp_stmts)
                decompiler._deferred_cp_stmts = []
        i += 1

    if decompiler._deferred_cp_stmts:
        stmts.extend(decompiler._deferred_cp_stmts)
        decompiler._deferred_cp_stmts = []

    flushed = decompiler._flush_pending_spie()
    if flushed:
        stmts.append(flushed)

    if getattr(region, '_switch_break_exit', False) and not jmp_handled:
        decompiler._collect_pre_stmts(stmts)
        stmts.append(BreakStmt())

    return stmts

def _generate_block_for_update(block, instructions, decompiler, obj):
    stmts = []
    i = block.start_idx
    while i < block.end_idx:
        instr = instructions[i]
        if instr.op in (VM.JF, VM.JNF, VM.EXTRY):
            i += 1
            continue
        if instr.op == VM.JMP:
            i += 1
            continue
        if instr.op == VM.ENTRY:
            i += 1
            continue
        stmt = decompiler._translate_instruction(instr, obj)
        decompiler._collect_pre_stmts(stmts)
        if stmt:
            stmts.append(stmt)
        i += 1
    flushed = decompiler._flush_pending_spie()
    if flushed:
        stmts.append(flushed)
    return stmts

def _generate_sequence(region: Region, cfg: CFG, instructions: List[Instruction],
                        decompiler: 'Decompiler', obj: CodeObject,
                        loop_context: Optional[Tuple[int, int, int]]) -> List[Stmt]:
    stmts = []
    for child in region.children:
        child_stmts = generate_code(child, cfg, instructions, decompiler, obj, loop_context)
        stmts.extend(child_stmts)

    _absorb_for_init(stmts)

    return stmts

def _absorb_for_init(stmts: List[Stmt]) -> None:
    import re
    i = 1
    while i < len(stmts):
        if isinstance(stmts[i], ForStmt) and stmts[i].init is None:
            prev = stmts[i - 1]
            if isinstance(prev, (ExprStmt, VarDeclStmt)):
                if _is_matching_for_init(prev, stmts[i]):
                    if isinstance(prev, VarDeclStmt):
                        var_name = prev.name
                        used_after = False
                        pattern = re.compile(r'\b' + re.escape(var_name) + r'\b')
                        for j in range(i + 1, len(stmts)):
                            if pattern.search(stmts[j].to_source()):
                                used_after = True
                                break
                        if used_after:
                            if prev.value is not None:
                                stmts[i].init = AssignExpr(VarExpr(var_name), prev.value)
                                prev.value = None
                            i += 1
                            continue
                        stmts[i].init = prev
                    else:
                        stmts[i].init = prev.expr
                    stmts.pop(i - 1)
                    continue
        i += 1

def _get_var_name_from_expr(expr) -> Optional[str]:
    if isinstance(expr, VarExpr):
        return expr.name
    if isinstance(expr, AssignExpr):
        return _get_var_name_from_expr(expr.target)
    return None

def _expr_contains_var(expr, var_name: str) -> bool:
    if expr is None:
        return False
    if isinstance(expr, VarExpr):
        return expr.name == var_name
    if isinstance(expr, BinaryExpr):
        return _expr_contains_var(expr.left, var_name) or _expr_contains_var(expr.right, var_name)
    if isinstance(expr, UnaryExpr):
        return _expr_contains_var(expr.operand, var_name)
    if isinstance(expr, AssignExpr):
        return _expr_contains_var(expr.target, var_name) or _expr_contains_var(expr.value, var_name)
    if isinstance(expr, TernaryExpr):
        return (_expr_contains_var(expr.cond, var_name)
                or _expr_contains_var(expr.true_val, var_name)
                or _expr_contains_var(expr.false_val, var_name))
    if isinstance(expr, PropertyExpr):
        return _expr_contains_var(expr.obj, var_name)
    if isinstance(expr, CommaExpr):
        return any(_expr_contains_var(e, var_name) for e in expr.exprs)
    return False

def _is_matching_for_init(prev_stmt: Stmt, for_stmt: ForStmt) -> bool:
    if isinstance(prev_stmt, VarDeclStmt):
        var_name = prev_stmt.name
    elif isinstance(prev_stmt, ExprStmt) and isinstance(prev_stmt.expr, AssignExpr):
        var_name = _get_var_name_from_expr(prev_stmt.expr.target)
    else:
        return False

    if var_name is None:
        return False

    in_cond = _expr_contains_var(for_stmt.condition, var_name)
    in_update = _expr_contains_var(for_stmt.update, var_name)

    return in_cond and in_update

def _process_condition_block_preamble(
    instructions: List[Instruction], decompiler: 'Decompiler', obj: CodeObject,
    start_idx: int, end_idx: int, clear_regs: bool = False
) -> Tuple[List[Stmt], Expr, Set[int], List[Tuple[int, int, 'Stmt']]]:
    if clear_regs:
        decompiler.regs.clear()

    preamble_stmts = []
    deferred_side_effects = []

    cond_side_effect_addrs = set()
    for j in range(start_idx, end_idx):
        instr = instructions[j]
        if instr.op in (VM.JF, VM.JNF):
            break
        if (instr.op in (VM.INC, VM.DEC) and len(instr.operands) == 1
                and instr.operands[0] < -2):
            cond_side_effect_addrs.add(instr.addr)

    preamble_end_idx = end_idx
    for j in range(start_idx, end_idx):
        if instructions[j].op in (VM.JF, VM.JNF):
            preamble_end_idx = j
            break

    pi = start_idx
    while pi < preamble_end_idx:
        instr = instructions[pi]
        if instr.addr in cond_side_effect_addrs:
            stmt = decompiler._translate_instruction(instr, obj)
            decompiler._collect_pre_stmts(preamble_stmts)
            if stmt:
                deferred_side_effects.append((len(preamble_stmts), instr.addr, stmt))
            pi += 1
            continue
        if (instr.op == VM.CP and len(instr.operands) >= 2
                and instr.operands[0] >= 0 and instr.operands[1] < -2
                and pi > start_idx and pi + 1 < preamble_end_idx):
            prev = instructions[pi - 1]
            next_instr = instructions[pi + 1]
            if (prev.op == VM.CP and len(prev.operands) >= 2
                    and prev.operands[0] == instr.operands[1]
                    and prev.operands[1] == instr.operands[0]
                    and next_instr.op in (VM.TYPEOF, VM.CHKINV)
                    and next_instr.operands[0] == instr.operands[0]):
                temp_reg = instr.operands[0]
                has_prior_write = False
                for scan_i in range(start_idx, pi - 1):
                    scan_instr = instructions[scan_i]
                    if (scan_instr.operands and scan_instr.operands[0] == temp_reg
                            and scan_instr.op != VM.JF and scan_instr.op != VM.JNF
                            and scan_instr.op != VM.JMP):
                        has_prior_write = True
                        break
                if has_prior_write:
                    pi += 1
                    continue
        swap_result = decompiler._try_detect_swap(instructions, obj, pi, preamble_end_idx)
        if swap_result:
            preamble_stmts.append(swap_result['stmt'])
            pi = swap_result['next_idx']
            continue
        stmt = decompiler._translate_instruction(instr, obj)
        decompiler._collect_pre_stmts(preamble_stmts)
        if stmt:
            preamble_stmts.append(stmt)
        pi += 1

    flushed = decompiler._flush_pending_spie()
    if flushed:
        preamble_stmts.append(flushed)

    cond = decompiler._get_condition(False)

    return preamble_stmts, cond, cond_side_effect_addrs, deferred_side_effects

def _emit_unmerged_side_effects(preamble_stmts: List[Stmt],
                                 deferred_side_effects: List[Tuple[int, int, 'Stmt']],
                                 merged_addrs: Set[int]) -> None:
    offset = 0
    for pos, addr, stmt in deferred_side_effects:
        if addr not in merged_addrs:
            preamble_stmts.insert(pos + offset, stmt)
            offset += 1

def _embed_assign_in_expr_tree(expr: Expr, target_value: Expr,
                                assign_expr: 'AssignExpr') -> bool:
    if isinstance(expr, BinaryExpr):
        if expr.left is target_value:
            expr.left = assign_expr
            return True
        if expr.right is target_value:
            expr.right = assign_expr
            return True
        if _embed_assign_in_expr_tree(expr.left, target_value, assign_expr):
            return True
        if _embed_assign_in_expr_tree(expr.right, target_value, assign_expr):
            return True
    elif isinstance(expr, UnaryExpr):
        if expr.operand is target_value:
            expr.operand = assign_expr
            return True
        return _embed_assign_in_expr_tree(expr.operand, target_value, assign_expr)
    elif isinstance(expr, CallExpr):
        if expr.func is target_value:
            expr.func = assign_expr
            return True
        if _embed_assign_in_expr_tree(expr.func, target_value, assign_expr):
            return True
        for i, arg in enumerate(expr.args):
            if arg is target_value:
                expr.args[i] = assign_expr
                return True
            if _embed_assign_in_expr_tree(arg, target_value, assign_expr):
                return True
    elif isinstance(expr, MethodCallExpr):
        if expr.obj is target_value:
            expr.obj = assign_expr
            return True
        if _embed_assign_in_expr_tree(expr.obj, target_value, assign_expr):
            return True
        for i, arg in enumerate(expr.args):
            if arg is target_value:
                expr.args[i] = assign_expr
                return True
            if _embed_assign_in_expr_tree(arg, target_value, assign_expr):
                return True
        if isinstance(expr.method, Expr):
            if expr.method is target_value:
                expr.method = assign_expr
                return True
            if _embed_assign_in_expr_tree(expr.method, target_value, assign_expr):
                return True
    elif isinstance(expr, PropertyExpr):
        if expr.obj is target_value:
            expr.obj = assign_expr
            return True
        if _embed_assign_in_expr_tree(expr.obj, target_value, assign_expr):
            return True
        if isinstance(expr.prop, Expr):
            if expr.prop is target_value:
                expr.prop = assign_expr
                return True
            if _embed_assign_in_expr_tree(expr.prop, target_value, assign_expr):
                return True
    elif isinstance(expr, TernaryExpr):
        for attr in ('cond', 'true_val', 'false_val'):
            child = getattr(expr, attr)
            if child is target_value:
                setattr(expr, attr, assign_expr)
                return True
            if _embed_assign_in_expr_tree(child, target_value, assign_expr):
                return True
    elif isinstance(expr, TypeofExpr):
        if expr.target is target_value:
            expr.target = assign_expr
            return True
        return _embed_assign_in_expr_tree(expr.target, target_value, assign_expr)
    elif isinstance(expr, IsValidExpr):
        if expr.target is target_value:
            expr.target = assign_expr
            return True
        return _embed_assign_in_expr_tree(expr.target, target_value, assign_expr)
    elif isinstance(expr, TypeCastExpr):
        if expr.operand is target_value:
            expr.operand = assign_expr
            return True
        return _embed_assign_in_expr_tree(expr.operand, target_value, assign_expr)
    return False

def _embed_assign_by_var_name(expr: Expr, var_name: str,
                               assign_expr: 'AssignExpr') -> bool:
    if isinstance(expr, BinaryExpr):
        if isinstance(expr.left, VarExpr) and expr.left.name == var_name:
            expr.left = assign_expr
            return True
        if _embed_assign_by_var_name(expr.left, var_name, assign_expr):
            return True
        if isinstance(expr.right, VarExpr) and expr.right.name == var_name:
            expr.right = assign_expr
            return True
        return _embed_assign_by_var_name(expr.right, var_name, assign_expr)
    elif isinstance(expr, UnaryExpr):
        return False
    elif isinstance(expr, PropertyExpr):
        if isinstance(expr.obj, VarExpr) and expr.obj.name == var_name:
            expr.obj = assign_expr
            return True
        return _embed_assign_by_var_name(expr.obj, var_name, assign_expr)
    elif isinstance(expr, MethodCallExpr):
        if isinstance(expr.obj, VarExpr) and expr.obj.name == var_name:
            expr.obj = assign_expr
            return True
        if _embed_assign_by_var_name(expr.obj, var_name, assign_expr):
            return True
        for i, arg in enumerate(expr.args):
            if isinstance(arg, VarExpr) and arg.name == var_name:
                expr.args[i] = assign_expr
                return True
            if _embed_assign_by_var_name(arg, var_name, assign_expr):
                return True
    elif isinstance(expr, CallExpr):
        for i, arg in enumerate(expr.args):
            if isinstance(arg, VarExpr) and arg.name == var_name:
                expr.args[i] = assign_expr
                return True
            if _embed_assign_by_var_name(arg, var_name, assign_expr):
                return True
    elif isinstance(expr, TypeofExpr):
        if isinstance(expr.target, VarExpr) and expr.target.name == var_name:
            expr.target = assign_expr
            return True
        return _embed_assign_by_var_name(expr.target, var_name, assign_expr)
    elif isinstance(expr, TypeCastExpr):
        if isinstance(expr.operand, VarExpr) and expr.operand.name == var_name:
            expr.operand = assign_expr
            return True
        return _embed_assign_by_var_name(expr.operand, var_name, assign_expr)
    return False

def _detect_assignment_in_condition(preamble_stmts: List[Stmt], cond: Expr,
                                     preamble_start_count: int = 0) -> Expr:
    if len(preamble_stmts) > preamble_start_count:
        last_stmt = preamble_stmts[-1]
        assign_expr = None
        was_var_decl = False
        if (isinstance(last_stmt, ExprStmt) and
                isinstance(last_stmt.expr, AssignExpr) and
                isinstance(last_stmt.expr.target, VarExpr)):
            assign_expr = last_stmt.expr
        elif isinstance(last_stmt, VarDeclStmt) and last_stmt.value is not None:
            assign_expr = AssignExpr(VarExpr(last_stmt.name), last_stmt.value)
            was_var_decl = True
        if assign_expr is not None:
            embedded = False
            if isinstance(cond, BinaryExpr) and cond.left is assign_expr.value:
                cond = BinaryExpr(assign_expr, cond.op, cond.right)
                embedded = True
            elif _embed_assign_in_expr_tree(cond, assign_expr.value, assign_expr):
                embedded = True
            elif (getattr(last_stmt, '_cp_aliased', False) and
                  isinstance(assign_expr.target, VarExpr) and
                  _embed_assign_by_var_name(
                      cond, assign_expr.target.name, assign_expr)):
                embedded = True
            if embedded:
                preamble_stmts.pop()
                if was_var_decl:
                    preamble_stmts.append(VarDeclStmt(last_stmt.name))
    return cond

def _generate_if(region: Region, cfg: CFG, instructions: List[Instruction],
                  decompiler: 'Decompiler', obj: CodeObject,
                  loop_context: Optional[Tuple[int, int, int]]) -> List[Stmt]:
    chain_blocks = getattr(region, '_condition_chain', None)
    if chain_blocks is not None:
        return _generate_compound_condition_if(
            region, cfg, instructions, decompiler, obj, loop_context
        )

    block = cfg.get_block(region.cond_block)
    if block is None:
        return []

    preamble_stmts, cond, cond_side_effect_addrs, deferred_se = _process_condition_block_preamble(
        instructions, decompiler, obj, block.start_idx, block.end_idx
    )
    cond = _detect_assignment_in_condition(preamble_stmts, cond)

    if block.terminator == 'jf':
        if_cond = decompiler._negate_expr(cond)
    else:
        if_cond = cond

    if_cond, merged_addrs = decompiler._apply_cond_side_effects(
        if_cond, instructions, block.start_idx, block.end_idx - 1
    )
    _emit_unmerged_side_effects(preamble_stmts, deferred_se, merged_addrs)

    if region.type == RegionType.IF_THEN_ELSE and region.then_region and region.else_region:
        ternary_result = _try_ternary_from_regions(
            region, cfg, instructions, decompiler, obj, if_cond
        )
        if ternary_result is not None:
            return preamble_stmts + ternary_result

    saved_regs = dict(decompiler.regs)
    saved_flag = decompiler.flag
    saved_flag_negated = decompiler.flag_negated

    then_stmts = []
    then_regs = saved_regs
    if region.then_region:
        decompiler.regs = dict(saved_regs)
        decompiler.flag = saved_flag
        decompiler.flag_negated = saved_flag_negated
        then_stmts = generate_code(region.then_region, cfg, instructions, decompiler, obj, loop_context)
        then_regs = dict(decompiler.regs)

    else_stmts = []
    else_regs = saved_regs
    if region.else_region:
        decompiler.regs = dict(saved_regs)
        decompiler.flag = saved_flag
        decompiler.flag_negated = saved_flag_negated
        else_stmts = generate_code(region.else_region, cfg, instructions, decompiler, obj, loop_context)
        else_regs = dict(decompiler.regs)

    decompiler.regs = dict(saved_regs)
    decompiler.flag = saved_flag
    decompiler.flag_negated = saved_flag_negated

    if region.type == RegionType.IF_THEN_ELSE:
        for reg in set(then_regs) | set(else_regs):
            if reg > 0 and reg not in saved_regs:
                then_val = then_regs.get(reg)
                else_val = else_regs.get(reg)
                if then_val is not None and else_val is not None:
                    decompiler.regs[reg] = then_val

    if decompiler._switch_break_stack:
        switch_break = decompiler._switch_break_stack[-1]

        def _check_region_exit_is_switch_break(r):
            if r is None:
                return False
            for bid in r.blocks:
                blk = cfg.get_block(bid)
                if blk and blk.terminator == 'jmp':
                    jmp_instr = instructions[blk.end_idx - 1]
                    if jmp_instr.op == VM.JMP:
                        target = jmp_instr.addr + jmp_instr.operands[0]
                        if target >= switch_break:
                            return True
            return False

        if not then_stmts and not else_stmts and region.type == RegionType.IF_THEN:
            if _check_region_exit_is_switch_break(region.then_region):
                false_exits_switch = False
                cond_blk = cfg.get_block(region.cond_block)
                if cond_blk and cond_blk.terminator in ('jf', 'jnf'):
                    jmp_instr = instructions[cond_blk.end_idx - 1]
                    false_target = jmp_instr.addr + jmp_instr.operands[0]
                    if false_target >= switch_break:
                        false_exits_switch = True
                if not false_exits_switch:
                    then_stmts = [BreakStmt()]

        if not then_stmts and else_stmts:
            if _check_region_exit_is_switch_break(region.then_region):
                then_stmts = [BreakStmt()]

        if then_stmts and not else_stmts and region.type == RegionType.IF_THEN:
            if _check_region_exit_is_switch_break(region.then_region):
                false_exits_switch = False
                cond_blk = cfg.get_block(region.cond_block)
                if cond_blk and cond_blk.terminator in ('jf', 'jnf'):
                    jmp_instr = instructions[cond_blk.end_idx - 1]
                    false_target = jmp_instr.addr + jmp_instr.operands[0]
                    if false_target >= switch_break:
                        false_exits_switch = True
                    else:
                        for bid in cfg.blocks:
                            blk = cfg.get_block(bid)
                            if blk and instructions[blk.start_idx].addr == false_target:
                                if blk.terminator == 'jmp':
                                    ft_jmp = instructions[blk.end_idx - 1]
                                    if ft_jmp.op == VM.JMP:
                                        ft_target = ft_jmp.addr + ft_jmp.operands[0]
                                        if ft_target >= switch_break:
                                            false_exits_switch = True
                                break
                if not false_exits_switch:
                    if not isinstance(then_stmts[-1], (BreakStmt, ReturnStmt, ContinueStmt)):
                        then_stmts.append(BreakStmt())

        if not else_stmts and then_stmts and region.type == RegionType.IF_THEN_ELSE:
            if _check_region_exit_is_switch_break(region.else_region):
                else_stmts = [BreakStmt()]

    if (loop_context and not else_stmts and then_stmts
            and region.type == RegionType.IF_THEN):
        last_instr = instructions[block.end_idx - 1]
        if last_instr.op in (VM.JF, VM.JNF):
            false_branch_addr = last_instr.addr + last_instr.operands[0]
            _, loop_exit_addr, continue_target = loop_context
            if false_branch_addr >= loop_exit_addr and false_branch_addr != continue_target:
                if not isinstance(then_stmts[-1], BreakStmt):
                    else_stmts = [BreakStmt()]

    if (loop_context and then_stmts and else_stmts
            and region.type == RegionType.IF_THEN_ELSE
            and not isinstance(then_stmts[-1], (BreakStmt, ReturnStmt, ContinueStmt))):
        _, loop_exit_addr, _ = loop_context
        _found_break = False
        if region.then_region:
            last_bid = max(region.then_region.blocks)
            blk = cfg.get_block(last_bid)
            if blk and blk.terminator == 'jmp':
                jmp_instr = instructions[blk.end_idx - 1]
                if jmp_instr.op == VM.JMP:
                    target = jmp_instr.addr + jmp_instr.operands[0]
                    if target >= loop_exit_addr:
                        _found_break = True
            if not _found_break and region.else_region and region.else_region.blocks:
                then_max_addr = max(
                    instructions[cfg.get_block(bid).end_idx - 1].addr
                    for bid in region.then_region.blocks
                    if cfg.get_block(bid) is not None
                )
                else_min_addr = min(
                    instructions[cfg.get_block(bid).start_idx].addr
                    for bid in region.else_region.blocks
                    if cfg.get_block(bid) is not None
                )
                for bid in cfg.blocks:
                    b = cfg.get_block(bid)
                    if b is None or b.id < 0 or b.terminator != 'jmp':
                        continue
                    b_addr = instructions[b.start_idx].addr
                    if then_max_addr < b_addr < else_min_addr:
                        jmp_instr = instructions[b.end_idx - 1]
                        if jmp_instr.op == VM.JMP:
                            target = jmp_instr.addr + jmp_instr.operands[0]
                            if target >= loop_exit_addr:
                                _found_break = True
                                break
        if _found_break:
            then_stmts.append(BreakStmt())

    if not then_stmts and not else_stmts and region.type == RegionType.IF_THEN:
        fall_through_id = block.end_idx
        if fall_through_id not in region.blocks:
            ft_block = cfg.get_block(fall_through_id)
            if (ft_block and ft_block.end_idx - ft_block.start_idx == 2
                    and instructions[ft_block.start_idx].op == VM.SRV
                    and instructions[ft_block.start_idx].operands[0] == 0
                    and instructions[ft_block.start_idx + 1].op == VM.RET):
                ft_end_addr = instructions[ft_block.end_idx - 1].addr
                jf_instr = instructions[block.end_idx - 1]
                jf_target = jf_instr.addr + jf_instr.operands[0]
                if jf_target > ft_end_addr:
                    then_stmts = [ReturnStmt(VoidExpr())]

    if not then_stmts and else_stmts:
        if_cond = decompiler._negate_expr(if_cond)
        then_stmts, else_stmts = else_stmts, then_stmts

    if_stmt = IfStmt(if_cond, then_stmts, else_stmts)
    result = preamble_stmts + [if_stmt]

    if (loop_context and not else_stmts and then_stmts
            and isinstance(then_stmts[-1], BreakStmt)):
        last_instr = instructions[block.end_idx - 1]
        if last_instr.op in (VM.JF, VM.JNF):
            false_branch_addr = last_instr.addr + last_instr.operands[0]
            _, loop_exit_addr, continue_target = loop_context
            if false_branch_addr >= loop_exit_addr and false_branch_addr != continue_target:
                break_stmt = then_stmts.pop()
                result.append(break_stmt)

    if loop_context and not else_stmts:
        last_instr = instructions[block.end_idx - 1]
        if last_instr.op in (VM.JF, VM.JNF):
            jump_target_addr = last_instr.addr + last_instr.operands[0]
            _, _, continue_target = loop_context
            if jump_target_addr == continue_target:
                result.append(ContinueStmt())

    return result

def _generate_compound_condition_if(region, cfg: CFG, instructions: List[Instruction],
                                     decompiler: 'Decompiler', obj: CodeObject,
                                     loop_context: Optional[Tuple[int, int, int]],
                                     return_condition_only: bool = False):
    chain_blocks_full = region._condition_chain
    body_block = region._chain_body_block
    else_block = region._chain_else_block
    nf_block_ids = getattr(region, '_chain_nf_block_ids', None)
    if nf_block_ids is None:
        old_nf = getattr(region, '_chain_nf_block', None)
        nf_block_ids = {old_nf} if old_nf is not None else set()

    def _resolve_jmp_trampoline(bid):
        blk = cfg.get_block(bid)
        if blk and blk.terminator == 'jmp' and len(blk.successors) == 1:
            if blk.end_idx - blk.start_idx == 1:
                return blk.successors[0]
        return bid
    resolved_body = _resolve_jmp_trampoline(body_block)
    resolved_else = _resolve_jmp_trampoline(else_block)

    nf_set = set(nf_block_ids)
    chain_blocks = [b for b in chain_blocks_full if b not in nf_set]
    chain_set = set(chain_blocks)
    chain_pos = {bid: i for i, bid in enumerate(chain_blocks)}
    n = len(chain_blocks)

    nf_terminators = {}
    for nf_id in nf_set:
        nf_blk = cfg.get_block(nf_id)
        if nf_blk:
            nf_terminators[nf_id] = nf_blk.terminator

    preamble_stmts = []
    conditions = []

    for bid in chain_blocks:
        block = cfg.get_block(bid)
        preamble_before = len(preamble_stmts)

        block_preamble, cond, _, deferred_se = _process_condition_block_preamble(
            instructions, decompiler, obj, block.start_idx, block.end_idx
        )
        preamble_stmts.extend(block_preamble)
        cond = _detect_assignment_in_condition(preamble_stmts, cond, preamble_before)

        cond, merged_addrs = decompiler._apply_cond_side_effects(
            cond, instructions, block.start_idx, block.end_idx - 1
        )
        _emit_unmerged_side_effects(preamble_stmts, deferred_se, merged_addrs)

        conditions.append(cond)

    for nf_id in nf_set:
        nf_blk = cfg.get_block(nf_id)
        if nf_blk:
            for instr in instructions[nf_blk.start_idx:nf_blk.end_idx]:
                if instr.op in (VM.JF, VM.JNF):
                    break
                decompiler._translate_instruction(instr, obj)

    _BODY = 'BODY'
    _ELSE = 'ELSE'
    _NF = 'NF'
    _CHAIN = 'CHAIN'
    _FALL_NF = 'FALL_NF'
    _FALL = 'FALL'

    def _classify_jump(bid):
        blk = cfg.get_block(bid)
        if blk.terminator == 'fall':
            if blk.successors:
                succ = blk.successors[0]
                if succ in nf_set:
                    return (_FALL_NF, succ)
            return (_FALL, None)

        if blk.terminator == 'jf':
            jump = blk.cond_true
        else:
            jump = blk.cond_false

        if jump == body_block or jump == resolved_body:
            return (_BODY, jump)
        if jump == else_block or jump == resolved_else:
            return (_ELSE, jump)
        if jump in nf_set:
            return (_NF, jump)
        if jump in chain_set:
            return (_CHAIN, jump)
        return (_ELSE, jump)

    def _resolve_nf(block_terminator, nf_id):
        nf_blk = cfg.get_block(nf_id)
        if nf_blk is None:
            return else_block
        if nf_blk.terminator == 'fall' and nf_blk.successors:
            next_id = nf_blk.successors[0]
            if next_id in nf_set:
                flipped = 'jnf' if block_terminator == 'jf' else 'jf'
                return _resolve_nf(flipped, next_id)
        if block_terminator == 'jf':
            resolved = nf_blk.cond_false
        else:
            resolved = nf_blk.cond_true
        return resolved if resolved is not None else else_block

    def _get_effective_cond(idx, subgroup_context=None, in_structural_not=False,
                            subgroup_end=None):
        bid = chain_blocks[idx]
        blk = cfg.get_block(bid)
        cond = conditions[idx]
        cls, raw_target = _classify_jump(bid)

        if cls == _CHAIN and subgroup_end is not None and raw_target in chain_pos:
            target_p = chain_pos[raw_target]
            if target_p > subgroup_end:
                is_jf = (blk.terminator == 'jf')
                if (subgroup_context == 'and' and is_jf) or \
                   (subgroup_context == 'or' and not is_jf):
                    return cond

        if cls == _FALL_NF:
            effective = _resolve_nf('jf', raw_target)
            if effective in chain_set:
                if subgroup_context == 'and':
                    should_negate = True
                elif subgroup_context == 'or':
                    should_negate = False
                else:
                    should_negate = True
            else:
                if in_structural_not:
                    should_negate = False
                else:
                    is_to_body = (effective == body_block or effective == resolved_body)
                    should_negate = not is_to_body
        elif cls == _CHAIN:
            is_jf = (blk.terminator == 'jf')
            if subgroup_context == 'and':
                should_negate = is_jf
            elif subgroup_context == 'or':
                should_negate = not is_jf
            else:
                should_negate = False
        elif cls == _NF:
            effective = _resolve_nf(blk.terminator, raw_target)
            if effective in chain_set:
                is_jf = (blk.terminator == 'jf')
                if subgroup_context == 'and':
                    should_negate = is_jf
                elif subgroup_context == 'or':
                    should_negate = not is_jf
                else:
                    should_negate = is_jf
            else:
                is_jf = (blk.terminator == 'jf')
                is_to_body = (effective == body_block)
                should_negate = (is_jf != is_to_body)
        elif cls in (_BODY, _ELSE):
            is_jf = (blk.terminator == 'jf')
            is_to_body = (cls == _BODY)
            should_negate = (is_jf != is_to_body)
        else:
            should_negate = False

        if should_negate:
            return decompiler._negate_expr(cond)
        return cond

    def _get_effective_jump(idx):
        bid = chain_blocks[idx]
        cls, raw_target = _classify_jump(bid)
        if cls == _NF:
            return _resolve_nf(cfg.get_block(bid).terminator, raw_target)
        if cls == _BODY:
            return body_block
        if cls == _ELSE:
            return else_block
        if cls == _CHAIN:
            return raw_target
        if cls == _FALL_NF:
            return None
        return else_block

    def _trace_to_exit(target, visited):
        if target is None:
            return True
        if target == body_block or target == resolved_body:
            return True
        if target == else_block or target == resolved_else:
            return False
        if target in chain_pos:
            if target in visited:
                warnings.warn(f"Cycle in subgroup type trace at block {target}")
                return True
            visited.add(target)
            idx = chain_pos[target]
            next_target = _get_effective_jump(idx)
            return _trace_to_exit(next_target, visited)
        warnings.warn(f"Cannot determine subgroup type: target {target} is not body/else/chain")
        return True

    def _is_structural_not_subgroup(start, target_pos):
        start_bid = chain_blocks[start]
        start_cls, start_raw = _classify_jump(start_bid)
        if start_cls not in (_NF, _FALL_NF):
            return False

        nf_bid = start_raw
        if nf_bid is None or nf_bid not in nf_set:
            return False

        nf_blk = cfg.get_block(nf_bid)
        if nf_blk is None:
            return False

        if nf_blk.terminator == 'jf':
            jf_target = nf_blk.cond_true
            if jf_target == body_block or jf_target == resolved_body:
                return False
            return True
        elif nf_blk.terminator == 'jnf':
            jnf_target = nf_blk.cond_false
            if jnf_target == body_block or jnf_target == resolved_body:
                return False
            return True
        elif nf_blk.terminator == 'fall':
            return False

        return False

    def _determine_subgroup_type(start, target_pos):
        boundary_idx = target_pos - 1
        if boundary_idx < start:
            return True
        boundary_target = _get_effective_jump(boundary_idx)
        return _trace_to_exit(boundary_target, set())

    def _reconstruct(start, end):
        if start >= end:
            return ConstExpr(True)
        if start == end - 1:
            return _get_effective_cond(start)

        jump = _get_effective_jump(start)

        if jump is None:
            rest = _reconstruct(start + 1, end)
            return BinaryExpr(_get_effective_cond(start), '&&', rest)

        if jump == body_block:
            rest = _reconstruct(start + 1, end)
            return BinaryExpr(_get_effective_cond(start), '||', rest)

        elif jump == else_block:
            rest = _reconstruct(start + 1, end)
            return BinaryExpr(_get_effective_cond(start), '&&', rest)

        elif jump in chain_pos:
            target_pos = chain_pos[jump]
            if target_pos <= start or target_pos >= end:
                rest = _reconstruct(start + 1, end)
                return BinaryExpr(_get_effective_cond(start), '&&', rest)

            expanded = True
            while expanded:
                expanded = False
                for si in range(start + 1, target_pos):
                    sj = _get_effective_jump(si)
                    if sj is not None and sj in chain_pos:
                        sp = chain_pos[sj]
                        if target_pos < sp < end:
                            target_pos = sp
                            expanded = True
                            break

            structural_not = _is_structural_not_subgroup(start, target_pos)
            use_and_subgroup = _determine_subgroup_type(start, target_pos)
            if structural_not:
                if use_and_subgroup:
                    inner = _reconstruct_and_subgroup(start, target_pos, _sn=True)
                else:
                    inner = _reconstruct_or_subgroup(start, target_pos, _sn=True)
                inner = UnaryExpr('!', inner)
                rest = _reconstruct(target_pos, end)
                return BinaryExpr(inner, '&&' if use_and_subgroup else '||', rest)
            elif use_and_subgroup:
                inner = _reconstruct_and_subgroup(start, target_pos)
                rest = _reconstruct(target_pos, end)
                return BinaryExpr(inner, '||', rest)
            else:
                inner = _reconstruct_or_subgroup(start, target_pos)
                rest = _reconstruct(target_pos, end)
                return BinaryExpr(inner, '&&', rest)
        else:
            rest = _reconstruct(start + 1, end)
            return BinaryExpr(_get_effective_cond(start), '&&', rest)

    def _reconstruct_and_subgroup(start, end, parent_context=None, _sn=False):
        if start >= end:
            return ConstExpr(True)
        if start == end - 1:
            ctx = parent_context if parent_context is not None else 'and'
            return _get_effective_cond(start, ctx, in_structural_not=_sn,
                                       subgroup_end=end)

        jump = _get_effective_jump(start)

        if jump is not None and jump in chain_pos:
            target_pos = chain_pos[jump]
            if start < target_pos < end:
                structural_not = _is_structural_not_subgroup(start, target_pos)
                use_and_inner = _determine_subgroup_type(start, target_pos)
                if structural_not:
                    if use_and_inner:
                        inner = _reconstruct_and_subgroup(start, target_pos, _sn=True)
                    else:
                        inner = _reconstruct_or_subgroup(start, target_pos, 'and', _sn=True)
                    inner = UnaryExpr('!', inner)
                    rest = _reconstruct_and_subgroup(target_pos, end, parent_context, _sn=_sn)
                    return BinaryExpr(inner, '&&' if use_and_inner else '||', rest)
                elif use_and_inner:
                    inner = _reconstruct_and_subgroup(start, target_pos, _sn=_sn)
                    rest = _reconstruct_and_subgroup(target_pos, end, parent_context, _sn=_sn)
                    return BinaryExpr(inner, '||', rest)
                else:
                    inner = _reconstruct_or_subgroup(start, target_pos, 'and', _sn=_sn)
                    rest = _reconstruct_and_subgroup(target_pos, end, parent_context, _sn=_sn)
                    return BinaryExpr(inner, '&&', rest)

        rest = _reconstruct_and_subgroup(start + 1, end, parent_context, _sn=_sn)
        return BinaryExpr(_get_effective_cond(start, 'and', in_structural_not=_sn,
                                              subgroup_end=end), '&&', rest)

    def _reconstruct_or_subgroup(start, end, parent_context=None, _sn=False):
        if start >= end:
            return ConstExpr(True)
        if start == end - 1:
            ctx = parent_context if parent_context is not None else 'or'
            return _get_effective_cond(start, ctx, in_structural_not=_sn,
                                       subgroup_end=end)

        jump = _get_effective_jump(start)

        if jump is not None and jump in chain_pos:
            target_pos = chain_pos[jump]
            if start < target_pos < end:
                structural_not = _is_structural_not_subgroup(start, target_pos)
                use_and_inner = _determine_subgroup_type(start, target_pos)
                if structural_not:
                    if use_and_inner:
                        inner = _reconstruct_and_subgroup(start, target_pos, _sn=True)
                    else:
                        inner = _reconstruct_or_subgroup(start, target_pos, _sn=True)
                    inner = UnaryExpr('!', inner)
                    rest = _reconstruct_or_subgroup(target_pos, end, parent_context, _sn=_sn)
                    return BinaryExpr(inner, '&&' if use_and_inner else '||', rest)
                elif use_and_inner:
                    inner = _reconstruct_and_subgroup(start, target_pos, _sn=_sn)
                    rest = _reconstruct_or_subgroup(target_pos, end, parent_context, _sn=_sn)
                    return BinaryExpr(inner, '||', rest)
                else:
                    inner = _reconstruct_or_subgroup(start, target_pos, _sn=_sn)
                    rest = _reconstruct_or_subgroup(target_pos, end, parent_context, _sn=_sn)
                    return BinaryExpr(inner, '&&', rest)

        rest = _reconstruct_or_subgroup(start + 1, end, parent_context, _sn=_sn)
        return BinaryExpr(_get_effective_cond(start, 'or', in_structural_not=_sn,
                                              subgroup_end=end), '||', rest)

    def _split_or_groups():
        if n <= 1:
            return [(0, n)]

        jump_positions = {}
        for i in range(n):
            ej = _get_effective_jump(i)
            if ej is not None and ej in chain_pos:
                jump_positions[i] = chain_pos[ej]

        def _is_or_success_at(idx):
            ej = _get_effective_jump(idx)
            if ej == body_block:
                return True
            bid = chain_blocks[idx]
            cls, raw_target = _classify_jump(bid)
            if cls == _FALL_NF and raw_target:
                nf_blk = cfg.get_block(raw_target)
                if nf_blk and nf_blk.terminator == 'jf' and nf_blk.cond_true == body_block:
                    return True
            return False

        boundaries = []
        group_start = 0
        max_chain_target = 0

        for i in range(n):
            ct = jump_positions.get(i)
            if ct is not None and ct > max_chain_target:
                max_chain_target = ct

            if max_chain_target <= i + 1 and _is_or_success_at(i):
                all_valid = True
                for q in range(group_start, i):
                    if _is_or_success_at(q):
                        covered = False
                        for r in range(group_start, q):
                            rt = jump_positions.get(r)
                            if rt is not None and rt > q:
                                covered = True
                                break
                        if not covered:
                            all_valid = False
                            break
                    else:
                        if q not in jump_positions:
                            all_valid = False
                            break

                if all_valid:
                    boundaries.append(i + 1)
                    group_start = i + 1
                    max_chain_target = i + 1

        if not boundaries:
            return [(0, n)]
        groups = []
        start = 0
        for end_pos in boundaries:
            groups.append((start, end_pos))
            start = end_pos
        if start < n:
            groups.append((start, n))
        return groups

    or_groups = _split_or_groups()

    if len(or_groups) <= 1:
        compound_cond = _reconstruct(0, n)
    else:
        group_exprs = []
        for g_start, g_end in or_groups:
            if g_start == g_end - 1:
                group_exprs.append(_get_effective_cond(g_start))
            else:
                has_inner_or = any(
                    _get_effective_jump(i) == body_block
                    for i in range(g_start, g_end - 1)
                )
                if has_inner_or:
                    group_exprs.append(_reconstruct(g_start, g_end))
                else:
                    group_exprs.append(_reconstruct_and_subgroup(g_start, g_end))
        compound_cond = group_exprs[0]
        for expr in group_exprs[1:]:
            compound_cond = BinaryExpr(compound_cond, '||', expr)

    if return_condition_only:
        return (compound_cond, preamble_stmts)

    if region.type == RegionType.IF_THEN_ELSE and region.then_region and region.else_region:
        ternary_result = _try_ternary_from_regions(
            region, cfg, instructions, decompiler, obj, compound_cond
        )
        if ternary_result is not None:
            return preamble_stmts + ternary_result

    saved_regs = dict(decompiler.regs)
    saved_flag = decompiler.flag
    saved_flag_negated = decompiler.flag_negated

    then_stmts = []
    then_regs = saved_regs
    if region.then_region:
        decompiler.regs = dict(saved_regs)
        decompiler.flag = saved_flag
        decompiler.flag_negated = saved_flag_negated
        then_stmts = generate_code(region.then_region, cfg, instructions, decompiler, obj, loop_context)
        then_regs = dict(decompiler.regs)

    else_stmts = []
    else_regs = saved_regs
    if region.else_region:
        decompiler.regs = dict(saved_regs)
        decompiler.flag = saved_flag
        decompiler.flag_negated = saved_flag_negated
        else_stmts = generate_code(region.else_region, cfg, instructions, decompiler, obj, loop_context)
        else_regs = dict(decompiler.regs)

    decompiler.regs = dict(saved_regs)
    decompiler.flag = saved_flag
    decompiler.flag_negated = saved_flag_negated

    if region.type == RegionType.IF_THEN_ELSE:
        for reg in set(then_regs) | set(else_regs):
            if reg > 0 and reg not in saved_regs:
                then_val = then_regs.get(reg)
                else_val = else_regs.get(reg)
                if then_val is not None and else_val is not None:
                    decompiler.regs[reg] = then_val

    if decompiler._switch_break_stack:
        switch_break = decompiler._switch_break_stack[-1]

        def _check_region_exit_is_switch_break_cc(r):
            if r is None:
                return False
            for bid in r.blocks:
                blk = cfg.get_block(bid)
                if blk and blk.terminator == 'jmp':
                    jmp_instr = instructions[blk.end_idx - 1]
                    if jmp_instr.op == VM.JMP:
                        target = jmp_instr.addr + jmp_instr.operands[0]
                        if target >= switch_break:
                            return True
            return False

        if then_stmts and not else_stmts:
            if _check_region_exit_is_switch_break_cc(region.then_region):
                false_exits_switch = False
                eb = cfg.get_block(else_block)
                if eb is not None:
                    false_branch_addr = instructions[eb.start_idx].addr
                    if false_branch_addr >= switch_break:
                        false_exits_switch = True
                    else:
                        if (eb.terminator == 'jmp'
                                and eb.end_idx - eb.start_idx == 1):
                            ft_jmp = instructions[eb.end_idx - 1]
                            if ft_jmp.op == VM.JMP:
                                ft_target = ft_jmp.addr + ft_jmp.operands[0]
                                if ft_target >= switch_break:
                                    false_exits_switch = True
                if not false_exits_switch:
                    if not isinstance(then_stmts[-1], (BreakStmt, ReturnStmt, ContinueStmt)):
                        then_stmts.append(BreakStmt())

        if not then_stmts and not else_stmts:
            if _check_region_exit_is_switch_break_cc(region.then_region):
                false_exits_switch = False
                eb = cfg.get_block(else_block)
                if eb is not None:
                    false_branch_addr = instructions[eb.start_idx].addr
                    if false_branch_addr >= switch_break:
                        false_exits_switch = True
                if not false_exits_switch:
                    then_stmts = [BreakStmt()]

        if not then_stmts and else_stmts:
            if _check_region_exit_is_switch_break_cc(region.then_region):
                then_stmts = [BreakStmt()]

        if not else_stmts and then_stmts and region.type == RegionType.IF_THEN_ELSE:
            if _check_region_exit_is_switch_break_cc(region.else_region):
                else_stmts = [BreakStmt()]

    if_stmt = IfStmt(compound_cond, then_stmts, else_stmts)
    result = preamble_stmts + [if_stmt]

    if (loop_context and not else_stmts and then_stmts
            and isinstance(then_stmts[-1], BreakStmt)):
        eb = cfg.get_block(else_block)
        if eb is not None:
            false_branch_addr = instructions[eb.start_idx].addr
            _, loop_exit_addr, continue_target = loop_context
            if false_branch_addr >= loop_exit_addr and false_branch_addr != continue_target:
                break_stmt = then_stmts.pop()
                result.append(break_stmt)

    if loop_context and not else_stmts:
        eb = cfg.get_block(else_block)
        if eb is not None:
            false_branch_addr = instructions[eb.start_idx].addr
            _, _, continue_target = loop_context
            if false_branch_addr == continue_target:
                result.append(ContinueStmt())

    return result

def _try_ternary_from_regions(region: Region, cfg: CFG, instructions: List[Instruction],
                               decompiler: 'Decompiler', obj: CodeObject,
                               condition: Expr) -> Optional[List[Stmt]]:
    then_blocks = sorted(region.then_region.blocks)
    else_blocks = sorted(region.else_region.blocks)

    if not then_blocks or not else_blocks:
        return None

    then_target = _find_branch_target_reg(cfg, instructions, then_blocks)
    else_target = _find_branch_target_reg(cfg, instructions, else_blocks)

    if then_target is not None and else_target is not None:
        then_reg, then_side = then_target
        else_reg, else_side = else_target

        if then_reg == else_reg and then_reg > 0:
            result = _try_register_ternary(
                region, cfg, instructions, decompiler, obj, condition, then_reg
            )
            if result is not None:
                return result

    if _is_flag_only_branch(cfg, instructions, then_blocks) and \
       _is_flag_only_branch(cfg, instructions, else_blocks):
        result = _try_flag_ternary(
            region, cfg, instructions, decompiler, obj, condition
        )
        if result is not None:
            return result

    return None

def _finalize_pending_literal(decompiler: 'Decompiler', reg: int) -> Optional[Expr]:
    if reg in decompiler.pending_dicts:
        items = decompiler.pending_dicts.pop(reg)
        result = DictExpr(items)
        decompiler.regs[reg] = result
        return result
    if reg in decompiler.pending_arrays:
        elements = decompiler.pending_arrays.pop(reg)
        decompiler.pending_counters.discard(reg + 1)
        result = ArrayExpr(elements)
        decompiler.regs[reg] = result
        return result
    return decompiler.regs.get(reg)

def _try_register_ternary(region: Region, cfg: CFG, instructions: List[Instruction],
                           decompiler: 'Decompiler', obj: CodeObject,
                           condition: Expr, target_reg: int) -> Optional[List[Stmt]]:
    snapshot = decompiler._save_speculative_state()

    decompiler._restore_speculative_state(snapshot)
    then_stmts = generate_code(region.then_region, cfg, instructions, decompiler, obj)
    true_expr = _finalize_pending_literal(decompiler, target_reg)

    decompiler._restore_speculative_state(snapshot)
    else_stmts = generate_code(region.else_region, cfg, instructions, decompiler, obj)
    false_expr = _finalize_pending_literal(decompiler, target_reg)

    decompiler._restore_speculative_state(snapshot)

    def _has_non_expr_stmts(stmts):
        return any(not isinstance(s, ExprStmt) for s in stmts)
    if _has_non_expr_stmts(then_stmts) or _has_non_expr_stmts(else_stmts):
        return None
    if true_expr is None or false_expr is None:
        return None

    if then_stmts:
        true_expr = CommaExpr([s.expr for s in then_stmts] + [true_expr])
    if else_stmts:
        false_expr = CommaExpr([s.expr for s in else_stmts] + [false_expr])

    ternary = TernaryExpr(condition, true_expr, false_expr)
    decompiler.regs[target_reg] = ternary

    return []

def _is_flag_only_branch(cfg: CFG, instructions: List[Instruction],
                          block_ids: List[int]) -> bool:
    _FLAG_OPS = {VM.CGT, VM.CLT, VM.CEQ, VM.CDEQ, VM.TT, VM.TF, VM.SETF, VM.SETNF, VM.NF}
    _CONTROL_OPS = {VM.JF, VM.JNF, VM.JMP, VM.NOP}
    _TEMP_WRITE_OPS = {VM.CONST, VM.CP, VM.CL, VM.GPD, VM.GPDS, VM.GPI, VM.GPIS,
                       VM.GLOBAL, VM.CHS, VM.LNOT, VM.INT, VM.REAL, VM.STR,
                       VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.MOD, VM.IDIV,
                       VM.BOR, VM.BAND, VM.BXOR, VM.SAL, VM.SAR, VM.SR,
                       VM.TYPEOF, VM.CALL, VM.CALLD, VM.CALLI, VM.NEW,
                       VM.CHGTHIS}

    has_flag_op = False

    for block_id in block_ids:
        block = cfg.get_block(block_id)
        if block is None:
            continue

        for idx in range(block.start_idx, block.end_idx):
            instr = instructions[idx]
            op = instr.op

            if op in _FLAG_OPS:
                has_flag_op = True
            elif op in _CONTROL_OPS:
                continue
            elif op in _TEMP_WRITE_OPS:
                if op == VM.CP and instr.operands[0] < -2:
                    return False
                if op in (VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.MOD, VM.IDIV,
                          VM.BOR, VM.BAND, VM.BXOR, VM.SAL, VM.SAR, VM.SR):
                    if instr.operands[0] < -2:
                        return False
                if op in (VM.CALL, VM.CALLD, VM.CALLI) and instr.operands[0] == 0:
                    return False
            elif op in (VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS, VM.SPI, VM.SPIE, VM.SPIS,
                        VM.SRV, VM.RET):
                return False

    return has_flag_op

def _try_flag_ternary(region: Region, cfg: CFG, instructions: List[Instruction],
                       decompiler: 'Decompiler', obj: CodeObject,
                       condition: Expr) -> Optional[List[Stmt]]:
    snapshot = decompiler._save_speculative_state()

    decompiler._restore_speculative_state(snapshot)
    then_stmts = generate_code(region.then_region, cfg, instructions, decompiler, obj)
    true_cond = decompiler._get_condition(False)

    decompiler._restore_speculative_state(snapshot)
    else_stmts = generate_code(region.else_region, cfg, instructions, decompiler, obj)
    false_cond = decompiler._get_condition(False)

    decompiler._restore_speculative_state(snapshot)

    if then_stmts or else_stmts:
        return None

    ternary_cond = TernaryExpr(condition, true_cond, false_cond)
    decompiler.flag = ternary_cond
    decompiler.flag_negated = False

    return []

def _find_branch_target_reg(cfg: CFG, instructions: List[Instruction],
                             block_ids: List[int]) -> Optional[Tuple[int, bool]]:
    target_reg = None
    has_side_effects = False
    last_was_flag_op = False
    new_target_reg = None
    local_new_regs = set()

    for block_id in block_ids:
        block = cfg.get_block(block_id)
        if block is None:
            continue

        for idx in range(block.start_idx, block.end_idx):
            instr = instructions[idx]
            op = instr.op
            ops = instr.operands

            if op == VM.CONST:
                if new_target_reg is not None and ops[0] == new_target_reg + 1:
                    new_target_reg = None
                else:
                    target_reg = ops[0]
                    new_target_reg = None
                last_was_flag_op = False
            elif op == VM.CP:
                r1 = ops[0]
                if r1 < -2:
                    has_side_effects = True
                target_reg = r1
                new_target_reg = None
                last_was_flag_op = False
            elif op == VM.CL:
                target_reg = ops[0]
                new_target_reg = None
                last_was_flag_op = False
            elif op in (VM.GPD, VM.GPDS, VM.GPI, VM.GPIS):
                target_reg = ops[0]
                new_target_reg = None
                last_was_flag_op = False
            elif op in (VM.CALL, VM.CALLD, VM.CALLI):
                if ops[0] == 0:
                    if op in (VM.CALLD, VM.CALLI) and len(ops) > 1 and ops[1] in local_new_regs:
                        target_reg = ops[1]
                    else:
                        has_side_effects = True
                        target_reg = ops[0]
                else:
                    target_reg = ops[0]
                new_target_reg = None
                last_was_flag_op = False
            elif op == VM.NEW:
                target_reg = ops[0]
                new_target_reg = ops[0]
                local_new_regs.add(ops[0])
                last_was_flag_op = False
            elif op in (VM.SETF, VM.SETNF):
                target_reg = ops[0]
                new_target_reg = None
                last_was_flag_op = False
            elif op == VM.GLOBAL:
                target_reg = ops[0]
                last_was_flag_op = False
            elif op in (VM.CHS, VM.LNOT):
                target_reg = ops[0]
                new_target_reg = None
                last_was_flag_op = False
            elif op in (VM.INT, VM.REAL, VM.STR):
                target_reg = ops[0]
                new_target_reg = None
                last_was_flag_op = False
            elif op in (VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.MOD, VM.IDIV,
                       VM.BOR, VM.BAND, VM.BXOR, VM.SAL, VM.SAR, VM.SR):
                r1 = ops[0]
                if r1 < -2:
                    has_side_effects = True
                target_reg = r1
                new_target_reg = None
                last_was_flag_op = False
            elif op in (VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS, VM.SPI, VM.SPIE, VM.SPIS):
                if ops[0] in local_new_regs:
                    target_reg = ops[0]
                else:
                    has_side_effects = True
                    target_reg = None
                new_target_reg = None
                last_was_flag_op = False
            elif op in (VM.INCPD, VM.DECPD, VM.INCPI, VM.DECPI, VM.INCP, VM.DECP):
                has_side_effects = True
                target_reg = None
                new_target_reg = None
                last_was_flag_op = False
            elif op in (VM.LORPD, VM.LANDPD, VM.BORPD, VM.BXORPD, VM.BANDPD,
                        VM.SARPD, VM.SALPD, VM.SRPD,
                        VM.ADDPD, VM.SUBPD, VM.MODPD, VM.DIVPD, VM.IDIVPD, VM.MULPD,
                        VM.LORPI, VM.LANDPI, VM.BORPI, VM.BXORPI, VM.BANDPI,
                        VM.SARPI, VM.SALPI, VM.SRPI,
                        VM.ADDPI, VM.SUBPI, VM.MODPI, VM.DIVPI, VM.IDIVPI, VM.MULPI,
                        VM.LORP, VM.LANDP, VM.BORP, VM.BXORP, VM.BANDP,
                        VM.SARP, VM.SALP, VM.SRP,
                        VM.ADDP, VM.SUBP, VM.MODP, VM.DIVP, VM.IDIVP, VM.MULP):
                has_side_effects = True
                target_reg = None
                new_target_reg = None
                last_was_flag_op = False
            elif op == VM.SRV:
                has_side_effects = True
                target_reg = None
                new_target_reg = None
                last_was_flag_op = False
            elif op == VM.RET:
                has_side_effects = True
                target_reg = None
                new_target_reg = None
                last_was_flag_op = False
            elif op in (VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT):
                last_was_flag_op = True
            elif op in (VM.TT, VM.TF):
                last_was_flag_op = True
            elif op in (VM.JF, VM.JNF, VM.JMP, VM.NOP, VM.NF):
                continue
            elif op == VM.TYPEOF:
                target_reg = ops[0]
                last_was_flag_op = False
            elif op == VM.CHGTHIS:
                target_reg = ops[0]
                new_target_reg = None
                last_was_flag_op = False

    if last_was_flag_op:
        return None

    if target_reg is not None and target_reg != 0:
        return (target_reg, has_side_effects)
    return None

def _find_subregion_by_header(region, target_block_id):
    if region is None:
        return None
    if region.header_block == target_block_id:
        return region
    if region.type == RegionType.SEQUENCE and region.children:
        for child in region.children:
            result = _find_subregion_by_header(child, target_block_id)
            if result is not None:
                return result
    for attr in ('then_region', 'else_region', 'body_region'):
        sub = getattr(region, attr, None)
        if sub is not None:
            result = _find_subregion_by_header(sub, target_block_id)
            if result is not None:
                return result
    return None

def _extract_counter_register(instructions: List, header) -> 'Optional[int]':
    for idx in range(header.end_idx - 1, header.start_idx - 1, -1):
        ins = instructions[idx]
        if ins.op in (VM.CLT, VM.CGT, VM.CEQ, VM.CDEQ):
            op0 = ins.operands[0]
            if op0 < 0:
                return op0
            if len(ins.operands) > 1 and ins.operands[1] < 0:
                return ins.operands[1]
            return None
    return None

def _writes_to_counter(ins, counter_reg: int) -> bool:
    if ins.op in (VM.INC, VM.DEC) and ins.operands[0] == counter_reg:
        return True
    if ins.op in (VM.ADD, VM.SUB) and ins.operands[0] == counter_reg:
        return True
    if ins.op in (VM.ADDPI, VM.SUBPI) and ins.operands[0] == counter_reg:
        return True
    if ins.op == VM.CP and ins.operands[0] == counter_reg:
        return True
    if ins.op == VM.CONST and ins.operands[0] == counter_reg:
        return True
    return False

def _generate_while(region: Region, cfg: CFG, instructions: List[Instruction],
                     decompiler: 'Decompiler', obj: CodeObject) -> List[Stmt]:
    loop_info = region.loop_info
    header = cfg.get_block(region.header_block)
    if header is None or loop_info is None:
        return []

    compound_chain_used = False
    chain_result = _detect_condition_chain(cfg, region.header_block, instructions)
    if chain_result is not None:
        chain_blocks_list, chain_body, chain_else, chain_nf_ids = chain_result
        chain_body_in_body = chain_body in loop_info.body_blocks
        chain_else_in_exit = chain_else in loop_info.exit_blocks
        if not chain_else_in_exit:
            ce_blk = cfg.get_block(chain_else)
            if (ce_blk and ce_blk.terminator == 'jmp'
                    and len(ce_blk.successors) == 1
                    and ce_blk.end_idx - ce_blk.start_idx == 1):
                chain_else_in_exit = ce_blk.successors[0] in loop_info.exit_blocks
        if not chain_else_in_exit:
            cb_blk = cfg.get_block(chain_body)
            if (cb_blk and cb_blk.terminator == 'jmp'
                    and len(cb_blk.successors) == 1
                    and cb_blk.end_idx - cb_blk.start_idx == 1):
                cb_resolved = cb_blk.successors[0]
                if (cb_resolved in loop_info.exit_blocks
                        and chain_else in loop_info.body_blocks):
                    chain_body, chain_else = chain_else, chain_body
                    chain_body_in_body = True
                    chain_else_in_exit = True
        if chain_body_in_body and chain_else_in_exit:
            class _ChainInfo:
                pass
            ci = _ChainInfo()
            ci._condition_chain = chain_blocks_list
            ci._chain_body_block = chain_body
            ci._chain_else_block = chain_else
            ci._chain_nf_block_ids = chain_nf_ids if chain_nf_ids else set()
            result = _generate_compound_condition_if(
                ci, cfg, instructions, decompiler, obj, None,
                return_condition_only=True
            )
            loop_cond, preamble_stmts = result
            body_sub = _find_subregion_by_header(region.body_region, chain_body)
            if body_sub is not None:
                region.body_region = body_sub
            else:
                region.body_region = Region(
                    type=RegionType.BLOCK,
                    header_block=chain_body,
                    blocks={chain_body}
                )
            region._compound_extracted = ([], [])
            compound_chain_used = True

    if not compound_chain_used:
        preamble_stmts, cond, _, deferred_se = _process_condition_block_preamble(
            instructions, decompiler, obj, header.start_idx, header.end_idx
        )
        cond = _detect_assignment_in_condition(preamble_stmts, cond)

        if header.terminator == 'jnf':
            loop_cond = cond
        else:
            loop_cond = decompiler._negate_expr(cond)

        loop_cond, merged_addrs = decompiler._apply_cond_side_effects(
            loop_cond, instructions, header.start_idx, header.end_idx - 1
        )
        _emit_unmerged_side_effects(preamble_stmts, deferred_se, merged_addrs)

    extra_cond_blocks = []
    extra_preamble_stmts = []
    body_region = region.body_region
    if hasattr(region, '_compound_extracted'):
        extra_cond_blocks, extra_preamble_stmts = region._compound_extracted
    elif body_region and loop_info.exit_blocks:
        extra_cond_blocks, extra_preamble_stmts = _extract_compound_conditions(
            cfg, instructions, decompiler, obj, region, body_region, loop_info
        )
        region._compound_extracted = (extra_cond_blocks, extra_preamble_stmts)

    preamble_stmts.extend(extra_preamble_stmts)
    for extra_cond in extra_cond_blocks:
        loop_cond = BinaryExpr(loop_cond, '&&', extra_cond)

    body_region = region.body_region

    loop_start_addr = instructions[header.start_idx].addr
    exit_blocks = sorted(loop_info.exit_blocks)
    loop_exit_addr = instructions[cfg.get_block(exit_blocks[0]).start_idx].addr if exit_blocks else (
        instructions[header.end_idx].addr if header.end_idx < len(instructions) else loop_start_addr + 1000
    )
    tail = cfg.get_block(loop_info.back_edge_source)
    tail_start_addr = instructions[tail.start_idx].addr if tail else None
    _has_tail_jmp = False
    if tail and tail_start_addr is not None and tail_start_addr != loop_start_addr:
        for bid in loop_info.body_blocks:
            if bid == loop_info.back_edge_source or bid == loop_info.header:
                continue
            blk = cfg.get_block(bid)
            if blk is None:
                continue
            last_ins_idx = blk.end_idx - 1
            if last_ins_idx >= blk.start_idx:
                last_ins = instructions[last_ins_idx]
                if last_ins.op == VM.JMP:
                    target = last_ins.addr + last_ins.operands[0]
                    if target == tail_start_addr:
                        _has_tail_jmp = True
                        break

    _tail_is_pure = False
    if _has_tail_jmp and tail:
        _counter_reg = _extract_counter_register(instructions, header)
        if _counter_reg is not None:
            first_tail_ins = instructions[tail.start_idx]
            if _writes_to_counter(first_tail_ins, _counter_reg):
                _tail_is_pure = True
            elif (first_tail_ins.op == VM.CONST
                  and tail.start_idx + 1 < tail.end_idx
                  and _writes_to_counter(instructions[tail.start_idx + 1], _counter_reg)):
                _tail_is_pure = True

    _body_bypasses_tail = False
    if _has_tail_jmp:
        for bid in loop_info.body_blocks:
            if bid == loop_info.back_edge_source or bid == loop_info.header:
                continue
            blk = cfg.get_block(bid)
            if blk is None:
                continue
            last_ins_idx = blk.end_idx - 1
            if last_ins_idx >= blk.start_idx:
                last_ins = instructions[last_ins_idx]
                if last_ins.op == VM.JMP:
                    target = last_ins.addr + last_ins.operands[0]
                    if target == loop_start_addr:
                        _body_bypasses_tail = True
                        break

    has_for_continue = _has_tail_jmp and _tail_is_pure and not _body_bypasses_tail
    continue_target = tail_start_addr if has_for_continue else loop_start_addr

    if has_for_continue:
        tail_block_id = loop_info.back_edge_source
        body_loop_context = (loop_start_addr, loop_exit_addr, tail_start_addr)

        saved_update_addr = decompiler._for_loop_update_addr
        saved_skip_tail_bid = decompiler._for_loop_skip_tail_bid
        decompiler._for_loop_update_addr = tail_start_addr
        decompiler._for_loop_skip_tail_bid = tail_block_id
        decompiler.loop_context_stack.append(body_loop_context)
        try:
            body_stmts = generate_code(
                body_region, cfg, instructions, decompiler, obj,
                loop_context=body_loop_context
            ) if body_region else []
        finally:
            decompiler.loop_context_stack.pop()
            decompiler._for_loop_update_addr = saved_update_addr
            decompiler._for_loop_skip_tail_bid = saved_skip_tail_bid

        while body_stmts and isinstance(body_stmts[-1], ContinueStmt):
            body_stmts.pop()

        update_expr = None
        tail_block = cfg.get_block(tail_block_id)
        if tail_block:
            update_stmts = _generate_block_for_update(
                tail_block, instructions, decompiler, obj
            )
            if update_stmts:
                all_expr = all(isinstance(s, ExprStmt) for s in update_stmts)
                if all_expr:
                    if len(update_stmts) == 1:
                        update_expr = update_stmts[0].expr
                    else:
                        update_expr = CommaExpr([s.expr for s in update_stmts])
                elif isinstance(update_stmts[-1], ExprStmt):
                    update_expr = update_stmts[-1].expr
                    if len(update_stmts) > 1:
                        body_stmts.extend(update_stmts[:-1])

        if update_expr is not None:
            for_stmt = ForStmt(init=None, condition=loop_cond, update=update_expr, body=body_stmts)
            return preamble_stmts + [for_stmt]
        else:
            if tail_block:
                fallback_stmts = _generate_block_for_update(
                    tail_block, instructions, decompiler, obj
                )
                body_stmts.extend(fallback_stmts)
            return preamble_stmts + [WhileStmt(loop_cond, body_stmts)]
    else:
        body_loop_context = (loop_start_addr, loop_exit_addr, continue_target)

        decompiler.loop_context_stack.append(body_loop_context)
        try:
            body_stmts = generate_code(
                body_region, cfg, instructions, decompiler, obj,
                loop_context=body_loop_context
            ) if body_region else []
        finally:
            decompiler.loop_context_stack.pop()

        while body_stmts and isinstance(body_stmts[-1], ContinueStmt):
            body_stmts.pop()

        if preamble_stmts and tail and tail.terminator in ('jf', 'jnf'):
            real_body = []
            for s in body_stmts:
                if isinstance(s, IfStmt) and not s.then_body and not s.else_body:
                    continue
                real_body.append(s)
            return [DoWhileStmt(loop_cond, preamble_stmts + real_body)]

        return preamble_stmts + [WhileStmt(loop_cond, body_stmts)]

def _extract_compound_conditions(cfg: CFG, instructions: List[Instruction],
                                   decompiler, obj: CodeObject,
                                   while_region: Region,
                                   body_region: Region, loop_info: LoopInfo
                                   ) -> Tuple[List[Expr], List[Stmt]]:
    extra_conds = []
    all_preamble_stmts = []

    exit_target_ids = loop_info.exit_blocks

    current_region = body_region

    while True:
        if current_region.type == RegionType.SEQUENCE and current_region.children:
            first_child = current_region.children[0]
        else:
            first_child = current_region

        if first_child.type not in (RegionType.IF_THEN, RegionType.IF_THEN_ELSE):
            break

        cond_block = cfg.get_block(first_child.cond_block)
        if cond_block is None:
            break

        then_exits_loop = False
        else_exits_loop = False

        if cond_block.cond_false in exit_target_ids:
            else_exits_loop = True
        if cond_block.cond_true in exit_target_ids:
            then_exits_loop = True

        if not then_exits_loop and not else_exits_loop:
            break

        block_preamble, extra_cond, _, deferred_se = _process_condition_block_preamble(
            instructions, decompiler, obj, cond_block.start_idx, cond_block.end_idx
        )
        extra_cond = _detect_assignment_in_condition(block_preamble, extra_cond)
        all_preamble_stmts.extend(block_preamble)

        if cond_block.terminator == 'jnf' and else_exits_loop:
            pass
        elif cond_block.terminator == 'jf' and then_exits_loop:
            extra_cond = decompiler._negate_expr(extra_cond)
        elif cond_block.terminator == 'jnf' and then_exits_loop:
            extra_cond = decompiler._negate_expr(extra_cond)
        elif cond_block.terminator == 'jf' and else_exits_loop:
            pass

        extra_cond, merged_addrs = decompiler._apply_cond_side_effects(
            extra_cond, instructions, cond_block.start_idx, cond_block.end_idx - 1
        )
        _emit_unmerged_side_effects(all_preamble_stmts, deferred_se, merged_addrs)

        extra_conds.append(extra_cond)

        keep_region = None
        if else_exits_loop and first_child.then_region:
            keep_region = first_child.then_region
        elif then_exits_loop and first_child.else_region:
            keep_region = first_child.else_region
        elif then_exits_loop and cond_block.terminator == 'jf' and first_child.then_region:
            if current_region.type != RegionType.SEQUENCE:
                keep_region = first_child.then_region

        if current_region.type == RegionType.SEQUENCE:
            current_region.children.pop(0)
            if keep_region:
                current_region.children.insert(0, keep_region)
            if current_region.children:
                continue
            break
        else:
            if keep_region:
                while_region.body_region = keep_region
                body_region = keep_region
                current_region = body_region
            else:
                break
            continue

        break

    return extra_conds, all_preamble_stmts

def _is_compound_dowhile(loop_info: LoopInfo) -> bool:
    return (loop_info.loop_type == 'do_while'
            and len(loop_info.all_back_edge_sources) > 1)

def _build_compound_dowhile_cond(
        cond_ranges: List[Tuple[int, int, str]],
        header_addr: int,
        instructions: List[Instruction],
        decompiler: 'Decompiler', obj: CodeObject) -> Tuple[Expr, List[Stmt]]:
    or_groups = []
    and_chain = []
    all_preamble = []

    for start_idx, end_idx, terminator in cond_ranges:
        for idx in range(start_idx, end_idx):
            instr = instructions[idx]
            if instr.op in (VM.JF, VM.JNF, VM.JMP):
                break
            stmt = decompiler._translate_instruction(instr, obj)
            decompiler._collect_pre_stmts(all_preamble)
            if stmt:
                all_preamble.append(stmt)

        cond = decompiler._get_condition(False)

        last_instr = instructions[end_idx - 1]
        if last_instr.op in (VM.JF, VM.JNF):
            target = last_instr.addr + last_instr.operands[0]
        else:
            target = None

        if target == header_addr:
            if terminator == 'jf':
                and_chain.append(cond)
            else:
                and_chain.append(decompiler._negate_expr(cond))
            if and_chain:
                group = and_chain[0]
                for c in and_chain[1:]:
                    group = BinaryExpr(group, '&&', c)
                or_groups.append(group)
                and_chain = []
        else:
            if terminator == 'jnf':
                and_chain.append(cond)
            else:
                and_chain.append(decompiler._negate_expr(cond))

    if and_chain:
        group = and_chain[0]
        for c in and_chain[1:]:
            group = BinaryExpr(group, '&&', c)
        or_groups.append(group)

    if or_groups:
        result = or_groups[0]
        for g in or_groups[1:]:
            result = BinaryExpr(result, '||', g)
        return result, all_preamble
    return ConstExpr(True), all_preamble

def _strip_dowhile_merge_continues(stmts):
    while stmts and isinstance(stmts[-1], ContinueStmt):
        stmts.pop()
    if not stmts:
        return
    last = stmts[-1]
    if isinstance(last, IfStmt):
        if last.then_body:
            _strip_dowhile_merge_continues(last.then_body)
        if last.else_body:
            _strip_dowhile_merge_continues(last.else_body)
        if not last.then_body and last.else_body:
            last.condition = _negate_condition(last.condition)
            last.then_body = last.else_body
            last.else_body = []
    elif isinstance(last, TryStmt):
        if last.try_body:
            _strip_dowhile_merge_continues(last.try_body)
        if last.catch_body:
            _strip_dowhile_merge_continues(last.catch_body)

def _negate_condition(expr):
    if isinstance(expr, UnaryExpr) and expr.op == '!':
        return expr.operand
    if isinstance(expr, BinaryExpr):
        _neg_map = {
            '==': '!=', '!=': '==',
            '<': '>=', '>=': '<',
            '>': '<=', '<=': '>',
            '===': '!==', '!==': '===',
        }
        if expr.op in _neg_map:
            return BinaryExpr(expr.left, _neg_map[expr.op], expr.right)
    return UnaryExpr('!', expr)

def _generate_do_while(region: Region, cfg: CFG, instructions: List[Instruction],
                        decompiler: 'Decompiler', obj: CodeObject) -> List[Stmt]:
    loop_info = region.loop_info
    if loop_info is None:
        return []

    header = cfg.get_block(region.header_block)
    if header is None:
        return []

    tail = cfg.get_block(loop_info.back_edge_source)
    if tail is None:
        return []

    loop_start_addr = instructions[header.start_idx].addr
    exit_idx = tail.end_idx
    if exit_idx < len(instructions):
        loop_exit_addr = instructions[exit_idx].addr
    else:
        loop_exit_addr = loop_start_addr + 1000
    is_self_loop = (loop_info.header == loop_info.back_edge_source)
    if is_self_loop:
        continue_target_addr = loop_start_addr
    else:
        continue_target_addr = instructions[tail.start_idx].addr
    body_loop_context = (loop_start_addr, loop_exit_addr, continue_target_addr)

    if is_self_loop:
        cond_start_idx = header.start_idx
        back_jump_idx = header.end_idx - 1

        for j in range(back_jump_idx - 1, header.start_idx - 1, -1):
            instr = instructions[j]
            if instr.op in (VM.TT, VM.TF, VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT):
                cond_start_idx = j
                for k in range(j - 1, header.start_idx - 1, -1):
                    prev = instructions[k]
                    if prev.op in _DOWHILE_COND_PREAMBLE_OPS:
                        cond_start_idx = k
                    else:
                        break
                break

        body_stmts = []
        decompiler.loop_context_stack.append(body_loop_context)
        try:
            i = header.start_idx
            while i < cond_start_idx:
                instr = instructions[i]
                swap_result = decompiler._try_detect_swap(instructions, obj, i, cond_start_idx)
                if swap_result:
                    body_stmts.append(swap_result['stmt'])
                    i = swap_result['next_idx']
                    continue
                stmt = decompiler._translate_instruction(instr, obj)
                decompiler._collect_pre_stmts(body_stmts)
                if stmt:
                    body_stmts.append(stmt)
                i += 1
            flushed = decompiler._flush_pending_spie()
            if flushed:
                body_stmts.append(flushed)
        finally:
            decompiler.loop_context_stack.pop()

        _strip_dowhile_merge_continues(body_stmts)

        cond_preamble, cond, _, deferred_se_self = _process_condition_block_preamble(
            instructions, decompiler, obj, cond_start_idx, back_jump_idx + 1,
            clear_regs=False
        )
        cond = _detect_assignment_in_condition(cond_preamble, cond)
        body_stmts.extend(cond_preamble)
    else:
        if _is_compound_dowhile(loop_info):
            cond_start_idx = header.start_idx
            back_jump_idx = header.end_idx - 1
            for j in range(back_jump_idx - 1, header.start_idx - 1, -1):
                instr = instructions[j]
                if instr.op in (VM.TT, VM.TF, VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT):
                    cond_start_idx = j
                    for k in range(j - 1, header.start_idx - 1, -1):
                        prev = instructions[k]
                        if prev.op in _DOWHILE_COND_PREAMBLE_OPS:
                            cond_start_idx = k
                        else:
                            break
                    break

            body_stmts = []
            decompiler.loop_context_stack.append(body_loop_context)
            try:
                i = header.start_idx
                while i < cond_start_idx:
                    instr = instructions[i]
                    swap_result = decompiler._try_detect_swap(
                        instructions, obj, i, cond_start_idx)
                    if swap_result:
                        body_stmts.append(swap_result['stmt'])
                        i = swap_result['next_idx']
                        continue
                    stmt = decompiler._translate_instruction(instr, obj)
                    decompiler._collect_pre_stmts(body_stmts)
                    if stmt:
                        body_stmts.append(stmt)
                    i += 1
                flushed = decompiler._flush_pending_spie()
                if flushed:
                    body_stmts.append(flushed)
            finally:
                decompiler.loop_context_stack.pop()

            _strip_dowhile_merge_continues(body_stmts)

            cond_ranges = []
            cond_ranges.append((cond_start_idx, header.end_idx, header.terminator))
            tail_bid = loop_info.back_edge_source
            for bid in sorted(loop_info.body_blocks,
                              key=lambda b: cfg.get_block(b).start_idx
                              if cfg.get_block(b) else float('inf')):
                if bid == loop_info.header:
                    continue
                block = cfg.get_block(bid)
                if block and block.start_idx >= header.end_idx:
                    cond_ranges.append(
                        (block.start_idx, block.end_idx, block.terminator))
            if tail_bid not in loop_info.body_blocks:
                tb = cfg.get_block(tail_bid)
                if tb:
                    cond_ranges.append(
                        (tb.start_idx, tb.end_idx, tb.terminator))

            loop_cond, cond_preamble = _build_compound_dowhile_cond(
                cond_ranges, loop_start_addr,
                instructions, decompiler, obj)
            body_stmts.extend(cond_preamble)
        else:
            decompiler.loop_context_stack.append(body_loop_context)
            try:
                body_stmts = generate_code(
                    region.body_region, cfg, instructions, decompiler, obj,
                    loop_context=body_loop_context
                ) if region.body_region else []
            finally:
                decompiler.loop_context_stack.pop()

            _strip_dowhile_merge_continues(body_stmts)

            back_jump_idx = tail.end_idx - 1
            tail_cond_start = tail.start_idx
            for j in range(back_jump_idx - 1, tail.start_idx - 1, -1):
                instr = instructions[j]
                if instr.op in (VM.TT, VM.TF, VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT):
                    tail_cond_start = j
                    for k in range(j - 1, tail.start_idx - 1, -1):
                        prev = instructions[k]
                        if prev.op in _DOWHILE_COND_PREAMBLE_OPS:
                            tail_cond_start = k
                        else:
                            break
                    break

            if tail_cond_start > tail.start_idx:
                decompiler.loop_context_stack.append(body_loop_context)
                try:
                    i = tail.start_idx
                    while i < tail_cond_start:
                        instr = instructions[i]
                        swap_result = decompiler._try_detect_swap(
                            instructions, obj, i, tail_cond_start)
                        if swap_result:
                            body_stmts.append(swap_result['stmt'])
                            i = swap_result['next_idx']
                            continue
                        stmt = decompiler._translate_instruction(instr, obj)
                        decompiler._collect_pre_stmts(body_stmts)
                        if stmt:
                            body_stmts.append(stmt)
                        i += 1
                    flushed = decompiler._flush_pending_spie()
                    if flushed:
                        body_stmts.append(flushed)
                finally:
                    decompiler.loop_context_stack.pop()

            cond_preamble, cond, _, deferred_se_multi = _process_condition_block_preamble(
                instructions, decompiler, obj, tail_cond_start, back_jump_idx + 1,
                clear_regs=True
            )
            cond = _detect_assignment_in_condition(cond_preamble, cond)
            body_stmts.extend(cond_preamble)

            if tail.terminator == 'jf':
                loop_cond = cond
            else:
                loop_cond = decompiler._negate_expr(cond)

            loop_cond, merged_addrs = decompiler._apply_cond_side_effects(
                loop_cond, instructions, tail_cond_start, back_jump_idx
            )
            _emit_unmerged_side_effects(body_stmts, deferred_se_multi, merged_addrs)

    if is_self_loop:
        if tail.terminator == 'jf':
            loop_cond = cond
        else:
            loop_cond = decompiler._negate_expr(cond)
        loop_cond, merged_addrs = decompiler._apply_cond_side_effects(
            loop_cond, instructions, cond_start_idx, back_jump_idx
        )
        _emit_unmerged_side_effects(body_stmts, deferred_se_self, merged_addrs)

    chain_conds = []
    while body_stmts:
        last = body_stmts[-1]
        if (isinstance(last, IfStmt) and
                not last.then_body and not last.else_body):
            chain_conds.append(last.condition)
            body_stmts.pop()
        else:
            break
    if chain_conds:
        chain_conds.reverse()
        chain_conds.append(loop_cond)
        combined = chain_conds[0]
        for c in chain_conds[1:]:
            combined = BinaryExpr(combined, '&&', c)
        loop_cond = combined

    return [DoWhileStmt(loop_cond, body_stmts)]

def _generate_infinite(region: Region, cfg: CFG, instructions: List[Instruction],
                        decompiler: 'Decompiler', obj: CodeObject) -> List[Stmt]:
    loop_info = region.loop_info
    if loop_info is None:
        return []

    header = cfg.get_block(region.header_block)
    if header is None:
        return []

    loop_start_addr = instructions[header.start_idx].addr
    exit_blocks = sorted(loop_info.exit_blocks)
    if exit_blocks:
        exit_block = cfg.get_block(exit_blocks[0])
        loop_exit_addr = instructions[exit_block.start_idx].addr if exit_block else loop_start_addr + 1000
    else:
        max_end_idx = 0
        for bid in loop_info.body_blocks:
            blk = cfg.get_block(bid)
            if blk and blk.end_idx > max_end_idx:
                max_end_idx = blk.end_idx
        if max_end_idx < len(instructions):
            loop_exit_addr = instructions[max_end_idx].addr
        else:
            loop_exit_addr = loop_start_addr + 10000

    body_loop_context = (loop_start_addr, loop_exit_addr, loop_start_addr)

    decompiler.loop_context_stack.append(body_loop_context)
    try:
        body_stmts = generate_code(
            region.body_region, cfg, instructions, decompiler, obj,
            loop_context=body_loop_context
        ) if region.body_region else []
    finally:
        decompiler.loop_context_stack.pop()

    while body_stmts and isinstance(body_stmts[-1], ContinueStmt):
        body_stmts.pop()

    return [WhileStmt(ConstExpr(True), body_stmts)]

def _generate_switch(region: Region, cfg: CFG, instructions: List[Instruction],
                      decompiler: 'Decompiler', obj: CodeObject,
                      loop_context: Optional[Tuple[int, int, int]]) -> List[Stmt]:
    if not region.switch_cases:
        return []

    first_block = cfg.get_block(region.header_block)
    if first_block is None:
        return []

    ref_reg = region.switch_ref_reg
    switch_break_addr = region.switch_break_target

    switch_last_in_loop = False
    if loop_context:
        loop_start_addr = loop_context[0]
        max_switch_idx = max(
            (cfg.get_block(bid).end_idx for bid in region.blocks
             if cfg.get_block(bid) is not None),
            default=0
        )
        max_switch_addr = instructions[max_switch_idx - 1].addr if max_switch_idx > 0 else 0
        has_exit_to_after_switch = False
        for bid in region.blocks:
            blk = cfg.get_block(bid)
            if blk is None or blk.terminator != 'jmp':
                continue
            for succ_id in blk.successors:
                succ = cfg.get_block(succ_id)
                if succ is None:
                    continue
                succ_addr = instructions[succ.start_idx].addr
                if succ_addr > max_switch_addr and succ_addr != loop_start_addr and succ_addr != loop_context[2]:
                    has_exit_to_after_switch = True
                    break
            if has_exit_to_after_switch:
                break
        switch_last_in_loop = not has_exit_to_after_switch

    preamble_stmts = []
    switch_expr = None

    for i in range(first_block.start_idx, first_block.end_idx):
        instr = instructions[i]
        if instr.op == VM.CEQ:
            switch_expr = decompiler.regs.get(ref_reg)
            break
        stmt = decompiler._translate_instruction(instr, obj)
        decompiler._collect_pre_stmts(preamble_stmts)
        if stmt:
            preamble_stmts.append(stmt)

    if switch_expr is None:
        switch_expr = VarExpr(f'%{ref_reg}')

    cases = []

    for sc in region.switch_cases:
        case_val_expr = None
        if sc.value_expr is not None and sc.cond_block_id is not None:
            cb = cfg.get_block(sc.cond_block_id)
            if cb is not None:
                ceq_reg = sc.value_expr
                written_regs = set()
                for idx in range(cb.start_idx, cb.end_idx):
                    instr = instructions[idx]
                    if instr.op == VM.CEQ:
                        if ceq_reg in written_regs:
                            case_val_expr = decompiler.regs.get(ceq_reg)
                        elif ceq_reg == 0:
                            case_val_expr = VoidExpr()
                        else:
                            case_val_expr = decompiler.regs.get(ceq_reg)
                        break
                    decompiler._translate_instruction(instr, obj)
                    if (len(instr.operands) > 0 and
                            instr.op not in (VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT,
                                             VM.TT, VM.TF, VM.NF, VM.JMP, VM.JF, VM.JNF,
                                             VM.SETF, VM.SETNF)):
                        written_regs.add(instr.operands[0])

        if sc.value_expr is not None and case_val_expr is None:
            case_val_expr = VarExpr(f'%{sc.value_expr}')

        body_stmts = []
        if sc.body_is_continue:
            body_stmts = [ContinueStmt()]
        elif sc.body_region is not None:
            if switch_break_addr is not None:
                decompiler._switch_break_stack.append(switch_break_addr)
            body_stmts = generate_code(
                sc.body_region, cfg, instructions, decompiler, obj,
                loop_context=loop_context
            )
            if switch_break_addr is not None:
                decompiler._switch_break_stack.pop()

            if body_stmts and isinstance(body_stmts[-1], BreakStmt):
                if switch_break_addr is not None:
                    last_body_blocks = sorted(sc.body_blocks)
                    is_switch_break = False
                    for bbid in reversed(last_body_blocks):
                        bb = cfg.get_block(bbid)
                        if bb and bb.terminator == 'jmp' and bb.successors:
                            target_bid = bb.successors[0]
                            if target_bid == region.exit_block:
                                is_switch_break = True
                            break
                    if is_switch_break or loop_context is None:
                        body_stmts.pop()
                        sc.has_break = True
            elif (body_stmts and isinstance(body_stmts[-1], ContinueStmt)
                  and switch_last_in_loop and not sc.body_is_continue):
                continue_target = loop_context[2] if loop_context else None
                has_post_switch_code = (switch_break_addr is not None
                                        and continue_target is not None
                                        and switch_break_addr < continue_target)
                if not (sc.has_continue and has_post_switch_code):
                    body_stmts.pop()
                    sc.has_break = True

            elif (body_stmts and switch_break_addr is not None
                  and not isinstance(body_stmts[-1], (BreakStmt, ContinueStmt, ReturnStmt))
                  and not sc.has_continue):
                for bbid in sorted(sc.body_blocks, reverse=True):
                    bb = cfg.get_block(bbid)
                    if bb and bb.terminator == 'jmp':
                        jmp_instr = instructions[bb.end_idx - 1]
                        if jmp_instr.op == VM.JMP:
                            target = jmp_instr.addr + jmp_instr.operands[0]
                            if target >= switch_break_addr:
                                sc.has_break = True
                            break

        if sc.fall_through and body_stmts and isinstance(body_stmts[-1], BreakStmt):
            body_stmts.pop()

        if sc.has_continue and not sc.has_break and not sc.fall_through and not sc.body_is_continue:
            if not body_stmts or not isinstance(body_stmts[-1], ContinueStmt):
                body_stmts.append(ContinueStmt())
        elif sc.has_break and not sc.fall_through:
            body_stmts.append(BreakStmt())

        cases.append((case_val_expr, body_stmts))

    return preamble_stmts + [SwitchStmt(switch_expr, cases)]

def _generate_try_catch(region: Region, cfg: CFG, instructions: List[Instruction],
                         decompiler: 'Decompiler', obj: CodeObject,
                         loop_context: Optional[Tuple[int, int, int]]) -> List[Stmt]:
    entry_block = cfg.get_block(region.header_block)
    if entry_block is None:
        return []

    if region.try_region is None and region.catch_region is None:
        block_ids = sorted(region.blocks)
        min_idx = min(cfg.get_block(b).start_idx for b in block_ids if cfg.get_block(b))
        max_idx = max(cfg.get_block(b).end_idx for b in block_ids if cfg.get_block(b))
        return decompiler._generate_structured_code(
            instructions, obj, min_idx, max_idx, loop_context=loop_context
        )

    preamble_end = entry_block.end_idx
    for idx in range(entry_block.start_idx, entry_block.end_idx):
        if instructions[idx].op == VM.ENTRY:
            preamble_end = idx
            break
    preamble_stmts = []
    idx = entry_block.start_idx
    while idx < preamble_end:
        instr = instructions[idx]
        swap_result = decompiler._try_detect_swap(instructions, obj, idx, preamble_end)
        if swap_result:
            preamble_stmts.append(swap_result['stmt'])
            idx = swap_result['next_idx']
            continue
        stmt = decompiler._translate_instruction(instr, obj)
        decompiler._collect_pre_stmts(preamble_stmts)
        if stmt:
            preamble_stmts.append(stmt)
        idx += 1
    flushed = decompiler._flush_pending_spie()
    if flushed:
        preamble_stmts.append(flushed)

    catch_var_name = None
    has_catch_cp = False
    catch_block = cfg.get_block(region.catch_block)
    if catch_block:
        first_catch_instr = instructions[catch_block.start_idx]
        if first_catch_instr.op == VM.CP:
            dest_reg = first_catch_instr.operands[0]
            src_reg = first_catch_instr.operands[1]
            if src_reg == region.exception_reg and dest_reg < -2:
                decompiler._current_addr = first_catch_instr.addr
                catch_var_name = decompiler._get_local_name(dest_reg)
                has_catch_cp = True

    if catch_var_name is None:
        if region.exception_reg is not None and region.exception_reg < -2:
            catch_var_name = decompiler._get_local_name(region.exception_reg)
        else:
            catch_var_name = '_e'

    saved_regs = dict(decompiler.regs)
    saved_flag = decompiler.flag
    saved_flag_negated = decompiler.flag_negated

    try_stmts = []
    catch_exit_addr = None
    if region.exit_block is not None:
        exit_blk = cfg.get_block(region.exit_block)
        if exit_blk:
            catch_exit_addr = instructions[exit_blk.start_idx].addr
    tc_loop_context = loop_context
    if (loop_context and decompiler._for_loop_update_addr is not None
            and loop_context[2] != decompiler._for_loop_update_addr
            and catch_exit_addr != decompiler._for_loop_update_addr):
        tc_loop_context = (loop_context[0], loop_context[1], decompiler._for_loop_update_addr)

    if region.try_region:
        try_stmts = generate_code(region.try_region, cfg, instructions, decompiler, obj, loop_context)

    decompiler.regs = dict(saved_regs)
    decompiler.flag = saved_flag
    decompiler.flag_negated = saved_flag_negated

    catch_was_already_declared = catch_var_name in decompiler.declared_vars
    decompiler.declared_vars.add(catch_var_name)

    catch_stmts = []
    if region.catch_region:
        catch_stmts = generate_code(region.catch_region, cfg, instructions, decompiler, obj, tc_loop_context)
        if has_catch_cp and catch_stmts:
            catch_stmts = catch_stmts[1:]

    decompiler.regs = dict(saved_regs)
    decompiler.flag = saved_flag
    decompiler.flag_negated = saved_flag_negated

    if not catch_was_already_declared:
        decompiler.declared_vars.discard(catch_var_name)

    try_stmt = TryStmt(try_stmts, catch_var_name, catch_stmts)
    return preamble_stmts + [try_stmt]
