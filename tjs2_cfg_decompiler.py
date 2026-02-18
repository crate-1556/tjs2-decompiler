import sys
from typing import List, Optional

from tjs2_decompiler import (
    Decompiler, BytecodeLoader, CodeObject, Instruction, Stmt,
    ReturnStmt, decode_instructions
)
from tjs2_cfg import (
    build_cfg, compute_dominators, compute_postdominators
)
from tjs2_structuring import (
    detect_loops, build_region_tree, generate_code
)

class CFGDecompiler(Decompiler):

    def __init__(self, loader: BytecodeLoader):
        super().__init__(loader)

    def _decompile_instructions(self, instructions: List[Instruction],
                                 obj: CodeObject) -> List[Stmt]:
        if not instructions:
            return []

        if not hasattr(self, '_pending_spie'):
            self._reset_state()

        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 10000))
        try:

            self._analyze_control_flow(instructions)

            cfg = build_cfg(instructions)

            compute_dominators(cfg)
            compute_postdominators(cfg)

            loops = detect_loops(cfg, instructions)

            region_tree = build_region_tree(cfg, instructions, loops)

            stmts = generate_code(
                region_tree, cfg, instructions, self, obj,
                is_top_level=True
            )

            while stmts and isinstance(stmts[-1], ReturnStmt) and stmts[-1].value is None:
                stmts.pop()

            return stmts
        finally:
            sys.setrecursionlimit(old_limit)
