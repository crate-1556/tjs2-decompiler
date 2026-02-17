#!/usr/bin/env python3

import os
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from enum import IntEnum
from abc import ABC, abstractmethod


class VM(IntEnum):
    NOP = 0; CONST = 1; CP = 2; CL = 3; CCL = 4; TT = 5; TF = 6
    CEQ = 7; CDEQ = 8; CLT = 9; CGT = 10; SETF = 11; SETNF = 12
    LNOT = 13; NF = 14; JF = 15; JNF = 16; JMP = 17
    INC = 18; INCPD = 19; INCPI = 20; INCP = 21
    DEC = 22; DECPD = 23; DECPI = 24; DECP = 25
    LOR = 26; LORPD = 27; LORPI = 28; LORP = 29
    LAND = 30; LANDPD = 31; LANDPI = 32; LANDP = 33
    BOR = 34; BORPD = 35; BORPI = 36; BORP = 37
    BXOR = 38; BXORPD = 39; BXORPI = 40; BXORP = 41
    BAND = 42; BANDPD = 43; BANDPI = 44; BANDP = 45
    SAR = 46; SARPD = 47; SARPI = 48; SARP = 49
    SAL = 50; SALPD = 51; SALPI = 52; SALP = 53
    SR = 54; SRPD = 55; SRPI = 56; SRP = 57
    ADD = 58; ADDPD = 59; ADDPI = 60; ADDP = 61
    SUB = 62; SUBPD = 63; SUBPI = 64; SUBP = 65
    MOD = 66; MODPD = 67; MODPI = 68; MODP = 69
    DIV = 70; DIVPD = 71; DIVPI = 72; DIVP = 73
    IDIV = 74; IDIVPD = 75; IDIVPI = 76; IDIVP = 77
    MUL = 78; MULPD = 79; MULPI = 80; MULP = 81
    BNOT = 82; TYPEOF = 83; TYPEOFD = 84; TYPEOFI = 85
    EVAL = 86; EEXP = 87; CHKINS = 88; ASC = 89; CHR = 90
    NUM = 91; CHS = 92; INV = 93; CHKINV = 94
    INT = 95; REAL = 96; STR = 97; OCTET = 98
    CALL = 99; CALLD = 100; CALLI = 101; NEW = 102
    GPD = 103; SPD = 104; SPDE = 105; SPDEH = 106
    GPI = 107; SPI = 108; SPIE = 109
    GPDS = 110; SPDS = 111; GPIS = 112; SPIS = 113
    SETP = 114; GETP = 115; DELD = 116; DELI = 117
    SRV = 118; RET = 119; ENTRY = 120; EXTRY = 121
    THROW = 122; CHGTHIS = 123; GLOBAL = 124
    ADDCI = 125; REGMEMBER = 126; DEBUGGER = 127


class DataType(IntEnum):
    VOID = 0; OBJECT = 1; INTER_OBJECT = 2; STRING = 3; OCTET = 4
    REAL = 5; BYTE = 6; SHORT = 7; INTEGER = 8; LONG = 9


class ContextType(IntEnum):
    TOP_LEVEL = 0; FUNCTION = 1; EXPR_FUNCTION = 2; PROPERTY = 3
    PROPERTY_SETTER = 4; PROPERTY_GETTER = 5; CLASS = 6; SUPER_CLASS_GETTER = 7


BINARY_OP_SYMBOLS = {
    VM.LOR: '||', VM.LAND: '&&', VM.BOR: '|', VM.BXOR: '^', VM.BAND: '&',
    VM.SAR: '>>', VM.SAL: '<<', VM.SR: '>>>',
    VM.ADD: '+', VM.SUB: '-', VM.MUL: '*', VM.DIV: '/', VM.MOD: '%', VM.IDIV: '\\',
    VM.CEQ: '==', VM.CDEQ: '===', VM.CLT: '<', VM.CGT: '>',
}

OP_PRECEDENCE = {
    '||': 1, '&&': 2, '|': 3, '^': 4, '&': 5,
    '==': 6, '===': 6, '!=': 6, '!==': 6,
    '<': 7, '>': 7, '<=': 7, '>=': 7, 'instanceof': 7,
    '>>': 8, '<<': 8, '>>>': 8,
    '+': 9, '-': 9, '\\': 10, '*': 10, '/': 10, '%': 10,
}


@dataclass
class CodeObject:
    index: int
    name: str
    parent: int
    context_type: int
    max_variable_count: int
    variable_reserve_count: int
    max_frame_count: int
    func_decl_arg_count: int
    func_decl_unnamed_arg_array_base: int
    func_decl_collapse_base: int
    prop_setter: int
    prop_getter: int
    super_class_getter: int
    code: List[int] = field(default_factory=list)
    data: List[Any] = field(default_factory=list)
    properties: List[Tuple[int, int]] = field(default_factory=list)
    source_positions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class Instruction:
    addr: int
    op: int
    operands: List[int]
    size: int


class Expr(ABC):
    @abstractmethod
    def to_source(self) -> str:
        pass

    def precedence(self) -> int:
        return 100


@dataclass
class ConstExpr(Expr):
    value: Any

    def to_source(self) -> str:
        if self.value is None:
            return 'void'
        elif isinstance(self.value, str):
            if self.value.startswith('//') and '/' in self.value[2:]:
                return self._format_regex(self.value)
            escaped = self.value.replace('\\', '\\\\').replace('"', '\\"')
            escaped = escaped.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return f'"{escaped}"'
        elif isinstance(self.value, bool):
            return 'true' if self.value else 'false'
        elif isinstance(self.value, float):
            import math
            if math.isinf(self.value):
                return 'Infinity' if self.value > 0 else '-Infinity'
            if math.isnan(self.value):
                return 'NaN'
            if self.value == int(self.value):
                return f'{int(self.value)}.0'
            return str(self.value)
        elif isinstance(self.value, int):
            return str(self.value)
        elif isinstance(self.value, bytes):
            return f"<% {self.value.hex()} %>"
        return str(self.value)

    def _format_regex(self, s: str) -> str:
        rest = s[2:]
        slash_pos = rest.find('/')
        if slash_pos == -1:
            return f'"{s}"'
        flags = rest[:slash_pos]
        pattern = rest[slash_pos + 1:]
        escaped_pattern = pattern.replace('/', '\\/')
        return f'/{escaped_pattern}/{flags}'


@dataclass
class VarExpr(Expr):
    name: str

    def to_source(self) -> str:
        return self.name


@dataclass
class ThisExpr(Expr):
    def to_source(self) -> str:
        return 'this'


@dataclass
class WithThisExpr(Expr):
    def to_source(self) -> str:
        return 'this'


@dataclass
class GlobalExpr(Expr):
    def to_source(self) -> str:
        return 'global'


@dataclass
class VoidExpr(Expr):
    def to_source(self) -> str:
        return 'void'


@dataclass
class OmittedArgExpr(Expr):
    def to_source(self) -> str:
        return ''


@dataclass
class NullExpr(Expr):
    def to_source(self) -> str:
        return 'null'


@dataclass
class FuncRefExpr(Expr):
    obj_index: int
    loader: Any

    def to_source(self) -> str:
        if self.loader and 0 <= self.obj_index < len(self.loader.objects):
            obj = self.loader.objects[self.obj_index]
            if obj.name:
                return obj.name
        return f'<func#{self.obj_index}>'


@dataclass
class AnonFuncExpr(Expr):
    args: List[str]
    body: str

    def to_source(self) -> str:
        args_str = ', '.join(self.args)
        body_stripped = self.body.strip()
        if body_stripped.startswith('return ') and '\n' not in body_stripped:
            return f'function({args_str}) {{ {body_stripped} }}'
        else:
            return f'function({args_str}) {{\n{self.body}\n}}'


@dataclass
class BinaryExpr(Expr):
    left: Expr
    op: str
    right: Expr

    def to_source(self) -> str:
        left_src = self._wrap_if_needed(self.left, 'left')
        right_src = self._wrap_if_needed(self.right, 'right')
        return f'{left_src} {self.op} {right_src}'

    def _wrap_if_needed(self, expr: Expr, side: str) -> str:
        src = expr.to_source()
        if isinstance(expr, BinaryExpr):
            my_prec = OP_PRECEDENCE.get(self.op, 0)
            expr_prec = OP_PRECEDENCE.get(expr.op, 0)
            if expr_prec < my_prec or (expr_prec == my_prec and side == 'right'):
                return f'({src})'
        if isinstance(expr, TernaryExpr):
            return f'({src})'
        if isinstance(expr, AssignExpr):
            return f'({src})'
        return src

    def precedence(self) -> int:
        return OP_PRECEDENCE.get(self.op, 0)


@dataclass
class UnaryExpr(Expr):
    op: str
    operand: Expr
    prefix: bool = True

    def to_source(self) -> str:
        src = self.operand.to_source()
        if isinstance(self.operand, (BinaryExpr, InstanceofExpr)):
            src = f'({src})'
        if self.prefix:
            return f'{self.op}{src}'
        return f'{src}{self.op}'


@dataclass
class PropertyExpr(Expr):
    obj: Expr
    prop: Union[str, Expr]

    def to_source(self) -> str:
        if isinstance(self.obj, WithThisExpr) and isinstance(self.prop, str):
            if self.prop.isidentifier() and not self.prop.startswith('%'):
                return self.prop
            return f'this["{self.prop}"]'

        obj_src = self.obj.to_source()
        if isinstance(self.obj, (BinaryExpr, TernaryExpr, InContextOfExpr, AssignExpr, TypeofExpr, UnaryExpr, InstanceofExpr)):
            obj_src = f'({obj_src})'

        if isinstance(self.prop, str):
            if self.prop.isidentifier() and not self.prop.startswith('%'):
                return f'{obj_src}.{self.prop}'
            return f'{obj_src}["{self.prop}"]'
        else:
            return f'{obj_src}[{self.prop.to_source()}]'


@dataclass
class CallExpr(Expr):
    func: Expr
    args: List[Expr]
    is_new: bool = False

    def to_source(self) -> str:
        args_src = ', '.join(a.to_source() for a in self.args)
        func_src = self.func.to_source()
        if self.is_new:
            return f'new {func_src}({args_src})'
        if isinstance(self.func, InContextOfExpr):
            func_src = f'({func_src})'
        return f'{func_src}({args_src})'


@dataclass
class MethodCallExpr(Expr):
    obj: Expr
    method: Union[str, Expr]
    args: List[Expr]

    def to_source(self) -> str:
        if isinstance(self.obj, WithThisExpr) and isinstance(self.method, str):
            args_src = ', '.join(a.to_source() for a in self.args)
            if self.method.isidentifier():
                return f'{self.method}({args_src})'
            return f'this["{self.method}"]({args_src})'

        obj_src = self.obj.to_source()
        if isinstance(self.obj, (BinaryExpr, TernaryExpr, InContextOfExpr, AssignExpr, TypeofExpr, UnaryExpr, InstanceofExpr)):
            obj_src = f'({obj_src})'

        args_src = ', '.join(a.to_source() for a in self.args)

        if isinstance(self.method, str):
            if self.method.isidentifier():
                return f'{obj_src}.{self.method}({args_src})'
            return f'{obj_src}["{self.method}"]({args_src})'
        else:
            return f'{obj_src}[{self.method.to_source()}]({args_src})'


@dataclass
class AssignExpr(Expr):
    target: Expr
    value: Expr
    op: str = '='

    def to_source(self) -> str:
        return f'{self.target.to_source()} {self.op} {self.value.to_source()}'


@dataclass
class TernaryExpr(Expr):
    cond: Expr
    true_val: Expr
    false_val: Expr

    def to_source(self) -> str:
        cond_src = self.cond.to_source()
        if isinstance(self.cond, (BinaryExpr, TernaryExpr)):
            cond_src = f'({cond_src})'
        return f'{cond_src} ? {self.true_val.to_source()} : {self.false_val.to_source()}'


@dataclass
class ArrayExpr(Expr):
    elements: List[Expr]

    def to_source(self) -> str:
        if not self.elements:
            return '[]'
        return '[' + ', '.join(e.to_source() for e in self.elements) + ']'


@dataclass
class DictExpr(Expr):
    items: List[Tuple[Expr, Expr]]

    def to_source(self) -> str:
        if not self.items:
            return '%[]'
        pairs = []
        for k, v in self.items:
            k_src = k.to_source()
            v_src = v.to_source()
            pairs.append(f'{k_src} => {v_src}')
        return '%[' + ', '.join(pairs) + ']'


@dataclass
class DeleteExpr(Expr):
    target: Expr

    def to_source(self) -> str:
        return f'delete {self.target.to_source()}'


@dataclass
class TypeofExpr(Expr):
    target: Expr

    def to_source(self) -> str:
        target_src = self.target.to_source()
        if isinstance(self.target, (BinaryExpr, TernaryExpr, InContextOfExpr)):
            target_src = f'({target_src})'
        return f'typeof {target_src}'


@dataclass
class IsValidExpr(Expr):
    target: Expr

    def to_source(self) -> str:
        return f'isvalid {self.target.to_source()}'


@dataclass
class InstanceofExpr(Expr):
    left: Expr
    right: Expr

    def to_source(self) -> str:
        return f'{self.left.to_source()} instanceof {self.right.to_source()}'


@dataclass
class InContextOfExpr(Expr):
    func: Expr
    context: Expr

    def to_source(self) -> str:
        ctx_src = self.context.to_source()
        if isinstance(self.context, AssignExpr):
            ctx_src = f'({ctx_src})'
        return f'{self.func.to_source()} incontextof {ctx_src}'


@dataclass
class SwapExpr(Expr):
    left: Expr
    right: Expr

    def to_source(self) -> str:
        return f'{self.left.to_source()} <-> {self.right.to_source()}'


class Stmt(ABC):
    @abstractmethod
    def to_source(self, indent: int = 0) -> str:
        pass


@dataclass
class ExprStmt(Stmt):
    expr: Expr

    def to_source(self, indent: int = 0) -> str:
        return '    ' * indent + self.expr.to_source() + ';'


@dataclass
class VarDeclStmt(Stmt):
    name: str
    value: Optional[Expr] = None

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent + f'var {self.name}'
        if self.value:
            return prefix + f' = {self.value.to_source()};'
        return prefix + ';'


@dataclass
class ReturnStmt(Stmt):
    value: Optional[Expr] = None

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent + 'return'
        if self.value:
            return prefix + f' {self.value.to_source()};'
        return prefix + ';'


@dataclass
class ThrowStmt(Stmt):
    value: Expr

    def to_source(self, indent: int = 0) -> str:
        return '    ' * indent + f'throw {self.value.to_source()};'


@dataclass
class IfStmt(Stmt):
    condition: Expr
    then_body: List[Stmt]
    else_body: List[Stmt] = field(default_factory=list)

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent
        lines = [f'{prefix}if ({self.condition.to_source()}) {{']
        for stmt in self.then_body:
            lines.append(stmt.to_source(indent + 1))
        if self.else_body:
            if len(self.else_body) == 1 and isinstance(self.else_body[0], IfStmt):
                lines.append(f'{prefix}}} else ' + self.else_body[0].to_source(indent).lstrip())
                return '\n'.join(lines)
            lines.append(f'{prefix}}} else {{')
            for stmt in self.else_body:
                lines.append(stmt.to_source(indent + 1))
        lines.append(f'{prefix}}}')
        return '\n'.join(lines)


@dataclass
class WhileStmt(Stmt):
    condition: Expr
    body: List[Stmt]

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent
        lines = [f'{prefix}while ({self.condition.to_source()}) {{']
        for stmt in self.body:
            lines.append(stmt.to_source(indent + 1))
        lines.append(f'{prefix}}}')
        return '\n'.join(lines)


@dataclass
class DoWhileStmt(Stmt):
    condition: Expr
    body: List[Stmt]

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent
        lines = [f'{prefix}do {{']
        for stmt in self.body:
            lines.append(stmt.to_source(indent + 1))
        lines.append(f'{prefix}}} while ({self.condition.to_source()});')
        return '\n'.join(lines)


@dataclass
class ForStmt(Stmt):
    init: Optional[Expr]
    condition: Optional[Expr]
    update: Optional[Expr]
    body: List[Stmt]

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent
        init_src = self.init.to_source() if self.init else ''
        cond_src = self.condition.to_source() if self.condition else ''
        update_src = self.update.to_source() if self.update else ''
        lines = [f'{prefix}for ({init_src}; {cond_src}; {update_src}) {{']
        for stmt in self.body:
            lines.append(stmt.to_source(indent + 1))
        lines.append(f'{prefix}}}')
        return '\n'.join(lines)


@dataclass
class TryStmt(Stmt):
    try_body: List[Stmt]
    catch_var: str
    catch_body: List[Stmt]

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent
        lines = [f'{prefix}try {{']
        for stmt in self.try_body:
            lines.append(stmt.to_source(indent + 1))
        lines.append(f'{prefix}}} catch ({self.catch_var}) {{')
        for stmt in self.catch_body:
            lines.append(stmt.to_source(indent + 1))
        lines.append(f'{prefix}}}')
        return '\n'.join(lines)


@dataclass
class _WithMarkerStmt(Stmt):
    expr: Expr

    def to_source(self, indent: int = 0) -> str:
        return ''


@dataclass
class WithStmt(Stmt):
    expr: Expr
    body: List[Stmt]

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent
        lines = [f'{prefix}with ({self.expr.to_source()}) {{']
        for stmt in self.body:
            lines.append(stmt.to_source(indent + 1))
        lines.append(f'{prefix}}}')
        return '\n'.join(lines)


@dataclass
class BreakStmt(Stmt):
    def to_source(self, indent: int = 0) -> str:
        return '    ' * indent + 'break;'


@dataclass
class ContinueStmt(Stmt):
    def to_source(self, indent: int = 0) -> str:
        return '    ' * indent + 'continue;'


@dataclass
class SwitchStmt(Stmt):
    value: Expr
    cases: List[Tuple[Optional[Expr], List[Stmt]]]

    def to_source(self, indent: int = 0) -> str:
        prefix = '    ' * indent
        lines = [f'{prefix}switch ({self.value.to_source()}) {{']
        for case_val, case_body in self.cases:
            if case_val is None:
                lines.append(f'{prefix}default:')
            else:
                lines.append(f'{prefix}case {case_val.to_source()}:')
            for stmt in case_body:
                lines.append(stmt.to_source(indent + 1))
        lines.append(f'{prefix}}}')
        return '\n'.join(lines)


class BytecodeLoader:

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.byte_array: List[int] = []
        self.short_array: List[int] = []
        self.long_array: List[int] = []
        self.long_long_array: List[int] = []
        self.double_array: List[float] = []
        self.string_array: List[str] = []
        self.octet_array: List[bytes] = []
        self.objects: List[CodeObject] = []
        self.toplevel: int = -1

    def read_i8(self) -> int:
        val = struct.unpack_from('<b', self.data, self.pos)[0]
        self.pos += 1
        return val

    def read_u8(self) -> int:
        val = self.data[self.pos]
        self.pos += 1
        return val

    def read_i16(self) -> int:
        val = struct.unpack_from('<h', self.data, self.pos)[0]
        self.pos += 2
        return val

    def read_u16(self) -> int:
        val = struct.unpack_from('<H', self.data, self.pos)[0]
        self.pos += 2
        return val

    def read_i32(self) -> int:
        val = struct.unpack_from('<i', self.data, self.pos)[0]
        self.pos += 4
        return val

    def read_u32(self) -> int:
        val = struct.unpack_from('<I', self.data, self.pos)[0]
        self.pos += 4
        return val

    def read_i64(self) -> int:
        val = struct.unpack_from('<q', self.data, self.pos)[0]
        self.pos += 8
        return val

    def read_f64(self) -> float:
        val = struct.unpack_from('<d', self.data, self.pos)[0]
        self.pos += 8
        return val

    def load(self) -> bool:
        if self.data[0:8] != b'TJS2100\x00':
            return False

        self.pos = 8
        file_size = self.read_u32()

        if self.data[self.pos:self.pos+4] != b'DATA':
            return False
        self.pos += 4
        data_size = self.read_u32()
        self._read_data_area()

        if self.data[self.pos:self.pos+4] != b'OBJS':
            return False
        self.pos += 4
        objs_size = self.read_u32()
        self._read_objects()

        return True

    def _read_data_area(self):
        count = self.read_u32()
        if count > 0:
            for _ in range(count):
                self.byte_array.append(self.read_i8())
            padding = (4 - (count % 4)) % 4
            self.pos += padding

        count = self.read_u32()
        for _ in range(count):
            self.short_array.append(self.read_i16())
        if count % 2:
            self.pos += 2

        count = self.read_u32()
        for _ in range(count):
            self.long_array.append(self.read_i32())

        count = self.read_u32()
        for _ in range(count):
            self.long_long_array.append(self.read_i64())

        count = self.read_u32()
        for _ in range(count):
            self.double_array.append(self.read_f64())

        count = self.read_u32()
        for _ in range(count):
            length = self.read_u32()
            chars = []
            for _ in range(length):
                chars.append(self.read_u16())
            try:
                s = ''.join(chr(c) for c in chars)
            except:
                s = f'<raw:{chars}>'
            self.string_array.append(s)
            if length % 2:
                self.pos += 2

        count = self.read_u32()
        for _ in range(count):
            length = self.read_u32()
            self.octet_array.append(self.data[self.pos:self.pos+length])
            self.pos += ((length + 3) // 4) * 4

    def _read_objects(self):
        self.toplevel = self.read_i32()
        obj_count = self.read_u32()

        for obj_idx in range(obj_count):
            if self.data[self.pos:self.pos+4] != b'TJS2':
                raise ValueError(f"Invalid object tag at {self.pos}")
            self.pos += 4

            obj_size = self.read_u32()
            parent = self.read_i32()
            name_idx = self.read_u32()
            context_type = self.read_u32()
            max_variable_count = self.read_u32()
            variable_reserve_count = self.read_u32()
            max_frame_count = self.read_u32()
            func_decl_arg_count = self.read_u32()
            func_decl_unnamed_arg_array_base = self.read_u32()
            func_decl_collapse_base = self.read_i32()
            prop_setter = self.read_i32()
            prop_getter = self.read_i32()
            super_class_getter = self.read_i32()

            src_pos_count = self.read_u32()
            source_positions = []
            if src_pos_count > 0:
                code_positions = [self.read_u32() for _ in range(src_pos_count)]
                source_pos = [self.read_u32() for _ in range(src_pos_count)]
                source_positions = list(zip(code_positions, source_pos))

            code_count = self.read_u32()
            code = [self.read_i16() for _ in range(code_count)]
            if code_count % 2:
                self.pos += 2

            data_count = self.read_u32()
            data = []
            for _ in range(data_count):
                dtype = self.read_i16()
                index = self.read_i16()
                data.append(self._resolve_data(dtype, index, obj_idx))

            scg_count = self.read_u32()
            for _ in range(scg_count):
                self.read_u32()

            prop_count = self.read_u32()
            properties = []
            for _ in range(prop_count):
                pname = self.read_u32()
                pobj = self.read_u32()
                properties.append((pname, pobj))

            name = self.string_array[name_idx] if name_idx < len(self.string_array) else ""

            self.objects.append(CodeObject(
                index=obj_idx, name=name, parent=parent, context_type=context_type,
                max_variable_count=max_variable_count,
                variable_reserve_count=variable_reserve_count,
                max_frame_count=max_frame_count,
                func_decl_arg_count=func_decl_arg_count,
                func_decl_unnamed_arg_array_base=func_decl_unnamed_arg_array_base,
                func_decl_collapse_base=func_decl_collapse_base,
                prop_setter=prop_setter, prop_getter=prop_getter,
                super_class_getter=super_class_getter,
                code=code, data=data, properties=properties,
                source_positions=source_positions
            ))

    def _resolve_data(self, dtype: int, index: int, current_obj: int) -> Any:
        if dtype == DataType.VOID:
            return None
        elif dtype == DataType.OBJECT:
            return ('object', None)
        elif dtype == DataType.INTER_OBJECT:
            return ('inter_object', index)
        elif dtype == DataType.STRING:
            return self.string_array[index] if index < len(self.string_array) else f'str[{index}]'
        elif dtype == DataType.OCTET:
            return self.octet_array[index] if index < len(self.octet_array) else f'octet[{index}]'
        elif dtype == DataType.REAL:
            return self.double_array[index] if index < len(self.double_array) else 0.0
        elif dtype == DataType.BYTE:
            return self.byte_array[index] if index < len(self.byte_array) else 0
        elif dtype == DataType.SHORT:
            return self.short_array[index] if index < len(self.short_array) else 0
        elif dtype == DataType.INTEGER:
            return self.long_array[index] if index < len(self.long_array) else 0
        elif dtype == DataType.LONG:
            return self.long_long_array[index] if index < len(self.long_long_array) else 0
        return ('unknown', dtype, index)


def get_instruction_size(code: List[int], pos: int) -> int:
    if pos >= len(code):
        return 1

    op = code[pos]
    if op < 0 or op > 127:
        return 1

    if op in (VM.NOP, VM.NF, VM.RET, VM.EXTRY, VM.REGMEMBER, VM.DEBUGGER):
        return 1

    if op in (VM.TT, VM.TF, VM.SETF, VM.SETNF, VM.LNOT, VM.BNOT, VM.ASC, VM.CHR,
              VM.NUM, VM.CHS, VM.CL, VM.INV, VM.CHKINV, VM.TYPEOF,
              VM.EVAL, VM.EEXP, VM.INT, VM.REAL, VM.STR, VM.OCTET,
              VM.JF, VM.JNF, VM.JMP, VM.SRV, VM.THROW, VM.GLOBAL,
              VM.INC, VM.DEC):
        return 2

    if op in (VM.CONST, VM.CP, VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT, VM.CHKINS,
              VM.CHGTHIS, VM.ADDCI, VM.CCL, VM.ENTRY, VM.SETP, VM.GETP,
              VM.INCP, VM.DECP):
        return 3

    binary_ops_base = [VM.LOR, VM.LAND, VM.BOR, VM.BXOR, VM.BAND,
                       VM.SAR, VM.SAL, VM.SR, VM.ADD, VM.SUB,
                       VM.MOD, VM.DIV, VM.IDIV, VM.MUL]
    for base_op in binary_ops_base:
        if op == base_op:
            return 3
        elif op == base_op + 1:
            return 5
        elif op == base_op + 2:
            return 5
        elif op == base_op + 3:
            return 4

    if op in (VM.INCPD, VM.DECPD, VM.INCPI, VM.DECPI,
              VM.GPD, VM.GPDS, VM.GPI, VM.GPIS,
              VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS, VM.SPI, VM.SPIE, VM.SPIS,
              VM.DELD, VM.DELI, VM.TYPEOFD, VM.TYPEOFI):
        return 4

    if op in (VM.CALL, VM.NEW):
        if pos + 3 < len(code):
            argc = code[pos + 3]
            if argc == -1:
                return 4
            elif argc == -2:
                if pos + 4 < len(code):
                    real_argc = code[pos + 4]
                    return 5 + real_argc * 2
                return 5
            return 4 + max(0, argc)
        return 4

    if op in (VM.CALLD, VM.CALLI):
        if pos + 4 < len(code):
            argc = code[pos + 4]
            if argc == -1:
                return 5
            elif argc == -2:
                if pos + 5 < len(code):
                    real_argc = code[pos + 5]
                    return 6 + real_argc * 2
                return 6
            return 5 + max(0, argc)
        return 5

    return 1


def decode_instructions(code: List[int]) -> List[Instruction]:
    instructions = []
    pos = 0
    while pos < len(code):
        op = code[pos]
        size = get_instruction_size(code, pos)
        operands = list(code[pos+1:pos+size]) if size > 1 else []
        instructions.append(Instruction(pos, op, operands, size))
        pos += size
    return instructions


class Decompiler:

    def __init__(self, loader: BytecodeLoader):
        self.loader = loader
        self.current_obj: Optional[CodeObject] = None
        self.regs: Dict[int, Expr] = {}
        self.local_vars: Dict[int, str] = {}
        self.var_counter = 0
        self.flag: Optional[Expr] = None
        self.flag_negated = False
        self.declared_vars: Set[str] = set()
        self.pending_arrays: Dict[int, List[Expr]] = {}
        self.pending_dicts: Dict[int, List[Tuple[Expr, Expr]]] = {}
        self.pending_counters: Set[int] = set()
        self.loop_context_stack: List[Tuple[int, int]] = []

    def decompile(self) -> str:
        lines = []

        self._class_children = {}
        class_indices = set()
        for obj in self.loader.objects:
            if obj.context_type == ContextType.CLASS:
                class_indices.add(obj.index)
                self._class_children[obj.index] = []
        for obj in self.loader.objects:
            if obj.parent in class_indices:
                self._class_children[obj.parent].append(obj)
        prop_indices = {obj.index for obj in self.loader.objects
                        if obj.context_type == ContextType.PROPERTY and obj.parent in class_indices}

        handled_by_parent = set()
        for children in self._class_children.values():
            for child in children:
                handled_by_parent.add(child.index)
        for obj in self.loader.objects:
            if obj.parent in prop_indices:
                handled_by_parent.add(obj.index)

        if self.loader.toplevel >= 0:
            top_obj = self.loader.objects[self.loader.toplevel]
            top_stmts = self._decompile_object(top_obj)
            for stmt in top_stmts:
                lines.append(stmt.to_source(0))

        for obj in self.loader.objects:
            if obj.index == self.loader.toplevel:
                continue

            if obj.index in handled_by_parent:
                continue

            if obj.context_type in (ContextType.PROPERTY_SETTER, ContextType.PROPERTY_GETTER,
                                    ContextType.SUPER_CLASS_GETTER):
                continue

            if obj.context_type == ContextType.EXPR_FUNCTION:
                continue

            lines.append('')
            obj_src = self._decompile_object_definition(obj)
            lines.append(obj_src)

        return '\n'.join(lines)

    def _decompile_object_definition(self, obj: CodeObject) -> str:
        if obj.context_type == ContextType.FUNCTION:
            return self._decompile_function(obj)
        elif obj.context_type == ContextType.EXPR_FUNCTION:
            return self._decompile_lambda(obj)
        elif obj.context_type == ContextType.CLASS:
            return self._decompile_class(obj)
        elif obj.context_type == ContextType.PROPERTY:
            return self._decompile_property(obj)
        else:
            return self._decompile_function(obj)

    @staticmethod
    def _contains_with_this(node) -> bool:
        if isinstance(node, WithThisExpr):
            return True
        if isinstance(node, (Expr, Stmt)):
            if hasattr(node, '__dataclass_fields__'):
                for field_name in node.__dataclass_fields__:
                    val = getattr(node, field_name)
                    if Decompiler._contains_with_this(val):
                        return True
        elif isinstance(node, (list, tuple)):
            for item in node:
                if Decompiler._contains_with_this(item):
                    return True
        return False

    def _wrap_with_blocks(self, stmts: List[Stmt]) -> List[Stmt]:
        if not stmts:
            return stmts

        marker_indices = [(i, s) for i, s in enumerate(stmts)
                          if isinstance(s, _WithMarkerStmt)]
        if not marker_indices:
            return stmts

        result = list(stmts)
        for idx, marker in reversed(marker_indices):
            candidates = result[idx + 1:]

            last_with_idx = -1
            for j, stmt in enumerate(candidates):
                if self._contains_with_this(stmt):
                    last_with_idx = j

            if last_with_idx >= 0:
                body = candidates[:last_with_idx + 1]
                continuation = candidates[last_with_idx + 1:]
            else:
                body = candidates
                continuation = []

            with_stmt = WithStmt(marker.expr, body)
            result = result[:idx] + [with_stmt] + continuation

        return result

    def _prepend_context_var_decls(self, obj: CodeObject, stmts: list) -> list:
        if self._context_var_names:
            undeclared = [name for name in self._context_var_names
                         if name not in self.declared_vars]
            if undeclared:
                data_order = {v: i for i, v in enumerate(obj.data) if isinstance(v, str)}
                undeclared.sort(key=lambda n: data_order.get(n, float('inf')))
                stmts = [VarDeclStmt(name) for name in undeclared] + stmts
        return stmts

    def _decompile_function(self, obj: CodeObject) -> str:
        self._reset_state()
        self.current_obj = obj

        args = self._build_args(obj)
        args_str = ', '.join(args)

        for i, arg in enumerate(args):
            if arg == '*':
                continue
            reg = -(3 + i)
            self.regs[reg] = VarExpr(arg)
            self.local_vars[reg] = arg
            self.declared_vars.add(arg)

        stmts = self._decompile_object(obj)

        stmts = self._wrap_with_blocks(stmts)

        stmts = self._prepend_context_var_decls(obj, stmts)

        lines = [f'function {obj.name or "anonymous"}({args_str}) {{']
        for stmt in stmts:
            lines.append(stmt.to_source(1))
        lines.append('}')

        return '\n'.join(lines)

    def _decompile_lambda(self, obj: CodeObject) -> str:
        self._reset_state()
        self.current_obj = obj

        args = self._build_args(obj)
        args_str = ', '.join(args)

        for i, arg in enumerate(args):
            if arg == '*':
                continue
            reg = -(3 + i)
            self.regs[reg] = VarExpr(arg)
            self.local_vars[reg] = arg
            self.declared_vars.add(arg)

        stmts = self._decompile_object(obj)

        stmts = self._wrap_with_blocks(stmts)

        stmts = self._prepend_context_var_decls(obj, stmts)

        if len(stmts) == 1 and isinstance(stmts[0], ReturnStmt):
            ret = stmts[0]
            if ret.value:
                return f'function({args_str}) {{ return {ret.value.to_source()}; }}'
            return f'function({args_str}) {{}}'

        lines = [f'function({args_str}) {{']
        for stmt in stmts:
            lines.append(stmt.to_source(1))
        lines.append('}')

        return '\n'.join(lines)

    def _decompile_anon_func(self, obj: CodeObject) -> AnonFuncExpr:
        saved_regs = dict(self.regs)
        saved_local_vars = dict(self.local_vars)
        saved_declared = set(self.declared_vars)
        saved_obj = self.current_obj
        saved_var_counter = self.var_counter
        saved_flag = self.flag
        saved_flag_negated = self.flag_negated
        saved_pending_dicts = dict(self.pending_dicts)
        saved_pending_arrays = dict(self.pending_arrays)
        saved_pending_counters = set(self.pending_counters)
        saved_loop_headers = dict(self.loop_headers) if hasattr(self, 'loop_headers') else {}
        saved_jump_targets = dict(self.jump_targets) if hasattr(self, 'jump_targets') else {}
        saved_back_edges = set(self.back_edges) if hasattr(self, 'back_edges') else set()
        saved_loop_context_stack = list(self.loop_context_stack) if hasattr(self, 'loop_context_stack') else []
        saved_with_cp_addrs = set(self._with_cp_addrs)
        saved_in_with = self._in_with

        self._reset_state()
        self.current_obj = obj

        args = self._build_args(obj)

        for i, arg in enumerate(args):
            if arg == '*':
                continue
            reg = -(3 + i)
            self.regs[reg] = VarExpr(arg)
            self.local_vars[reg] = arg
            self.declared_vars.add(arg)

        stmts = self._decompile_object(obj)

        body_lines = []
        for stmt in stmts:
            body_lines.append(stmt.to_source(1))
        body = '\n'.join(body_lines)

        self.regs = saved_regs
        self.local_vars = saved_local_vars
        self.declared_vars = saved_declared
        self.current_obj = saved_obj
        self.var_counter = saved_var_counter
        self.flag = saved_flag
        self.flag_negated = saved_flag_negated
        self.pending_dicts = saved_pending_dicts
        self.pending_arrays = saved_pending_arrays
        self.pending_counters = saved_pending_counters
        self.loop_headers = saved_loop_headers
        self.jump_targets = saved_jump_targets
        self.back_edges = saved_back_edges
        self.loop_context_stack = saved_loop_context_stack
        self._with_cp_addrs = saved_with_cp_addrs
        self._in_with = saved_in_with

        return AnonFuncExpr(args, body)

    def _decompile_class(self, obj: CodeObject, indent: int = 0) -> str:
        prefix = '    ' * indent
        inner = '    ' * (indent + 1)
        lines = [f'{prefix}class {obj.name or "anonymous"} {{']

        self._reset_state()
        self.current_obj = obj
        stmts = self._decompile_object(obj)
        for stmt in stmts:
            lines.append(stmt.to_source(indent + 1))

        children = getattr(self, '_class_children', {}).get(obj.index, [])
        for child_obj in children:
            if child_obj.context_type == ContextType.SUPER_CLASS_GETTER:
                continue

            lines.append('')
            if child_obj.context_type == ContextType.FUNCTION:
                self._reset_state()
                self.current_obj = child_obj
                args = self._build_args(child_obj)
                for i, arg in enumerate(args):
                    if arg == '*':
                        continue
                    reg = -(3 + i)
                    self.regs[reg] = VarExpr(arg)
                    self.local_vars[reg] = arg
                    self.declared_vars.add(arg)
                body_stmts = self._decompile_object(child_obj)
                body_stmts = self._wrap_with_blocks(body_stmts)
                body_stmts = self._prepend_context_var_decls(child_obj, body_stmts)
                lines.append(f'{inner}function {child_obj.name}({", ".join(args)}) {{')
                for stmt in body_stmts:
                    lines.append(stmt.to_source(indent + 2))
                lines.append(f'{inner}}}')
            elif child_obj.context_type == ContextType.PROPERTY:
                lines.append(self._decompile_property(child_obj, indent=indent + 1))
            elif child_obj.context_type == ContextType.CLASS:
                lines.append(self._decompile_class(child_obj, indent=indent + 1))

        lines.append(f'{prefix}}}')
        return '\n'.join(lines)

    def _decompile_property(self, obj: CodeObject, indent: int = 0) -> str:
        prefix = '    ' * indent
        lines = [f'{prefix}property {obj.name} {{']

        if obj.prop_getter >= 0 and obj.prop_getter < len(self.loader.objects):
            getter_obj = self.loader.objects[obj.prop_getter]
            self._reset_state()
            self.current_obj = getter_obj
            getter_stmts = self._decompile_object(getter_obj)
            getter_stmts = self._wrap_with_blocks(getter_stmts)
            lines.append(f'{prefix}    getter() {{')
            for stmt in getter_stmts:
                lines.append(stmt.to_source(indent + 2))
            lines.append(f'{prefix}    }}')

        if obj.prop_setter >= 0 and obj.prop_setter < len(self.loader.objects):
            setter_obj = self.loader.objects[obj.prop_setter]
            self._reset_state()
            self.current_obj = setter_obj
            args = self._build_args(setter_obj)
            for i, arg in enumerate(args):
                reg = -(3 + i)
                self.regs[reg] = VarExpr(arg)
                self.local_vars[reg] = arg
                self.declared_vars.add(arg)
            setter_stmts = self._decompile_object(setter_obj)
            setter_stmts = self._wrap_with_blocks(setter_stmts)
            lines.append(f'{prefix}    setter({", ".join(args)}) {{')
            for stmt in setter_stmts:
                lines.append(stmt.to_source(indent + 2))
            lines.append(f'{prefix}    }}')

        lines.append(f'{prefix}}}')
        return '\n'.join(lines)

    def _build_args(self, obj: CodeObject) -> List[str]:
        args = []
        for i in range(obj.func_decl_arg_count):
            args.append(f'arg{i}')
        if obj.func_decl_collapse_base >= 0:
            args.append('*')
        elif obj.func_decl_unnamed_arg_array_base > 0:
            args.append('*')
        return args

    def _reset_state(self):
        self.regs = {}
        self.local_vars = {}
        self.var_counter = 0
        self.flag = None
        self.flag_negated = False
        self.declared_vars = set()
        self.pending_arrays = {}
        self.pending_dicts = {}
        self.pending_counters = set()
        self._context_var_names = set()
        self._prev_instruction = None
        self._with_cp_addrs = set()
        self._in_with = False
        self._pending_spie = None
        self._pre_stmts = []

    def _detect_with_blocks(self, instructions: List[Instruction]):
        self._with_cp_addrs = set()

        if not instructions:
            return

        from tjs2_decompiler import VM

        cp_candidates = []
        for i, instr in enumerate(instructions):
            if instr.op == VM.CP and len(instr.operands) >= 2:
                dest, src = instr.operands[0], instr.operands[1]
                if dest > 0 and src < -2:
                    cp_candidates.append((i, dest, src, instr.addr))

        if not cp_candidates:
            return


        _data_idx_at_1 = {VM.CONST, VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS}
        _data_idx_at_2 = {VM.GPD, VM.GPDS, VM.TYPEOFD, VM.CALLD}
        _count_at_2 = {VM.CALL, VM.NEW}
        _count_at_3 = {VM.CALLI, VM.CALLD}
        _jump_ops = {VM.JF, VM.JNF, VM.JMP}

        def _get_read_regs(instr):
            op = instr.op
            ops = instr.operands
            if not ops:
                return set()

            if op in _jump_ops:
                return set()

            result = set()
            for pos, val in enumerate(ops):
                if pos == 1 and op in _data_idx_at_1:
                    continue
                if pos == 2 and op in _data_idx_at_2:
                    continue
                if pos == 2 and op in _count_at_2:
                    continue
                if pos == 3 and op in _count_at_3:
                    continue

                if pos == 0:
                    _write_dest_ops = {
                        VM.CP, VM.CONST, VM.CL, VM.CCL,
                        VM.GPD, VM.GPDS, VM.GPI, VM.GPIS,
                        VM.CALL, VM.CALLD, VM.CALLI, VM.NEW,
                    }
                    if op in _write_dest_ops:
                        continue

                result.add(val)
            return result

        for cp_idx, dest_reg, src_reg, cp_addr in cp_candidates:
            is_read = False
            for j in range(cp_idx + 1, len(instructions)):
                read_regs = _get_read_regs(instructions[j])
                if dest_reg in read_regs:
                    is_read = True
                    break

            if not is_read:
                self._with_cp_addrs.add(cp_addr)

    def _decompile_object(self, obj: CodeObject) -> List[Stmt]:
        if not obj.code:
            return []

        if self.current_obj is None:
            self.current_obj = obj

        instructions = decode_instructions(obj.code)

        self._detect_with_blocks(instructions)

        return self._decompile_instructions(instructions, obj)

    def _decompile_instructions(self, instructions: List[Instruction], obj: CodeObject) -> List[Stmt]:
        if not instructions:
            return []

        self._analyze_control_flow(instructions)

        return self._generate_structured_code(instructions, obj, 0, len(instructions), is_top_level=True)

    def _analyze_control_flow(self, instructions: List[Instruction]):
        self.jump_targets = {}
        self.back_edges = set()
        self.loop_headers = {}

        addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}

        for instr in instructions:
            if instr.op in (VM.JF, VM.JNF, VM.JMP):
                target = instr.addr + instr.operands[0]
                is_cond = instr.op != VM.JMP

                if target not in self.jump_targets:
                    self.jump_targets[target] = []
                self.jump_targets[target].append((instr.addr, is_cond))

                if target < instr.addr:
                    self.back_edges.add((instr.addr, target))

        for back_from, back_to in self.back_edges:
            if back_to not in self.loop_headers or back_from > self.loop_headers[back_to]:
                self.loop_headers[back_to] = back_from

    def _generate_structured_code(self, instructions: List[Instruction], obj: CodeObject,
                                    start_idx: int, end_idx: int, is_top_level: bool = False,
                                    loop_context: Optional[Tuple[int, int]] = None) -> List[Stmt]:
        stmts = []
        addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}
        i = start_idx

        while i < end_idx:
            instr = instructions[i]

            if instr.addr in self.loop_headers:
                loop_result = self._process_loop(instructions, obj, i, end_idx)
                if loop_result:
                    stmts.append(loop_result['stmt'])
                    i = loop_result['next_idx']
                    continue

            if instr.op == VM.ENTRY:
                try_result = self._process_try(instructions, obj, i, end_idx)
                if try_result:
                    stmts.append(try_result['stmt'])
                    i = try_result['next_idx']
                    continue

            if instr.op in (VM.JF, VM.JNF):
                target = instr.addr + instr.operands[0]

                if target < instr.addr and target in addr_to_idx:
                    current_loop = loop_context or (self.loop_context_stack[-1] if self.loop_context_stack else None)
                    if current_loop and target == current_loop[0]:
                        cond = self._get_condition(False)
                        if instr.op == VM.JNF:
                            cond = self._negate_expr(cond)

                        next_idx = i + 1
                        if next_idx < end_idx and instructions[next_idx].op == VM.JMP:
                            next_jmp = instructions[next_idx]
                            jmp_target = next_jmp.addr + next_jmp.operands[0]
                            if jmp_target >= current_loop[1]:
                                inverted_cond = self._negate_expr(cond)
                                stmts.append(IfStmt(inverted_cond, [BreakStmt()], []))
                                i = next_idx + 1
                                continue

                        stmts.append(IfStmt(cond, [ContinueStmt()], []))
                        i += 1
                        continue
                    else:
                        cond = self._get_condition(False)
                        loop_cond = cond if instr.op == VM.JF else self._negate_expr(cond)
                        i += 1
                        continue


                sc_result = self._try_process_short_circuit(instructions, obj, i, end_idx, addr_to_idx)
                if sc_result is not None:
                    i = sc_result
                    continue

                switch_result = self._process_switch(instructions, obj, i, end_idx)
                if switch_result:
                    if switch_result.get('stmt') is not None:
                        stmts.append(switch_result['stmt'])
                    i = switch_result['next_idx']
                    continue

                if_result = self._process_if(instructions, obj, i, end_idx)
                if if_result:
                    if if_result.get('stmt') is not None:
                        stmts.append(if_result['stmt'])
                    i = if_result['next_idx']
                    continue

            if instr.op == VM.JMP:
                target = instr.addr + instr.operands[0]

                current_loop = loop_context or (self.loop_context_stack[-1] if self.loop_context_stack else None)
                if current_loop:
                    loop_start_addr, loop_exit_addr = current_loop
                    if target >= loop_exit_addr:
                        stmts.append(BreakStmt())
                        i += 1
                        continue
                    elif target == loop_start_addr:
                        stmts.append(ContinueStmt())
                        i += 1
                        continue

                if target < instr.addr:
                    i += 1
                    continue
                i += 1
                continue

            swap_result = self._try_detect_swap(instructions, obj, i, end_idx)
            if swap_result:
                stmts.append(swap_result['stmt'])
                i = swap_result['next_idx']
                continue

            stmt = self._translate_instruction(instr, obj)
            self._collect_pre_stmts(stmts)
            if stmt:
                stmts.append(stmt)
            i += 1

        flushed = self._flush_pending_spie()
        if flushed:
            stmts.append(flushed)

        if is_top_level:
            while stmts and isinstance(stmts[-1], ReturnStmt) and stmts[-1].value is None:
                stmts.pop()

        return stmts

    def _process_loop(self, instructions: List[Instruction], obj: CodeObject,
                      start_idx: int, end_idx: int) -> Optional[Dict]:
        loop_start = instructions[start_idx].addr
        loop_end_addr = self.loop_headers.get(loop_start)

        if loop_end_addr is None:
            return None

        addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}

        if loop_end_addr not in addr_to_idx:
            return None
        back_jump_idx = addr_to_idx[loop_end_addr]
        if back_jump_idx < start_idx or back_jump_idx >= end_idx:
            return None
        back_jump = instructions[back_jump_idx]

        if back_jump.op == VM.JMP:
            loop_exit_addr = back_jump.addr + back_jump.size

            cond_jmp_idx = None
            for j in range(start_idx, min(start_idx + 10, back_jump_idx)):
                instr = instructions[j]
                if instr.op in (VM.JF, VM.JNF):
                    target = instr.addr + instr.operands[0]
                    if target >= loop_exit_addr:
                        cond_jmp_idx = j
                        break

            if cond_jmp_idx is not None:
                all_exit_jmp_indices = [cond_jmp_idx]
                for j in range(cond_jmp_idx + 1, min(cond_jmp_idx + 20, back_jump_idx)):
                    instr_j = instructions[j]
                    if instr_j.op in (VM.JF, VM.JNF):
                        target_j = instr_j.addr + instr_j.operands[0]
                        if target_j >= loop_exit_addr:
                            all_exit_jmp_indices.append(j)
                        else:
                            break
                    elif instr_j.op == VM.JMP:
                        break

                last_cond_jmp_idx = all_exit_jmp_indices[-1]

                conditions = []
                seg_start = start_idx
                for exit_jmp_idx in all_exit_jmp_indices:
                    for j in range(seg_start, exit_jmp_idx):
                        self._translate_instruction(instructions[j], obj)

                    cond = self._get_condition(False)
                    cond_instr_j = instructions[exit_jmp_idx]
                    if cond_instr_j.op == VM.JNF:
                        seg_cond = cond
                    else:
                        seg_cond = self._negate_expr(cond)

                    seg_cond, _merged = self._apply_cond_side_effects(
                        seg_cond, instructions, seg_start, exit_jmp_idx)

                    conditions.append(seg_cond)
                    seg_start = exit_jmp_idx + 1

                loop_cond = conditions[0]
                for c in conditions[1:]:
                    loop_cond = BinaryExpr(loop_cond, '&&', c)

                body_start = last_cond_jmp_idx + 1
                body_end = back_jump_idx

                body_loop_context = (loop_start, loop_exit_addr)
                self.loop_context_stack.append(body_loop_context)
                try:
                    body_stmts = self._generate_structured_code(instructions, obj, body_start, body_end,
                                                                loop_context=body_loop_context)
                finally:
                    self.loop_context_stack.pop()

                while_stmt = WhileStmt(loop_cond, body_stmts)
                return {'stmt': while_stmt, 'next_idx': back_jump_idx + 1}

            else:

                saved_loop_end = self.loop_headers.pop(loop_start, None)

                body_loop_context = (loop_start, loop_exit_addr)
                self.loop_context_stack.append(body_loop_context)
                try:
                    body_stmts = self._generate_structured_code(instructions, obj, start_idx, back_jump_idx,
                                                                loop_context=body_loop_context)
                finally:
                    self.loop_context_stack.pop()

                if saved_loop_end is not None:
                    self.loop_headers[loop_start] = saved_loop_end

                infinite_cond = ConstExpr(True)
                while_stmt = WhileStmt(infinite_cond, body_stmts)
                return {'stmt': while_stmt, 'next_idx': back_jump_idx + 1}

        elif back_jump.op in (VM.JF, VM.JNF):

            cond_start_idx = start_idx

            j = back_jump_idx - 1
            while j >= start_idx and instructions[j].op == VM.NF:
                cond_start_idx = j
                j -= 1
            if j >= start_idx and instructions[j].op in (VM.TT, VM.TF, VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT):
                cond_start_idx = j
                for k in range(j - 1, start_idx - 1, -1):
                    prev = instructions[k]
                    if prev.op in (VM.CONST, VM.GPD, VM.GPI, VM.GPDS, VM.GPIS,
                                   VM.CP, VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.MOD,
                                   VM.BAND, VM.BOR, VM.BXOR, VM.BNOT,
                                   VM.INC, VM.DEC, VM.TYPEOF, VM.CHKINS):
                        cond_start_idx = k
                    else:
                        break

            saved_loop_end = self.loop_headers.pop(loop_start, None)

            do_while_exit_addr = back_jump.addr + back_jump.size

            body_loop_context = (loop_start, do_while_exit_addr)
            self.loop_context_stack.append(body_loop_context)
            try:
                body_stmts = self._generate_structured_code(instructions, obj, start_idx, cond_start_idx,
                                                            loop_context=body_loop_context)
            finally:
                self.loop_context_stack.pop()

            if saved_loop_end is not None:
                self.loop_headers[loop_start] = saved_loop_end

            self.regs.clear()

            for j in range(cond_start_idx, back_jump_idx):
                self._translate_instruction(instructions[j], obj)

            cond = self._get_condition(False)

            if back_jump.op == VM.JF:
                loop_cond = cond
            else:
                loop_cond = self._negate_expr(cond)

            do_while_stmt = DoWhileStmt(loop_cond, body_stmts)
            return {'stmt': do_while_stmt, 'next_idx': back_jump_idx + 1}

        return None

    def _apply_cond_side_effects(self, cond: Expr, instructions: List['Instruction'],
                                  start_idx: int, end_idx: int) -> Tuple[Expr, Set[int]]:
        merged_addrs = set()
        for j in range(start_idx, end_idx):
            instr = instructions[j]
            if instr.op in (VM.INC, VM.DEC) and len(instr.operands) == 1:
                r = instr.operands[0]
                if r < -2:
                    op = '++' if instr.op == VM.INC else '--'
                    var_name = self._get_local_name(r)

                    is_postfix = False
                    if j > start_idx:
                        prev = instructions[j - 1]
                        if (prev.op == VM.CP and prev.operands[1] == r and
                            prev.operands[0] >= 0):
                            is_postfix = True

                    if is_postfix:
                        merged_addrs.add(instr.addr)
                        continue

                    side_effect_expr = UnaryExpr(op, VarExpr(var_name), prefix=True)
                    new_cond = self._replace_var_in_expr(cond, var_name, side_effect_expr)
                    if new_cond is not cond:
                        cond = new_cond
                        merged_addrs.add(instr.addr)

        return cond, merged_addrs

    def _replace_var_in_expr(self, expr: Expr, var_name: str, replacement: Expr) -> Expr:
        if isinstance(expr, VarExpr) and expr.name == var_name:
            return replacement
        if isinstance(expr, BinaryExpr):
            new_left = self._replace_var_in_expr(expr.left, var_name, replacement)
            if new_left is not expr.left:
                return BinaryExpr(new_left, expr.op, expr.right)
            new_right = self._replace_var_in_expr(expr.right, var_name, replacement)
            if new_right is not expr.right:
                return BinaryExpr(expr.left, expr.op, new_right)
        if isinstance(expr, UnaryExpr):
            new_operand = self._replace_var_in_expr(expr.operand, var_name, replacement)
            if new_operand is not expr.operand:
                return UnaryExpr(expr.op, new_operand, expr.prefix)
        return expr

    def _try_detect_swap(self, instructions: List[Instruction], obj: CodeObject,
                         start_idx: int, end_idx: int) -> Optional[Dict]:
        if start_idx + 4 <= end_idx:
            i0, i1, i2, i3 = instructions[start_idx:start_idx + 4]
            if (i0.op == VM.GPD and i1.op == VM.GPD and
                i2.op in (VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS) and
                i3.op in (VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS)):

                r1, obj1, prop1_idx = i0.operands[0], i0.operands[1], i0.operands[2]
                r2, obj2, prop2_idx = i1.operands[0], i1.operands[1], i1.operands[2]
                obj3, prop3_idx, val3 = i2.operands[0], i2.operands[1], i2.operands[2]
                obj4, prop4_idx, val4 = i3.operands[0], i3.operands[1], i3.operands[2]

                if (obj1 == obj2 == obj3 == obj4 and
                    prop1_idx == prop3_idx and prop2_idx == prop4_idx and
                    r1 == val4 and r2 == val3):

                    prop1 = obj.data[prop1_idx] if prop1_idx < len(obj.data) else f'prop{prop1_idx}'
                    prop2 = obj.data[prop2_idx] if prop2_idx < len(obj.data) else f'prop{prop2_idx}'

                    def get_obj_expr(reg):
                        if reg == -1:
                            return ThisExpr()
                        elif reg == -2:
                            return ThisExpr()
                        elif reg < -2:
                            return VarExpr(self._get_local_name(reg))
                        else:
                            return self.regs.get(reg, VarExpr(f'%{reg}'))

                    obj_expr = get_obj_expr(obj1)
                    left = PropertyExpr(obj_expr, prop1 if isinstance(prop1, str) else str(prop1))
                    right = PropertyExpr(obj_expr, prop2 if isinstance(prop2, str) else str(prop2))

                    swap_stmt = ExprStmt(SwapExpr(left, right))
                    return {'stmt': swap_stmt, 'next_idx': start_idx + 4}

        if start_idx + 2 < end_idx:
            i0, i1, i2 = instructions[start_idx:start_idx + 3]

            if i0.op == VM.CP and i1.op == VM.CP and i2.op == VM.CP:
                temp, src1 = i0.operands[0], i0.operands[1]
                dest1, src2 = i1.operands[0], i1.operands[1]
                dest2, src3 = i2.operands[0], i2.operands[1]

                if (temp > 0 and src3 == temp and
                    src1 == dest1 and src2 == dest2 and
                    src1 < -2 and src2 < -2):

                    left = VarExpr(self._get_local_name(src1))
                    right = VarExpr(self._get_local_name(src2))

                    swap_stmt = ExprStmt(SwapExpr(left, right))
                    return {'stmt': swap_stmt, 'next_idx': start_idx + 3}

        if start_idx + 5 <= end_idx:
            i0, i1, i2, i3, i4 = instructions[start_idx:start_idx + 5]
            if (i0.op in (VM.GPD, VM.GPDS) and
                i1.op in (VM.GPI, VM.GPIS) and
                i2.op in (VM.GPD, VM.GPDS) and
                i3.op in (VM.SPI, VM.SPIE, VM.SPIS) and
                i4.op == VM.CP):

                r1, obj1, prop1 = i0.operands[0], i0.operands[1], i0.operands[2]
                r2, gpi_obj, idx_reg = i1.operands[0], i1.operands[1], i1.operands[2]
                r3, obj2, prop2 = i2.operands[0], i2.operands[1], i2.operands[2]
                spi_obj, spi_idx, spi_val = i3.operands[0], i3.operands[1], i3.operands[2]
                cp_dest, cp_src = i4.operands[0], i4.operands[1]

                if (obj1 == obj2 and prop1 == prop2 and
                    gpi_obj == r1 and spi_obj == r3 and
                    idx_reg == spi_idx and
                    cp_src == r2 and
                    cp_dest == spi_val and
                    cp_dest < -2):

                    def get_obj_expr(reg):
                        if reg == -1 or reg == -2:
                            return ThisExpr()
                        elif reg < -2:
                            return VarExpr(self._get_local_name(reg))
                        else:
                            return self.regs.get(reg, VarExpr(f'%{reg}'))

                    left = VarExpr(self._get_local_name(cp_dest))
                    obj_expr = get_obj_expr(obj1)
                    prop_name = obj.data[prop1] if prop1 < len(obj.data) else f'prop{prop1}'
                    container_expr = PropertyExpr(obj_expr, prop_name if isinstance(prop_name, str) else str(prop_name))
                    idx_expr = get_obj_expr(idx_reg)
                    right = PropertyExpr(container_expr, idx_expr)

                    swap_stmt = ExprStmt(SwapExpr(left, right))
                    return {'stmt': swap_stmt, 'next_idx': start_idx + 5}

        return None

    def _try_detect_logical_expr(self, instructions: List[Instruction], obj: CodeObject,
                                   cond_idx: int, end_idx: int, addr_to_idx: Dict[int, int]) -> Optional[Dict]:
        jf_instr = instructions[cond_idx]
        if jf_instr.op not in (VM.JF, VM.JNF):
            return None

        target = jf_instr.addr + jf_instr.operands[0]
        target_idx = addr_to_idx.get(target)
        if target_idx is None or target_idx >= end_idx:
            return None

        target_instr = instructions[target_idx]
        if target_instr.op != VM.SETF:
            return None

        fall_through_idx = cond_idx + 1
        for j in range(fall_through_idx, target_idx):
            instr = instructions[j]
            if instr.op in (VM.JF, VM.JNF, VM.JMP, VM.RET, VM.SRV, VM.ENTRY):
                return None

        left_expr = self._get_condition(False)

        saved_regs = dict(self.regs)
        saved_flag = self.flag
        saved_flag_negated = self.flag_negated

        for j in range(fall_through_idx, target_idx):
            self._translate_instruction(instructions[j], obj)

        right_expr = self._get_condition(False)

        self.regs = saved_regs
        self.flag = saved_flag
        self.flag_negated = saved_flag_negated

        if jf_instr.op == VM.JF:
            logical_expr = BinaryExpr(left_expr, '||', right_expr)
        else:
            logical_expr = BinaryExpr(left_expr, '&&', right_expr)

        setf_reg = target_instr.operands[0]
        self.regs[setf_reg] = logical_expr

        next_idx = target_idx + 1

        return {'stmt': None, 'next_idx': next_idx}

    def _try_detect_value_logical_chain(self, instructions: List[Instruction], obj: CodeObject,
                                          cond_idx: int, end_idx: int, addr_to_idx: Dict[int, int]) -> Optional[Dict]:
        first_instr = instructions[cond_idx]
        if first_instr.op not in (VM.JF, VM.JNF):
            return None

        jump_op = first_instr.op
        target = first_instr.addr + first_instr.operands[0]
        target_idx = addr_to_idx.get(target)
        if target_idx is None or target_idx >= end_idx:
            return None

        target_instr = instructions[target_idx]
        if target_instr.op not in (VM.SETF, VM.SETNF):
            return None

        jump_indices = [cond_idx]
        for j in range(cond_idx + 1, target_idx):
            instr = instructions[j]
            if instr.op == jump_op:
                this_target = instr.addr + instr.operands[0]
                if this_target == target:
                    jump_indices.append(j)
                else:
                    return None
            elif instr.op in (VM.JF, VM.JNF):
                if instr.op != jump_op:
                    return None
            elif instr.op in (VM.JMP, VM.RET, VM.SRV, VM.ENTRY):
                return None

        conditions = [self._get_condition(False)]

        saved_regs = dict(self.regs)
        saved_flag = self.flag
        saved_flag_negated = self.flag_negated

        for i, jump_idx in enumerate(jump_indices[:-1]):
            next_jump_idx = jump_indices[i + 1]
            self.regs = dict(saved_regs)
            self.flag = saved_flag
            self.flag_negated = saved_flag_negated
            for j in range(jump_idx + 1, next_jump_idx):
                self._translate_instruction(instructions[j], obj)
            conditions.append(self._get_condition(False))

        last_jump_idx = jump_indices[-1]
        self.regs = dict(saved_regs)
        self.flag = saved_flag
        self.flag_negated = saved_flag_negated
        for j in range(last_jump_idx + 1, target_idx):
            self._translate_instruction(instructions[j], obj)
        conditions.append(self._get_condition(False))

        self.regs = saved_regs
        self.flag = saved_flag
        self.flag_negated = saved_flag_negated

        if jump_op == VM.JNF:
            combined = conditions[0]
            for cond in conditions[1:]:
                combined = BinaryExpr(combined, '&&', cond)
        else:
            combined = conditions[0]
            for cond in conditions[1:]:
                combined = BinaryExpr(combined, '||', cond)

        if target_instr.op == VM.SETNF:
            result_expr = self._negate_expr(combined)
        else:
            result_expr = combined

        reg = target_instr.operands[0]
        self.regs[reg] = result_expr

        return {'stmt': None, 'next_idx': target_idx + 1}

    def _try_detect_or_chain(self, instructions: List[Instruction], obj: CodeObject,
                              start_idx: int, end_idx: int, addr_to_idx: Dict[int, int]) -> Optional[Dict]:
        jf_instr = instructions[start_idx]
        if jf_instr.op != VM.JF:
            return None

        jf_target = jf_instr.addr + jf_instr.operands[0]
        jf_target_idx = addr_to_idx.get(jf_target)
        if jf_target_idx is None:
            return None

        jf_indices = [start_idx]
        jnf_idx = None
        jnf_target = None

        for j in range(start_idx + 1, jf_target_idx):
            instr = instructions[j]
            if instr.op == VM.JF:
                this_target = instr.addr + instr.operands[0]
                if this_target == jf_target:
                    jf_indices.append(j)
            elif instr.op == VM.JNF:
                jnf_idx = j
                jnf_target = instr.addr + instr.operands[0]
                break

        if jnf_idx is None or jnf_target is None:
            return None

        jnf_target_idx = addr_to_idx.get(jnf_target)
        if jnf_target_idx is None:
            return None

        if not (jnf_idx < jf_target_idx < jnf_target_idx):
            return None

        and_jnf_indices = []
        real_then_start = jf_target_idx
        for j in range(jf_target_idx, jnf_target_idx):
            instr = instructions[j]
            if instr.op == VM.JNF:
                this_target = instr.addr + instr.operands[0]
                if this_target == jnf_target:
                    and_jnf_indices.append(j)
                    real_then_start = j + 1
                else:
                    break
            elif instr.op in (VM.JF, VM.JMP, VM.ENTRY):
                break

        then_end_idx = jnf_target_idx
        else_end_idx = jnf_target_idx
        has_else = jnf_target_idx < end_idx

        for j in range(real_then_start, jnf_target_idx):
            instr = instructions[j]
            if instr.op == VM.JMP:
                jmp_target = instr.addr + instr.operands[0]
                jmp_target_idx = addr_to_idx.get(jmp_target)
                if jmp_target_idx and jmp_target_idx > jnf_target_idx:
                    then_end_idx = j
                    else_end_idx = min(jmp_target_idx, end_idx)
                    has_else = True
                    break

        or_conditions = []

        or_conditions.append(self._get_condition(False))

        saved_regs = dict(self.regs)
        saved_flag = self.flag
        saved_flag_negated = self.flag_negated

        for i, jf_idx in enumerate(jf_indices[:-1]):
            next_jf_idx = jf_indices[i + 1]
            self.regs = dict(saved_regs)
            self.flag = saved_flag
            self.flag_negated = saved_flag_negated

            for j in range(jf_idx + 1, next_jf_idx):
                self._translate_instruction(instructions[j], obj)

            or_conditions.append(self._get_condition(False))

        last_jf_idx = jf_indices[-1]
        self.regs = dict(saved_regs)
        self.flag = saved_flag
        self.flag_negated = saved_flag_negated

        for j in range(last_jf_idx + 1, jnf_idx):
            self._translate_instruction(instructions[j], obj)

        or_conditions.append(self._get_condition(False))

        and_conditions = []
        if and_jnf_indices:
            seg_start = jf_target_idx
            for and_jnf_idx in and_jnf_indices:
                self.regs = dict(saved_regs)
                self.flag = saved_flag
                self.flag_negated = saved_flag_negated
                for j in range(seg_start, and_jnf_idx):
                    self._translate_instruction(instructions[j], obj)
                and_conditions.append(self._get_condition(False))
                seg_start = and_jnf_idx + 1

        self.regs = saved_regs
        self.flag = saved_flag
        self.flag_negated = saved_flag_negated

        combined_or = or_conditions[0]
        for cond in or_conditions[1:]:
            combined_or = BinaryExpr(combined_or, '||', cond)

        if and_conditions:
            combined_cond = combined_or
            for cond in and_conditions:
                combined_cond = BinaryExpr(combined_cond, '&&', cond)
        else:
            combined_cond = combined_or

        then_stmts = self._generate_structured_code(instructions, obj, real_then_start, then_end_idx)

        else_stmts = []
        if has_else and else_end_idx > jnf_target_idx:
            else_stmts = self._generate_structured_code(instructions, obj, jnf_target_idx, else_end_idx)

        result_stmt = IfStmt(combined_cond, then_stmts, else_stmts)

        next_idx = else_end_idx if has_else else jnf_target_idx
        return {'stmt': result_stmt, 'next_idx': next_idx}

    def _try_detect_and_chain(self, instructions: List[Instruction], obj: CodeObject,
                                start_idx: int, end_idx: int, addr_to_idx: Dict[int, int]) -> Optional[Dict]:
        jnf_instr = instructions[start_idx]
        if jnf_instr.op != VM.JNF:
            return None

        jnf_target = jnf_instr.addr + jnf_instr.operands[0]
        jnf_target_idx = addr_to_idx.get(jnf_target)
        if jnf_target_idx is None:
            return None

        if jnf_target_idx >= end_idx:
            return None

        jnf_indices = [start_idx]
        scan_idx = start_idx + 1
        while scan_idx < jnf_target_idx:
            instr = instructions[scan_idx]
            if instr.op == VM.JNF:
                this_target = instr.addr + instr.operands[0]
                if this_target == jnf_target:
                    jnf_indices.append(scan_idx)
                else:
                    break
            elif instr.op == VM.JF:
                break
            elif instr.op in (VM.JMP, VM.ENTRY, VM.RET, VM.SRV):
                break
            scan_idx += 1

        if len(jnf_indices) < 2:
            return None

        first_cond = self._get_condition(False)
        conditions = [first_cond]

        saved_regs = dict(self.regs)
        saved_flag = self.flag
        saved_flag_negated = self.flag_negated

        for i, jnf_idx in enumerate(jnf_indices[:-1]):
            next_jnf_idx = jnf_indices[i + 1]
            self.regs = dict(saved_regs)
            self.flag = saved_flag
            self.flag_negated = saved_flag_negated
            for j in range(jnf_idx + 1, next_jnf_idx):
                self._translate_instruction(instructions[j], obj)
            conditions.append(self._get_condition(False))

        self.regs = saved_regs
        self.flag = saved_flag
        self.flag_negated = saved_flag_negated

        combined_cond = conditions[0]
        for cond in conditions[1:]:
            combined_cond = BinaryExpr(combined_cond, '&&', cond)

        last_jnf_idx = jnf_indices[-1]
        then_start_idx = last_jnf_idx + 1

        then_end_idx = jnf_target_idx
        else_end_idx = jnf_target_idx
        has_else = False

        current_loop = self.loop_context_stack[-1] if self.loop_context_stack else None
        best_jmp_target_idx = -1

        for j in range(then_start_idx, jnf_target_idx):
            instr = instructions[j]
            if instr.op == VM.JMP:
                jmp_target = instr.addr + instr.operands[0]
                jmp_target_idx = addr_to_idx.get(jmp_target)
                if current_loop:
                    loop_start_addr, loop_exit_addr = current_loop
                    if jmp_target >= loop_exit_addr:
                        continue
                    elif jmp_target == loop_start_addr:
                        continue
                if jmp_target_idx and jmp_target_idx > jnf_target_idx:
                    if jmp_target_idx >= best_jmp_target_idx:
                        has_else = True
                        then_end_idx = j
                        else_end_idx = min(jmp_target_idx, end_idx)
                        best_jmp_target_idx = jmp_target_idx

        if has_else:
            ternary_result = self._try_detect_ternary(
                instructions, obj, then_start_idx, then_end_idx,
                jnf_target_idx, else_end_idx, combined_cond
            )
            if ternary_result:
                return ternary_result

        then_stmts = self._generate_structured_code(instructions, obj, then_start_idx, then_end_idx)
        else_stmts = []
        if has_else:
            else_stmts = self._generate_structured_code(instructions, obj, jnf_target_idx, else_end_idx)

        if_stmt = IfStmt(combined_cond, then_stmts, else_stmts)
        next_idx = else_end_idx if has_else else jnf_target_idx
        return {'stmt': if_stmt, 'next_idx': next_idx}

    def _try_process_short_circuit(self, instructions: List[Instruction], obj: CodeObject,
                                    cond_idx: int, end_idx: int, addr_to_idx: Dict) -> Optional[int]:
        instr = instructions[cond_idx]
        target_addr = instr.addr + instr.operands[0]

        setf_idx = None
        setf_addr = None
        scan_start = cond_idx + 1
        scan_limit = min(scan_start + 30, end_idx)
        for j in range(scan_start, scan_limit):
            if instructions[j].op in (VM.SETF, VM.SETNF):
                setf_idx = j
                setf_addr = instructions[j].addr
                break

        if setf_idx is None:
            return None

        cond_addr = instr.addr
        for j in range(cond_idx, setf_idx):
            inst = instructions[j]
            if inst.op in (VM.JF, VM.JNF):
                jmp_target = inst.addr + inst.operands[0]
                if jmp_target == setf_addr:
                    continue
                if cond_addr < jmp_target < setf_addr:
                    continue
                return None
            if inst.op in (VM.JMP, VM.RET, VM.SRV, VM.ENTRY):
                return None

        segments = []
        seg_start = cond_idx
        for j in range(cond_idx, setf_idx):
            inst = instructions[j]
            if inst.op in (VM.JF, VM.JNF):
                jmp_target = inst.addr + inst.operands[0]
                segments.append((seg_start, j, inst.op, jmp_target))
                seg_start = j + 1

        if seg_start < setf_idx:
            segments.append((seg_start, setf_idx, None, None))

        conditions = []
        for seg_start_idx, seg_end_idx, jmp_op, jmp_target in segments:
            for j in range(seg_start_idx, seg_end_idx):
                self._translate_instruction(instructions[j], obj)
            cond = self._get_condition(False)
            conditions.append((cond, jmp_op, jmp_target))

        def find_target_seg(target_addr_val):
            for k in range(len(segments)):
                s_start_idx = segments[k][0]
                if s_start_idx < len(instructions) and instructions[s_start_idx].addr == target_addr_val:
                    return k
            return None

        def build_expr(start, end):
            if start >= end:
                return ConstExpr(True)
            if start == end - 1:
                cond, jmp_op, jmp_target_val = conditions[start]
                return cond

            cond, jmp_op, jmp_target_val = conditions[start]

            if jmp_op == VM.JF:
                if jmp_target_val == setf_addr:
                    rest = build_expr(start + 1, end)
                    return BinaryExpr(cond, '||', rest)
                else:
                    target_seg = find_target_seg(jmp_target_val)
                    if target_seg is not None and target_seg > start + 1:
                        or_group = build_or_group(start, target_seg)
                        if target_seg >= end:
                            return or_group
                        rest = build_expr(target_seg, end)
                        return BinaryExpr(or_group, '&&', rest)
                    rest = build_expr(start + 1, end)
                    return BinaryExpr(cond, '||', rest)
            elif jmp_op == VM.JNF:
                if jmp_target_val == setf_addr:
                    rest = build_expr(start + 1, end)
                    return BinaryExpr(cond, '&&', rest)
                else:
                    target_seg = find_target_seg(jmp_target_val)
                    if target_seg is not None and target_seg > start + 1:
                        and_group = build_and_group(start, target_seg)
                        if target_seg >= end:
                            return and_group
                        rest = build_expr(target_seg, end)
                        return BinaryExpr(and_group, '||', rest)
                    rest = build_expr(start + 1, end)
                    return BinaryExpr(cond, '&&', rest)
            else:
                return cond

        def build_and_group(start, end):
            if start >= end:
                return ConstExpr(True)
            if start == end - 1:
                return conditions[start][0]
            cond, _, _ = conditions[start]
            rest = build_and_group(start + 1, end)
            return BinaryExpr(cond, '&&', rest)

        def build_or_group(start, end):
            if start >= end:
                return ConstExpr(True)
            if start == end - 1:
                return conditions[start][0]
            cond, _, _ = conditions[start]
            rest = build_or_group(start + 1, end)
            return BinaryExpr(cond, '||', rest)

        compound = build_expr(0, len(conditions))

        setf_instr = instructions[setf_idx]
        setf_reg = setf_instr.operands[0]
        if setf_instr.op == VM.SETNF:
            compound = self._negate_expr(compound)
        self.regs[setf_reg] = compound

        return setf_idx + 1

    def _process_if(self, instructions: List[Instruction], obj: CodeObject,
                    cond_idx: int, end_idx: int) -> Optional[Dict]:
        cond_instr = instructions[cond_idx]
        if cond_instr.op not in (VM.JF, VM.JNF):
            return None

        target = cond_instr.addr + cond_instr.operands[0]
        fall_through_idx = cond_idx + 1

        addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}

        if target not in addr_to_idx:
            return None

        target_idx = addr_to_idx[target]

        if target_idx < cond_idx:
            current_loop = self.loop_context_stack[-1] if self.loop_context_stack else None
            if current_loop and target == current_loop[0]:
                cond = self._get_condition(False)
                if cond_instr.op == VM.JNF:
                    cond = self._negate_expr(cond)
                stmt = IfStmt(cond, [ContinueStmt()], [])
                return {'stmt': stmt, 'next_idx': cond_idx + 1}
            return None

        logical_result = self._try_detect_logical_expr(instructions, obj, cond_idx, end_idx, addr_to_idx)
        if logical_result:
            return logical_result

        chain_result = self._try_detect_value_logical_chain(instructions, obj, cond_idx, end_idx, addr_to_idx)
        if chain_result:
            return chain_result

        if cond_instr.op == VM.JF:
            or_result = self._try_detect_or_chain(instructions, obj, cond_idx, end_idx, addr_to_idx)
            if or_result:
                return or_result

        if cond_instr.op == VM.JNF:
            and_result = self._try_detect_and_chain(instructions, obj, cond_idx, end_idx, addr_to_idx)
            if and_result:
                return and_result

        cond = self._get_condition(False)

        if cond_instr.op == VM.JNF:
            if_cond = cond
        else:
            if_cond = self._negate_expr(cond)

        if target_idx >= end_idx:
            then_stmts = self._generate_structured_code(instructions, obj, fall_through_idx, end_idx)
            if_stmt = IfStmt(if_cond, then_stmts, [])
            return {'stmt': if_stmt, 'next_idx': end_idx}

        has_else = False
        else_end_idx = target_idx
        then_end_idx = target_idx

        if target_idx > fall_through_idx and instructions[target_idx - 1].op == VM.JMP:
            jmp_instr = instructions[target_idx - 1]
            jmp_target = jmp_instr.addr + jmp_instr.operands[0]
            jmp_target_idx = addr_to_idx.get(jmp_target)

            is_break_or_continue = False
            current_loop = self.loop_context_stack[-1] if self.loop_context_stack else None
            if current_loop:
                loop_start_addr, loop_exit_addr = current_loop
                if jmp_target >= loop_exit_addr:
                    is_break_or_continue = True
                elif jmp_target == loop_start_addr:
                    is_break_or_continue = True

            if not is_break_or_continue and jmp_target_idx is not None and jmp_target_idx > target_idx:
                has_else = True
                then_end_idx = target_idx - 1
                else_end_idx = min(jmp_target_idx, end_idx)

        if has_else:
            ternary_result = self._try_detect_ternary(
                instructions, obj, fall_through_idx, then_end_idx,
                target_idx, else_end_idx, if_cond
            )
            if ternary_result:
                return ternary_result

        then_stmts = self._generate_structured_code(instructions, obj, fall_through_idx, then_end_idx)

        else_stmts = []
        if has_else:
            else_stmts = self._generate_structured_code(instructions, obj, target_idx, else_end_idx)

        if_stmt = IfStmt(if_cond, then_stmts, else_stmts)
        next_idx = else_end_idx if has_else else target_idx

        return {'stmt': if_stmt, 'next_idx': next_idx}

    def _process_try(self, instructions: List[Instruction], obj: CodeObject,
                     start_idx: int, end_idx: int) -> Optional[Dict]:
        entry_instr = instructions[start_idx]
        if entry_instr.op != VM.ENTRY:
            return None

        catch_offset = entry_instr.operands[0]
        exception_reg = entry_instr.operands[1]
        catch_addr = entry_instr.addr + catch_offset

        addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}

        if catch_addr not in addr_to_idx:
            return None
        catch_idx = addr_to_idx[catch_addr]

        extry_idx = None
        for j in range(start_idx + 1, catch_idx):
            if instructions[j].op == VM.EXTRY:
                extry_idx = j

        if extry_idx is None:
            extry_idx = catch_idx - 2

        skip_catch_idx = end_idx
        for j in range(extry_idx + 1, catch_idx):
            if instructions[j].op == VM.JMP:
                jmp_target = instructions[j].addr + instructions[j].operands[0]
                if jmp_target >= catch_addr:
                    if jmp_target in addr_to_idx:
                        skip_catch_idx = addr_to_idx[jmp_target]
                    break

        try_body_start = start_idx + 1
        try_body_end = extry_idx
        try_stmts = self._generate_structured_code(instructions, obj, try_body_start, try_body_end)

        catch_var_name = None
        catch_body_start = catch_idx
        if catch_idx < len(instructions):
            first_catch_instr = instructions[catch_idx]
            if first_catch_instr.op == VM.CP:
                dest_reg = first_catch_instr.operands[0]
                src_reg = first_catch_instr.operands[1]
                if src_reg == exception_reg and dest_reg < -2:
                    catch_var_name = self._get_local_name(dest_reg)
                    catch_body_start = catch_idx + 1

        if catch_var_name is None:
            if exception_reg < -2:
                catch_var_name = self._get_local_name(exception_reg)
            else:
                catch_var_name = f'%{exception_reg}'

        catch_stmts = self._generate_structured_code(instructions, obj, catch_body_start, skip_catch_idx)

        try_stmt = TryStmt(try_stmts, catch_var_name, catch_stmts)
        return {'stmt': try_stmt, 'next_idx': skip_catch_idx}

    def _process_switch(self, instructions: List[Instruction], obj: CodeObject,
                        start_idx: int, end_idx: int) -> Optional[Dict]:
        addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}

        jnf_instr = instructions[start_idx]
        if jnf_instr.op != VM.JNF:
            return None

        ceq_idx = start_idx - 1
        if ceq_idx < 0 or instructions[ceq_idx].op != VM.CEQ:
            return None
        ref_reg = instructions[ceq_idx].operands[0]

        case_count = 0
        scan_idx = ceq_idx
        case_infos = []

        while scan_idx < end_idx:
            instr = instructions[scan_idx]

            if instr.op == VM.CEQ and instr.operands[0] == ref_reg:
                jnf_idx = scan_idx + 1
                if jnf_idx < end_idx and instructions[jnf_idx].op in (VM.JF, VM.JNF):
                    pass
                else:
                    jnf_idx = None

                if jnf_idx is not None:
                    if instructions[jnf_idx].op != VM.JNF:
                        break

                    jnf_target = instructions[jnf_idx].addr + instructions[jnf_idx].operands[0]
                    case_value_reg = instr.operands[1]
                    case_value_expr = self.regs.get(case_value_reg, VarExpr(f'%{case_value_reg}'))
                    case_infos.append({
                        'ceq_idx': scan_idx,
                        'jnf_idx': jnf_idx,
                        'case_value': case_value_expr,
                        'jnf_target': jnf_target,
                        'jnf_target_idx': addr_to_idx.get(jnf_target)
                    })
                    case_count += 1
                    scan_idx = jnf_idx + 1
                    continue

            if instr.op == VM.CONST:
                self._translate_instruction(instr, obj)
                scan_idx += 1
                continue

            if instr.op == VM.JMP:
                scan_idx += 1
                continue

            if case_infos:
                last_jnf_target = case_infos[-1]['jnf_target']
                last_jnf_target_idx = addr_to_idx.get(last_jnf_target)
                if last_jnf_target_idx is not None and last_jnf_target_idx > scan_idx:
                    found_next_case = False
                    for ahead_idx in range(last_jnf_target_idx, end_idx):
                        ahead_instr = instructions[ahead_idx]
                        if ahead_instr.op == VM.CEQ and ahead_instr.operands[0] == ref_reg:
                            scan_idx = last_jnf_target_idx
                            found_next_case = True
                            break
                        if ahead_instr.op == VM.CONST:
                            continue
                        break
                    if not found_next_case:
                        break
                    if scan_idx == last_jnf_target_idx:
                        continue

            break

        if case_count < 2:
            return None


        switch_end_addr = 0
        body_regions = []


        for i, case_info in enumerate(case_infos):
            jnf_idx = case_info['jnf_idx']

            if jnf_idx + 1 < end_idx and instructions[jnf_idx + 1].op == VM.JMP:
                body_jmp = instructions[jnf_idx + 1]
                body_addr = body_jmp.addr + body_jmp.operands[0]
                case_info['body_start'] = body_addr
            elif jnf_idx + 1 < end_idx:
                case_info['body_start'] = instructions[jnf_idx + 1].addr
            else:
                case_info['body_start'] = case_info['jnf_target']

        last_case = case_infos[-1]
        default_or_end_addr = last_case['jnf_target']
        first_case_addr = case_infos[0]['ceq_idx']
        first_case_instr_addr = instructions[case_infos[0]['ceq_idx']].addr

        backward_default = default_or_end_addr < first_case_instr_addr

        scan_start = last_case['jnf_idx'] + 1
        if not backward_default:
            for j in range(scan_start, end_idx):
                instr = instructions[j]
                if instr.op == VM.JMP:
                    jmp_target = instr.addr + instr.operands[0]
                    if jmp_target > switch_end_addr:
                        switch_end_addr = jmp_target
                if instr.addr >= default_or_end_addr:
                    break

        if not backward_default:
            default_or_end_idx_for_scan = addr_to_idx.get(default_or_end_addr, end_idx)
            for j in range(default_or_end_idx_for_scan, end_idx):
                instr = instructions[j]
                if instr.op == VM.JMP:
                    jmp_target = instr.addr + instr.operands[0]
                    if jmp_target == default_or_end_addr and instr.addr > default_or_end_addr:
                        if j + 1 < len(instructions):
                            true_end = instructions[j + 1].addr
                            if true_end > switch_end_addr:
                                switch_end_addr = true_end
                        break

        if switch_end_addr == 0:
            if backward_default:
                for j in range(scan_start, end_idx):
                    instr = instructions[j]
                    if instr.op == VM.RET:
                        if j + 1 < end_idx:
                            k = j + 1
                            while k < end_idx and instructions[k].op == VM.JMP:
                                k += 1
                            if k < end_idx:
                                switch_end_addr = instructions[k].addr
                            else:
                                switch_end_addr = instructions[end_idx - 1].addr + 1
                        break
                if switch_end_addr == 0:
                    switch_end_addr = instructions[end_idx - 1].addr + 1
            else:
                switch_end_addr = default_or_end_addr

        switch_end_idx = addr_to_idx.get(switch_end_addr, end_idx)


        body_to_cases: Dict[int, List[Dict]] = {}
        for case_info in case_infos:
            body_start = case_info.get('body_start', case_info['jnf_target'])
            if body_start not in body_to_cases:
                body_to_cases[body_start] = []
            body_to_cases[body_start].append(case_info)

        sorted_bodies = sorted(body_to_cases.keys())

        if_chain = []
        ref_expr = self.regs.get(ref_reg, VarExpr(f'%{ref_reg}'))

        for body_addr in sorted_bodies:
            cases = body_to_cases[body_addr]

            conditions = []
            for case in cases:
                cond = BinaryExpr(ref_expr, '==', case['case_value'])
                conditions.append(cond)

            if len(conditions) == 1:
                combined_cond = conditions[0]
            else:
                combined_cond = conditions[0]
                for c in conditions[1:]:
                    combined_cond = BinaryExpr(combined_cond, '||', c)

            if_chain.append({
                'condition': combined_cond,
                'body_addr': body_addr
            })

        for i, item in enumerate(if_chain):
            body_addr = item['body_addr']
            body_start_idx = addr_to_idx.get(body_addr, end_idx)

            body_end_idx = switch_end_idx

            if i + 1 < len(if_chain):
                next_body_addr = if_chain[i + 1]['body_addr']
                next_body_idx = addr_to_idx.get(next_body_addr, end_idx)

                fall_through_jmp_idx = None
                fall_through_target_addr = None
                for j in range(body_start_idx, min(next_body_idx, end_idx)):
                    instr = instructions[j]
                    if instr.op == VM.RET:
                        body_end_idx = j + 1
                        break
                    if instr.op == VM.JMP:
                        jmp_target = instr.addr + instr.operands[0]
                        if jmp_target >= switch_end_addr:
                            body_end_idx = j
                            break
                        elif jmp_target == next_body_addr:
                            fall_through_jmp_idx = j
                            fall_through_target_addr = jmp_target
                else:
                    if fall_through_jmp_idx is not None:
                        body_end_idx = fall_through_jmp_idx
                        fall_through_idx = addr_to_idx.get(fall_through_target_addr, end_idx)
                        fall_through_end_idx = switch_end_idx
                        for j in range(fall_through_idx, switch_end_idx):
                            instr = instructions[j]
                            if instr.op == VM.JMP:
                                jmp_target = instr.addr + instr.operands[0]
                                if jmp_target >= switch_end_addr:
                                    fall_through_end_idx = j
                                    break
                        item['fall_through_start'] = fall_through_idx
                        item['fall_through_end'] = fall_through_end_idx
                    else:
                        for next_case in body_to_cases.get(next_body_addr, []):
                            if next_case['ceq_idx'] < next_body_idx:
                                body_end_idx = next_case['ceq_idx'] - 1
                                break
                        else:
                            body_end_idx = next_body_idx
            else:
                if default_or_end_addr < switch_end_addr:
                    for j in range(body_start_idx, switch_end_idx):
                        instr = instructions[j]
                        if instr.op == VM.RET:
                            body_end_idx = j + 1
                            break
                        if instr.op == VM.JMP:
                            jmp_target = instr.addr + instr.operands[0]
                            if jmp_target >= switch_end_addr:
                                body_end_idx = j
                                break

            item['body_start_idx'] = body_start_idx
            item['body_end_idx'] = body_end_idx

        has_default = False
        default_body_start_idx = None
        default_body_end_idx = None

        if not backward_default and default_or_end_addr < switch_end_addr:
            has_default = True
            default_body_start_idx = addr_to_idx.get(default_or_end_addr, end_idx)
            default_body_end_idx = switch_end_idx

        result_stmt = None

        for i, item in enumerate(if_chain):
            cond = item['condition']
            body_start = item['body_start_idx']
            body_end = item['body_end_idx']

            self.regs.clear()

            body_stmts = self._generate_structured_code(instructions, obj, body_start, body_end)

            if 'fall_through_start' in item and 'fall_through_end' in item:
                fall_through_stmts = self._generate_structured_code(
                    instructions, obj, item['fall_through_start'], item['fall_through_end'])
                body_stmts.extend(fall_through_stmts)

            if result_stmt is None:
                result_stmt = IfStmt(cond, body_stmts, [])
            else:
                current = result_stmt
                while current.else_body and len(current.else_body) == 1 and isinstance(current.else_body[0], IfStmt):
                    current = current.else_body[0]
                current.else_body = [IfStmt(cond, body_stmts, [])]

        if has_default and result_stmt is not None:
            saved_loop = self.loop_headers.pop(default_or_end_addr, None)
            try:
                default_stmts = self._generate_structured_code(instructions, obj,
                                                               default_body_start_idx,
                                                               default_body_end_idx)
            finally:
                if saved_loop is not None:
                    self.loop_headers[default_or_end_addr] = saved_loop
            current = result_stmt
            while current.else_body and len(current.else_body) == 1 and isinstance(current.else_body[0], IfStmt):
                current = current.else_body[0]
            current.else_body = default_stmts

        if result_stmt is None:
            return None

        return {'stmt': result_stmt, 'next_idx': switch_end_idx}

    def _try_detect_ternary(self, instructions: List[Instruction], obj: CodeObject,
                           then_start: int, then_end: int,
                           else_start: int, else_end: int,
                           condition: Expr) -> Optional[Dict]:
        then_result = self._analyze_branch_for_ternary(instructions, then_start, then_end, obj)
        if then_result is None:
            return None

        then_target_reg, then_has_side_effects = then_result
        if then_has_side_effects:
            return None

        else_result = self._analyze_branch_for_ternary(instructions, else_start, else_end, obj)
        if else_result is None:
            return None

        else_target_reg, else_has_side_effects = else_result
        if else_has_side_effects:
            return None

        if then_target_reg != else_target_reg:
            return None

        if then_target_reg <= 0:
            return None

        saved_regs = dict(self.regs)

        self.regs = dict(saved_regs)
        for i in range(then_start, then_end):
            self._translate_instruction(instructions[i], obj)
        true_expr = self.regs.get(then_target_reg, VoidExpr())

        self.regs = dict(saved_regs)

        nested_ternary = self._try_detect_nested_ternary(
            instructions, obj, else_start, else_end, else_target_reg)

        if nested_ternary is not None:
            false_expr = nested_ternary
        else:
            for i in range(else_start, else_end):
                self._translate_instruction(instructions[i], obj)
            false_expr = self.regs.get(else_target_reg, VoidExpr())

        self.regs = dict(saved_regs)
        ternary = TernaryExpr(condition, true_expr, false_expr)
        self.regs[then_target_reg] = ternary

        return {'stmt': None, 'next_idx': else_end, 'is_ternary': True}

    def _try_detect_nested_ternary(self, instructions: List[Instruction], obj: CodeObject,
                                   start_idx: int, end_idx: int,
                                   expected_target_reg: int) -> Optional[Expr]:
        if start_idx >= end_idx:
            return None

        addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}

        jnf_idx = None
        for i in range(start_idx, end_idx):
            instr = instructions[i]
            if instr.op in (VM.JF, VM.JNF):
                jnf_target = instr.addr + instr.operands[0]
                jnf_target_idx = addr_to_idx.get(jnf_target)
                if jnf_target_idx is not None and start_idx < jnf_target_idx < end_idx:
                    jnf_idx = i
                    break

        if jnf_idx is None:
            return None

        jnf_instr = instructions[jnf_idx]
        nested_else_addr = jnf_instr.addr + jnf_instr.operands[0]
        nested_else_idx = addr_to_idx.get(nested_else_addr)

        if nested_else_idx is None:
            return None

        nested_then_start = jnf_idx + 1

        nested_then_end = nested_else_idx
        nested_end_idx = end_idx

        for j in range(nested_then_start, nested_else_idx):
            instr = instructions[j]
            if instr.op == VM.JMP:
                jmp_target = instr.addr + instr.operands[0]
                jmp_target_idx = addr_to_idx.get(jmp_target)
                if jmp_target_idx is not None and jmp_target_idx > nested_else_idx:
                    nested_then_end = j
                    nested_end_idx = jmp_target_idx
                    break

        saved_regs = dict(self.regs)
        for i in range(start_idx, jnf_idx):
            self._translate_instruction(instructions[i], obj)

        nested_cond = self._get_condition(False)
        if jnf_instr.op == VM.JNF:
            nested_if_cond = nested_cond
        else:
            nested_if_cond = self._negate_expr(nested_cond)

        nested_then_result = self._analyze_branch_for_ternary(
            instructions, nested_then_start, nested_then_end, obj)
        nested_else_result = self._analyze_branch_for_ternary(
            instructions, nested_else_idx, nested_end_idx, obj)

        if nested_then_result is None or nested_else_result is None:
            self.regs = dict(saved_regs)
            return None

        nested_then_reg, nested_then_side = nested_then_result
        nested_else_reg, nested_else_side = nested_else_result

        if nested_then_reg != nested_else_reg:
            self.regs = dict(saved_regs)
            return None

        if nested_then_side or nested_else_side:
            self.regs = dict(saved_regs)
            return None

        self.regs = dict(saved_regs)
        for i in range(start_idx, jnf_idx):
            self._translate_instruction(instructions[i], obj)
        for i in range(nested_then_start, nested_then_end):
            self._translate_instruction(instructions[i], obj)
        nested_true_expr = self.regs.get(nested_then_reg, VoidExpr())

        self.regs = dict(saved_regs)
        for i in range(start_idx, jnf_idx):
            self._translate_instruction(instructions[i], obj)

        further_nested = self._try_detect_nested_ternary(
            instructions, obj, nested_else_idx, nested_end_idx, nested_else_reg)

        if further_nested is not None:
            nested_false_expr = further_nested
        else:
            for i in range(nested_else_idx, nested_end_idx):
                self._translate_instruction(instructions[i], obj)
            nested_false_expr = self.regs.get(nested_else_reg, VoidExpr())

        self.regs = dict(saved_regs)

        return TernaryExpr(nested_if_cond, nested_true_expr, nested_false_expr)

    def _analyze_branch_for_ternary(self, instructions: List[Instruction],
                                    start_idx: int, end_idx: int,
                                    obj: CodeObject) -> Optional[Tuple[int, bool]]:
        if start_idx >= end_idx:
            return None

        target_reg = None
        has_side_effects = False

        for i in range(start_idx, end_idx):
            instr = instructions[i]
            op = instr.op
            ops = instr.operands

            if op == VM.CONST:
                target_reg = ops[0]
            elif op == VM.CP:
                r1 = ops[0]
                if r1 < -2:
                    has_side_effects = True
                target_reg = r1
            elif op == VM.CL:
                target_reg = ops[0]
            elif op in (VM.GPD, VM.GPDS, VM.GPI, VM.GPIS):
                target_reg = ops[0]
            elif op == VM.CALL:
                if ops[0] == 0:
                    has_side_effects = True
                target_reg = ops[0]
            elif op == VM.CALLD:
                if ops[0] == 0:
                    has_side_effects = True
                target_reg = ops[0]
            elif op == VM.CALLI:
                if ops[0] == 0:
                    has_side_effects = True
                target_reg = ops[0]
            elif op == VM.NEW:
                target_reg = ops[0]
            elif op in (VM.SETF, VM.SETNF):
                target_reg = ops[0]
            elif op == VM.GLOBAL:
                target_reg = ops[0]
            elif op == VM.CHS:
                target_reg = ops[0]
            elif op == VM.LNOT:
                target_reg = ops[0]
            elif op in (VM.INT, VM.REAL, VM.STR):
                target_reg = ops[0]
            elif op in (VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.MOD, VM.IDIV,
                       VM.BOR, VM.BAND, VM.BXOR, VM.SAL, VM.SAR, VM.SR):
                r1 = ops[0]
                if r1 < -2:
                    has_side_effects = True
                target_reg = r1
            elif op in (VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS, VM.SPI, VM.SPIE, VM.SPIS):
                has_side_effects = True
            elif op == VM.SRV:
                has_side_effects = True
            elif op == VM.RET:
                has_side_effects = True
            elif op in (VM.JF, VM.JNF, VM.JMP):
                continue
            elif op in (VM.TT, VM.TF, VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT):
                continue
            elif op in (VM.NOP, VM.NF):
                continue

        if target_reg is not None and target_reg != 0:
            return (target_reg, has_side_effects)

        return None

    def _get_condition(self, negate: bool = False) -> Expr:
        if self.flag is None:
            return ConstExpr(True)

        cond = self.flag
        if self.flag_negated:
            negate = not negate

        if negate:
            cond = self._negate_expr(cond)

        return cond

    def _negate_expr(self, expr: Expr) -> Expr:
        if isinstance(expr, UnaryExpr) and expr.op == '!':
            return expr.operand
        if isinstance(expr, BinaryExpr):
            inversions = {
                '==': '!=', '!=': '==',
                '===': '!==', '!==': '===',
                '<': '>=', '>=': '<',
                '>': '<=', '<=': '>',
            }
            if expr.op in inversions:
                return BinaryExpr(expr.left, inversions[expr.op], expr.right)
            if expr.op == '&&':
                return BinaryExpr(self._negate_expr(expr.left), '||', self._negate_expr(expr.right))
            if expr.op == '||':
                return BinaryExpr(self._negate_expr(expr.left), '&&', self._negate_expr(expr.right))
        return UnaryExpr('!', expr)

    def _flush_pending_spie(self):
        if self._pending_spie is not None:
            pending = self._pending_spie
            self._pending_spie = None
            return ExprStmt(AssignExpr(pending['target'], pending['value']))
        return None

    def _collect_pre_stmts(self, stmts: list):
        if self._pre_stmts:
            stmts.extend(self._pre_stmts)
            self._pre_stmts.clear()

    def _translate_instruction(self, instr: Instruction, obj: CodeObject) -> Optional[Stmt]:
        if self._pending_spie is not None:
            pending = self._pending_spie
            op_check = instr.op
            ops_check = instr.operands
            if (op_check == VM.CP and len(ops_check) >= 2 and
                    ops_check[1] == pending['value_reg'] and ops_check[0] < -2):
                self._pending_spie = None
                self._prev_instruction = instr
                r1_cp = ops_check[0]
                name = self._get_local_name(r1_cp)
                chain_value = AssignExpr(pending['target'], pending['value'])
                self.regs[r1_cp] = VarExpr(name)
                self.regs[pending['value_reg']] = VarExpr(name)
                if name not in self.declared_vars:
                    self.declared_vars.add(name)
                    return VarDeclStmt(name, chain_value)
                return ExprStmt(AssignExpr(VarExpr(name), chain_value))
            elif op_check in (VM.CALL, VM.CALLD, VM.CALLI):
                vreg = pending['value_reg']
                has_pending_use = False
                if op_check == VM.CALL:
                    if len(ops_check) > 1 and ops_check[1] == vreg:
                        has_pending_use = True
                elif op_check in (VM.CALLD, VM.CALLI):
                    if len(ops_check) > 1 and ops_check[1] == vreg:
                        has_pending_use = True
                if not has_pending_use:
                    if op_check == VM.CALL:
                        argc = ops_check[2] if len(ops_check) > 2 else 0
                        for k in range(argc):
                            arg_idx = 3 + k
                            if arg_idx < len(ops_check) and ops_check[arg_idx] == vreg:
                                has_pending_use = True
                                break
                    elif op_check == VM.CALLD:
                        argc = ops_check[3] if len(ops_check) > 3 else 0
                        for k in range(argc):
                            arg_idx = 4 + k
                            if arg_idx < len(ops_check) and ops_check[arg_idx] == vreg:
                                has_pending_use = True
                                break
                    elif op_check == VM.CALLI:
                        argc = ops_check[3] if len(ops_check) > 3 else 0
                        for k in range(argc):
                            arg_idx = 4 + k
                            if arg_idx < len(ops_check) and ops_check[arg_idx] == vreg:
                                has_pending_use = True
                                break
                if has_pending_use:
                    self._pending_spie = None
                    self.regs[vreg] = AssignExpr(pending['target'], pending['value'])
                else:
                    self._pending_spie = None
                    self._pre_stmts.append(ExprStmt(AssignExpr(pending['target'], pending['value'])))
            elif op_check == VM.CHGTHIS:
                vreg = pending['value_reg']
                if len(ops_check) >= 2 and ops_check[1] == vreg:
                    self._pending_spie = None
                    self.regs[vreg] = AssignExpr(pending['target'], pending['value'])
                else:
                    self._pending_spie = None
                    self._pre_stmts.append(ExprStmt(AssignExpr(pending['target'], pending['value'])))
            else:
                vreg = pending['value_reg']
                _REG_WRITING_OPS = frozenset({
                    VM.CONST, VM.CP, VM.CL, VM.CCL,
                    VM.GPD, VM.GPI, VM.GPDS, VM.GPIS,
                    VM.ADD, VM.SUB, VM.MUL, VM.DIV, VM.MOD,
                    VM.BAND, VM.BOR, VM.BXOR, VM.SAR, VM.SAL,
                    VM.INC, VM.DEC,
                    VM.ASC, VM.CHR, VM.NUM, VM.INT, VM.REAL, VM.STR, VM.OCTET,
                    VM.TYPEOF, VM.TYPEOFD, VM.TYPEOFI,
                    VM.LNOT, VM.BNOT,
                    VM.NEW,
                })
                _SIDE_EFFECT_OPS = frozenset({
                    VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS,
                    VM.SPI, VM.SPIE, VM.SPIS,
                    VM.INCPD, VM.DECPD, VM.INCPI, VM.DECPI,
                    VM.INCP, VM.DECP,
                    VM.DELD, VM.DELI,
                    VM.INC, VM.DEC,
                    VM.THROW, VM.SRV, VM.RET, VM.INV, VM.EVAL,
                    VM.ENTRY, VM.EXTRY,
                    VM.JMP, VM.JF, VM.JNF,
                })
                writes_to_vreg = (op_check in _REG_WRITING_OPS and
                                  len(ops_check) > 0 and ops_check[0] == vreg)
                is_side_effect = op_check in _SIDE_EFFECT_OPS
                if writes_to_vreg or is_side_effect:
                    self._pending_spie = None
                    self._pre_stmts.append(ExprStmt(AssignExpr(pending['target'], pending['value'])))

        prev_instr = self._prev_instruction
        self._prev_instruction = instr
        op = instr.op
        ops = instr.operands
        data = obj.data

        def get_data(idx: int) -> Any:
            return data[idx] if 0 <= idx < len(data) else None

        def get_reg(r: int) -> Expr:
            if r == 0:
                return VoidExpr()

            if r == -1:
                return ThisExpr()
            if r == -2:
                if self._in_with:
                    return WithThisExpr()
                return ThisExpr()
            if r < -2:
                name = self._get_local_name(r)
                return VarExpr(name)

            if r in self.pending_dicts:
                items = self.pending_dicts.pop(r)
                result = DictExpr(items)
                self.regs[r] = result
                return result

            if r in self.pending_arrays:
                elements = self.pending_arrays.pop(r)
                self.pending_counters.discard(r + 1)
                result = ArrayExpr(elements)
                self.regs[r] = result
                return result

            if r in self.regs:
                return self.regs[r]
            name = self._get_temp_name(r)
            return VarExpr(name)

        def set_reg(r: int, expr: Expr):
            self.regs[r] = expr

        def make_const(val: Any) -> Expr:
            if val is None:
                return VoidExpr()
            if isinstance(val, tuple):
                if val[0] == 'object':
                    return NullExpr()
                if val[0] == 'inter_object':
                    obj_idx = val[1]
                    if 0 <= obj_idx < len(self.loader.objects):
                        ref_obj = self.loader.objects[obj_idx]
                        if ref_obj.context_type == 2:
                            return self._decompile_anon_func(ref_obj)
                    return FuncRefExpr(obj_idx, self.loader)
            return ConstExpr(val)

        if op in (VM.NOP, VM.EXTRY, VM.REGMEMBER, VM.DEBUGGER):
            return None

        if op == VM.NF:
            self.flag_negated = not self.flag_negated
            return None

        if op == VM.CONST:
            r, idx = ops[0], ops[1]
            val = get_data(idx)
            set_reg(r, make_const(val))
            return None

        if op == VM.CL:
            r = ops[0]
            set_reg(r, VoidExpr())
            if r < -2:
                name = self._get_local_name(r)
                if name not in self.declared_vars:
                    self.declared_vars.add(name)
                    return VarDeclStmt(name)
            return None

        if op == VM.CCL:
            r, count = ops[0], ops[1]
            for i in range(count):
                set_reg(r + i, VoidExpr())
            return None

        if op == VM.GLOBAL:
            set_reg(ops[0], GlobalExpr())
            return None

        if op == VM.TT:
            self.flag = get_reg(ops[0])
            self.flag_negated = False
            return None

        if op == VM.TF:
            self.flag = get_reg(ops[0])
            self.flag_negated = True
            return None

        if op in (VM.CEQ, VM.CDEQ, VM.CLT, VM.CGT):
            left = get_reg(ops[0])
            right = get_reg(ops[1])
            op_sym = BINARY_OP_SYMBOLS.get(op, '==')
            self.flag = BinaryExpr(left, op_sym, right)
            self.flag_negated = False
            return None

        if op == VM.SETF:
            cond = self._get_condition(False)
            set_reg(ops[0], cond)
            return None

        if op == VM.SETNF:
            cond = self._get_condition(True)
            set_reg(ops[0], cond)
            return None

        if op == VM.CP:
            r1, r2 = ops[0], ops[1]
            src = get_reg(r2)

            if instr.addr in self._with_cp_addrs:
                self._in_with = True
                set_reg(r1, src)
                return _WithMarkerStmt(src)

            if r1 < -2:
                name = self._get_local_name(r1)
                set_reg(r1, VarExpr(name))

                if r2 >= 0 and (
                    isinstance(src, (DictExpr, ArrayExpr)) or
                    (isinstance(src, CallExpr) and src.is_new)
                ):
                    set_reg(r2, VarExpr(name))

                if name not in self.declared_vars:
                    self.declared_vars.add(name)
                    return VarDeclStmt(name, src)
                return ExprStmt(AssignExpr(VarExpr(name), src))

            set_reg(r1, src)
            return None

        if op == VM.LNOT:
            r = ops[0]
            operand = get_reg(r)
            set_reg(r, UnaryExpr('!', operand))
            return None

        if op == VM.BNOT:
            r = ops[0]
            operand = get_reg(r)
            set_reg(r, UnaryExpr('~', operand))
            return None

        if op == VM.CHS:
            r = ops[0]
            operand = get_reg(r)
            set_reg(r, UnaryExpr('-', operand))
            return None

        if op == VM.INT:
            r = ops[0]
            set_reg(r, CallExpr(VarExpr('int'), [get_reg(r)]))
            return None

        if op == VM.REAL:
            r = ops[0]
            set_reg(r, CallExpr(VarExpr('real'), [get_reg(r)]))
            return None

        if op == VM.STR:
            r = ops[0]
            set_reg(r, CallExpr(VarExpr('string'), [get_reg(r)]))
            return None

        if op == VM.OCTET:
            r = ops[0]
            set_reg(r, CallExpr(VarExpr('octet'), [get_reg(r)]))
            return None

        if op == VM.ASC:
            r = ops[0]
            set_reg(r, UnaryExpr('#', get_reg(r)))
            return None

        if op == VM.CHR:
            r = ops[0]
            set_reg(r, UnaryExpr('$', get_reg(r)))
            return None

        if op == VM.NUM:
            r = ops[0]
            set_reg(r, UnaryExpr('+', get_reg(r)))
            return None

        if op in (VM.INC, VM.DEC):
            r = ops[0]
            if r in self.pending_counters:
                return None
            target = get_reg(r)
            op_sym = '++' if op == VM.INC else '--'

            if r < -2 and prev_instr is not None:
                if (prev_instr.op == VM.CP and
                    prev_instr.operands[1] == r and
                    prev_instr.operands[0] >= 0):
                    temp_reg = prev_instr.operands[0]
                    set_reg(temp_reg, UnaryExpr(op_sym, target, prefix=False))
                    return None

            if r >= -2:
                set_reg(r, UnaryExpr(op_sym, target, prefix=True))
            return ExprStmt(UnaryExpr(op_sym, target, prefix=True))

        if op in (VM.INCPD, VM.DECPD):
            r1, r2, idx = ops[0], ops[1], ops[2]
            obj_expr = get_reg(r2)
            prop = get_data(idx)
            if isinstance(prop, str):
                target = PropertyExpr(obj_expr, prop)
            else:
                target = PropertyExpr(obj_expr, make_const(prop))
            op_sym = '++' if op == VM.INCPD else '--'

            if r1 == 0 and prev_instr is not None:
                if (prev_instr.op in (VM.GPD, VM.GPDS) and
                    prev_instr.operands[1] == r2 and
                    prev_instr.operands[2] == idx and
                    prev_instr.operands[0] >= 0):
                    temp_reg = prev_instr.operands[0]
                    set_reg(temp_reg, UnaryExpr(op_sym, target, prefix=False))
                    return None

            result = UnaryExpr(op_sym, target, prefix=True)
            if r1 != 0:
                set_reg(r1, result)
                return None
            return ExprStmt(result)

        if op in (VM.INCPI, VM.DECPI):
            r1, r2, r3 = ops[0], ops[1], ops[2]
            obj_expr = get_reg(r2)
            idx_expr = get_reg(r3)
            target = PropertyExpr(obj_expr, idx_expr)
            op_sym = '++' if op == VM.INCPI else '--'

            if r1 == 0 and prev_instr is not None:
                if (prev_instr.op in (VM.GPI, VM.GPIS) and
                    prev_instr.operands[1] == r2 and
                    prev_instr.operands[2] == r3 and
                    prev_instr.operands[0] >= 0):
                    temp_reg = prev_instr.operands[0]
                    set_reg(temp_reg, UnaryExpr(op_sym, target, prefix=False))
                    return None

            result = UnaryExpr(op_sym, target, prefix=True)
            if r1 != 0:
                set_reg(r1, result)
                return None
            return ExprStmt(result)

        if op in (VM.INCP, VM.DECP):
            r1, r2 = ops[0], ops[1]
            target = get_reg(r2)
            op_sym = '++' if op == VM.INCP else '--'

            if r1 == 0 and prev_instr is not None:
                if (prev_instr.op == VM.GETP and
                    prev_instr.operands[1] == r2 and
                    prev_instr.operands[0] >= 0):
                    temp_reg = prev_instr.operands[0]
                    set_reg(temp_reg, UnaryExpr(op_sym, target, prefix=False))
                    return None

            result = UnaryExpr(op_sym, target, prefix=True)
            if r1 != 0:
                set_reg(r1, result)
                return None
            return ExprStmt(result)

        binary_ops_base = {
            VM.LOR: '||', VM.LAND: '&&', VM.BOR: '|', VM.BXOR: '^', VM.BAND: '&',
            VM.SAR: '>>', VM.SAL: '<<', VM.SR: '>>>',
            VM.ADD: '+', VM.SUB: '-', VM.MUL: '*', VM.DIV: '/', VM.MOD: '%', VM.IDIV: '\\'
        }

        for base_op, op_sym in binary_ops_base.items():
            if op == base_op:
                r1, r2 = ops[0], ops[1]
                target = get_reg(r1)
                right = get_reg(r2)
                result = BinaryExpr(target, op_sym, right)
                if r1 < -2:
                    return ExprStmt(AssignExpr(target, right, f'{op_sym}='))
                set_reg(r1, result)
                return None

        for base_op, op_sym in binary_ops_base.items():
            if op == base_op + 1:
                r1, r2, idx, r3 = ops[0], ops[1], ops[2], ops[3]
                obj_expr = get_reg(r2)
                prop = get_data(idx)
                if isinstance(prop, str):
                    target = PropertyExpr(obj_expr, prop)
                else:
                    target = PropertyExpr(obj_expr, make_const(prop))
                value = get_reg(r3)
                stmt = ExprStmt(AssignExpr(target, value, f'{op_sym}='))
                if r1 != 0:
                    set_reg(r1, target)
                return stmt

        for base_op, op_sym in binary_ops_base.items():
            if op == base_op + 2:
                r1, r2, r3, r4 = ops[0], ops[1], ops[2], ops[3]
                obj_expr = get_reg(r2)
                idx_expr = get_reg(r3)
                target = PropertyExpr(obj_expr, idx_expr)
                value = get_reg(r4)
                stmt = ExprStmt(AssignExpr(target, value, f'{op_sym}='))
                if r1 != 0:
                    set_reg(r1, target)
                return stmt

        for base_op, op_sym in binary_ops_base.items():
            if op == base_op + 3:
                r1, r2, r3 = ops[0], ops[1], ops[2]
                target = get_reg(r2)
                value = get_reg(r3)
                stmt = ExprStmt(AssignExpr(target, value, f'{op_sym}='))
                if r1 != 0:
                    set_reg(r1, target)
                return stmt

        if op in (VM.GPD, VM.GPDS):
            r1, r2, idx = ops[0], ops[1], ops[2]
            prop = get_data(idx)
            obj_expr = get_reg(r2)
            if isinstance(prop, str):
                set_reg(r1, PropertyExpr(obj_expr, prop))
            else:
                set_reg(r1, PropertyExpr(obj_expr, make_const(prop)))
            return None

        if op in (VM.GPI, VM.GPIS):
            r1, r2, r3 = ops[0], ops[1], ops[2]
            obj_expr = get_reg(r2)
            idx_expr = get_reg(r3)
            set_reg(r1, PropertyExpr(obj_expr, idx_expr))
            return None

        if op in (VM.SPD, VM.SPDE, VM.SPDEH, VM.SPDS):
            r1, idx, r3 = ops[0], ops[1], ops[2]
            prop = get_data(idx)
            value = get_reg(r3)

            obj_expr = get_reg(r1)
            if isinstance(prop, str):
                target = PropertyExpr(obj_expr, prop)
            else:
                target = PropertyExpr(obj_expr, make_const(prop))

            if r3 > 0 and (
                isinstance(value, (DictExpr, ArrayExpr)) or
                (isinstance(value, CallExpr) and value.is_new)
            ):
                self._pending_spie = {
                    'target': target,
                    'value': value,
                    'value_reg': r3,
                }
                return None

            if r3 > 0:
                set_reg(r3, target)

            return ExprStmt(AssignExpr(target, value))

        if op in (VM.SPI, VM.SPIE, VM.SPIS):
            r1, r2, r3 = ops[0], ops[1], ops[2]

            if r1 in self.pending_dicts:
                key_expr = get_reg(r2)
                value_expr = get_reg(r3)
                self.pending_dicts[r1].append((key_expr, value_expr))
                return None

            if r1 in self.pending_arrays:
                self.pending_counters.add(r2)
                value_expr = get_reg(r3)
                self.pending_arrays[r1].append(value_expr)
                return None

            obj_expr = get_reg(r1)
            idx_expr = get_reg(r2)
            value = get_reg(r3)
            target = PropertyExpr(obj_expr, idx_expr)

            if r3 > 0 and (
                isinstance(value, (DictExpr, ArrayExpr)) or
                (isinstance(value, CallExpr) and value.is_new)
            ):
                self._pending_spie = {
                    'target': target,
                    'value': value,
                    'value_reg': r3,
                }
                return None

            return ExprStmt(AssignExpr(target, value))

        if op == VM.CALL:
            r1, r2 = ops[0], ops[1]
            argc = ops[2]
            func_expr = get_reg(r2)
            args = self._parse_call_args(ops, 3, argc)
            result = CallExpr(func_expr, args)
            set_reg(r1, result)
            if r1 == 0:
                return ExprStmt(result)
            return None

        if op == VM.CALLD:
            r1, r2, idx = ops[0], ops[1], ops[2]
            argc = ops[3]
            obj_expr = get_reg(r2)
            method = get_data(idx)
            args = self._parse_call_args(ops, 4, argc)

            if (r1 == 0 and method == '_compile' and argc == 1 and
                isinstance(obj_expr, CallExpr) and
                isinstance(obj_expr.func, PropertyExpr) and
                isinstance(obj_expr.func.prop, str) and
                obj_expr.func.prop == 'RegExp' and
                len(obj_expr.args) == 0):
                pattern_arg = args[0] if args else None
                if isinstance(pattern_arg, ConstExpr) and isinstance(pattern_arg.value, str):
                    pattern_str = pattern_arg.value
                    if pattern_str.startswith('//'):
                        set_reg(r2, ConstExpr(pattern_str))
                        return None

            result = MethodCallExpr(obj_expr, method if isinstance(method, str) else make_const(method), args)
            set_reg(r1, result)
            if r1 == 0:
                return ExprStmt(result)
            return None

        if op == VM.CALLI:
            r1, r2, r3 = ops[0], ops[1], ops[2]
            argc = ops[3]
            obj_expr = get_reg(r2)
            method_expr = get_reg(r3)
            args = self._parse_call_args(ops, 4, argc)
            result = MethodCallExpr(obj_expr, method_expr, args)
            set_reg(r1, result)
            if r1 == 0:
                return ExprStmt(result)
            return None

        if op == VM.NEW:
            r1, r2 = ops[0], ops[1]
            argc = ops[2]
            ctor = get_reg(r2)
            args = self._parse_call_args(ops, 3, argc)
            result = CallExpr(ctor, args, is_new=True)

            if argc == 0 and isinstance(ctor, PropertyExpr):
                ctor_name = ctor.prop if isinstance(ctor.prop, str) else None
                if ctor_name == 'Dictionary':
                    self.pending_dicts[r1] = []
                    set_reg(r1, result)
                    return None
                elif ctor_name == 'Array':
                    self.pending_arrays[r1] = []
                    set_reg(r1, result)
                    return None

            set_reg(r1, result)
            return None

        if op == VM.CHKINS:
            r1, r2 = ops[0], ops[1]
            left = get_reg(r1)
            right = get_reg(r2)
            result = InstanceofExpr(left, right)
            set_reg(r1, result)
            self.flag = result
            self.flag_negated = False
            return None

        if op == VM.CHGTHIS:
            r1, r2 = ops[0], ops[1]
            func = get_reg(r1)
            ctx = get_reg(r2)
            set_reg(r1, InContextOfExpr(func, ctx))
            return None

        if op == VM.TYPEOF:
            r = ops[0]
            set_reg(r, TypeofExpr(get_reg(r)))
            return None

        if op == VM.CHKINV:
            r = ops[0]
            set_reg(r, IsValidExpr(get_reg(r)))
            return None

        if op == VM.TYPEOFD:
            r1, r2, idx = ops[0], ops[1], ops[2]
            obj_expr = get_reg(r2)
            prop = get_data(idx)
            target = PropertyExpr(obj_expr, prop if isinstance(prop, str) else make_const(prop))
            set_reg(r1, TypeofExpr(target))
            return None

        if op == VM.TYPEOFI:
            r1, r2, r3 = ops[0], ops[1], ops[2]
            obj_expr = get_reg(r2)
            idx_expr = get_reg(r3)
            target = PropertyExpr(obj_expr, idx_expr)
            set_reg(r1, TypeofExpr(target))
            return None

        if op == VM.DELD:
            r1, r2, idx = ops[0], ops[1], ops[2]
            obj_expr = get_reg(r2)
            prop = get_data(idx)
            target = PropertyExpr(obj_expr, prop if isinstance(prop, str) else make_const(prop))
            return ExprStmt(DeleteExpr(target))

        if op == VM.DELI:
            r1, r2, r3 = ops[0], ops[1], ops[2]
            obj_expr = get_reg(r2)
            idx_expr = get_reg(r3)
            target = PropertyExpr(obj_expr, idx_expr)
            return ExprStmt(DeleteExpr(target))

        if op == VM.SRV:
            r = ops[0]
            if r == 0:
                return ReturnStmt(None)
            return ReturnStmt(get_reg(r))

        if op == VM.RET:
            return None

        if op == VM.THROW:
            return ThrowStmt(get_reg(ops[0]))

        if op == VM.ENTRY:
            return None

        if op == VM.INV:
            r = ops[0]
            return ExprStmt(CallExpr(VarExpr('invalidate'), [get_reg(r)]))

        if op == VM.ADDCI:
            return None

        if op == VM.EVAL:
            r = ops[0]
            set_reg(r, UnaryExpr('!', get_reg(r), prefix=False))
            return None

        if op == VM.EEXP:
            r = ops[0]
            return ExprStmt(UnaryExpr('!', get_reg(r), prefix=False))

        return None

    def _parse_call_args(self, ops: List[int], start_idx: int, argc: int) -> List[Expr]:
        args = []

        def get_arg_expr(reg: int) -> Expr:
            if reg == 0:
                return OmittedArgExpr()
            if reg == -1:
                return ThisExpr()
            if reg == -2:
                return ThisExpr()
            if reg < -2:
                return VarExpr(self._get_local_name(reg))
            if reg in self.pending_dicts:
                items = self.pending_dicts.pop(reg)
                result = DictExpr(items)
                self.regs[reg] = result
                return result
            if reg in self.pending_arrays:
                elements = self.pending_arrays.pop(reg)
                self.pending_counters.discard(reg + 1)
                result = ArrayExpr(elements)
                self.regs[reg] = result
                return result
            if reg in self.regs:
                return self.regs[reg]
            return VarExpr(self._get_temp_name(reg))

        if argc == -1:
            args.append(VarExpr('...'))
        elif argc == -2:
            real_argc = ops[start_idx] if start_idx < len(ops) else 0
            for i in range(real_argc):
                arg_type = ops[start_idx + 1 + i * 2] if start_idx + 1 + i * 2 < len(ops) else 0
                arg_reg = ops[start_idx + 2 + i * 2] if start_idx + 2 + i * 2 < len(ops) else 0
                arg_expr = get_arg_expr(arg_reg)
                if arg_type == 1:
                    args.append(UnaryExpr('*', arg_expr, prefix=False))
                elif arg_type == 2:
                    args.append(VarExpr('*'))
                else:
                    args.append(arg_expr)
        else:
            for i in range(argc):
                if start_idx + i < len(ops):
                    arg_reg = ops[start_idx + i]
                    args.append(get_arg_expr(arg_reg))

        return args

    def _get_local_name(self, reg: int) -> str:
        if reg in self.local_vars:
            return self.local_vars[reg]

        name = f'local{self.var_counter}'
        self.var_counter += 1
        self.local_vars[reg] = name
        return name

    def _get_temp_name(self, reg: int) -> str:
        if reg in self.local_vars:
            return self.local_vars[reg]

        name = f'tmp{reg}'
        self.local_vars[reg] = name
        return name


def disassemble_object(obj: CodeObject, loader: BytecodeLoader) -> str:
    lines = []
    ctx_names = ['TopLevel', 'Function', 'ExprFunction', 'Property',
                'PropertySetter', 'PropertyGetter', 'Class', 'SuperClassGetter']
    ctx = ctx_names[obj.context_type] if obj.context_type < len(ctx_names) else str(obj.context_type)
    lines.append(f"; Object [{obj.index}] {obj.name or '(anonymous)'} ({ctx})")
    lines.append(f";   Args: {obj.func_decl_arg_count}, MaxVar: {obj.max_variable_count}")

    instructions = decode_instructions(obj.code)
    for instr in instructions:
        op_name = VM(instr.op).name if 0 <= instr.op <= 127 else f'OP_{instr.op}'
        ops_str = ', '.join(str(o) for o in instr.operands)

        extra = ''
        if instr.op == VM.CONST and len(instr.operands) >= 2:
            idx = instr.operands[1]
            if 0 <= idx < len(obj.data):
                val = obj.data[idx]
                if isinstance(val, str):
                    extra = f'  ; "{val[:30]}..."' if len(val) > 30 else f'  ; "{val}"'
                else:
                    extra = f'  ; {val}'
        elif instr.op in (VM.JF, VM.JNF, VM.JMP, VM.ENTRY) and instr.operands:
            target = instr.addr + instr.operands[0]
            extra = f'  ; -> {target}'

        lines.append(f'{instr.addr:4d}: {op_name:12s} {ops_str}{extra}')

    return '\n'.join(lines)


def is_tjs2_bytecode(filepath):
    try:
        with open(filepath, 'rb') as f:
            return f.read(8) == b'TJS2100\x00'
    except (OSError, IOError):
        return False


def decompile_file(input_path, output_path=None, disasm=False, info=False, obj_idx=None, debug_cfg=False):
    with open(input_path, 'rb') as f:
        data = f.read()

    loader = BytecodeLoader(data)
    if not loader.load():
        print(f"Error: Invalid TJS2 bytecode file: {input_path}", file=sys.stderr)
        return False

    if disasm:
        if obj_idx is not None:
            if 0 <= obj_idx < len(loader.objects):
                print(disassemble_object(loader.objects[obj_idx], loader))
            else:
                print(f"Error: Object index {obj_idx} out of range", file=sys.stderr)
        else:
            for obj in loader.objects:
                print(disassemble_object(obj, loader))
                print()
        return True

    if info:
        print(f"TJS2 Bytecode Information: {input_path}")
        print(f"  Strings: {len(loader.string_array)}")
        print(f"  Objects: {len(loader.objects)}")
        print(f"  Top-level: {loader.toplevel}")
        print()
        for obj in loader.objects:
            ctx_names = ['TopLevel', 'Function', 'ExprFunction', 'Property',
                        'PropertySetter', 'PropertyGetter', 'Class', 'SuperClassGetter']
            ctx = ctx_names[obj.context_type] if obj.context_type < len(ctx_names) else str(obj.context_type)
            print(f"  [{obj.index}] {obj.name or '(anonymous)'}: {ctx}, {len(obj.code)} codes")
        return True

    from tjs2_cfg_decompiler import CFGDecompiler
    decompiler = CFGDecompiler(loader, debug_cfg=debug_cfg)
    source = decompiler.decompile()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(source)
    else:
        print(source)
    return True


def decompile_directory(input_dir, output_dir, recursive=False):
    import pathlib
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)

    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    pattern = '**/*' if recursive else '*'
    files = [f for f in input_path.glob(pattern) if f.is_file() and is_tjs2_bytecode(f)]

    if not files:
        print(f"No TJS2 bytecode files found in {input_dir}", file=sys.stderr)
        return

    ok = 0
    fail = 0
    for filepath in sorted(files):
        rel = filepath.relative_to(input_path)
        out_file = output_path / rel
        try:
            if decompile_file(str(filepath), str(out_file)):
                print(f"  OK: {rel}")
                ok += 1
            else:
                print(f"FAIL: {rel}")
                fail += 1
        except Exception as e:
            print(f"FAIL: {rel} ({e})")
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed, {ok + fail} total")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='TJS2 Bytecode Decompiler')
    parser.add_argument('input', help='Input bytecode file or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively decompile directory')
    parser.add_argument('-i', '--info', action='store_true', help='Show file info')
    parser.add_argument('-d', '--disasm', action='store_true', help='Disassemble only')
    parser.add_argument('--obj', type=int, help='Object index to disassemble')
    parser.add_argument('--debug-cfg', action='store_true', help='Print CFG debug info')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        if not args.output:
            print("Error: -o <output_dir> is required for directory mode", file=sys.stderr)
            sys.exit(1)
        decompile_directory(args.input, args.output, recursive=args.recursive)
        return

    if not decompile_file(args.input, args.output, disasm=args.disasm, info=args.info,
                          obj_idx=args.obj, debug_cfg=getattr(args, 'debug_cfg', False)):
        sys.exit(1)


if __name__ == '__main__':
    main()
