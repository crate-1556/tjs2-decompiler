import re

MAX_LINE_LENGTH = 120
INDENT_STR = '    '

def format_source(source: str) -> str:
    max_line = MAX_LINE_LENGTH

    source = _fix_anon_func_indent(source)

    lines = source.split('\n')
    result = []
    for line in lines:
        if len(line) <= max_line:
            result.append(line)
        else:
            formatted = _format_long_line(line, max_line)
            final = []
            for fl in formatted:
                if len(fl) > max_line and fl != line:
                    sub = _format_long_line(fl, max_line)
                    final.extend(sub)
                else:
                    final.append(fl)
            result.extend(final)
    source = '\n'.join(result)

    source = _fix_anon_func_indent(source)

    source = _format_dict_short_keys(source)

    source = _restore_default_params(source)

    source, inheritance_map = _restore_extends(source)

    if inheritance_map:
        source = _restore_super_calls(source, inheritance_map)

    source = _merge_else_if(source)

    return source

def _fix_anon_func_indent(source: str) -> str:
    while True:
        new_source = _fix_anon_func_indent_pass(source)
        if new_source == source:
            break
        source = new_source
    return source

def _scan_brace_depth(line, initial_depth):
    depth = initial_depth
    in_string = None
    j = 0
    while j < len(line):
        ch = line[j]
        if in_string:
            if ch == '\\':
                j += 2
                continue
            if ch == in_string:
                in_string = None
        elif ch in ('"', "'"):
            in_string = ch
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return True, 0
        j += 1
    return False, depth

def _fix_anon_func_indent_pass(source: str) -> str:
    lines = source.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        rstripped = line.rstrip()

        m = re.search(r'function\s*\([^)]*\)\s*\{$', rstripped)
        if m:
            base_indent = len(line) - len(line.lstrip())

            first_body = i + 1
            while first_body < len(lines) and not lines[first_body].strip():
                first_body += 1

            if first_body < len(lines):
                next_indent = len(lines[first_body]) - len(lines[first_body].lstrip())
                first_content = lines[first_body].strip()

                if first_content.startswith('}'):
                    expected_indent = base_indent
                else:
                    expected_indent = base_indent + 4

                if next_indent < expected_indent:
                    shift = expected_indent - next_indent
                    result.append(line)
                    i += 1

                    brace_depth = 1
                    while i < len(lines) and brace_depth > 0:
                        body_line = lines[i]
                        reached_zero, brace_depth = _scan_brace_depth(body_line, brace_depth)

                        if body_line.strip():
                            result.append(' ' * shift + body_line.rstrip())
                        else:
                            result.append('')
                        i += 1

                        if reached_zero:
                            break
                    continue

        result.append(line)
        i += 1

    return '\n'.join(result)

def _get_indent(line: str) -> tuple:
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]
    return indent, stripped

def _format_long_line(line: str, max_line: int = MAX_LINE_LENGTH) -> list:
    indent, content = _get_indent(line)
    inner_indent = indent + INDENT_STR

    for formatter in [
        _try_format_dict,
        _try_format_array,
        _try_format_condition,
        _try_format_string_concat,
        _try_format_call,
        _try_format_return_expr,
        _try_format_assignment_condition,
        _try_format_incontextof_call,
        _try_format_comma_continuation,
    ]:
        result = formatter(content, indent, inner_indent)
        if result:
            return result

    return [line]

def _find_matching_bracket(text: str, open_pos: int, open_char: str, close_char: str) -> int:
    depth = 1
    pos = open_pos + 1
    in_string = None
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if in_string:
            if ch == '\\':
                pos += 2
                continue
            if ch == in_string:
                in_string = None
        else:
            if ch in ('"', "'"):
                in_string = ch
            elif ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
        pos += 1
    return pos - 1 if depth == 0 else -1

def _split_top_level(text: str, separator: str = ',') -> list:
    parts = []
    current = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_string = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                current.append(ch)
                i += 1
                if i < len(text):
                    current.append(text[i])
                i += 1
                continue
            if ch == in_string:
                in_string = None
            current.append(ch)
        else:
            if ch in ('"', "'"):
                in_string = ch
                current.append(ch)
            elif ch == '(':
                depth_paren += 1
                current.append(ch)
            elif ch == ')':
                depth_paren -= 1
                current.append(ch)
            elif ch == '[':
                depth_bracket += 1
                current.append(ch)
            elif ch == ']':
                depth_bracket -= 1
                current.append(ch)
            elif ch == '{':
                depth_brace += 1
                current.append(ch)
            elif ch == '}':
                depth_brace -= 1
                current.append(ch)
            elif (ch == separator[0] and text[i:i+len(separator)] == separator
                  and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0):
                parts.append(''.join(current).strip())
                current = []
                i += len(separator)
                continue
            else:
                current.append(ch)
        i += 1
    remaining = ''.join(current).strip()
    if remaining:
        parts.append(remaining)
    return parts

def _try_format_dict(content: str, indent: str, inner_indent: str) -> list:
    m = re.search(r'%\[', content)
    if not m:
        return None

    start = m.start()

    if start > 0 and content[start - 1] == '[':
        return None

    bracket_start = start + 1
    bracket_end = _find_matching_bracket(content, bracket_start, '[', ']')

    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    prefix_text = content[:start]

    if bracket_end >= 0:
        dict_content = content[bracket_start + 1:bracket_end]
        suffix = content[bracket_end + 1:]

        entries = _split_top_level(dict_content, ',')
        if len(entries) <= 1:
            return None

        lines = [f'{indent}{prefix_text}%[']
        for i, entry in enumerate(entries):
            comma = ',' if i < len(entries) - 1 else ''
            lines.append(f'{inner_indent}{entry.strip()}{comma}')
        lines.append(f'{indent}]{suffix}')
        return lines
    else:
        dict_content = content[bracket_start + 1:]
        entries = _split_top_level(dict_content, ',')
        if len(entries) <= 1:
            return None

        lines = [f'{indent}{prefix_text}%[']
        for i, entry in enumerate(entries):
            comma = ',' if i < len(entries) - 1 else ''
            lines.append(f'{inner_indent}{entry.strip()}{comma}')
        return lines

def _try_format_array(content: str, indent: str, inner_indent: str) -> list:
    for m in re.finditer(r'\[', content):
        bracket_start = m.start()
        if bracket_start > 0 and content[bracket_start - 1] == '%':
            continue
        if bracket_start > 0:
            before = content[:bracket_start].rstrip()
            if before and before[-1] not in ('=', ',', '(', '[', ' '):
                if re.match(r'\w', before[-1]):
                    continue

        bracket_end = _find_matching_bracket(content, bracket_start, '[', ']')
        if bracket_end < 0:
            continue

        array_content = content[bracket_start + 1:bracket_end]
        elements = _split_top_level(array_content, ',')
        if len(elements) <= 1:
            continue

        prefix_text = content[:bracket_start]
        suffix = content[bracket_end + 1:]

        full_line = indent + content
        if len(full_line) <= MAX_LINE_LENGTH:
            return None

        has_dicts = any(e.strip().startswith('%[') for e in elements)

        lines = [f'{indent}{prefix_text}[']
        if has_dicts:
            for i, elem in enumerate(elements):
                comma = ',' if i < len(elements) - 1 else ''
                lines.append(f'{inner_indent}{elem.strip()}{comma}')
        else:
            current_group = []
            current_len = len(inner_indent)

            for i, elem in enumerate(elements):
                elem_str = elem.strip()
                add_len = len(elem_str) + (2 if current_group else 0)

                if current_group and current_len + add_len > MAX_LINE_LENGTH:
                    lines.append(f'{inner_indent}{", ".join(current_group)},')
                    current_group = [elem_str]
                    current_len = len(inner_indent) + len(elem_str)
                else:
                    current_group.append(elem_str)
                    current_len += add_len

            if current_group:
                lines.append(f'{inner_indent}{", ".join(current_group)}')

        lines.append(f'{indent}]{suffix}')
        return lines

    return None

def _try_format_call(content: str, indent: str, inner_indent: str) -> list:
    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    control_keywords = {'if', 'while', 'for', 'switch', 'catch', 'with'}

    best = None
    for m in re.finditer(r'(\w+)\s*\(', content):
        name = m.group(1)
        if name in control_keywords:
            continue
        paren_start = content.index('(', m.start())
        paren_end = _find_matching_bracket(content, paren_start, '(', ')')
        if paren_end < 0:
            continue
        args_text = content[paren_start + 1:paren_end]
        args = _split_top_level(args_text, ',')
        if len(args) <= 1:
            continue
        call_end_col = len(indent) + paren_end + 1
        if call_end_col > MAX_LINE_LENGTH:
            best = (paren_start, paren_end, args)
            break

    if not best:
        for m in re.finditer(r'(\w+)\s*\(', content):
            name = m.group(1)
            if name in control_keywords:
                continue
            paren_start = content.index('(', m.start())
            paren_end = _find_matching_bracket(content, paren_start, '(', ')')
            if paren_end < 0:
                continue
            args_text = content[paren_start + 1:paren_end]
            args = _split_top_level(args_text, ',')
            if len(args) >= 3:
                best = (paren_start, paren_end, args)
                break

    if not best:
        return None

    paren_start, paren_end, args = best
    prefix_text = content[:paren_start]
    suffix = content[paren_end + 1:]

    lines = [f'{indent}{prefix_text}(']
    current_group = []
    current_len = len(inner_indent)

    for arg in args:
        arg_str = arg.strip()
        add_len = len(arg_str) + (2 if current_group else 0)

        if current_group and current_len + add_len > MAX_LINE_LENGTH:
            lines.append(f'{inner_indent}{", ".join(current_group)},')
            current_group = [arg_str]
            current_len = len(inner_indent) + len(arg_str)
        else:
            current_group.append(arg_str)
            current_len += add_len

    if current_group:
        lines.append(f'{inner_indent}{", ".join(current_group)}')

    lines.append(f'{indent}){suffix}')
    return lines

def _try_format_condition(content: str, indent: str, inner_indent: str) -> list:
    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    m = re.match(r'(&&\s+|[|][|]\s+)', content)
    if m:
        return _try_format_condition_continuation(content, indent, inner_indent, m)

    m = re.match(r'((?:}\s*else\s+)?(?:if|while|for))\s*\(', content)
    if not m:
        m = re.match(r'(return\s+)', content)
        if m:
            return _try_format_return_condition(content, indent, inner_indent, m)
        return None

    keyword = m.group(1)
    paren_start = content.index('(', m.start())
    paren_end = _find_matching_bracket(content, paren_start, '(', ')')
    if paren_end < 0:
        return None

    condition = content[paren_start + 1:paren_end]
    suffix = content[paren_end + 1:]

    parts = _split_condition(condition)
    if len(parts) <= 1:
        return None

    cont_indent = inner_indent + INDENT_STR
    lines = [f'{indent}{keyword} ({parts[0]}']
    for part in parts[1:]:
        lines.append(f'{cont_indent}{part}')
    lines[-1] = lines[-1] + ')' + suffix
    return lines

def _try_format_return_condition(content, indent, inner_indent, m):
    prefix = m.group(1)
    rest = content[m.end():]
    if rest.endswith(';'):
        rest = rest[:-1]
        suffix = ';'
    else:
        suffix = ''

    parts = _split_condition(rest)
    if len(parts) <= 1:
        return None

    cont_indent = inner_indent + INDENT_STR
    lines = [f'{indent}{prefix}{parts[0]}']
    for part in parts[1:]:
        lines.append(f'{cont_indent}{part}')
    lines[-1] = lines[-1] + suffix
    return lines

def _try_format_condition_continuation(content, indent, inner_indent, m):
    op_prefix = m.group(1)
    rest = content[m.end():]

    paren_wrap = ''
    suffix = ''
    if rest.startswith('('):
        close = _find_matching_bracket(rest, 0, '(', ')')
        if close >= 0:
            paren_wrap = '('
            inner_rest = rest[1:close]
            suffix = rest[close:]
        else:
            inner_rest = rest
    else:
        inner_rest = rest
        if inner_rest.endswith(';'):
            suffix = ';'
            inner_rest = inner_rest[:-1]

    parts = _split_condition(inner_rest)
    if len(parts) <= 1:
        return None

    cont_indent = indent + INDENT_STR
    lines = [f'{indent}{op_prefix}{paren_wrap}{parts[0]}']
    for part in parts[1:]:
        lines.append(f'{cont_indent}{part}')
    lines[-1] += suffix
    return lines

def _split_condition(condition: str) -> list:
    parts = []
    current = []
    depth = 0
    in_string = None
    i = 0
    text = condition
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                current.append(ch)
                i += 1
                if i < len(text):
                    current.append(text[i])
                i += 1
                continue
            if ch == in_string:
                in_string = None
            current.append(ch)
        elif ch in ('"', "'"):
            in_string = ch
            current.append(ch)
        elif ch in ('(', '[', '{'):
            depth += 1
            current.append(ch)
        elif ch in (')', ']', '}'):
            depth -= 1
            current.append(ch)
        elif depth == 0 and i + 1 < len(text) and text[i:i+2] in ('&&', '||'):
            parts.append(''.join(current).strip())
            op = text[i:i+2]
            current = [op + ' ']
            i += 2
            continue
        else:
            current.append(ch)
        i += 1

    remaining = ''.join(current).strip()
    if remaining:
        parts.append(remaining)
    return parts

def _try_format_string_concat(content: str, indent: str, inner_indent: str) -> list:
    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    has_string = '"' in content or "'" in content
    if not has_string:
        return None

    patterns = [
        r'(var\s+\w+\s*=\s*)',
        r'(\w+(?:\.\w+)*\s*\+=\s*)',
        r'(\w+(?:\.\w+)*\s*=\s*)',
        r'(throw\s+new\s+\w+\()',
        r'(dm\()',
        r'(System\.inform\()',
        r'(Debug\.notice\()',
        r'(\w+\.sprintf\()',
        r'(return\s+)',
    ]

    for pat in patterns:
        m = re.match(pat, content)
        if not m:
            continue

        prefix_part = m.group(1)
        rest = content[m.end():]

        parts = _split_at_plus(rest)
        if len(parts) <= 1:
            continue

        lines = [f'{indent}{prefix_part}{parts[0]}']
        for part in parts[1:]:
            lines.append(f'{inner_indent}+ {part}')
        return lines

    return None

def _try_format_return_expr(content: str, indent: str, inner_indent: str) -> list:
    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    m = re.match(r'return\s+', content)
    if not m:
        return None

    rest = content[m.end():]
    if rest.endswith(';'):
        rest = rest[:-1]
        suffix = ';'
    else:
        suffix = ''

    if '"' in rest or "'" in rest:
        parts = _split_at_plus(rest)
        if len(parts) > 1:
            lines = [f'{indent}return {parts[0]}']
            for part in parts[1:]:
                lines.append(f'{inner_indent}+ {part}')
            lines[-1] += suffix
            return lines

    return None

def _try_format_assignment_condition(content: str, indent: str, inner_indent: str) -> list:
    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    m = re.match(r'((?:var\s+)?\w+(?:\.\w+)*\s*=\s*)', content)
    if not m:
        return None

    prefix = m.group(1)
    rest = content[m.end():]
    if '&&' not in rest and '||' not in rest:
        return None

    if rest.endswith(';'):
        rest = rest[:-1]
        suffix = ';'
    else:
        suffix = ''

    parts = _split_condition(rest)
    if len(parts) <= 1:
        return None

    cont_indent = inner_indent + INDENT_STR
    lines = [f'{indent}{prefix}{parts[0]}']
    for part in parts[1:]:
        lines.append(f'{cont_indent}{part}')
    lines[-1] += suffix
    return lines

def _try_format_incontextof_call(content: str, indent: str, inner_indent: str) -> list:
    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    m = re.search(r'\)\s*\(', content)
    if not m:
        return None

    call_paren_start = m.end() - 1
    paren_end = _find_matching_bracket(content, call_paren_start, '(', ')')
    if paren_end < 0:
        return None

    args_text = content[call_paren_start + 1:paren_end]
    args = _split_top_level(args_text, ',')
    if len(args) <= 1:
        return None

    prefix_text = content[:call_paren_start]
    suffix = content[paren_end + 1:]

    lines = [f'{indent}{prefix_text}(']
    current_group = []
    current_len = len(inner_indent)

    for arg in args:
        arg_str = arg.strip()
        add_len = len(arg_str) + (2 if current_group else 0)
        if current_group and current_len + add_len > MAX_LINE_LENGTH:
            lines.append(f'{inner_indent}{", ".join(current_group)},')
            current_group = [arg_str]
            current_len = len(inner_indent) + len(arg_str)
        else:
            current_group.append(arg_str)
            current_len += add_len

    if current_group:
        lines.append(f'{inner_indent}{", ".join(current_group)}')

    lines.append(f'{indent}){suffix}')
    return lines

def _try_format_comma_continuation(content: str, indent: str, inner_indent: str) -> list:
    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    split_points = []
    in_string = None
    depth = 0
    i = 0
    while i < len(content):
        ch = content[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == in_string:
                in_string = None
        elif ch in ('"', "'"):
            in_string = ch
        elif ch in ('(', '['):
            depth += 1
        elif ch in (')', ']'):
            depth -= 1
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        elif ch == ',' and depth <= 0:
            split_points.append(i)
        i += 1

    if len(split_points) < 1:
        return None

    parts = []
    prev = 0
    for sp in split_points:
        parts.append(content[prev:sp].strip())
        prev = sp + 1
    parts.append(content[prev:].strip())

    if len(parts) <= 1:
        return None

    lines = []
    for i, part in enumerate(parts):
        comma = ',' if i < len(parts) - 1 else ''
        if i == 0:
            lines.append(f'{indent}{part}{comma}')
        else:
            lines.append(f'{inner_indent}{part}{comma}')
    return lines

def _split_at_plus(text: str) -> list:
    parts = []
    current = []
    depth = 0
    in_string = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                current.append(ch)
                i += 1
                if i < len(text):
                    current.append(text[i])
                i += 1
                continue
            if ch == in_string:
                in_string = None
            current.append(ch)
        elif ch in ('"', "'"):
            in_string = ch
            current.append(ch)
        elif ch in ('(', '[', '{'):
            depth += 1
            current.append(ch)
        elif ch in (')', ']', '}'):
            depth -= 1
            current.append(ch)
        elif depth == 0 and ch == '+' and i + 1 < len(text) and text[i+1] != '+':
            if i + 1 < len(text) and text[i+1] == '=':
                current.append(ch)
            else:
                part = ''.join(current).rstrip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                while i < len(text) and text[i] == ' ':
                    i += 1
                continue
        else:
            current.append(ch)
        i += 1

    remaining = ''.join(current).strip()
    if remaining:
        parts.append(remaining)
    return parts

_TJS2_RESERVED = frozenset({
    'if', 'else', 'while', 'for', 'do', 'class', 'function', 'var', 'const',
    'return', 'break', 'continue', 'switch', 'case', 'default', 'try', 'catch',
    'throw', 'new', 'delete', 'typeof', 'instanceof', 'void', 'this', 'super',
    'global', 'true', 'false', 'null', 'in', 'incontextof', 'invalidate',
    'isvalid', 'int', 'real', 'string', 'enum', 'goto', 'with', 'static',
    'setter', 'getter', 'property',
})

_VALID_IDENT_RE = re.compile(r'^[a-zA-Z_]\w*$')

def _format_dict_short_keys(source: str) -> str:
    def _replace_dict_key(m):
        key = m.group(2)
        if _VALID_IDENT_RE.match(key) and key not in _TJS2_RESERVED:
            pos = m.start() - 1
            while pos >= 0 and source[pos] in ' \t\n\r':
                pos -= 1
            if pos >= 0 and source[pos] in ('[', ','):
                return f'{key}: '
        return m.group(0)

    return re.sub(r'(["\'])([a-zA-Z_]\w*)\1(\s*=>\s*)', _replace_dict_key, source)

def _restore_default_params(source: str) -> str:
    lines = source.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        m = re.match(r'^(\s*)(function\s+\w*\s*)\(([^)]*)\)\s*\{', line)
        if not m:
            result.append(line)
            i += 1
            continue

        indent_str = m.group(1)
        func_prefix = m.group(2)
        params_str = m.group(3)
        params = [p.strip() for p in params_str.split(',')] if params_str.strip() else []

        defaults = {}
        j = i + 1
        while j < len(lines):
            block_line = lines[j].strip()
            sm = re.match(
                r'if\s*\((arg\d+)\s*===\s*void\)\s*\{\s*$', block_line
            )
            if sm:
                param_name = sm.group(1)
                if j + 1 < len(lines):
                    assign_line = lines[j + 1].strip()
                    am = re.match(
                        r'(arg\d+)\s*=\s*(.+?)\s*;\s*$', assign_line
                    )
                    if am and am.group(1) == param_name:
                        if j + 2 < len(lines) and lines[j + 2].strip() == '}':
                            val = am.group(2)
                            if _is_safe_default_value(val):
                                if param_name in defaults:
                                    break
                                defaults[param_name] = val
                                j += 3
                                continue
                break
            elif block_line == '':
                j += 1
                continue
            else:
                break

        if defaults:
            new_params = []
            for p in params:
                if p in defaults:
                    new_params.append(f'{p} = {defaults[p]}')
                else:
                    new_params.append(p)
            result.append(f'{indent_str}{func_prefix}({", ".join(new_params)}) {{')
            i = j
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)

def _is_safe_default_value(val: str) -> bool:
    if re.match(r'^-?\d+(\.\d+)?$', val):
        return True
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return True
    if val in ('void', 'null'):
        return True
    if val == '%[]':
        return True
    if val == '[]':
        return True
    if _VALID_IDENT_RE.match(val):
        return True
    return False

def _restore_extends(source: str) -> tuple:
    lines = source.split('\n')
    result = []
    inheritance_map = {}
    i = 0
    while i < len(lines):
        line = lines[i]

        m = re.match(r'^(\s*)class\s+(\w+)\s*\{', line)
        if not m:
            result.append(line)
            i += 1
            continue

        indent_str = m.group(1)
        class_name = m.group(2)

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j < len(lines):
            next_line = lines[j].strip()
            em = re.match(r'^\((?:global\.)?(\w+)\s+incontextof\s+this\)\(\)\s*;?\s*$', next_line)
            if em:
                parent_name = em.group(1)
                inheritance_map[class_name] = parent_name
                result.append(f'{indent_str}class {class_name} extends {parent_name} {{')
                for k in range(i + 1, j):
                    result.append(lines[k])
                i = j + 1
                continue

        result.append(line)
        i += 1

    return '\n'.join(result), inheritance_map

def _is_inside_string(line: str, pos: int) -> bool:
    in_single = False
    in_double = False
    i = 0
    while i < pos:
        ch = line[i]
        if ch == '\\' and (in_single or in_double):
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        i += 1
    return in_single or in_double

def _restore_super_in_line(line: str, parent_class: str) -> str:
    pattern = 'global.' + parent_class + '.'
    parts = []
    last_end = 0
    start = 0
    while True:
        pos = line.find(pattern, start)
        if pos == -1:
            break
        if _is_inside_string(line, pos):
            start = pos + len(pattern)
            continue
        parts.append(line[last_end:pos])
        parts.append('super.')
        last_end = pos + len(pattern)
        start = last_end
    parts.append(line[last_end:])
    return ''.join(parts)

def _restore_super_calls(source: str, inheritance_map: dict) -> str:
    if not inheritance_map:
        return source

    lines = source.split('\n')
    result = []
    current_class = None
    current_parent = None
    brace_depth = 0
    class_stack = []

    for line in lines:
        stripped = line.strip()

        cm = re.match(r'^(\s*)class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{', line)
        if cm:
            if current_class is not None:
                class_stack.append((current_class, current_parent, brace_depth))
            current_class = cm.group(2)
            current_parent = inheritance_map.get(current_class)
            brace_depth = 1
            result.append(line)
            continue

        if current_class is not None:
            in_string = None
            j = 0
            while j < len(stripped):
                ch = stripped[j]
                if in_string:
                    if ch == '\\':
                        j += 2
                        continue
                    if ch == in_string:
                        in_string = None
                elif ch in ('"', "'"):
                    in_string = ch
                elif ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        if class_stack:
                            current_class, current_parent, brace_depth = class_stack.pop()
                        else:
                            current_class = None
                            current_parent = None
                        break
                j += 1

        if current_parent and ('global.' + current_parent + '.') in line:
            line = _restore_super_in_line(line, current_parent)

        result.append(line)

    return '\n'.join(result)

def _count_braces_in_line(line: str) -> int:
    delta = 0
    in_str = False
    str_char = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            if ch == '\\':
                i += 2
                continue
            if ch == str_char:
                in_str = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            str_char = ch
        elif ch == '{':
            delta += 1
        elif ch == '}':
            delta -= 1
        elif ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
            break
        i += 1
    return delta

def _merge_else_if(source: str) -> str:
    while True:
        merged = _merge_else_if_pass(source)
        if merged == source:
            break
        source = merged
    return source

def _merge_else_if_pass(source: str) -> str:
    lines = source.split('\n')
    result = []
    i = 0
    INDENT = '    '

    while i < len(lines):
        stripped = lines[i].rstrip()

        m = re.match(r'^(\s*)\} else \{\s*$', stripped)
        if not m:
            result.append(lines[i])
            i += 1
            continue

        base_indent = m.group(1)
        inner_indent = base_indent + INDENT

        if i + 1 >= len(lines):
            result.append(lines[i])
            i += 1
            continue

        next_stripped = lines[i + 1].rstrip()
        if not (next_stripped.startswith(inner_indent + 'if (')
                and (len(next_stripped) == len(inner_indent)
                     or next_stripped[len(inner_indent)] != ' ')):
            result.append(lines[i])
            i += 1
            continue

        depth = 1
        close_idx = -1
        for j in range(i + 1, len(lines)):
            depth += _count_braces_in_line(lines[j])
            if depth == 0:
                close_idx = j
                break

        if close_idx < 0:
            result.append(lines[i])
            i += 1
            continue

        if lines[close_idx].rstrip() != base_indent + '}':
            result.append(lines[i])
            i += 1
            continue

        inner_depth = 1
        if_end_line = -1
        for j in range(i + 1, close_idx):
            inner_depth += _count_braces_in_line(lines[j])
            if inner_depth == 1:
                if_end_line = j
                break

        if if_end_line != close_idx - 1:
            result.append(lines[i])
            i += 1
            continue

        if_content = lines[i + 1].rstrip()[len(inner_indent):]
        result.append(base_indent + '} else ' + if_content)

        for k in range(i + 2, close_idx):
            old = lines[k]
            if old.startswith(inner_indent):
                result.append(base_indent + old[len(inner_indent):])
            elif old.strip() == '':
                result.append(old)
            else:
                result.append(old)

        i = close_idx + 1

    return '\n'.join(result)
