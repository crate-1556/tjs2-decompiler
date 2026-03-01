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
            pending = formatted
            for _pass in range(3):
                still_long = False
                next_pending = []
                for fl in pending:
                    if len(fl) > max_line:
                        sub = _format_long_line(fl, max_line)
                        if len(sub) > 1 or (len(sub) == 1 and sub[0] != fl):
                            next_pending.extend(sub)
                            still_long = True
                        else:
                            next_pending.append(fl)
                    else:
                        next_pending.append(fl)
                pending = next_pending
                if not still_long:
                    break
            result.extend(pending)
    source = '\n'.join(result)

    source = _fix_anon_func_indent(source)

    source = _format_dict_short_keys(source)

    source = _restore_default_params(source)

    source, inheritance_map = _restore_extends(source)

    if inheritance_map:
        source = _restore_super_calls(source, inheritance_map)

    source = _merge_else_if(source)

    source = _collapse_empty_blocks(source)

    source = _normalize_blank_lines(source)

    source = _rename_catch_var(source)

    source = _wrap_toplevel_local_vars(source)

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
        _try_format_ternary,
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

    call_pat = re.compile(r'(?:new\s+)?(?:\w+\.)*\w+\s*\(')
    best = None
    for m in call_pat.finditer(content):
        match_text = m.group(0).rstrip('(').rstrip()
        name_parts = match_text.split()
        name = name_parts[-1].split('.')[-1] if name_parts else ''
        if name in control_keywords:
            continue
        paren_start = m.end() - 1
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
        for m in call_pat.finditer(content):
            match_text = m.group(0).rstrip('(').rstrip()
            name_parts = match_text.split()
            name = name_parts[-1].split('.')[-1] if name_parts else ''
            if name in control_keywords:
                continue
            paren_start = m.end() - 1
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
    if any(len(l) > MAX_LINE_LENGTH for l in lines):
        greedy = _greedy_condition_wrap(
            f'{keyword} (', condition, ')' + suffix, indent)
        if greedy is not None:
            return greedy
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
    if any(len(l) > MAX_LINE_LENGTH for l in lines):
        greedy = _greedy_condition_wrap(prefix, rest, suffix, indent)
        if greedy is not None:
            return greedy
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
        if inner_rest:
            return _greedy_condition_wrap(
                f'{op_prefix}{paren_wrap}', inner_rest, suffix, indent)
        return None

    cont_indent = indent + INDENT_STR
    first_line = f'{indent}{op_prefix}{paren_wrap}{parts[0]}'
    lines = [first_line]
    for part in parts[1:]:
        lines.append(f'{cont_indent}{part}')
    lines[-1] += suffix
    if any(len(l) > MAX_LINE_LENGTH for l in lines):
        greedy = _greedy_condition_wrap(
            f'{op_prefix}{paren_wrap}', inner_rest, suffix, indent)
        if greedy is not None:
            return greedy
    return lines

def _greedy_condition_wrap(prefix: str, inner_rest: str, suffix: str,
                           indent: str) -> list:
    segments = _find_all_logical_segments(inner_rest)
    if len(segments) <= 1:
        return None

    cont_indent = indent + INDENT_STR
    first_seg = segments[0]
    current_line = f'{indent}{prefix}{first_seg}'
    lines = []

    for seg in segments[1:]:
        candidate = current_line + ' ' + seg
        if len(candidate) <= MAX_LINE_LENGTH:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = f'{cont_indent}{seg}'

    current_line += suffix
    lines.append(current_line)
    return lines

def _find_all_logical_segments(text: str) -> list:
    segments = []
    current = []
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
        elif i + 1 < len(text) and text[i:i+2] in ('&&', '||'):
            segments.append(''.join(current).strip())
            op = text[i:i+2]
            current = [op + ' ']
            i += 2
            continue
        else:
            current.append(ch)
        i += 1

    remaining = ''.join(current).strip()
    if remaining:
        segments.append(remaining)
    return segments

def _try_format_ternary(content: str, indent: str, inner_indent: str) -> list:
    full_line = indent + content
    if len(full_line) <= MAX_LINE_LENGTH:
        return None

    q_pos = _find_top_level_ternary_q(content)
    if q_pos < 0:
        return None

    c_pos = _find_ternary_colon(content, q_pos + 1)
    if c_pos < 0:
        return None

    cond_part = content[:q_pos].rstrip()
    true_part = content[q_pos + 1:c_pos].strip()
    false_part_with_suffix = content[c_pos + 1:].strip()

    ternary_rest = f'? {true_part} : {false_part_with_suffix}'
    first_line = f'{indent}{cond_part}'
    if len(first_line) <= MAX_LINE_LENGTH and len(inner_indent + ternary_rest) <= MAX_LINE_LENGTH:
        return [first_line, f'{inner_indent}{ternary_rest}']

    lines = [f'{indent}{cond_part}']
    q_line = f'{inner_indent}? {true_part}'
    c_line = f'{inner_indent}: {false_part_with_suffix}'
    merged = f'{indent}{cond_part} ? {true_part}'
    if len(merged) <= MAX_LINE_LENGTH:
        lines = [merged]
        lines.append(c_line)
    else:
        lines.append(q_line)
        lines.append(c_line)
    return lines

def _find_top_level_ternary_q(text: str) -> int:
    depth = 0
    in_string = None
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == in_string:
                in_string = None
        elif ch in ('"', "'"):
            in_string = ch
        elif ch in ('(', '[', '{'):
            depth += 1
        elif ch in (')', ']', '}'):
            depth -= 1
        elif depth == 0 and ch == '?':
            if i + 1 < len(text) and text[i + 1] == '.':
                i += 2
                continue
            return i
        i += 1
    return -1

def _find_ternary_colon(text: str, start: int) -> int:
    depth = 0
    in_string = None
    ternary_depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == in_string:
                in_string = None
        elif ch in ('"', "'"):
            in_string = ch
        elif ch in ('(', '[', '{'):
            depth += 1
        elif ch in (')', ']', '}'):
            depth -= 1
        elif depth == 0 and ch == '?':
            if not (i + 1 < len(text) and text[i + 1] == '.'):
                ternary_depth += 1
        elif depth == 0 and ch == ':':
            if ternary_depth > 0:
                ternary_depth -= 1
            else:
                return i
        i += 1
    return -1

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
        r'((?:new\s+)?(?:\w+\.)*\w+\()',
        r'(return\s+)',
        r'(filter:\s*\[)',
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

        result_lines = [f'{indent}{prefix_part}{parts[0]}']
        for part in parts[1:]:
            candidate = result_lines[-1] + ' + ' + part
            if len(candidate) <= MAX_LINE_LENGTH:
                result_lines[-1] = candidate
            else:
                result_lines.append(f'{inner_indent}+ {part}')
        new_long = sum(1 for l in result_lines if len(l) > MAX_LINE_LENGTH)
        if new_long > 1:
            continue
        return result_lines

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
    new_long = sum(1 for l in lines if len(l) > MAX_LINE_LENGTH)
    if new_long > 1:
        return None
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

        m = re.match(r'^(\s*)class\s+(\w+)\s*\{(?:\s*//\s*@scg:(\d+))?\s*$', line)
        if not m:
            result.append(line)
            i += 1
            continue

        indent_str = m.group(1)
        class_name = m.group(2)
        scg_count = int(m.group(3)) if m.group(3) else 1

        parents = []
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        first_incontextof_j = j

        while j < len(lines) and len(parents) < scg_count:
            next_line = lines[j].strip()
            em = re.match(r'^\((?:global\.)?(\w+)\s+incontextof\s+this\)\(\)\s*;?\s*$', next_line)
            if em:
                parents.append(em.group(1))
                j += 1
            else:
                break

        if parents:
            inheritance_map[class_name] = parents
            extends_str = ', '.join(parents)
            result.append(f'{indent_str}class {class_name} extends {extends_str} {{')
            for k in range(i + 1, first_incontextof_j):
                result.append(lines[k])
            i = j
            continue

        if m.group(3):
            result.append(f'{indent_str}class {class_name} {{')
        else:
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
    current_parents = []
    brace_depth = 0
    class_stack = []

    for line in lines:
        stripped = line.strip()

        cm = re.match(r'^(\s*)class\s+(\w+)(?:\s+extends\s+[\w,\s]+)?\s*\{', line)
        if cm:
            if current_class is not None:
                class_stack.append((current_class, current_parents, brace_depth))
            current_class = cm.group(2)
            current_parents = inheritance_map.get(current_class, [])
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
                            current_class, current_parents, brace_depth = class_stack.pop()
                        else:
                            current_class = None
                            current_parents = []
                        break
                j += 1

        if brace_depth >= 2 and len(current_parents) == 1:
            first_parent = current_parents[0]
            if ('global.' + first_parent + '.') in line:
                line = _restore_super_in_line(line, first_parent)

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

        scan = i + 1
        if scan < len(lines):
            ln = lines[scan].rstrip()
            if (ln.startswith(inner_indent)
                    and re.match(r'^var\s+\w+\s*;\s*$', ln[len(inner_indent):])):
                result.append(lines[i])
                i += 1
                continue

        if_line_idx = scan

        next_stripped = lines[if_line_idx].rstrip() if if_line_idx < len(lines) else ''
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
        saw_open = False
        for j in range(if_line_idx, close_idx):
            inner_depth += _count_braces_in_line(lines[j])
            if inner_depth > 1:
                saw_open = True
            if saw_open and inner_depth == 1:
                if_end_line = j
                break

        if if_end_line != close_idx - 1:
            result.append(lines[i])
            i += 1
            continue

        if_content = lines[if_line_idx].rstrip()[len(inner_indent):]
        result.append(base_indent + '} else ' + if_content)

        for k in range(if_line_idx + 1, close_idx):
            old = lines[k]
            if old.startswith(inner_indent):
                result.append(base_indent + old[len(inner_indent):])
            elif old.strip() == '':
                result.append(old)
            else:
                result.append(old)

        i = close_idx + 1

    return '\n'.join(result)

def _collapse_empty_blocks(source: str) -> str:
    lines = source.split('\n')
    result = []
    i = 0
    while i < len(lines):
        if (i + 1 < len(lines) and
            lines[i].rstrip().endswith('{') and
            lines[i + 1].strip() == '}'):
            result.append(lines[i].rstrip() + ' }')
            i += 2
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)

def _normalize_blank_lines(source: str) -> str:
    lines = source.split('\n')
    result = _compress_consecutive_blanks(lines)
    result = _insert_structural_blanks(result)
    result = _separate_multiline_stmts(result)
    result = _fix_registration_blanks(result)
    result = _compress_consecutive_blanks(result)
    while result and result[0].strip() == '':
        result.pop(0)
    while result and result[-1].strip() == '':
        result.pop()
    return '\n'.join(result)

def _compress_consecutive_blanks(lines):
    result = []
    prev_blank = False
    for line in lines:
        if line.strip() == '':
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        result.append(line)
    return result

def _insert_structural_blanks(lines):
    result = []
    depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        open_count = _count_structural_braces(line, '{')
        close_count = _count_structural_braces(line, '}')

        if stripped.startswith('}'):
            line_depth = depth - close_count
            if line_depth < 0:
                line_depth = 0
        else:
            line_depth = depth

        if result and stripped != '' and stripped != '}':
            prev = result[-1].strip()
            prev_is_blank = (prev == '')

            if not prev_is_blank:
                need_blank = False

                if line_depth == 0:
                    is_definition = (stripped.startswith('function ') or
                                    stripped.startswith('class ') or
                                    stripped.startswith('property '))
                    is_registration = (stripped.startswith('this.') or
                                      stripped.startswith('global.'))
                    if prev == '}':
                        if not is_registration:
                            need_blank = True
                    if is_definition:
                        if prev != '' and not prev.startswith('//') and prev != '{':
                            need_blank = True

                elif line_depth == 1:
                    if prev == '}':
                        if (stripped.startswith('function ') or
                            stripped.startswith('class ') or
                            stripped.startswith('property ') or
                            stripped.startswith('var ') or
                            stripped.startswith('this.')):
                            need_blank = True
                    if (prev.endswith(';') and
                        _is_var_decl(result[-1]) and
                        (stripped.startswith('function ') or
                         stripped.startswith('property '))):
                        need_blank = True

                if need_blank:
                    result.append('')

        if stripped == '}' or stripped.startswith('} else'):
            if result and result[-1].strip() == '':
                result.pop()

        if stripped == '' and result:
            prev_stripped = result[-1].strip()
            if (prev_stripped.endswith('{') and
                (prev_stripped.startswith('class ') or
                 (' extends ' in prev_stripped and '{' in prev_stripped))):
                continue

        result.append(line)

        depth += open_count - close_count
        if depth < 0:
            depth = 0

    return result

def _separate_multiline_stmts(lines):
    depth = 0
    line_info = []
    for line in lines:
        s = line.strip()
        opens = _count_structural_braces(line, '{')
        closes = _count_structural_braces(line, '}')
        if s.startswith('}'):
            line_depth = max(0, depth - closes)
        else:
            line_depth = depth
        new_depth = max(0, depth + opens - closes)
        line_info.append((line_depth, new_depth, s, opens))
        depth = new_depth

    max_depth = max((info[1] for info in line_info), default=0) + 1
    stmt_start = [-1] * (max_depth + 1)
    stmts = []

    for i, (line_depth, new_depth, s, opens) in enumerate(line_info):
        if s == '':
            d = line_depth
            if stmt_start[d] >= 0:
                stmts.append((stmt_start[d], i - 1, d))
                stmt_start[d] = -1
            continue

        if stmt_start[line_depth] < 0:
            stmt_start[line_depth] = i

        is_terminating = (s.endswith(';') or s.endswith('{ }') or
                          s == '}' or (s.endswith('}') and opens == 0))
        if is_terminating and stmt_start[new_depth] >= 0:
            stmts.append((stmt_start[new_depth], i, new_depth))
            stmt_start[new_depth] = -1

    for d in range(max_depth + 1):
        if stmt_start[d] >= 0:
            stmts.append((stmt_start[d], len(lines) - 1, d))

    stmts.sort()

    from collections import defaultdict
    by_depth = defaultdict(list)
    for start, end, d in stmts:
        by_depth[d].append((start, end))

    insert_before = set()
    for d, depth_stmts in by_depth.items():
        depth_stmts.sort()
        for j in range(1, len(depth_stmts)):
            prev_s, prev_e = depth_stmts[j - 1]
            curr_s, curr_e = depth_stmts[j]

            if curr_s > prev_e + 1:
                continue

            prev_len = prev_e - prev_s + 1
            curr_len = curr_e - curr_s + 1

            if (prev_len >= 3 or curr_len >= 3) and d == 0:
                insert_before.add(curr_s)

    result = []
    for i, line in enumerate(lines):
        if i in insert_before:
            result.append('')
        result.append(line)

    return result

def _count_structural_braces(line, brace_char):
    count = 0
    in_string = None
    i = 0
    s = line
    while i < len(s):
        ch = s[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == in_string:
                in_string = None
        else:
            if ch == '/' and i + 1 < len(s) and s[i+1] == '/':
                break
            if ch == '"' or ch == "'":
                in_string = ch
            elif ch == brace_char:
                count += 1
        i += 1
    return count

def _is_var_decl(line):
    stripped = line.strip()
    return stripped.startswith('var ')

_REGISTRATION_RE = re.compile(
    r'^(this|global)\.(\w+)\s*=\s*(\w+)\s*(incontextof\s+\w+\s*)?;$'
)

def _is_registration_line(line):
    s = line.strip()
    m = _REGISTRATION_RE.match(s)
    if not m:
        return False
    prop_name = m.group(2)
    value_name = m.group(3)
    return prop_name == value_name

def _fix_registration_blanks(lines):
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        if _is_registration_line(line):
            while result and result[-1].strip() == '':
                result.pop()
            result.append(line)
        else:
            result.append(line)

    return result

def _find_catch_brace(source, open_pos):
    depth = 0
    i = open_pos
    n = len(source)
    while i < n:
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1

def _rename_catch_var(source):
    pattern = re.compile(r'\bcatch\s*\(\s*([a-zA-Z_]\w*)\s*\)')
    offset = 0

    while True:
        m = pattern.search(source, offset)
        if not m:
            break

        var_name = m.group(1)
        if var_name == 'e':
            offset = m.end()
            continue

        brace_pos = source.find('{', m.end())
        if brace_pos == -1:
            offset = m.end()
            continue

        if source[m.end():brace_pos].strip():
            offset = m.end()
            continue

        close_brace = _find_catch_brace(source, brace_pos)
        if close_brace == -1:
            offset = m.end()
            continue

        catch_decl_new = 'catch (e)'
        source = source[:m.start()] + catch_decl_new + source[m.end():]
        delta = len(catch_decl_new) - (m.end() - m.start())
        brace_pos += delta
        close_brace += delta

        body = source[brace_pos:close_brace + 1]
        word_re = re.compile(r'\b' + re.escape(var_name) + r'\b')
        new_body = word_re.sub('e', body)
        source = source[:brace_pos] + new_body + source[close_brace + 1:]
        close_brace += len(new_body) - len(body)

        offset = close_brace

    return source

_TOPLEVEL_LOCAL_DECL_RE = re.compile(r'^var (local\d+(?:_\d+)?)\b')
_DEF_START_RE = re.compile(r'^(class |function |property )')

def _wrap_toplevel_local_vars(source: str) -> str:
    lines = source.split('\n')

    local_names: set = set()
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if indent == 0:
            m = _TOPLEVEL_LOCAL_DECL_RE.match(stripped)
            if m:
                local_names.add(m.group(1))

    if not local_names:
        return source

    names_alt = '|'.join(re.escape(n)
                         for n in sorted(local_names, key=len, reverse=True))
    ref_re = re.compile(r'\b(' + names_alt + r')\b')

    in_definition = [False] * len(lines)
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        indent = (len(lines[i]) - len(stripped)) if stripped else 999
        if indent == 0 and _DEF_START_RE.match(stripped):
            def_start = i
            for j in range(i + 1, len(lines)):
                s = lines[j].rstrip()
                if s == '}':
                    for k in range(def_start, j + 1):
                        in_definition[k] = True
                    i = j + 1
                    break
            else:
                for k in range(def_start, len(lines)):
                    in_definition[k] = True
                break
        else:
            i += 1

    tsb_starts = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if indent == 0 and not in_definition[i]:
            tsb_starts.append(i)

    if not tsb_starts:
        return source

    tsbs = []
    for idx, start in enumerate(tsb_starts):
        if idx + 1 < len(tsb_starts):
            end = tsb_starts[idx + 1] - 1
        else:
            end = len(lines) - 1
        while end > start and not lines[end].strip():
            end -= 1
        tsbs.append((start, end))

    first_tsb_idx = None
    last_tsb_idx = None
    for t_idx, (start, end) in enumerate(tsbs):
        for i in range(start, end + 1):
            if in_definition[i]:
                continue
            if ref_re.search(lines[i]):
                if first_tsb_idx is None:
                    first_tsb_idx = t_idx
                last_tsb_idx = t_idx
                break

    if first_tsb_idx is None:
        return source

    wrap_start = tsbs[first_tsb_idx][0]
    wrap_end = tsbs[last_tsb_idx][1]

    for i in range(wrap_end + 1, len(lines)):
        stripped = lines[i].lstrip()
        if not stripped:
            continue
        if in_definition[i]:
            break
        indent = len(lines[i]) - len(stripped)
        if indent > 0:
            wrap_end = i
            continue
        if stripped.startswith('}'):
            wrap_end = i
            continue
        break

    result = []
    for i, line in enumerate(lines):
        if i == wrap_start:
            if result and result[-1].strip():
                result.append('')
            result.append('{')
        if wrap_start <= i <= wrap_end:
            if line.strip():
                result.append(INDENT_STR + line)
            else:
                result.append('')
        else:
            result.append(line)
        if i == wrap_end:
            result.append('}')
            next_non_blank = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    next_non_blank = j
                    break
            if next_non_blank is not None and next_non_blank == i + 1:
                result.append('')

    return '\n'.join(result)
