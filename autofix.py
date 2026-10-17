"""Small deterministic Python rewrites grounded in AST source positions."""
import ast
import io
import tokenize


_EVAL_REVIEW = (
    "# REVIEW: eval replaced by ast.literal_eval, only literals are accepted"
)


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for index, character in enumerate(source):
        if character == "\n":
            offsets.append(index + 1)
    if offsets[-1] != len(source):
        offsets.append(len(source))
    return offsets


def _source_offset(
    source: str, line_offsets: list[int], line: int, byte_column: int,
) -> int:
    line_start = line_offsets[line - 1]
    line_end = source.find("\n", line_start)
    if line_end == -1:
        line_end = len(source)
    line_text = source[line_start:line_end]
    char_column = len(line_text.encode("utf-8")[:byte_column].decode("utf-8"))
    return line_start + char_column


def _import_insertion_offset(source: str, tree: ast.Module, offsets: list[int]) -> int:
    body = tree.body
    position = 0
    body_index = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(
        body[0].value, ast.Constant
    ) and isinstance(body[0].value.value, str):
        position = offsets[body[0].end_lineno]
        body_index = 1
    while body_index < len(body):
        node = body[body_index]
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        ):
            position = offsets[node.end_lineno]
            body_index += 1
            continue
        break
    if position:
        return position

    lines = source.splitlines(keepends=True)
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        stripped = line.lstrip()
        if (
            line_index == 0 and stripped.startswith("#!")
        ) or "coding" in stripped[:32] and stripped.startswith("#"):
            position += len(line)
            line_index += 1
            continue
        break
    return position


def _comment_insertions(
    source: str, lines_with_eval: set[int],
) -> list[tuple[int, str]]:
    comments: dict[int, int] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.start[1]

    insertions = []
    for line_number in sorted(lines_with_eval):
        line = source.splitlines(keepends=True)[line_number - 1]
        line_without_newline = line.rstrip("\r\n")
        comment_column = comments.get(line_number)
        if comment_column is None:
            insertion = len(line_without_newline)
            insertions.append((sum(
                len(part) for part in source.splitlines(keepends=True)[:line_number - 1]
            ) + insertion, " " + _EVAL_REVIEW))
        else:
            line_offset = sum(
                len(part) for part in source.splitlines(keepends=True)[:line_number - 1]
            )
            insertions.append(
                (line_offset + comment_column, _EVAL_REVIEW + " ")
            )
    return insertions


def apply_python_autofixes(source: str) -> tuple[str, list[dict]]:
    """Rewrite supported eval/yaml calls and return the changed code and records."""
    tree = ast.parse(source)
    offsets = _line_offsets(source)
    edits: list[tuple[int, int, str]] = []
    rewritten: list[dict] = []
    eval_lines: set[int] = set()
    needs_ast_import = False
    has_plain_ast_import = any(
        isinstance(node, ast.Import)
        and any(alias.name == "ast" and alias.asname is None for alias in node.names)
        for node in tree.body
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "eval":
            start = _source_offset(
                source, offsets, node.func.lineno, node.func.col_offset
            )
            end = _source_offset(
                source, offsets, node.func.end_lineno, node.func.end_col_offset
            )
            edits.append((start, end, "ast.literal_eval"))
            rewritten.append({
                "rule": "eval -> ast.literal_eval",
                "line": node.lineno,
            })
            eval_lines.add(node.end_lineno)
            needs_ast_import = True
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "yaml"
            and node.func.attr == "load"
            and len(node.args) < 2
            and not any(keyword.arg == "Loader" for keyword in node.keywords)
        ):
            start = _source_offset(
                source, offsets, node.func.lineno, node.func.col_offset
            )
            end = _source_offset(
                source, offsets, node.func.end_lineno, node.func.end_col_offset
            )
            edits.append((start, end, "yaml.safe_load"))
            rewritten.append({
                "rule": "yaml.load -> yaml.safe_load",
                "line": node.lineno,
            })

    if not rewritten:
        return source, []

    edits.extend(
        (position, position, content)
        for position, content in _comment_insertions(source, eval_lines)
    )
    if needs_ast_import and not has_plain_ast_import:
        insertion = _import_insertion_offset(source, tree, offsets)
        import_text = "import ast\n"
        if insertion and source[:insertion].endswith("\n\n"):
            import_text = "import ast\n"
        elif insertion and not source[:insertion].endswith("\n"):
            import_text = "\nimport ast\n"
        edits.append((insertion, insertion, import_text))

    fixed = source
    for start, end, replacement in sorted(edits, reverse=True):
        fixed = fixed[:start] + replacement + fixed[end:]
    rewritten.sort(key=lambda item: (item["line"], item["rule"]))
    return fixed, rewritten
