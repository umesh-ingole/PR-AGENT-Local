import ast

import pytest

from autofix import apply_python_autofixes


@pytest.mark.parametrize(
    "source",
    [
        'def parse(value):\n    return eval(value)\n',
        '"""Module docs."""\n\nimport ast\n\ndef parse(value):\n    return eval(value)\n',
    '"""Module docs."""\nfrom __future__ import annotations\n\ndef parse(value):\n    return eval(value)\n',
    '#!/usr/bin/env python\n# coding: utf-8\ndef parse(value):\n    return eval(value)\n',
])
def test_eval_is_rewritten_and_comment_is_added_with_valid_import_order(source):
    fixed, rewrites = apply_python_autofixes(source)

    assert "return ast.literal_eval(value)" in fixed
    assert fixed.count("import ast") == 1
    assert (
        "# REVIEW: eval replaced by ast.literal_eval, only literals are accepted"
        in fixed
    )
    assert rewrites == [{"rule": "eval -> ast.literal_eval", "line": source.count("\n", 0, source.index("eval(")) + 1}]
    ast.parse(fixed)


def test_eval_rewrite_adds_comment_before_existing_trailing_comment():
    source = "result = eval(value)  # existing note\n"

    fixed, _ = apply_python_autofixes(source)

    assert (
        "result = ast.literal_eval(value)  "
        "# REVIEW: eval replaced by ast.literal_eval, only literals are accepted "
        "# existing note"
    ) in fixed
    ast.parse(fixed)


def test_yaml_load_without_loader_becomes_safe_load():
    fixed, rewrites = apply_python_autofixes(
        "import yaml\nresult = yaml.load(contents)\n"
    )

    assert "yaml.safe_load(contents)" in fixed
    assert rewrites == [{"rule": "yaml.load -> yaml.safe_load", "line": 2}]


def test_yaml_load_with_loader_is_unchanged():
    source = (
        "import yaml\n"
        "result = yaml.load(contents, Loader=yaml.SafeLoader)\n"
    )

    fixed, rewrites = apply_python_autofixes(source)

    assert fixed == source
    assert rewrites == []


def test_existing_literal_eval_is_untouched():
    source = "import ast\nresult = ast.literal_eval(value)\n"

    fixed, rewrites = apply_python_autofixes(source)

    assert fixed == source
    assert rewrites == []


def test_eval_source_positions_handle_unicode_before_call():
    source = 'message = "café"\nresult = eval(value)\n'

    fixed, rewrites = apply_python_autofixes(source)

    assert "result = ast.literal_eval(value)" in fixed
    assert rewrites == [{"rule": "eval -> ast.literal_eval", "line": 2}]
