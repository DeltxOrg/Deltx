"""Conservative Python semantic fingerprints for checkpoint selection."""

import ast
import io
import tokenize


def fingerprint(source: str) -> str:
    """Ignore only recognized docstrings, comments and formatting.

    Failed AST parses use tokens preserving NEWLINE/INDENT/DEDENT. If even
    tokenization fails, compare the original source so uncertainty keeps changes.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        try:
            tokens = []
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type in {
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.ENCODING,
                    tokenize.ENDMARKER,
                }:
                    continue
                value = token.string
                if token.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE}:
                    value = ""
                tokens.append((token.type, value))
            return repr(tokens)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return "unparsed:" + source
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def meaningful_change(old: str, new: str) -> bool:
    """Keep even a one-line semantic change; empty/docstring-only files are inert."""
    return fingerprint(old) != fingerprint(new)
