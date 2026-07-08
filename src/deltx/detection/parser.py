"""Python source code parser for the AI authorship detection module.

Transforms raw Python source (as a string) into a :class:`ParsedSource`
structure that bundles lexical tokens, the AST, extracted identifiers, and a
handful of line/indentation statistics. This is the single ingestion point for
all three detection feature families, so it is deliberately resilient: malformed
or unparseable input never raises out of :meth:`PythonSourceParser.parse`;
instead the result is flagged ``is_valid=False`` with whatever could still be
recovered.
"""

import ast
import builtins
import io
import logging
import tokenize
from pathlib import Path

from deltx.detection.models import ParsedSource

logger = logging.getLogger(__name__)

# Built-in names (``print``, ``len``, ``range``, ...) are language vocabulary,
# not author-chosen identifiers, so they are excluded from stylometric analysis.
_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))

# Token types that carry no lexical signal for the language model and are
# dropped from the token list. COMMENT is handled separately (counted, not kept).
_SKIP_TOKEN_TYPES: frozenset[int] = frozenset(
    {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.ENCODING,
    }
)


class PythonSourceParser:
    """Parses Python source into structured representations for feature extraction."""

    def parse(self, source_code: str, file_path: Path) -> ParsedSource:
        """Parse a Python source string into a :class:`ParsedSource`.

        Args:
            source_code: Raw Python source of a single file.
            file_path: Path the source originated from (used for reporting only).

        Returns:
            A :class:`ParsedSource`. ``is_valid`` is ``False`` when the input is
            empty, cannot be decoded, or fails to parse as Python; partial data
            (e.g. lexical tokens) is still populated where recoverable.

        Raises:
            ParsingError: Never raised directly; retained for interface symmetry
                with callers that treat parsing as fallible. All parse failures
                are reported via ``is_valid=False``.
        """
        # Empty (or whitespace-only) source has nothing to analyze.
        if not source_code or not source_code.strip():
            return self._empty_result(source_code, file_path)

        tokens, comment_line_nums, tokens_valid = self._tokenize(source_code)
        ast_tree, identifiers, node_types, depths, ast_valid = self._parse_ast(
            source_code, file_path
        )
        loc, comment_lines, total_lines, indent_levels = self._line_metrics(
            source_code, comment_line_nums
        )

        return ParsedSource(
            source_code=source_code,
            file_path=file_path,
            tokens=tokens,
            ast_tree=ast_tree,
            identifiers=identifiers,
            lines_of_code=loc,
            comment_lines=comment_lines,
            total_lines=total_lines,
            indent_levels=indent_levels,
            ast_node_types=node_types,
            ast_depths=depths,
            is_valid=tokens_valid and ast_valid,
        )

    def _empty_result(self, source_code: str, file_path: Path) -> ParsedSource:
        """Build a fully-zeroed invalid result for empty input."""
        return ParsedSource(
            source_code=source_code,
            file_path=file_path,
            tokens=[],
            ast_tree=None,
            identifiers=[],
            lines_of_code=0,
            comment_lines=0,
            total_lines=0,
            indent_levels=[],
            ast_node_types=[],
            ast_depths=[],
            is_valid=False,
        )

    def _tokenize(
        self, source_code: str
    ) -> tuple[list[str], set[int], bool]:
        """Lexically tokenize the source.

        Returns:
            A tuple of ``(tokens, comment_line_numbers, is_valid)``. Tokens
            collected before any lexical error are retained so that even
            unparseable code yields a usable token stream.
        """
        tokens: list[str] = []
        comment_line_nums: set[int] = set()
        is_valid = True
        readline = io.StringIO(source_code).readline
        try:
            for tok in tokenize.generate_tokens(readline):
                if tok.type == tokenize.COMMENT:
                    # Count only full-line comments (nothing but whitespace
                    # precedes the '#'); inline comments belong to a code line.
                    if tok.line[: tok.start[1]].strip() == "":
                        comment_line_nums.add(tok.start[0])
                    continue
                if tok.type in _SKIP_TOKEN_TYPES:
                    continue
                tokens.append(tok.string)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            is_valid = False
        except UnicodeDecodeError:
            logger.warning("Non-decodable source encountered during tokenization")
            is_valid = False
        return tokens, comment_line_nums, is_valid

    def _parse_ast(
        self, source_code: str, file_path: Path
    ) -> tuple[ast.Module | None, list[str], list[str], list[int], bool]:
        """Parse and walk the AST.

        Returns:
            ``(ast_tree, identifiers, node_types, depths, is_valid)``. On a parse
            failure the tree is ``None``, the derived lists are empty, and
            ``is_valid`` is ``False``.
        """
        try:
            ast_tree = ast.parse(source_code)
        except (SyntaxError, ValueError) as exc:
            # ValueError covers e.g. source containing null bytes.
            logger.warning("AST parse failed for %s: %s", file_path, exc)
            return None, [], [], [], False

        import_names = self._collect_import_names(ast_tree)
        identifiers: list[str] = []
        node_types: list[str] = []
        depths: list[int] = []
        self._traverse(
            ast_tree, 0, import_names, identifiers, node_types, depths
        )
        return ast_tree, identifiers, node_types, depths, True

    @staticmethod
    def _collect_import_names(ast_tree: ast.Module) -> set[str]:
        """Collect top-level names bound by ``import`` / ``from`` statements.

        These are module references (``os`` in ``os.getcwd()``), not
        author-chosen identifiers, so they are excluded from the identifier list.
        """
        import_names: set[str] = set()
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    bound = alias.asname or alias.name
                    import_names.add(bound.split(".")[0])
        return import_names

    def _traverse(
        self,
        node: ast.AST,
        depth: int,
        import_names: set[str],
        identifiers: list[str],
        node_types: list[str],
        depths: list[int],
    ) -> None:
        """Recursively record node type, depth, and identifiers for ``node``.

        Depth is computed by explicit recursion (the AST root is depth 0) rather
        than via :func:`ast.walk`, which discards structural depth.
        """
        node_types.append(type(node).__name__)
        depths.append(depth)

        if isinstance(node, ast.Name):
            if node.id not in _BUILTIN_NAMES and node.id not in import_names:
                identifiers.append(node.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            identifiers.append(node.name)
        elif isinstance(node, ast.arg):
            identifiers.append(node.arg)

        for child in ast.iter_child_nodes(node):
            self._traverse(
                child, depth + 1, import_names, identifiers, node_types, depths
            )

    @staticmethod
    def _line_metrics(
        source_code: str, comment_line_nums: set[int]
    ) -> tuple[int, int, int, list[int]]:
        """Compute line-based statistics.

        Returns:
            ``(lines_of_code, comment_lines, total_lines, indent_levels)`` where
            ``lines_of_code`` counts non-empty, non-comment-only lines and
            ``indent_levels`` holds the leading-whitespace width of each such
            line.
        """
        lines = source_code.splitlines()
        total_lines = len(lines)
        comment_lines = len(comment_line_nums)
        loc = 0
        indent_levels: list[int] = []
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue  # blank line
            if line_no in comment_line_nums:
                continue  # comment-only line
            loc += 1
            indent_levels.append(len(line) - len(line.lstrip()))
        return loc, comment_lines, total_lines, indent_levels
