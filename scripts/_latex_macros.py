r"""_latex_macros.py — \newcommand / \def expansion for math verification.

Real-world LaTeX papers define short aliases (``\calC``, ``\RR``, ``\opE``) and
use them throughout displayed equations. `sympy.parsing.latex.parse_latex`
does not know these macros and fails with opaque parser errors — the user's
ESORICS report showed a huge share of `unverifiable_kind=tool` targets came
from exactly this cause.

This module re-implements just enough LaTeX macro substitution to cover:

  * ``\newcommand{\foo}{body}`` and ``\newcommand\foo{body}``
  * ``\newcommand{\foo}[n]{body}`` with positional arguments ``#1..#9``
  * ``\renewcommand`` with the same signatures
  * ``\def\foo{body}`` and ``\def\foo#1..#n{body}`` (single-body form only —
    we do not support TeX's full argument-pattern grammar).

Expansion is iterative (fixed-point or iteration cap, whichever first) so
nested definitions like ``\newcommand{\calC}{\mathcal{C}}\newcommand
{\calCstar}{\calC^{\ast}}`` resolve cleanly.

This is an intentionally small subset — complex TeX parsing belongs to a real
engine. When a macro expansion would introduce unbalanced braces or exceed
``max_iterations``, the caller is expected to skip the expansion for that
equation only (with ``evidence.macro_expansion_note``) rather than crash the
whole verifier.
"""

from __future__ import annotations

import re
from typing import Iterable

__all__ = [
    "extract_newcommands",
    "expand_macros",
    "MacroExpansionError",
]


class MacroExpansionError(Exception):
    """Raised for unrecoverable expansion problems (unbalanced braces, etc.)."""


# ---------------- brace utilities ----------------


def _read_brace_group(tex: str, open_index: int) -> tuple[int, int] | None:
    r"""Return ``(inside_start, after_close_index)`` for a ``{...}`` group.

    ``open_index`` must point at a literal ``{``. Returns None if the group is
    unterminated. Correctly ignores ``\{``/``\}``.
    """
    if open_index >= len(tex) or tex[open_index] != "{":
        return None
    depth = 0
    i = open_index
    n = len(tex)
    while i < n:
        ch = tex[i]
        if ch == "\\" and i + 1 < n:
            # Skip the escaped character — do not let it toggle brace depth.
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (open_index + 1, i + 1)
        i += 1
    return None


def _skip_ws(tex: str, i: int) -> int:
    n = len(tex)
    while i < n and tex[i] in " \t\r\n":
        i += 1
    return i


def _read_optional_arg_count(tex: str, i: int) -> tuple[int, int]:
    """If ``tex[i:]`` starts with ``[N]`` return ``(N, new_i)`` else ``(0, i)``."""
    if i >= len(tex) or tex[i] != "[":
        return (0, i)
    close = tex.find("]", i + 1)
    if close < 0:
        return (0, i)
    inner = tex[i + 1:close].strip()
    if inner.isdigit():
        return (int(inner), close + 1)
    return (0, i)


def _read_macro_name(tex: str, i: int) -> tuple[str | None, int]:
    """Read ``\name`` (letters-only macro name) starting at ``tex[i:]``.

    Returns ``(name_without_backslash, new_index_after_name)`` or ``(None, i)``
    when no name is present.
    """
    n = len(tex)
    if i >= n or tex[i] != "\\":
        return (None, i)
    j = i + 1
    while j < n and tex[j].isalpha():
        j += 1
    if j == i + 1:
        return (None, i)
    return (tex[i + 1:j], j)


# ---------------- extraction ----------------


_NEWCOMMAND_HEADER_RE = re.compile(
    r"\\(?:new|renew|provide)command\*?\s*"
)
_DEF_HEADER_RE = re.compile(r"\\def\s*")


def _extract_newcommand_style(tex: str, macros: dict[str, tuple[int, str]]) -> None:
    for m in _NEWCOMMAND_HEADER_RE.finditer(tex):
        idx = m.end()
        # Name may be ``{\foo}`` or bare ``\foo``.
        if idx < len(tex) and tex[idx] == "{":
            group = _read_brace_group(tex, idx)
            if not group:
                continue
            inside = tex[group[0]:group[1] - 1]
            name, _ = _read_macro_name(inside.strip(), 0)
            if name is None:
                continue
            idx = group[1]
        else:
            name, new_idx = _read_macro_name(tex, idx)
            if name is None:
                continue
            idx = new_idx

        idx = _skip_ws(tex, idx)
        n_args, idx = _read_optional_arg_count(tex, idx)
        idx = _skip_ws(tex, idx)
        # Some \newcommand templates include a default value for the first arg
        # as a second optional [default]; consume it benignly.
        _, idx = _read_optional_arg_count(tex, idx)
        idx = _skip_ws(tex, idx)
        if idx >= len(tex) or tex[idx] != "{":
            continue
        group = _read_brace_group(tex, idx)
        if not group:
            continue
        body = tex[group[0]:group[1] - 1]
        macros[name] = (n_args, body)


def _extract_def_style(tex: str, macros: dict[str, tuple[int, str]]) -> None:
    for m in _DEF_HEADER_RE.finditer(tex):
        idx = m.end()
        name, idx = _read_macro_name(tex, idx)
        if name is None:
            continue
        # Simplified \def: count successive #N patterns until a ``{``.
        n = len(tex)
        n_args = 0
        while idx < n and tex[idx] == "#":
            if idx + 1 < n and tex[idx + 1].isdigit():
                n_args = max(n_args, int(tex[idx + 1]))
                idx += 2
            else:
                break
        idx = _skip_ws(tex, idx)
        if idx >= n or tex[idx] != "{":
            continue
        group = _read_brace_group(tex, idx)
        if not group:
            continue
        body = tex[group[0]:group[1] - 1]
        macros[name] = (n_args, body)


def extract_newcommands(tex: str) -> dict[str, tuple[int, str]]:
    """Extract all ``\\newcommand``/``\\renewcommand``/``\\def`` definitions.

    Returns a dict ``{macro_name: (n_args, body)}``. Duplicate names keep the
    last definition — matching LaTeX semantics for ``\\renewcommand``.
    """
    macros: dict[str, tuple[int, str]] = {}
    _extract_newcommand_style(tex, macros)
    _extract_def_style(tex, macros)
    return macros


# ---------------- expansion ----------------


def _find_macro_invocation(tex: str, name: str, start: int = 0) -> int:
    r"""Return index of the next ``\name`` that is not followed by another
    letter (so ``\cal`` does not match inside ``\calC``).
    """
    n = len(tex)
    i = start
    while True:
        pos = tex.find("\\" + name, i)
        if pos < 0:
            return -1
        end = pos + 1 + len(name)
        if end < n and tex[end].isalpha():
            i = end
            continue
        return pos


def _substitute_single_call(
    tex: str,
    pos: int,
    name: str,
    n_args: int,
    body: str,
) -> tuple[str, int] | None:
    """Replace one ``\name[args]`` invocation at ``pos``.

    Returns ``(new_tex, new_cursor)``. ``new_cursor`` is the position just past
    the inserted body so we keep scanning from the right place — the expanded
    body itself will be re-scanned by the outer loop.
    """
    name_end = pos + 1 + len(name)
    args: list[str] = []
    cursor = name_end
    for _ in range(n_args):
        cursor = _skip_ws(tex, cursor)
        if cursor >= len(tex):
            return None
        if tex[cursor] == "{":
            group = _read_brace_group(tex, cursor)
            if not group:
                return None
            args.append(tex[group[0]:group[1] - 1])
            cursor = group[1]
        else:
            # Bare-token argument: one character OR one ``\cmd``.
            if tex[cursor] == "\\":
                cname, end = _read_macro_name(tex, cursor)
                if cname is None:
                    return None
                args.append(tex[cursor:end])
                cursor = end
            else:
                args.append(tex[cursor])
                cursor += 1

    expanded = body
    for k in range(9, 0, -1):  # replace #9 before #1 to avoid overlap issues
        token = f"#{k}"
        if token in expanded:
            replacement = args[k - 1] if k - 1 < len(args) else ""
            expanded = expanded.replace(token, replacement)

    new_tex = tex[:pos] + expanded + tex[cursor:]
    return new_tex, pos + len(expanded)


def _pass_once(tex: str, macros: dict[str, tuple[int, str]]) -> tuple[str, bool]:
    changed = False
    # Sort longest name first so ``\calCstar`` is tried before ``\cal``. Our
    # _find_macro_invocation guard prevents ``\cal`` ever matching inside
    # ``\calC``, but ordering still matters when two defined names share a
    # prefix (``\RR`` vs ``\R``).
    names = sorted(macros.keys(), key=lambda s: (-len(s), s))
    cursor_by_name: dict[str, int] = {n: 0 for n in names}
    i = 0
    # Single-pass: walk the string linearly, expanding the first macro seen
    # and restarting (so nested expansions are re-processed in the next pass).
    for name in names:
        n_args, body = macros[name]
        pos = _find_macro_invocation(tex, name, cursor_by_name[name])
        if pos < 0:
            continue
        subst = _substitute_single_call(tex, pos, name, n_args, body)
        if subst is None:
            # Skip this invocation (could be malformed or argument unresolved).
            cursor_by_name[name] = pos + 1 + len(name)
            continue
        tex, _ = subst
        return tex, True
    return tex, changed


def expand_macros(
    tex: str,
    macros: dict[str, tuple[int, str]],
    max_iterations: int = 5,
) -> str:
    """Expand known macros in ``tex`` until fixed-point or ``max_iterations``.

    Raises ``MacroExpansionError`` only when:
      * the expansion attempts produce a string with unbalanced braces (count
        mismatch), OR
      * ``max_iterations`` is exceeded without reaching a fixed point — this
        usually indicates a macro that expands to itself, e.g. a buggy
        ``\\newcommand\\foo{\\foo+1}``.

    Callers are expected to catch ``MacroExpansionError`` and attach a note
    to the offending target; crashing the whole verifier would be worse than
    returning the original string.
    """
    if not macros:
        return tex

    current = tex
    for _ in range(max_iterations):
        updated, changed = _pass_once(current, macros)
        if not changed:
            current = updated
            break
        current = updated
    else:
        raise MacroExpansionError(
            f"macro expansion did not converge after {max_iterations} passes"
        )

    if current.count("{") != current.count("}"):
        raise MacroExpansionError(
            f"macro expansion produced unbalanced braces "
            f"({current.count('{')} open, {current.count('}')} close)"
        )

    return current


def safe_expand(
    tex: str,
    macros: dict[str, tuple[int, str]],
    max_iterations: int = 5,
) -> tuple[str, str | None]:
    """Expand and swallow ``MacroExpansionError``.

    Returns ``(expanded_or_original, note_or_None)``. Verifiers use this when
    they prefer to fall back to the un-expanded source rather than fail the
    equation outright.
    """
    try:
        return expand_macros(tex, macros, max_iterations=max_iterations), None
    except MacroExpansionError as exc:
        return tex, str(exc)


def iter_macro_names(macros: dict[str, tuple[int, str]]) -> Iterable[str]:
    """Convenience for tests and logs."""
    return sorted(macros.keys())
