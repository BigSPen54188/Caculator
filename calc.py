"""Smart Calculator — Optimized for readability, performance, and robustness.

Supports: Equations, Systems, Inequalities, ODEs (Analytical/Numerical),
          Automatic Complex Solutions.

Environment: sympy >= 1.14, scipy >= 1.18, numpy >= 2.0, mpmath >= 1.3, mypy >= 2.2
"""

import logging
import re
import types
import unittest
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, TypeAlias

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sympy import (
    Derivative, Dummy, Eq, Function, S, Subs, Symbol,
    cancel, dsolve, expand, factor, lambdify, nroots, parse_expr, pi,
    real_roots, solve, solveset, symbols,
    E, I, oo, zoo, nan, nsolve, Poly, Pow, Mul,
    exp, sin, cos, tan, log, sinh, cosh, tanh,
)
from sympy.core.containers import Tuple as SympyTuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Equality, Relational
from sympy.parsing.sympy_parser import (
    auto_symbol, convert_xor, implicit_multiplication_application,
    standard_transformations,
)
from sympy.polys.polyerrors import PolynomialError
from sympy.solvers.inequalities import reduce_inequalities
from sympy.solvers.solveset import (
    ConditionSet, FiniteSet, ImageSet, Interval, Union,
    linsolve, nonlinsolve,
)

mp.mp.dps = 20

EVALF_PREC: Final[int] = 15
MP_NSTR_DIGITS: Final[int] = 12
NSOLVE_PREC: Final[int] = 20
NSOLVE_TOL: Final[float] = 1e-14
MAX_POLY_DEGREE: Final[int] = 20
MAX_ROOT_ABS: Final[float] = 1e15
MAX_INPUT_LEN: Final[int] = 5000
TOL_VERIFY: Final[float] = 1e-6
TOL_REAL_FILTER: Final[float] = 1e-10
TOL_NSOLVE_DEDUP: Final[float] = 1e-12
TOL_FMT_ZERO: Final[float] = 1e-15

_TRANS: Final[tuple[Any, ...]] = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    auto_symbol,
)

_SYM_MAP: Final[dict[str, Symbol]] = {
    str(s): s
    for s in symbols("x y z a b c t u v w alpha beta gamma delta epsilon theta lambda mu phi psi")
}
_SYM_MAP.update({"pi": pi, "e": E, "I": I, "oo": oo})

_CONSTANTS: Final[frozenset[Symbol]] = frozenset((pi, E, I, oo, zoo, nan))

SymExpr: TypeAlias = Expr | Eq | Relational

_TRANSCENDENTAL_FUNCS: Final[frozenset[Any]] = frozenset((
    exp, sin, cos, tan, log, sinh, cosh, tanh, AppliedUndef,
))

_NSOLVE_REAL_GUESSES: Final[tuple[float, ...]] = (
    0.0, 0.5, -0.5, 1.0, -1.0,
)

_NSOLVE_COMPLEX_GUESSES: Final[tuple[complex, ...]] = (
    1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j,
    2j, -2j,
)

_RE_DERIV_PARAM: Final = re.compile(r"([a-zA-Z_]\w*)('+)\(([^)]+)\)")
_RE_DERIV_SIMPLE: Final = re.compile(r"(?<![a-zA-Z_'])\b([a-zA-Z_]\w*)('+(?!\())")
_RE_DIFF_PARAM: Final = re.compile(r"([a-zA-Z_]\w*)\(([^)]+)\)\.diff\(([^)]+)\)")
_RE_DIFF_SIMPLE: Final = re.compile(r"([a-zA-Z_]\w*)\(([^)]+)\)\.diff\(\)")
_RE_DIFF_NOARG: Final = re.compile(r"(?<!\))\b([a-zA-Z_]\w*)\.diff\(([^)]+)\)")
_RE_DIFF_NOARG_EMPTY: Final = re.compile(r"(?<!\))\b([a-zA-Z_]\w*)\.diff\(\)")

_RE_NUMERIC_FLAG: Final = re.compile(r"(?i)\bnumeric\b")
_RE_INIT: Final = re.compile(r"(?i)\binit\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)*)\b")
_RE_TSPAN: Final = re.compile(r"(?i)\btspan\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b")
_RE_METHOD: Final = re.compile(r"(?i)\bmethod\s*=\s*(\w+)\b")
_RE_RTOL: Final = re.compile(r"(?i)\brtol\s*=\s*([0-9.eE+-]+)\b")
_RE_ATOL: Final = re.compile(r"(?i)\batol\s*=\s*([0-9.eE+-]+)\b")
_RE_MAXSTEP: Final = re.compile(r"(?i)\bmax_step\s*=\s*([0-9.eE+-]+)\b")
_RE_DOMAIN_REAL: Final = re.compile(r"(?i)\bdomain\s*=\s*real\b")
_RE_DOMAIN_COMPLEX: Final = re.compile(r"(?i)\bdomain\s*=\s*complex\b")
_RE_FOR_CLAUSE: Final = re.compile(r"\bfor\s+")
_RE_MULTIPLE_EQ: Final = re.compile(r"=.*=")
_RE_WHITESPACE: Final = re.compile(r"\s+")
_RE_OPTION_DELIM: Final = re.compile(r",\s*,")
_RE_VALID_VAR: Final = re.compile(r"^[a-zA-Z_]\w*$")

_REL_OPS: Final[frozenset[str]] = frozenset((">=", "<=", ">", "<", "==", "!="))

logger = logging.getLogger(__name__)

_VALID_ODE_METHODS: Final[frozenset[str]] = frozenset((
    "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"
))


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Solver options specified by user through input string."""
    numeric_mode: bool
    init_val: list[float] | None
    t_span: tuple[float, float]
    method: str
    rtol: float
    atol: float
    max_step: float | None


@dataclass(frozen=True, slots=True)
class ParsedInput:
    """Preprocessed input data."""
    raw: str
    domain: Any
    for_vars: tuple[Symbol, ...] | None


# ── Caching optimizations ─────────────────────────────────────────────

@lru_cache(maxsize=256)
def _get_func_pattern(name: str) -> re.Pattern[str] | None:
    """Compile and cache regex patterns for function names to avoid unbounded growth."""
    if not name or not name.isidentifier():
        return None
    return re.compile(rf"(?<![a-zA-Z0-9_]){re.escape(name)}(?![a-zA-Z0-9_('])")


_BASE_LOCAL_DICT: Final[Mapping[str, Any]] = types.MappingProxyType({"Derivative": Derivative, "Subs": Subs})
_BASE_SYM_DICT: Final[Mapping[str, Any]] = types.MappingProxyType(dict(_SYM_MAP))


@lru_cache(maxsize=128)
def _build_local_dict_cached(func_names: frozenset[str], indep_var: str | None) -> dict[str, Any]:
    """Build local symbol dictionary for parsing with immutable key caching."""
    local_dict = dict(_BASE_SYM_DICT)
    local_dict.update(_BASE_LOCAL_DICT)
    if func_names and indep_var:
        for name in func_names:
            local_dict[name] = Function(name)
    return local_dict


def _build_local_dict(func_names: Collection[str] | None, indep_var: str | None) -> dict[str, Any]:
    """Build local symbol dictionary for parsing.

    Parameters
    ----------
    func_names : Collection[str] | None
        Function names to register as Function objects.
    indep_var : str | None
        Independent variable name for constructing function call forms.

    Returns
    -------
    dict[str, Any]
        Local dictionary copy containing symbol mappings and function definitions.
    """
    return _build_local_dict_cached(
        frozenset(func_names) if func_names else frozenset(),
        indep_var
    )


# ── Derivative shorthand preprocessing ────────────────────────────────

def _deriv_param_repl(m: re.Match[str], indep_var: str) -> str:
    fn, primes, var = m.group(1), m.group(2), m.group(3)
    order = len(primes)
    if not _RE_VALID_VAR.match(var):
        iv = indep_var
        if order == 1:
            return f"Subs(Derivative({fn}({iv}), {iv}), {iv}, {var})"
        return f"Subs(Derivative({fn}({iv}), ({iv}, {order})), {iv}, {var})"
    if order == 1:
        return f"Derivative({fn}({var}), {var})"
    return f"Derivative({fn}({var}), ({var}, {order}))"


def _deriv_simple_repl(m: re.Match[str], func_names: frozenset[str], indep_var: str) -> str:
    fn, primes = m.group(1), m.group(2)
    if fn not in func_names:
        return m.group(0)
    order = len(primes)
    if order == 1:
        return f"Derivative({fn}({indep_var}), {indep_var})"
    return f"Derivative({fn}({indep_var}), ({indep_var}, {order}))"


def _diff_param_repl(m: re.Match[str]) -> str:
    fn, arg, var_spec = m.group(1), m.group(2), m.group(3)
    var_spec = var_spec.strip()
    if "," in var_spec:
        parts = [p.strip() for p in var_spec.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            return f"Derivative({fn}({arg}), ({parts[0]}, {parts[1]}))"
        return f"Derivative({fn}({arg}), {', '.join(parts)})"
    return f"Derivative({fn}({arg}), {var_spec})"


def _diff_simple_repl(m: re.Match[str]) -> str:
    fn, arg = m.group(1), m.group(2)
    return f"Derivative({fn}({arg}), {arg})"


def _diff_noarg_repl(m: re.Match[str], indep_var: str) -> str:
    fn, var_spec = m.group(1), m.group(2)
    var_spec = var_spec.strip()
    if "," in var_spec:
        parts = [p.strip() for p in var_spec.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            return f"Derivative({fn}({indep_var}), ({parts[0]}, {parts[1]}))"
        return f"Derivative({fn}({indep_var}), {', '.join(parts)})"
    return f"Derivative({fn}({indep_var}), {var_spec})"


def _diff_noarg_empty_repl(m: re.Match[str], indep_var: str) -> str:
    return f"Derivative({m.group(1)}({indep_var}), {indep_var})"


def _collect_func_names_and_indep_var(expr_str: str) -> tuple[set[str], str | None]:
    """Extract function names and independent variable from expression string."""
    func_names: set[str] = set()
    all_vars: set[str] = set()

    for m in _RE_DERIV_PARAM.finditer(expr_str):
        func_names.add(m.group(1))
        var = m.group(3)
        if _RE_VALID_VAR.match(var):
            all_vars.add(var)

    for m in _RE_DERIV_SIMPLE.finditer(expr_str):
        func_names.add(m.group(1))

    for pat, g1, g2 in (
        (_RE_DIFF_PARAM, 1, 3),
        (_RE_DIFF_SIMPLE, 1, 2),
        (_RE_DIFF_NOARG, 1, 2),
        (_RE_DIFF_NOARG_EMPTY, 1, None),
    ):
        for m in pat.finditer(expr_str):
            func_names.add(m.group(g1))
            if g2 is not None:
                var_spec = m.group(g2).strip()
                v = var_spec.split(",")[0].strip() if "," in var_spec else var_spec
                if _RE_VALID_VAR.match(v):
                    all_vars.add(v)

    if len(all_vars) > 1:
        raise ValueError(f"Multiple independent variables detected: {sorted(all_vars)}")
    indep_var = next(iter(all_vars)) if all_vars else None
    return func_names, indep_var


def preprocess_ode(expr_str: str) -> tuple[str, frozenset[str], str]:
    """Convert derivative shorthand (y', y.diff(x), etc.) to standard SymPy Derivative syntax.

    Note: If user expression contains ordinary variables with same name as functions
    (e.g., using y as variable in non-ODE context while also inputting y'), the variable
    will be coerced to function form y(x). This is a reasonable trade-off for ODE scenarios,
    but caution is needed when mixing function notation and variable notation.

    Returns
    -------
    tuple[str, frozenset[str], str]
        (transformed expression, function names set, independent variable name)
    """
    if "'" not in expr_str and ".diff(" not in expr_str:
        return expr_str, frozenset(), "x"

    func_names, indep_var = _collect_func_names_and_indep_var(expr_str)
    if indep_var is None:
        indep_var = "x"

    iv = indep_var
    fnames = frozenset(func_names)

    expr_str = _RE_DERIV_PARAM.sub(lambda m: _deriv_param_repl(m, iv), expr_str)
    expr_str = _RE_DERIV_SIMPLE.sub(lambda m: _deriv_simple_repl(m, fnames, iv), expr_str)
    expr_str = _RE_DIFF_PARAM.sub(_diff_param_repl, expr_str)
    expr_str = _RE_DIFF_SIMPLE.sub(_diff_simple_repl, expr_str)
    expr_str = _RE_DIFF_NOARG.sub(lambda m: _diff_noarg_repl(m, iv), expr_str)
    expr_str = _RE_DIFF_NOARG_EMPTY.sub(lambda m: _diff_noarg_empty_repl(m, iv), expr_str)

    # Conservative replacement: only replace recognized function names, avoid replacing already parenthesized calls
    for name in func_names:
        pat = _get_func_pattern(name)
        if pat is not None:
            expr_str = pat.sub(f"{name}({iv})", expr_str)

    return expr_str, fnames, iv


# ── Expression splitting ──────────────────────────────────────────────

def split_top_level(expr_str: str, delimiter: str = ",") -> list[str]:
    """Bracket-aware top-level delimiter splitting with quote string and escape support.

    Handles basic escaped quotes (e.g., \\"), but not nested quotes or octal/hex escapes.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_str = False
    str_char = ""
    for i, char in enumerate(expr_str):
        if in_str:
            current.append(char)
            if char == str_char:
                # Count consecutive backslashes before current position
                backslash_count = 0
                for j in range(i - 1, -1, -1):
                    if expr_str[j] == '\\':
                        backslash_count += 1
                    else:
                        break
                if backslash_count % 2 == 0:
                    in_str = False
            continue
        if char in '"\'':
            in_str = True
            str_char = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Mismatched closing bracket at position {i}: {char!r}")
            current.append(char)
        elif char == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current.clear()
        else:
            current.append(char)
    if depth != 0:
        raise ValueError("Mismatched brackets in expression")
    if current:
        parts.append("".join(current).strip())
    return parts


def _has_eq_sign(expr_str: str) -> bool:
    """Check if string contains an equality sign (excluding relational operators)."""
    if "=" not in expr_str:
        return False
    return not any(op in expr_str for op in _REL_OPS)


# ── Expression parsing ────────────────────────────────────────────────

def _parse_equality(expr_str: str, local_dict: dict[str, Any]) -> Eq:
    """Parse equality of form a = b into SymPy Eq object."""
    if _RE_MULTIPLE_EQ.search(expr_str):
        raise ValueError("Multiple equal signs detected; chained equalities are not supported")
    left, right = expr_str.split("=", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError("Empty side in equality")
    lhs = parse_expr(left, local_dict=local_dict.copy(), transformations=_TRANS)
    rhs = parse_expr(right, local_dict=local_dict.copy(), transformations=_TRANS)
    return Eq(lhs, rhs, evaluate=False)


def parse_single(
    expr_str: str, func_names: Collection[str] | None = None, indep_var: str | None = None
) -> Any:
    """Parse a single expression or equality."""
    local_dict = _build_local_dict(
        tuple(sorted(func_names)) if func_names else (), indep_var
    )
    if _has_eq_sign(expr_str):
        return _parse_equality(expr_str, local_dict)
    return parse_expr(expr_str, local_dict=local_dict.copy(), transformations=_TRANS)


@lru_cache(maxsize=512)
def parse(expr_str: str) -> Any:
    """Parse input string into SymPy expression, supporting comma-separated multi-expressions."""
    # Note: convert_xor transformer already handles ^ operator, no manual replacement needed
    preprocessed, func_names, indep_var = preprocess_ode(expr_str)
    parts = split_top_level(preprocessed, ",")
    if len(parts) > 1:
        local_dict = _build_local_dict(tuple(sorted(func_names)), indep_var)
        eqs: list[Any] = []
        for p in parts:
            if not p:
                continue
            if _has_eq_sign(p):
                eqs.append(_parse_equality(p, local_dict))
            else:
                eqs.append(parse_expr(p, local_dict=local_dict.copy(), transformations=_TRANS))
        return eqs
    return parse_single(preprocessed, func_names, indep_var)


def extract_vars(expr: Any) -> tuple[Symbol, ...]:
    """Extract free variables from expression (excluding constants)."""
    if isinstance(expr, bool):
        return ()
    if isinstance(expr, list):
        syms: set[Symbol] = set()
        for e in expr:
            if hasattr(e, "free_symbols"):
                syms.update(e.free_symbols)
        return tuple(sorted(syms - _CONSTANTS, key=lambda s: s.name))
    if not hasattr(expr, "free_symbols"):
        return ()
    return tuple(sorted(expr.free_symbols - _CONSTANTS, key=lambda s: s.name))


@lru_cache(maxsize=512)
def preprocess(raw: str) -> ParsedInput:
    """Preprocess raw input: extract domain, for-variables, and other metadata."""
    raw = raw.strip()
    domain = S.Complexes
    if _RE_DOMAIN_REAL.search(raw):
        domain = S.Reals
        raw = _RE_DOMAIN_REAL.sub("", raw)
    elif _RE_DOMAIN_COMPLEX.search(raw):
        raw = _RE_DOMAIN_COMPLEX.sub("", raw)

    for_vars: tuple[Symbol, ...] | None = None
    m = _RE_FOR_CLAUSE.search(raw)
    if m:
        prefix = raw[:m.start()]
        if prefix.count("(") == prefix.count(")"):
            inp = prefix.strip()
            vars_part = raw[m.end():].strip()
            raw = inp
            for_vars_list: list[Symbol] = []
            for v in vars_part.split(","):
                v = v.strip()
                if not v or not _RE_VALID_VAR.match(v):
                    raise ValueError(f"Invalid variable name in 'for' clause: {v!r}")
                for_vars_list.append(_SYM_MAP.get(v, symbols(v)))
            for_vars = tuple(for_vars_list)

    return ParsedInput(raw=raw, domain=domain, for_vars=for_vars)


def is_ode(expr: Any) -> bool:
    """Check if expression contains an ODE (ordinary differential equation)."""
    if isinstance(expr, bool):
        return False
    if isinstance(expr, list):
        return any(is_ode(e) for e in expr)
    if isinstance(expr, Eq):
        expr = expr.lhs - expr.rhs
    return hasattr(expr, "has") and expr.has(Derivative)


# ── Numerical ODE solving ───────────────────────────────────────────

def _get_derivative_info(d: Derivative) -> tuple[Expr, Symbol, int]:
    """Extract argument, variable, and order from Derivative object.

    Compatible with sympy 1.14.0 where Derivative.args[1] may be a Tuple.
    """
    arg = d.args[0]
    var_info = d.args[1]
    if isinstance(var_info, (tuple, SympyTuple)):
        return arg, var_info[0], int(var_info[1])
    return arg, var_info, 1


@lru_cache(maxsize=128)
def _lambdify_cached(args: tuple[Any, ...], exprs: tuple[Any, ...], modules: str = "numpy") -> Callable[..., Any]:
    """Cache lambdify results to avoid repeated compilation.

    Parameters
    ----------
    args : tuple
        Symbol arguments for lambdify.
    exprs : tuple
        Expressions to compile.
    modules : str
        Numerical backend module, default "numpy".

    Returns
    -------
    Callable
        Compiled callable function.
    """
    fn: Callable[..., Any] = lambdify(list(args), list(exprs), modules=modules, cse=True, docstring_limit=0)
    return fn


# ── Numerical ODE solution wrapper ───────────────────────────────────

@dataclass(frozen=True, slots=True)
class OdeSolution:
    """Wrapper for scipy OdeResult to avoid dynamic attribute setting on immutable objects."""
    result: Any
    used_method: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.result, name)


def _build_rhs(
    ode_expr: Expr,
    f_var_name: str,
    t_sym: Symbol,
    max_order: int,
) -> tuple[Callable[..., NDArray[np.float64]] | None, str | None]:
    Symbol(f_var_name)
    Function(f_var_name)(t_sym)
    y_sym = Symbol(f_var_name)
    f_var = Function(f_var_name)(t_sym)

    deriv_expr = Derivative(f_var, (t_sym, max_order))
    sols = solve(ode_expr, deriv_expr, dict=True)
    if not sols:
        try:
            eq_form = Eq(ode_expr, 0) if not isinstance(ode_expr, Eq) else ode_expr
            sols = solve(eq_form, deriv_expr, dict=True)
        except (NotImplementedError, TypeError):
            pass
        if not sols:
            return None, "Cannot solve for highest order derivative"

    if len(sols) > 1:
        logger.warning(
            "ODE has %d branches for the highest-order derivative; using the first branch. "
            "Other branches may represent physically distinct solutions.",
            len(sols),
        )

    rhs_expr = sols[0][deriv_expr]

    if max_order == 1:
        rhs_expr = rhs_expr.subs(f_var, y_sym)
        # Defense: replace possible remaining bare symbol y (not y(t))
        dependent_sym = next(
            (s for s in rhs_expr.free_symbols if s.name == f_var_name),
            None
        )
        if dependent_sym is not None:
            rhs_expr = rhs_expr.subs(dependent_sym, y_sym)
        undefined = rhs_expr.free_symbols - {t_sym, y_sym}
        if undefined:
            return None, "Undefined parameters: " + ", ".join(str(s) for s in undefined)
        try:
            f_num = _lambdify_cached((t_sym, y_sym), (rhs_expr,))
        except (TypeError, ValueError) as e:
            return None, f"lambdify failed: {e}"

        def ode_func(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray([f_num(t, y[0])], dtype=float)

        return ode_func, None

    state_syms = tuple(Dummy(f"__calc_state_{i}") for i in range(max_order))
    subs_map: dict[Any, Any] = {f_var: state_syms[0]}
    # Explicitly look up dependent variable name from expression free symbols to avoid cache assumption conflicts
    dependent_sym = next(
        (s for s in ode_expr.free_symbols if s.name == f_var_name),
        Symbol(f_var_name)
    )
    subs_map[dependent_sym] = state_syms[0]
    # Critical fix: search Derivative objects in rhs_expr, not ode_expr
    for d in rhs_expr.find(Derivative):
        arg, var, order = _get_derivative_info(d)
        if arg == f_var and var == t_sym and 1 <= order < max_order:
            subs_map[d] = state_syms[order]
    rhs_expr_sub = rhs_expr.subs(subs_map)
    undefined = rhs_expr_sub.free_symbols - {t_sym} - set(state_syms)
    if undefined:
        return None, "Undefined parameters: " + ", ".join(str(s) for s in undefined)
    f_exprs = list(state_syms[1:]) + [rhs_expr_sub]
    try:
        f_num = _lambdify_cached((t_sym,) + state_syms, tuple(f_exprs))
    except (TypeError, ValueError) as e:
        return None, f"lambdify failed: {e}"

    def _ode_func_high(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        val = f_num(t, *y)
        return np.asarray(val, dtype=float).ravel()

    return _ode_func_high, None


def solve_ivp_numerical(
    ode_expr: Any,
    t_span: tuple[float, float] = (0.0, 10.0),
    initial_conditions: float | Collection[float] | None = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> Any:
    """Solve ODE numerically using scipy.integrate.solve_ivp.

    Parameters
    ----------
    ode_expr : sympy expression or Eq
        ODE expression containing Derivative.
    t_span : tuple[float, float]
        Integration interval (t0, tf).
    initial_conditions : float | Collection[float] | None
        Initial condition vector. Scalar for first-order ODE, sequence for higher-order.
    method : str
        Integration method, case-insensitive. Options: RK45, RK23, DOP853, Radau, BDF, LSODA.
    rtol, atol : float
        Relative and absolute tolerances.
    max_step : float | None
        Maximum step size, auto-calculated if None.

    Returns
    -------
    OdeSolution | str
        OdeSolution wrapper on success (with .success, .t, .y, .sol attributes),
        error message string on failure.
    """
    method = method.upper()
    if method not in _VALID_ODE_METHODS:
        return (
            f"Invalid ODE method: {method!r}. "
            f"Choose from {sorted(_VALID_ODE_METHODS)}"
        )
    if isinstance(ode_expr, Eq):
        ode_expr = ode_expr.lhs - ode_expr.rhs

    if not hasattr(ode_expr, "find"):
        return "Invalid ODE expression"

    derivs = [
        d for d in ode_expr.find(Derivative)
        if isinstance(d.args[0], AppliedUndef)
    ]
    if not derivs:
        return "No derivatives found in expression"

    func_names: set[str] = set()
    for d in derivs:
        arg = d.args[0]
        if isinstance(arg, AppliedUndef):
            func_names.add(str(arg.func))
    if len(func_names) > 1:
        return "Multiple dependent variables detected; system ODEs not supported for numerical solve"

    max_order = 1
    highest_deriv = None
    f_var = None
    t_sym = None
    for d in derivs:
        arg, var, order = _get_derivative_info(d)
        if order >= max_order:
            max_order = order
            highest_deriv = d
            f_var = arg
            t_sym = var

    if highest_deriv is None or t_sym is None or f_var is None:
        return "Could not determine ODE structure"

    # Pass the actual Symbol object to avoid reconstruction
    ode_func, err = _build_rhs(
        ode_expr, str(f_var.func), t_sym, max_order
    )
    if err:
        return err
    if ode_func is None:
        return "Cannot solve for highest order derivative"

    if initial_conditions is None:
        return "Numerical ODE requires initial conditions (use init=...)"

    t_span_len = abs(t_span[1] - t_span[0])
    if max_step is None:
        max_step = min(t_span_len / 50.0, 0.5)

    if isinstance(initial_conditions, (list, tuple)):
        y0 = list(initial_conditions)
        if len(y0) != max_order:
            return f"Initial conditions mismatch: expected {max_order} values for order-{max_order} ODE, got {len(y0)}"
    elif isinstance(initial_conditions, (int, float, np.floating, np.integer)):
        if max_order != 1:
            return f"Initial conditions mismatch: expected {max_order} values for order-{max_order} ODE, got 1"
        y0 = [float(initial_conditions)]
    else:
        return "Invalid initial_conditions type"

    solvers_to_try: list[str] = [method]
    if method in ("RK45", "RK23", "DOP853"):
        solvers_to_try.extend(["LSODA", "BDF", "Radau"])
    elif method == "LSODA":
        solvers_to_try.extend(["BDF", "Radau"])
    elif method in ("Radau", "BDF"):
        solvers_to_try.extend(["LSODA"] if method == "Radau" else ["Radau", "LSODA"])

    last_err: str = ""
    for solver in solvers_to_try:
        try:
            sol = solve_ivp(
                ode_func,
                t_span,
                y0,
                method=solver,
                rtol=rtol,
                atol=atol,
                dense_output=True,
                max_step=max_step,
            )
            if getattr(sol, "success", False):
                if solver != method:
                    logger.info("ODE solved successfully with fallback method '%s' (requested '%s')", solver, method)
                    return OdeSolution(result=sol, used_method=solver)
                return OdeSolution(result=sol, used_method=None)
            last_err = f"Solver '{solver}' returned without success"
        except Exception as exc:
            last_err = f"Solver '{solver}' failed: {exc}"
            continue

    return f"Numerical integration failed: {last_err}"


# ── Root solving helpers ─────────────────────────────────────────────

def _has_transcendental_atoms(expr: Any) -> bool:
    if not hasattr(expr, "has"):
        return False
    return bool(expr.has(*_TRANSCENDENTAL_FUNCS))


def _factor_roots(expr: Any, var: Symbol) -> list[Any] | None:
    """Try factoring then root finding."""
    try:
        fac = factor(expr)
    except (NotImplementedError, TypeError, ValueError, PolynomialError):
        return None
    if fac == expr:
        return None
    factors = list(fac.args) if isinstance(fac, Mul) else [fac]
    all_roots: set[Any] = set()
    for f in factors:
        base = f.base if isinstance(f, Pow) else f
        if getattr(base, "is_number", False):
            continue
        try:
            rts = solve(base, var, dict=True)
            if rts and not isinstance(rts, ConditionSet):
                for r in rts:
                    if isinstance(r, dict) and var in r:
                        all_roots.add(r[var])
        except (NotImplementedError, TypeError):
            continue
    return list(all_roots) if all_roots else None


def _verify_root(expr: Any, var: Symbol, root: Any, tol: float = TOL_VERIFY) -> bool:
    """Verify if numerical root satisfies original equation. Includes infinite value safety check."""
    try:
        residual = expr.subs(var, root).evalf(15)
        if residual is S.NaN:
            return False
        if isinstance(residual, Expr) and residual.is_number:
            # Key fix: check for finite value to avoid complex(oo) crash
            if hasattr(residual, "is_finite") and not residual.is_finite:
                return False
            return abs(complex(residual)) < tol
        return False
    except (TypeError, ValueError, OverflowError):
        return False


def _try_poly_roots(expr: Any, var: Symbol, domain: Any) -> list[Any] | None:
    """Try polynomial root finding, preferring real_roots (real domain) or nroots (complex domain)."""
    if _has_transcendental_atoms(expr):
        return None
    try:
        p = Poly(expr, var)
        deg = p.degree()
        if deg == 0:
            # Constant polynomial: zero means identity, non-zero means no solution
            if p.all_coeffs()[0] == 0:
                return ["All real numbers" if domain is S.Reals else "All complex numbers"]
            return ["No solution found"]
        if 0 < deg <= MAX_POLY_DEGREE:
            if domain is S.Reals:
                try:
                    rts = real_roots(p)
                    if rts:
                        return [r.evalf(EVALF_PREC) for r in rts]
                except (NotImplementedError, TypeError, ValueError):
                    pass
                roots = nroots(p)
                real_rts = []
                for r in roots:
                    rv = r.evalf(EVALF_PREC)
                    c = complex(rv)
                    # Numerical tolerance filtering: nroots returns approximate complex numbers,
                    # small imaginary part is treated as numerical noise.
                    # Different from symbolic path's is_real is not False semantics (numerical vs symbolic),
                    # but TOL_REAL_FILTER=1e-10 is sufficient to distinguish true complex roots from float noise.
                    if abs(c.imag) < TOL_REAL_FILTER:
                        real_rts.append(rv if c.imag == 0 else rv.as_real_imag()[0])
                return real_rts
            else:
                roots = nroots(p)
                if roots:
                    return [r.evalf(EVALF_PREC) for r in roots]
                return []
    except (PolynomialError, NotImplementedError, TypeError, ValueError):
        pass
    return None


def _normalize_root(val: Any, tol: float = TOL_NSOLVE_DEDUP) -> float | complex:
    c = complex(val)
    if abs(c.imag) < tol:
        return c.real
    return c


def _nsolve_single_real(expr: Any, var: Symbol, guess: float) -> float | None:
    """Try single real initial guess, return root or None. Thread-safe: no shared state."""
    try:
        sol = nsolve(expr, var, guess, tol=NSOLVE_TOL, maxsteps=100, prec=NSOLVE_PREC)
        if not hasattr(sol, "evalf"):
            return None
        sol_eval = sol.evalf(EVALF_PREC)
        # Do not rely on is_real attribute, directly compute complex imaginary part and compare with tolerance
        try:
            c = complex(sol_eval)
        except (TypeError, ValueError):
            return None
        if abs(c.imag) > TOL_REAL_FILTER:
            return None
        real_val = c.real
        if abs(real_val) > MAX_ROOT_ABS:
            return None
        if _verify_root(expr, var, sol):
            return real_val
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return None


def _nsolve_single_complex(expr: Any, var: Symbol, guess: complex) -> complex | None:
    """Try single complex initial guess, return root or None. Thread-safe: no shared state."""
    try:
        sol = nsolve(expr, var, guess, tol=NSOLVE_TOL, maxsteps=100, prec=NSOLVE_PREC)
        if not hasattr(sol, "evalf"):
            return None
        cplx_val = complex(sol.evalf(EVALF_PREC))
        if abs(cplx_val) > MAX_ROOT_ABS:
            return None
        if _verify_root(expr, var, sol):
            return cplx_val
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return None


def find_roots_nsolve(expr: Any, var: Symbol, domain: Any) -> list[Any]:
    """Search numerical roots from multiple initial guesses using nsolve. Serial execution to avoid SymPy thread-safety issues."""
    if isinstance(expr, Eq):
        expr = expr.lhs - expr.rhs

    tol = TOL_NSOLVE_DEDUP
    roots: set[float | complex] = set()

    if domain is S.Reals:
        for g in _NSOLVE_REAL_GUESSES:
            root: float | complex | None = _nsolve_single_real(expr, var, g)
            if root is not None:
                norm = _normalize_root(root, tol)
                if all(abs(norm - _normalize_root(r, tol)) > tol for r in roots):
                    roots.add(norm)
    else:
        for cg in _NSOLVE_COMPLEX_GUESSES:
            root = _nsolve_single_complex(expr, var, cg)
            if root is not None:
                norm = _normalize_root(root, tol)
                if all(abs(norm - _normalize_root(r, tol)) > tol for r in roots):
                    roots.add(norm)

    return list(roots)


def _is_expr_numeric_zero(expr: Any) -> bool:
    if isinstance(expr, bool):
        return False
    if isinstance(expr, Expr) and expr.is_zero is not None:
        return bool(expr.is_zero)
    try:
        return bool(expr == 0)
    except Exception:
        return False


def _is_polynomial_like(expr: Any) -> bool:
    if not hasattr(expr, "has"):
        return False
    return not expr.has(*_TRANSCENDENTAL_FUNCS)


# ── Core solving functions ──────────────────────────────────────────

def _try_symbolic_solve(expr: Any, var: Symbol, is_real_domain: bool) -> list[Any] | None:
    """Try symbolic solving, return solution list or None for failure/no valid solutions."""
    try:
        sol = solve(expr, var, dict=True)
        match sol:
            case dict() if var in sol:
                solutions = [sol[var]]
                if is_real_domain:
                    solutions = [s for s in solutions if s.is_real is not False]
                return solutions if solutions else None
            case list() if sol and not isinstance(sol, ConditionSet):
                solutions = [s[var] for s in sol if isinstance(s, dict) and var in s]
                if is_real_domain:
                    solutions = [s for s in solutions if s.is_real is not False]
                return solutions if solutions else None
    except (NotImplementedError, TypeError) as exc:
        logger.debug("solve() failed for %s: %s", expr, exc)
    return None


def _try_solveset(expr: Any, var: Symbol, domain: Any) -> Any | None:
    """Try solveset set-based solving."""
    try:
        sol_set = solveset(expr, var, domain)
        match sol_set:
            case FiniteSet():
                return list(sol_set)
            case _ if sol_set is S.EmptySet:
                return None
            case _ if sol_set is S.Reals:
                return "All real numbers"
            case _ if sol_set is S.Complexes:
                return "All complex numbers"
            case Union() | Interval() | ImageSet():
                return sol_set
            case ConditionSet():
                return f"Parametric solution: {sol_set}"
    except (NotImplementedError, TypeError) as exc:
        logger.debug("solveset() failed for %s: %s", expr, exc)
    return None


def solve_one(expr: Any, var: Symbol, domain: Any) -> Any:
    """Solve single-variable equation/expression.

    Solving strategy (by priority):
    1. Zero expression -> universal solution
    2. Polynomial roots (real_roots / nroots), including constant short-circuit
    3. Factorization root finding
    4. solve symbolic solving
    5. solveset set-based solving
    6. nsolve numerical fallback
    """
    if isinstance(expr, Eq):
        expr = expr.lhs - expr.rhs

    if _is_expr_numeric_zero(expr):
        return "All real numbers" if domain is S.Reals else "All complex numbers"

    poly_roots = _try_poly_roots(expr, var, domain)
    if poly_roots is not None:
        if not poly_roots:
            return "No real roots" if domain is S.Reals else "No complex roots"
        # Constant polynomial short-circuit result (string marker)
        if len(poly_roots) == 1 and isinstance(poly_roots[0], str):
            return poly_roots[0]
        return poly_roots

    is_transcendental = _has_transcendental_atoms(expr)
    is_real_domain = domain is S.Reals

    if not is_transcendental:
        if _is_polynomial_like(expr) and len(str(expr)) < 500:
            fac_roots = _factor_roots(expr, var)
            if fac_roots:
                if is_real_domain:
                    fac_roots = [r for r in fac_roots if r.is_real is not False]
                if fac_roots:
                    return fac_roots
                return "No real roots"

        sym_sol = _try_symbolic_solve(expr, var, is_real_domain)
        if sym_sol is not None:
            return sym_sol

        sol_set = _try_solveset(expr, var, domain)
        if sol_set is not None:
            return sol_set

    # Transcendental equations: try symbolic solve first (e.g., cos(x)=sin(x) can be solved directly)
    sym_sol = _try_symbolic_solve(expr, var, is_real_domain)
    if sym_sol is not None:
        return sym_sol

    num_roots = find_roots_nsolve(expr, var, domain)
    if num_roots:
        return num_roots
    return "No solution found"


def solve_multi(expr: Any, vars_list: Collection[Symbol]) -> Any:
    """Solve multi-variable equation system."""
    match expr:
        case Eq():
            eqs = [expr]
        case list():
            eqs = expr
        case _:
            eqs = [Eq(expr, 0, evaluate=False)]

    eqs_zero = []
    for e in eqs:
        if isinstance(e, Eq):
            eqs_zero.append(e.lhs - e.rhs)
        else:
            eqs_zero.append(e)

    try:
        sol = linsolve(eqs_zero, vars_list)
        if isinstance(sol, FiniteSet) and sol:
            return [dict(zip(vars_list, tup)) for tup in sol]
    except (NotImplementedError, TypeError, ValueError) as exc:
        logger.debug("linsolve() failed: %s", exc)

    try:
        sol = solve(eqs, *vars_list, dict=True)
        match sol:
            case dict():
                return [sol]
            case list() if sol and not isinstance(sol, ConditionSet):
                return sol
    except (NotImplementedError, TypeError) as exc:
        logger.debug("solve() failed for system: %s", exc)

    try:
        sol = nonlinsolve(eqs, vars_list)
        match sol:
            case FiniteSet() if sol and not isinstance(sol, ConditionSet):
                return [{v: val for v, val in zip(vars_list, s)} for s in sol]
    except (NotImplementedError, TypeError) as exc:
        logger.debug("nonlinsolve() failed: %s", exc)

    if len(vars_list) == len(eqs):
        try:
            guesses = [0.5] * len(vars_list)
            sol = nsolve(eqs_zero, vars_list, guesses, tol=NSOLVE_TOL, maxsteps=100, prec=NSOLVE_PREC)
            return [{v: sol[i] for i, v in enumerate(vars_list)}]
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            logger.debug("nsolve() failed for system: %s", exc)

    return "No solution found"


def _ics_key(item: tuple[Any, Any]) -> tuple[int, float, str]:
    """Sort initial conditions by key (derivative order, evaluation point numeric, evaluation point string).

    Numeric points sorted by float for correct numerical order (10.0 > 2.0),
    symbolic points (non-floatable) use inf to place after numeric points, sorted by string.
    """
    k = item[0]
    if isinstance(k, Subs):
        d = k.args[0]
        pt_raw = k.args[2] if len(k.args) > 2 else (0,)
        pt = pt_raw[0] if isinstance(pt_raw, (tuple, SympyTuple)) else pt_raw
        try:
            pt_num = float(pt)
        except (TypeError, ValueError):
            pt_num = float("inf")
        pt_str = str(pt)
        order = _get_derivative_info(d)[2] if isinstance(d, Derivative) else 0
        return order, pt_num, pt_str
    if isinstance(k, AppliedUndef):
        pt = k.args[0] if k.args else 0
        try:
            pt_num = float(pt)
        except (TypeError, ValueError):
            pt_num = float("inf")
        pt_str = str(pt)
        return 0, pt_num, pt_str
    return 0, float("inf"), ""


# ── Option extraction and ODE solving ───────────────────────────────

def _is_inside_brackets(text: str, start: int, end: int) -> bool:
    """Check if range [start, end) is fully inside any bracket pair."""
    depth = 0
    for i, ch in enumerate(text):
        if i >= end:
            break
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if start <= i < end and depth > 0:
            return True
    return False


def _extract_options(inp: str) -> tuple[str, SolverOptions]:
    """Extract solver options from input string.

    Uses interval merging strategy: locate all options and record their text intervals,
    remove from back to front to avoid index shifting.
    Additional bracket context validation: skip match if interval is inside brackets,
    avoiding accidental damage to expression content (e.g., variable name numeric or parameter init).
    Supports arbitrary order and mixed delimiter (comma or space) option combinations.
    """
    raw = inp.strip()

    numeric_mode = False
    init_val: list[float] | None = None
    t_span: tuple[float, float] = (0.0, 10.0)
    method = "RK45"
    rtol = 1e-6
    atol = 1e-9
    max_step: float | None = None
    intervals: list[tuple[int, int]] = []

    # numeric flag
    m = _RE_NUMERIC_FLAG.search(raw)
    if m and not _is_inside_brackets(raw, m.start(), m.end()):
        numeric_mode = True
        intervals.append((m.start(), m.end()))

    # init=...
    m = _RE_INIT.search(raw)
    if m and not _is_inside_brackets(raw, m.start(), m.end()):
        init_str = m.group(1)
        try:
            init_val = [float(v.strip()) for v in init_str.split(",")]
        except ValueError as exc:
            raise ValueError(f"Invalid init value: '{init_str.strip()}'") from exc
        intervals.append((m.start(), m.end()))

    # tspan=...
    m = _RE_TSPAN.search(raw)
    if m and not _is_inside_brackets(raw, m.start(), m.end()):
        try:
            t_span = (float(m.group(1)), float(m.group(2)))
        except ValueError as exc:
            raise ValueError(f"Invalid tspan: {m.group(0)}") from exc
        intervals.append((m.start(), m.end()))

    # method=...
    m = _RE_METHOD.search(raw)
    if m and not _is_inside_brackets(raw, m.start(), m.end()):
        method = m.group(1).upper()  # normalize to uppercase
        intervals.append((m.start(), m.end()))

    # rtol=...
    m = _RE_RTOL.search(raw)
    if m and not _is_inside_brackets(raw, m.start(), m.end()):
        rtol = float(m.group(1))
        intervals.append((m.start(), m.end()))

    # atol=...
    m = _RE_ATOL.search(raw)
    if m and not _is_inside_brackets(raw, m.start(), m.end()):
        atol = float(m.group(1))
        intervals.append((m.start(), m.end()))

    # max_step=...
    m = _RE_MAXSTEP.search(raw)
    if m and not _is_inside_brackets(raw, m.start(), m.end()):
        max_step = float(m.group(1))
        intervals.append((m.start(), m.end()))

    # Remove from back to front to avoid index shifting
    for start, end in sorted(intervals, reverse=True):
        raw = raw[:start] + " " + raw[end:]

    # Clean up residual spaces and commas
    raw = raw.strip(" ,")
    raw = _RE_WHITESPACE.sub(" ", raw)
    raw = _RE_OPTION_DELIM.sub(",", raw)
    raw = raw.strip(" ,")

    # Validate method name
    if method not in _VALID_ODE_METHODS:
        raise ValueError(
            f"Invalid ODE method: {method!r}. "
            f"Choose from {sorted(_VALID_ODE_METHODS)}"
        )

    options = SolverOptions(
        numeric_mode=numeric_mode,
        init_val=init_val,
        t_span=t_span,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    return raw, options


def _extract_ics_from_expr(expr: Any) -> dict[Any, Any]:
    """Extract initial conditions from single expression or expression list."""
    ics: dict[Any, Any] = {}
    items = expr if isinstance(expr, list) else [expr]
    for e in items:
        if not isinstance(e, Eq):
            continue
        if isinstance(e.lhs, Subs):
            ics[e.lhs] = e.rhs
        elif isinstance(e.lhs, AppliedUndef):
            ics[e.lhs] = e.rhs
    return ics


def _solve_ode_system(expr_list: list[Any], options: SolverOptions) -> Any:
    """Solve ODE system (currently supports single ODE only)."""
    ode_eqs: list[Eq] = []
    ics: dict[Any, Any] = {}

    for e in expr_list:
        if not isinstance(e, Eq):
            if hasattr(e, "has") and e.has(Derivative):
                e = Eq(e, 0, evaluate=False)
            else:
                continue
        if isinstance(e.lhs, Subs):
            ics[e.lhs] = e.rhs
        elif isinstance(e.lhs, AppliedUndef):
            ics[e.lhs] = e.rhs
        elif e.has(Derivative):
            ode_eqs.append(e)

    if not ode_eqs:
        return "No differential equation found"
    if len(ode_eqs) > 1:
        return "Systems of multiple differential equations are not supported yet. Please provide a single ODE."

    ode_eq = ode_eqs[0]

    if options.numeric_mode:
        init_val = options.init_val
        if init_val is None and ics:
            try:
                sorted_ics = sorted(ics.items(), key=_ics_key)
                init_val = [
                    float(v.evalf()) if hasattr(v, "evalf") else float(v)
                    for _, v in sorted_ics
                ]
            except (ValueError, TypeError):
                return "Invalid initial conditions"
        elif init_val is None:
            return "Numerical ODE requires initial conditions (init=...)"

        numeric_solution = solve_ivp_numerical(
            ode_eq,
            t_span=options.t_span,
            initial_conditions=init_val,
            method=options.method,
            rtol=options.rtol,
            atol=options.atol,
            max_step=options.max_step,
        )
        if isinstance(numeric_solution, str):
            return numeric_solution
        if isinstance(numeric_solution, OdeSolution):
            return numeric_solution
        if numeric_solution is not None and getattr(numeric_solution, "success", False):
            return numeric_solution
        return "Numerical solution failed"

    try:
        if ics:
            sol = dsolve(ode_eq, ics=ics, simplify=False)
        else:
            sol = dsolve(ode_eq, simplify=False)
        return cancel(expand(sol))
    except (NotImplementedError, TypeError) as exc:
        logger.debug("dsolve() failed: %s", exc)
        return "ODE analytical solution failed"


def max_order_from_expr(ode_eq: Any) -> int:
    """Extract the highest derivative order from ODE expression."""
    if not hasattr(ode_eq, "find"):
        return 1
    max_order = 1
    for d in ode_eq.find(Derivative):
        if isinstance(d.args[0], AppliedUndef):
            _, _, order = _get_derivative_info(d)
            max_order = max(max_order, order)
    return max_order


def _dispatch_ode(expr: Any, options: SolverOptions) -> Any:
    """Dispatch ODE solving based on options."""
    if isinstance(expr, list):
        return _solve_ode_system(expr, options)

    ics = _extract_ics_from_expr(expr)
    if ics:
        return _solve_ode_system([expr], options)

    if options.numeric_mode:
        numeric_solution = solve_ivp_numerical(
            expr,
            t_span=options.t_span,
            initial_conditions=options.init_val,
            method=options.method,
            rtol=options.rtol,
            atol=options.atol,
            max_step=options.max_step,
        )
        if isinstance(numeric_solution, str):
            return numeric_solution
        if isinstance(numeric_solution, OdeSolution):
            return numeric_solution
        if numeric_solution is not None and getattr(numeric_solution, "success", False):
            return numeric_solution
        return "Numerical solution failed"
    try:
        sol = dsolve(expr, simplify=False)
        return cancel(expand(sol))
    except (NotImplementedError, TypeError) as exc:
        logger.debug("dsolve() failed: %s", exc)
        return "ODE analytical solution failed"


def solve_problem(inp: str) -> Any:
    """Main solver entry: parse input and dispatch to corresponding solver.

    Parameters
    ----------
    inp
        User input mathematical expression string.

    Returns
    -------
    Any
        Solver result, type depends on problem type.
    """
    if not inp or not inp.strip():
        return "Input is empty"

    if len(inp) > MAX_INPUT_LEN:
        return f"Input too long (max {MAX_INPUT_LEN} characters)"

    # Intercept direct input of oo (infinity) as solving target
    stripped = inp.strip().lower()
    if stripped in ("oo", "inf", "infinity", "+oo", "-oo", "-inf", "-infinity"):
        return "Infinity is not a valid equation to solve"

    try:
        clean_inp, options = _extract_options(inp)
    except ValueError as e:
        return str(e)

    if not clean_inp:
        return "Input is empty"

    try:
        parsed = preprocess(clean_inp)
        expr = parse(parsed.raw)
    except Exception as e:
        logger.exception("Parsing error for input: %s", inp)
        return f"Parsing error: {e}"

    match expr:
        case bool():
            return expr
        case _ if is_ode(expr):
            return _dispatch_ode(expr, options)
        case Relational() if not isinstance(expr, Equality):
            vars_list = extract_vars(expr)
            if vars_list:
                try:
                    return reduce_inequalities(expr, list(vars_list))
                except (NotImplementedError, TypeError, ValueError) as exc:
                    logger.debug("reduce_inequalities() failed: %s", exc)
                    return "Inequality solving failed"
            try:
                return reduce_inequalities(expr)
            except (NotImplementedError, TypeError, ValueError) as exc:
                logger.debug("reduce_inequalities() failed: %s", exc)
                return "Inequality solving failed"

    vars_list = parsed.for_vars if parsed.for_vars is not None else extract_vars(expr)
    if not vars_list:
        return expr

    if isinstance(expr, list):
        return solve_multi(expr, vars_list)

    if len(vars_list) == 1:
        return solve_one(expr, vars_list[0], parsed.domain)
    return solve_multi(expr, vars_list)


# ── Formatting output ────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def _evalf_cached(expr: Expr, n: int = EVALF_PREC) -> Any:
    return expr.evalf(n)


def approx_of(x: Any, n: int = EVALF_PREC) -> str:
    match x:
        case Expr():
            return fmt_item(_evalf_cached(x, n))
    return fmt_item(x)


def _format_list_result(result: list[Any] | tuple[Any, ...]) -> str:
    lines = []
    for i, item in enumerate(result):
        match item:
            case dict():
                lines.append(f"Solution {i+1}:")
                for k, v in item.items():
                    exact = fmt_item(v)
                    approx = approx_of(v)
                    if exact == approx:
                        lines.append(f"  {k} = {exact}")
                    else:
                        lines.append(f"  {k} = {exact}  ≈  {approx}")
            case list() | tuple():
                lines.append(f"Group {i+1}:")
                lines.extend(_format_list_result(item).splitlines())
            case _:
                exact = fmt_item(item)
                approx = approx_of(item)
                if exact == approx:
                    lines.append(f"[{i}] {exact}")
                else:
                    lines.append(f"[{i}] {exact}  ≈  {approx}")
    return "\n".join(lines)


def _format_numerical_solution(result: Any, used_method: str | None = None) -> str:
    """Format numerical ODE solution result."""
    # result is the underlying OdeResult, used_method passed by caller

    if not result.success:
        return "Numerical solution failed"
    t = getattr(result, "t", None)
    y = getattr(result, "y", None)
    sol = getattr(result, "sol", None)
    if t is None or y is None or len(y) == 0 or len(t) == 0:
        return "Numerical solution failed"

    t_arr = np.asarray(t)
    y_arr = np.asarray(y)
    t0 = float(t_arr[0])
    t1 = float(t_arr[-1])
    n_components = y_arr.shape[0]

    parts = [f"Numerical solution: t∈[{t0:.2f}, {t1:.2f}]"]

    if used_method is not None:
        parts.append(f"(used: {used_method})")

    for comp in range(n_components):
        y_comp = y_arr[comp]
        y_min = float(np.min(y_comp))
        y_max = float(np.max(y_comp))
        label = f"y{comp}" if comp > 0 else "y"
        parts.append(f"{label}∈[{y_min:.4f}, {y_max:.4f}]")

    if sol is not None and callable(sol):
        t_mid = (t0 + t1) / 2
        y_mid = sol(t_mid)
        if isinstance(y_mid, np.ndarray):
            mid_vals = ", ".join(f"{float(y_mid[i]):.6f}" for i in range(n_components))
            parts.append(f"y({t_mid:.2f})≈[{mid_vals}]")
        else:
            parts.append(f"y({t_mid:.2f})≈{float(y_mid):.6f}")

    return ", ".join(parts)


def fmt_with_approx(result: Any) -> str:
    if isinstance(result, OdeSolution):
        return _format_numerical_solution(result.result, result.used_method)
    match result:
        case None:
            return "No solution"
        case str():
            return result
        case Eq():
            exact = fmt_item(result)
            approx_rhs = approx_of(result.rhs)
            return f"{exact}  ≈  {result.lhs} = {approx_rhs}"
        case list() | tuple():
            return _format_list_result(result)
        case dict():
            d_exact = {k: fmt_item(v) for k, v in result.items()}
            d_approx = {k: approx_of(v) for k, v in result.items()}
            lines = ["Solution:"]
            for k in d_exact:
                if d_exact[k] == d_approx[k]:
                    lines.append(f"  {k} = {d_exact[k]}")
                else:
                    lines.append(f"  {k} = {d_exact[k]}  ≈  {d_approx[k]}")
            return "\n".join(lines)
        case set() | frozenset():
            items = []
            for v in result:
                exact = fmt_item(v)
                approx = approx_of(v)
                if exact == approx:
                    items.append(exact)
                else:
                    items.append(f"{exact}  ≈  {approx}")
            return "{" + ", ".join(items) + "}"
        case _ if result is S.Reals:
            return "All real numbers"
        case _ if result is S.Complexes:
            return "All complex numbers"
        case Union() | Interval() | ImageSet():
            return str(result)
        case _:
            exact = fmt_item(result)
            approx = approx_of(result)
            if exact == approx:
                return exact
            return f"{exact}  ≈  {approx}"


def _mp_nstr(val: Any) -> str:
    if isinstance(val, mp.mpf) and not mp.isfinite(val):
        return str(val)
    s = mp.nstr(val, MP_NSTR_DIGITS, strip_zeros=True).rstrip(".")
    return "0" if s in ("", "-0") else s


def _safe_float_str(x: float) -> str:
    if not np.isfinite(x):
        if np.isposinf(x):
            return "∞"
        if np.isneginf(x):
            return "-∞"
        return "nan"
    if x == int(x):
        return str(int(x))
    return format(x, ".12g").rstrip(".") if abs(x) >= TOL_FMT_ZERO else "0"


def _format_complex_pair(re_val: float, im_val: float) -> str:
    if abs(im_val) < TOL_FMT_ZERO:
        return _safe_float_str(re_val)
    if abs(re_val) < TOL_FMT_ZERO:
        if im_val < 0:
            return f"-{_safe_float_str(-im_val)}*I"
        return f"{_safe_float_str(im_val)}*I"
    re_str = _safe_float_str(re_val)
    if im_val < 0:
        return f"{re_str} - {_safe_float_str(-im_val)}*I"
    return f"{re_str} + {_safe_float_str(im_val)}*I"


def fmt_item(x: Any) -> str:
    if isinstance(x, Expr):
        if x is I:
            return "I"
        if x is oo:
            return "∞"
        if x is -oo:
            return "-∞"
        if x is zoo:
            return "∞̃"
        if x is nan:
            return "nan"
        if x.is_Number:
            if x.is_real:
                try:
                    f = float(x)
                    return _safe_float_str(f)
                except (OverflowError, ValueError):
                    pass
                try:
                    return _mp_nstr(mp.mpf(str(_evalf_cached(x, EVALF_PREC))))
                except (TypeError, ValueError):
                    return str(x)
            re_part, im_part = x.as_real_imag()
            im_val = float(_evalf_cached(im_part, EVALF_PREC))
            if abs(im_val) < TOL_FMT_ZERO:
                return fmt_item(re_part)
            re_str = fmt_item(re_part)
            if im_val < 0:
                return f"{re_str} - {_safe_float_str(-im_val)}*I"
            return f"{re_str} + {_safe_float_str(im_val)}*I"
        return str(x)

    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        return _safe_float_str(float(x))

    if isinstance(x, complex):
        return _format_complex_pair(x.real, x.imag)

    if isinstance(x, mp.mpf):
        return _mp_nstr(x)

    if isinstance(x, mp.mpc):
        if mp.almosteq(x.imag, 0, abs_eps=TOL_FMT_ZERO):
            return fmt_item(x.real)
        if mp.almosteq(x.real, 0, abs_eps=TOL_FMT_ZERO):
            return _format_complex_pair(0.0, float(x.imag))
        return _format_complex_pair(float(x.real), float(x.imag))

    return str(x)


# ── Unit tests ───────────────────────────────────────────────────────

class TestSmartCalculator(unittest.TestCase):
    """Regression test suite covering key solving paths and edge cases."""

    def test_polynomial_quadratic(self) -> None:
        res = solve_problem("x**2 - 5*x + 6 = 0")
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2)
        fmt = fmt_with_approx(res)
        self.assertIn("2", fmt)
        self.assertIn("3", fmt)

    def test_polynomial_cubic_complex(self) -> None:
        res = solve_problem("x**3 - 1 = 0")
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 3)

    def test_constant_polynomial_no_solution(self) -> None:
        res = solve_problem("1 = 0 for x")
        self.assertEqual(res, "No solution found")

    def test_zero_polynomial(self) -> None:
        res = solve_problem("0 = 0 for x")
        self.assertEqual(res, "All complex numbers")

    def test_ode_analytical_first_order(self) -> None:
        res = solve_problem("y' + y = 0")
        self.assertIsInstance(res, Eq)
        self.assertIn("C1", fmt_with_approx(res))

    def test_ode_analytical_second_order_with_ics(self) -> None:
        res = solve_problem("y'' + y = 0, y(0)=1, y'(0)=0")
        self.assertIsInstance(res, Eq)
        self.assertIn("cos(x)", fmt_with_approx(res))

    def test_ode_numeric_first_order(self) -> None:
        res = solve_problem("y' + y = 0 numeric init=1")
        self.assertTrue(hasattr(res, "success"))
        self.assertTrue(res.success)

    def test_ode_numeric_second_order(self) -> None:
        res = solve_problem("y'' + y = 0 numeric init=1,0")
        self.assertTrue(hasattr(res, "success"))
        self.assertTrue(res.success)

    def test_ode_numeric_invalid_method(self) -> None:
        res = solve_problem("y' + y = 0 numeric init=1 method=INVALID")
        self.assertIn("Invalid ODE method", res)

    def test_system_linear(self) -> None:
        res = solve_problem("x + y = 10, 2*x - y = 5")
        self.assertIsInstance(res, list)
        self.assertEqual(res[0][Symbol("x")], 5)
        self.assertEqual(res[0][Symbol("y")], 5)

    def test_inequality(self) -> None:
        res = solve_problem("sin(x) > 0.5 domain=real")
        self.assertIsNotNone(res)

    def test_derivative_preprocessing(self) -> None:
        expr, funcs, iv = preprocess_ode("y'' + y = 0")
        self.assertIn("y", funcs)
        self.assertEqual(iv, "x")

    def test_mixed_option_delimiters(self) -> None:
        raw, opts = _extract_options("y' + y = 0 numeric, init=1, method=RK45")
        self.assertTrue(opts.numeric_mode)
        self.assertEqual(opts.init_val, [1.0])
        self.assertEqual(opts.method, "RK45")

    def test_transcendental_equation(self) -> None:
        res = solve_problem("cos(x) = sin(x)")
        self.assertIsInstance(res, list)
        fmt = fmt_with_approx(res)
        self.assertIn("0.785", fmt)

    def test_multiple_eq_rejection(self) -> None:
        res = solve_problem("x = 0 = 1")
        self.assertIn("Parsing error", res)

    def test_input_too_long(self) -> None:
        res = solve_problem("x" * 5001)
        self.assertEqual(res, "Input too long (max 5000 characters)")

    def test_infinity_input(self) -> None:
        res = solve_problem("oo")
        self.assertEqual(res, "Infinity is not a valid equation to solve")

    def test_numeric_flag_via_constant(self) -> None:
        raw, opts = _extract_options("numeric")
        self.assertTrue(opts.numeric_mode)
        self.assertEqual(raw, "")

    def test_high_order_ode_bare_symbol(self) -> None:
        # Verify bare symbol y is correctly replaced by state variable in high-order ODE
        expr_str, funcs, iv = preprocess_ode("y'' + y = 0")
        self.assertIn("y", funcs)
        res = solve_problem("y'' + y = 0 numeric init=1,0")
        self.assertTrue(hasattr(res, "success"))
        self.assertTrue(res.success)

    def test_complex_domain(self) -> None:
        res = solve_problem("x**2 + 1 = 0 domain=complex")
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2)

    def test_high_order_ode_third_order(self) -> None:
        res = solve_problem("y''' = y numeric init=1,1,1")
        self.assertTrue(hasattr(res, "success"))
        self.assertTrue(res.success)

    def test_split_top_level_escaped_quote(self) -> None:
        # Test escaped quote handling
        parts = split_top_level('a, "b\\",c", d')
        self.assertEqual(parts, ['a', '"b\\",c"', 'd'])

    def test_split_top_level_double_backslash(self) -> None:
        # Test double backslash (escaped backslash followed by quote end)
        parts = split_top_level('a, "C:\\\\", d')
        self.assertEqual(parts, ['a', '"C:\\\\"', 'd'])

    def test_split_top_level_single_backslash(self) -> None:
        # Standard escape rule: \\ -> literal \, then " ends string
        parts = split_top_level('a, "b\\\\",c", d')
        self.assertEqual(parts, ['a', '"b\\\\"', 'c", d'])

    def test_split_top_level_mismatched_bracket(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            split_top_level("(a, b")
        self.assertIn("Mismatched", str(ctx.exception))

    def test_fallback_method_hint(self) -> None:
        # Force fallback: specify a method that may fail for current problem
        # Here we mainly verify formatting output correctly displays used_method
        res = solve_problem("y' + y = 0 numeric init=1 method=RK45")
        self.assertTrue(hasattr(res, "success"))
        self.assertTrue(res.success)
        fmt = fmt_with_approx(res)
        # RK45 usually succeeds, so fallback is not triggered; do not check used_method
        self.assertIn("Numerical solution", fmt)

    def test_option_inside_brackets_ignored(self) -> None:
        # numeric inside brackets should be ignored; init=5 outside, should be extracted
        raw, opts = _extract_options("f(numeric) + init=5")
        self.assertFalse(opts.numeric_mode)
        self.assertEqual(opts.init_val, [5.0])
        self.assertEqual(raw.strip(), "f(numeric) +")

    def test_solve_multi_no_solution(self) -> None:
        res = solve_problem("x + y = 0, x + y = 1")
        self.assertEqual(res, "No solution found")

    def test_for_and_domain_mixed(self) -> None:
        res = solve_problem("x**2 + y**2 = 1 for x, y domain=real")
        self.assertIsInstance(res, list)

    def test_func_pattern_no_double_paren(self) -> None:
        # Verify _get_func_pattern does not turn y(x) into y(x)(x)
        pat = _get_func_pattern("y")
        self.assertIsNotNone(pat)
        assert pat is not None  # for mypy
        test = "Derivative(y(x), (x, 2)) + y = 0"
        result = pat.sub("y(x)", test)
        self.assertNotIn("y(x)(x)", result)
        self.assertIn("Derivative(y(x), (x, 2))", result)

    def test_transcendental_symbolic_first(self) -> None:
        # Transcendental equations should try symbolic solve first to avoid unnecessary nsolve
        res = solve_problem("cos(x) = sin(x)")
        self.assertIsInstance(res, list)
        fmt = fmt_with_approx(res)
        self.assertIn("0.785", fmt)

    def test_nsolve_reduced_guesses(self) -> None:
        # Verify nsolve returns valid results under fixed guess set
        cases = [
            "exp(x) = 2",
            "x**3 - 1 = 0",
            "sin(x) = 0",
            "cos(x) - x = 0",
        ]
        for expr in cases:
            with self.subTest(expr=expr):
                res = solve_problem(expr)
                # At least not "No solution found" or error string
                self.assertNotIn("No solution found", str(res))
                self.assertNotIn("error", str(res).lower())
                self.assertNotIn("failed", str(res).lower())

    def test_mapping_proxy_type_immutable(self) -> None:
        # Verify _BASE_LOCAL_DICT is MappingProxyType (immutable)
        self.assertIsInstance(_BASE_LOCAL_DICT, types.MappingProxyType)
        self.assertIsInstance(_BASE_SYM_DICT, types.MappingProxyType)

    # ── New tests: critical defect fixes ──

    def test_ode_solution_wrapper(self) -> None:
        """Verify OdeSolution wrapper correctly proxies attributes."""
        from scipy.integrate import solve_ivp
        sol = solve_ivp(lambda t, y: [-y[0]], [0, 1], [1.0])
        wrapped = OdeSolution(result=sol, used_method="RK45")
        self.assertTrue(wrapped.success)
        self.assertEqual(wrapped.used_method, "RK45")
        self.assertEqual(wrapped.result, sol)

    def test_verify_root_infinite(self) -> None:
        """Verify _verify_root does not crash at infinite values."""
        x = Symbol('x')
        expr = 1/x
        result = _verify_root(expr, x, 0)
        self.assertFalse(result)

    def test_mp_nstr_infinite(self) -> None:
        """Verify _mp_nstr is safe for non-finite mpf values."""
        inf_val = mp.mpf('inf')
        result = _mp_nstr(inf_val)
        self.assertIn(result, ("inf", "+inf"))

    def test_scientific_notation_init(self) -> None:
        """Verify scientific notation init parsing."""
        raw, opts = _extract_options("y' + y = 0 numeric init=1e-3,2.5e2")
        self.assertEqual(opts.init_val, [1e-3, 250.0])

    def test_scientific_notation_tspan(self) -> None:
        """Verify scientific notation tspan parsing."""
        raw, opts = _extract_options("y' + y = 0 numeric init=1 tspan=0,1e3")
        self.assertEqual(opts.t_span, (0.0, 1000.0))

    def test_no_rationalize_transform(self) -> None:
        """Verify rationalize transform is removed to avoid symbolic explosion."""
        from sympy.parsing.sympy_parser import rationalize
        self.assertNotIn(rationalize, _TRANS)
        res = parse_single("0.5")
        self.assertEqual(res, 1/2)


# ── Main program ─────────────────────────────────────────────────────


def main() -> None:
    print("Smart Calculator (SymPy + SciPy + mpmath) - High Precision Auto Output")
    print("Supports: Equations, Systems, Inequalities, ODEs (Analytical/Numerical), Automatic Complex Solutions")
    print("Examples:")
    print("  x**3-1=0")
    print("  x**2 - 5*x + 6 = 0")
    print("  cos(x) = sin(x)")
    print("  cos(x) - x = 0")
    print("  y' + y = 0")
    print("  y' + y = 0, y(0)=1")
    print("  y'' + y = 0, y(0)=1, y'(0)=0")
    print("  y' + y = 0 numeric init=1")
    print("  y'' + y = 0 numeric init=1,0")
    print("  y' + y = 0 numeric init=1 method=LSODA")
    print("  sin(x) > 0.5 domain=real")
    print("  x + y = 10, 2*x - y = 5")
    print("  x**2 + y**2 = 1 for x, y")
    print("Run tests: python calc_optimized.py --test")
    print("exit to quit")
    print()
    while True:
        try:
            cmd = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd.lower() in ("exit", "quit"):
            break
        if not cmd:
            continue
        if cmd == "--test":
            unittest.main(module=__name__, exit=False, verbosity=2)
            continue
        res = solve_problem(cmd)
        print("=>", fmt_with_approx(res))
        print()


if __name__ == "__main__":
    main()
