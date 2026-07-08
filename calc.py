import random
import re
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any, Final, TypeAlias, cast

import mpmath as mp
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sympy import (
    Derivative,
    Eq,
    Function,
    S,
    Subs,
    Symbol,
    Tuple as SymTuple,
    cancel,
    dsolve,
    expand,
    factor,
    lambdify,
    nsimplify,
    parse_expr,
    pi,
    real_roots,
    solve,
    solveset,
    symbols,
    E,
    I,
    oo,
    nsolve,
    Poly,
    Pow,
    Mul,
    nroots,
    exp,
    sin,
    cos,
    tan,
    log,
    sinh,
    cosh,
    tanh,
    sqrt,
)
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Equality, Relational
from sympy.parsing.sympy_parser import (
    auto_symbol,
    convert_xor,
    implicit_multiplication_application,
    rationalize,
    standard_transformations,
)
from sympy.polys.polyerrors import PolynomialError
from sympy.solvers.inequalities import reduce_inequalities
from sympy.solvers.solveset import (
    ConditionSet,
    FiniteSet,
    ImageSet,
    Interval,
    Union,
    linsolve,
    nonlinsolve,
)

mp.mp.dps = 50

# ----------------------------------------------------------------------
# 常量与类型别名
# ----------------------------------------------------------------------
TRANS: Final[tuple[Any, ...]] = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    rationalize,
    auto_symbol,
)

_SYMS: Final = symbols(
    "x y z a b c t u v w alpha beta gamma delta epsilon theta lambda mu phi psi"
)
SYM_MAP: Final[dict[str, Symbol]] = {str(s): s for s in _SYMS}
SYM_MAP.update({"pi": pi, "e": E, "I": I, "oo": oo})

SymExpr: TypeAlias = Expr | Eq | Relational

# 超越函数原子元组 - 用于 isinstance 快速判断
_TRANSCENDENTAL_FUNCS: Final[tuple[type, ...]] = (
    exp, sin, cos, tan, log, sinh, cosh, tanh, sqrt, AppliedUndef
)


# ----------------------------------------------------------------------
# 数据类
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SolverOptions:
    numeric_mode: bool
    init_val: list[float] | None
    t_span: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ParsedInput:
    raw: str
    domain: Any
    for_vars: list[Symbol] | None


# ----------------------------------------------------------------------
# 正则表达式 - 全部预编译
# ----------------------------------------------------------------------
_RE_DERIV_PARAM: Final = re.compile(r"([a-zA-Z_]\w*)('+)\(([^)]+)\)")
_RE_DERIV_SIMPLE: Final = re.compile(r"(?<![a-zA-Z_'])\b([a-zA-Z_]\w*)('+(?!\())")
_RE_DIFF_PARAM: Final = re.compile(r"([a-zA-Z_]\w*)\(([^)]+)\)\.diff\(([^)]+)\)")
_RE_DIFF_SIMPLE: Final = re.compile(r"([a-zA-Z_]\w*)\(([^)]+)\)\.diff\(\)")
_RE_DIFF_NOARG: Final = re.compile(r"(?<![a-zA-Z_]\))\b([a-zA-Z_]\w*)\.diff\(([^)]+)\)")
_RE_DIFF_NOARG_EMPTY: Final = re.compile(r"(?<![a-zA-Z_]\))\b([a-zA-Z_]\w*)\.diff\(\)")

_RE_NUMERIC_FLAG: Final = re.compile(r"(?i)\bnumeric\b")
_RE_INIT: Final = re.compile(r"init=([-0-9.,\s]+)")
_RE_TSPAN: Final = re.compile(r"tspan=([-0-9.]+),([-0-9.]+)")
_RE_CLEAN_INPUT: Final = re.compile(
    r"(?i)\bnumeric\b|init=[-0-9.,\s]+|tspan=[-0-9.]+,[-0-9.]+"
)
_RE_WHITESPACE: Final = re.compile(r"\s+")
_RE_VALID_VAR: Final = re.compile(r"^[a-zA-Z_]\w*$")

_REL_OPS: Final[frozenset[str]] = frozenset((">=", "<=", ">", "<", "==", "!="))


# 缓存函数名匹配模式，避免重复编译
@cache
def _get_func_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-zA-Z_'])\b{re.escape(name)}\b(?!\w|['(])")


# ----------------------------------------------------------------------
# 输入预处理 - 辅助函数
# ----------------------------------------------------------------------
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


def preprocess_ode(expr_str: str) -> tuple[str, frozenset[str], str]:
    """预处理 ODE 字符串，将简写转换为 SymPy 的 Derivative/Subs 形式。

    Returns:
        (处理后的表达式字符串, 函数名集合, 独立变量名)
    """
    func_names: set[str] = set()
    indep_var: str | None = None

    # 第一遍扫描：收集函数名和独立变量
    for m in _RE_DERIV_PARAM.finditer(expr_str):
        func_names.add(m.group(1))
        var = m.group(3)
        if indep_var is None and _RE_VALID_VAR.match(var):
            indep_var = var
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
            if indep_var is None and g2 is not None:
                var_spec = m.group(g2).strip()
                indep_var = var_spec.split(",")[0].strip() if "," in var_spec else var_spec

    if indep_var is None:
        indep_var = "x"

    # 第二遍：替换
    expr_str = _RE_DERIV_PARAM.sub(
        lambda m: _deriv_param_repl(m, indep_var), expr_str
    )
    expr_str = _RE_DERIV_SIMPLE.sub(
        lambda m: _deriv_simple_repl(m, frozenset(func_names), indep_var), expr_str
    )
    expr_str = _RE_DIFF_PARAM.sub(_diff_param_repl, expr_str)
    expr_str = _RE_DIFF_SIMPLE.sub(_diff_simple_repl, expr_str)
    expr_str = _RE_DIFF_NOARG.sub(
        lambda m: _diff_noarg_repl(m, indep_var), expr_str
    )
    expr_str = _RE_DIFF_NOARG_EMPTY.sub(
        lambda m: _diff_noarg_empty_repl(m, indep_var), expr_str
    )

    # 将裸函数名替换为函数调用
    for name in func_names:
        expr_str = _get_func_pattern(name).sub(f"{name}({indep_var})", expr_str)

    return expr_str, frozenset(func_names), indep_var


def split_top_level(expr_str: str, delimiter: str = ",") -> list[str]:
    """按分隔符分割字符串，但忽略括号内的分隔符。"""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in expr_str:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _has_eq_sign(expr_str: str) -> bool:
    if "=" not in expr_str:
        return False
    return not any(r in expr_str for r in _REL_OPS)


# 缓存本地字典构建，减少重复计算
@cache
def _build_local_dict(
    func_names: tuple[str, ...], indep_var: str | None
) -> dict[str, Any]:
    local_dict: dict[str, Any] = dict(SYM_MAP)
    if func_names and indep_var:
        for name in func_names:
            local_dict[name] = Function(name)
    return local_dict


def _parse_expr_safe(
    expr_str: str, local_dict: dict[str, Any], transformations: tuple[Any, ...]
) -> Any:
    return parse_expr(expr_str, local_dict=local_dict, transformations=transformations)


def parse_single(
    expr_str: str, func_names: Collection[str] | None = None, indep_var: str | None = None
) -> Any:
    expr_str = expr_str.replace("^", "**")
    local_dict = _build_local_dict(
        tuple(sorted(func_names)) if func_names else (), indep_var
    )
    if _has_eq_sign(expr_str):
        left, right = expr_str.split("=", 1)
        lhs = parse_expr(left, local_dict=local_dict, transformations=TRANS)
        rhs = parse_expr(right, local_dict=local_dict, transformations=TRANS)
        return Eq(lhs, rhs, evaluate=False)
    return _parse_expr_safe(expr_str, local_dict, TRANS)


def parse(expr_str: str) -> Any:
    expr_str = expr_str.replace("^", "**")
    preprocessed, func_names, indep_var = preprocess_ode(expr_str)
    parts = split_top_level(preprocessed, ",")
    if len(parts) > 1:
        local_dict = _build_local_dict(tuple(sorted(func_names)), indep_var)
        eqs: list[Any] = []
        for p in parts:
            if not p:
                continue
            if _has_eq_sign(p):
                left, right = p.split("=", 1)
                lhs = parse_expr(left, local_dict=local_dict, transformations=TRANS)
                rhs = parse_expr(right, local_dict=local_dict, transformations=TRANS)
                eqs.append(Eq(lhs, rhs, evaluate=False))
            else:
                eqs.append(_parse_expr_safe(p, local_dict, TRANS))
        return eqs
    return parse_single(preprocessed, func_names, indep_var)


# ----------------------------------------------------------------------
# 表达式工具
# ----------------------------------------------------------------------
def extract_vars(expr: Any) -> list[Symbol]:
    if isinstance(expr, bool):
        return []
    if isinstance(expr, list):
        syms = {s for e in expr if hasattr(e, "free_symbols") for s in e.free_symbols}
        return sorted(syms, key=str)
    if not hasattr(expr, "free_symbols"):
        return []
    return sorted(expr.free_symbols, key=str)


def preprocess(raw: str) -> ParsedInput:
    raw = raw.strip()
    domain = S.Complexes
    if "domain=real" in raw:
        domain = S.Reals
        raw = raw.replace("domain=real", "")
    elif "domain=complex" in raw:
        raw = raw.replace("domain=complex", "")
    for_vars: list[Symbol] | None = None
    if " for " in raw:
        inp, vars_part = raw.split(" for ", 1)
        raw = inp
        for_vars = [SYM_MAP.get(v.strip(), symbols(v.strip())) for v in vars_part.split(",")]
    return ParsedInput(raw=raw, domain=domain, for_vars=for_vars)


def normalize(expr: Any) -> Any:
    return cancel(expand(expr))


def is_ode(expr: Any) -> bool:
    if isinstance(expr, bool):
        return False
    if isinstance(expr, list):
        return any(is_ode(e) for e in expr)
    if isinstance(expr, Eq):
        expr = expr.lhs - expr.rhs
    return expr.has(Derivative) if hasattr(expr, "has") else False


def _get_derivative_info(d: Derivative) -> tuple[Expr, Symbol, int]:
    arg = d.args[0]
    var_info = d.args[1]
    if isinstance(var_info, SymTuple):
        return arg, var_info[0], int(var_info[1])
    return arg, var_info, 1


def normalize_ode(expr: Any) -> Any:
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, list):
        return [normalize_ode(e) for e in expr]
    if isinstance(expr, Eq):
        return Eq(normalize_ode(expr.lhs), normalize_ode(expr.rhs), evaluate=False)

    func_map: dict[Any, Any] = {}
    indep_var: Symbol | None = None
    for d in expr.find(Derivative):
        arg, var, _ = _get_derivative_info(d)
        if arg.func.is_Function:
            func_sym = arg.func
            if indep_var is None:
                indep_var = var
            elif indep_var != var:
                return expr
            func_map[func_sym] = arg

    if not func_map:
        return expr

    free_syms = expr.free_symbols
    for func_sym, f_var in func_map.items():
        sym = Symbol(str(func_sym))
        if sym in free_syms:
            expr = expr.subs(sym, f_var)
    return expr


# ----------------------------------------------------------------------
# 数值求解
# ----------------------------------------------------------------------
def _build_first_order_rhs(
    ode_expr: Any,
    highest_deriv: Derivative,
    f_var: Expr,
    t_sym: Symbol,
    y_sym: Symbol,
    max_order: int,
) -> tuple[Callable[..., NDArray[np.float64]], list[float], str | None]:
    """构建一阶 ODE 系统的右侧函数和默认初始条件。

    Returns:
        (ode_func, y0_default, error_message)
    """
    deriv_expr = Derivative(f_var, (t_sym, max_order))
    sols = solve(ode_expr, deriv_expr, dict=True)
    if not sols:
        return (lambda t, y: np.zeros(1)), [0.0], "无法解出最高阶导数"

    rhs_expr = sols[0][deriv_expr]

    if max_order == 1:
        rhs_expr = rhs_expr.subs(f_var, y_sym)
        undefined = rhs_expr.free_symbols - {t_sym, y_sym}
        if undefined:
            return (lambda t, y: np.zeros(1)), [0.0], f"方程含未定义参数: {', '.join(str(s) for s in sorted(undefined, key=str))}"
        try:
            f_num = lambdify(
                [t_sym, y_sym], rhs_expr, modules="numpy",
                cse=True, docstring_limit=0
            )
        except Exception as e:
            return (lambda t, y: np.zeros(1)), [0.0], f"lambdify 失败: {e}"

        def ode_func(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray([f_num(t, y[0])], dtype=float)

        return ode_func, [0.0], None

    # 高阶 ODE 转化为一阶系统
    state_syms = [Symbol(f"u{i}") for i in range(max_order)]
    subs_map: dict[Any, Any] = {f_var: state_syms[0]}
    for d in ode_expr.find(Derivative):
        arg, var, order = _get_derivative_info(d)
        if arg == f_var and var == t_sym and 1 <= order < max_order:
            subs_map[d] = state_syms[order]
    rhs_expr_sub = rhs_expr.subs(subs_map)
    undefined = rhs_expr_sub.free_symbols - {t_sym} - set(state_syms)
    if undefined:
        return (lambda t, y: np.zeros(max_order)), [0.0] * max_order, f"方程含未定义参数: {', '.join(str(s) for s in sorted(undefined, key=str))}"
    f_exprs = state_syms[1:] + [rhs_expr_sub]
    try:
        f_num = lambdify(
            [t_sym] + state_syms, f_exprs, modules="numpy",
            cse=True, docstring_limit=0
        )
    except Exception as e:
        return (lambda t, y: np.zeros(max_order)), [0.0] * max_order, f"lambdify 失败: {e}"

    def _ode_func_high(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        val = f_num(t, *y)
        return np.asarray(val, dtype=float).ravel()

    return _ode_func_high, [0.0] * max_order, None


def solve_ivp_numerical(
    ode_expr: Any,
    t_span: tuple[float, float] = (0.0, 10.0),
    initial_conditions: float | Collection[float] | None = None,
) -> Any:
    if isinstance(ode_expr, Eq):
        ode_expr = ode_expr.lhs - ode_expr.rhs

    if not hasattr(ode_expr, "find"):
        return None

    derivs = ode_expr.find(Derivative)
    if not derivs:
        return None

    max_order = 1
    highest_deriv: Derivative | None = None
    f_var: Expr | None = None
    t_sym: Symbol | None = None
    for d in derivs:
        arg, var, order = _get_derivative_info(d)
        if not hasattr(arg, "func") or not arg.func.is_Function:
            continue
        if order >= max_order:
            max_order = order
            highest_deriv = d
            f_var = arg
            t_sym = var

    if highest_deriv is None or t_sym is None or f_var is None:
        return None

    y_sym = Symbol(str(f_var.func))
    ode_func, y0_default, err = _build_first_order_rhs(
        ode_expr, highest_deriv, f_var, t_sym, y_sym, max_order
    )
    if err:
        return err

    t_span_len = abs(t_span[1] - t_span[0])
    max_step = min(t_span_len / 20.0, 0.5)

    # 解析初始条件
    if initial_conditions is None:
        y0 = y0_default
    elif isinstance(initial_conditions, (list, tuple)):
        y0 = list(initial_conditions)[:max_order]
        if len(y0) < max_order:
            y0.extend([0.0] * (max_order - len(y0)))
    elif isinstance(initial_conditions, (int, float)):
        y0 = [float(initial_conditions)] + [0.0] * (max_order - 1)
    else:
        y0 = y0_default

    try:
        sol = solve_ivp(
            ode_func,
            t_span,
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            dense_output=True,
            max_step=max_step,
        )
    except Exception as e:
        return f"数值积分失败: {e}"
    return sol


# ----------------------------------------------------------------------
# 代数求解
# ----------------------------------------------------------------------
def _has_transcendental_atoms(expr: Any) -> bool:
    """快速检测表达式是否包含超越函数原子（非多项式函数）。"""
    if not hasattr(expr, "atoms"):
        return False
    return any(isinstance(atom, _TRANSCENDENTAL_FUNCS) for atom in expr.atoms())


def _factor_roots(expr: Any, var: Symbol) -> list[Any] | None:
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
        try:
            rts = solve(base, var, dict=True)
            if rts and not isinstance(rts, ConditionSet):
                for r in rts:
                    if isinstance(r, dict) and var in r:
                        all_roots.add(r[var])
        except (NotImplementedError, TypeError):
            continue
    return list(all_roots) if all_roots else None


def _verify_root(expr: Any, var: Symbol, root: Any, tol: float = 1e-6) -> bool:
    try:
        residual = expr.subs(var, root).evalf()
        if residual is S.NaN:
            return False
        if isinstance(residual, Expr) and residual.is_Number:
            return abs(complex(residual)) < tol
        return True
    except Exception:
        return True


# 缓存 Poly 创建结果以加速重复调用
_POLY_CACHE: dict[tuple[int, str, str], list[Any] | None] = {}


def _try_poly_roots(expr: Any, var: Symbol, domain: Any) -> list[Any] | None:
    # 快速路径：检查是否为纯多项式
    if _has_transcendental_atoms(expr):
        return None
    
    cache_key = (id(expr), str(var), str(domain))
    if cache_key in _POLY_CACHE:
        return _POLY_CACHE[cache_key]
    
    try:
        p = Poly(expr, var)
        deg = p.degree()
        if 0 < deg <= 20:
            if domain == S.Reals:
                rts = real_roots(p)
                if rts:
                    result = [nsimplify(r) for r in rts]
                    _POLY_CACHE[cache_key] = result
                    return result
            else:
                roots = nroots(p)
                if roots:
                    result = [nsimplify(r) for r in roots]
                    _POLY_CACHE[cache_key] = result
                    return result
    except (PolynomialError, NotImplementedError, TypeError, ValueError):
        pass
    _POLY_CACHE[cache_key] = None
    return None


# 模块级缓存随机数生成器，避免重复创建
_RNG: Final = random.Random(42)


def find_roots_nsolve(expr: Any, var: Symbol, domain: Any) -> list[Any]:
    if isinstance(expr, Eq):
        expr = expr.lhs - expr.rhs

    poly_roots = _try_poly_roots(expr, var, domain)
    if poly_roots is not None:
        return poly_roots

    roots: set[float | complex] = set()
    tol = 1e-12
    MAX_ROOT_ABS = 1e15

    if domain == S.Reals:
        guesses = (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 100.0, -100.0)
        for g in guesses:
            try:
                sol = nsolve(expr, var, g, tol=1e-14, maxsteps=100, prec=50)
                sol_eval = sol.evalf()
                if sol_eval.is_real:
                    real_val = float(sol_eval)
                    if abs(real_val) > MAX_ROOT_ABS:
                        continue
                    if all(abs(real_val - r) > tol for r in roots):
                        if _verify_root(expr, var, sol):
                            roots.add(real_val)
            except (ValueError, TypeError, ZeroDivisionError):
                continue
    else:
        grid = (-2, -1, 0, 1, 2)
        offsets = [(_RNG.uniform(-0.5, 0.5), _RNG.uniform(-0.5, 0.5)) for _ in range(2)]
        for re in grid:
            for im in grid:
                for dr, di in offsets:
                    cg = complex(re + dr, im + di)
                    try:
                        sol = nsolve(expr, var, cg, tol=1e-14, maxsteps=100, prec=50)
                        cplx_val = complex(sol.evalf())
                        if abs(cplx_val) > MAX_ROOT_ABS:
                            continue
                        if all(abs(cplx_val - r) > tol for r in roots):
                            if _verify_root(expr, var, sol):
                                roots.add(cplx_val)
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
    return [nsimplify(r) for r in roots]


def _is_expr_numeric_zero(expr: Any) -> bool:
    if isinstance(expr, bool):
        return False
    if isinstance(expr, Expr) and expr.is_zero is not None:
        return bool(expr.is_zero)
    try:
        return bool(expr == 0)
    except Exception:
        return False


def _is_polynomial_like(expr: Any, var: Symbol) -> bool:
    """快速判断表达式是否像多项式（不含三角/指数/对数函数）。"""
    if not hasattr(expr, "atoms"):
        return False
    for atom in expr.atoms():
        if hasattr(atom, "is_Function") and atom.is_Function:
            return False
    return True


def solve_one(expr: Any, var: Symbol, domain: Any) -> Any:
    if isinstance(expr, Eq):
        expr = expr.lhs - expr.rhs

    if _is_expr_numeric_zero(expr):
        return "所有实数" if domain == S.Reals else "所有复数"

    # 快速路径 1：多项式根（nroots / real_roots）
    poly_roots = _try_poly_roots(expr, var, domain)
    if poly_roots is not None:
        return poly_roots

    # 快速路径 2：因式分解（仅对短且多项式样的表达式）
    expr_str = str(expr)
    if len(expr_str) < 500 and _is_polynomial_like(expr, var):
        fac_roots = _factor_roots(expr, var)
        if fac_roots:
            if domain == S.Reals:
                fac_roots = [r for r in fac_roots if r.is_real is not False]
            if fac_roots:
                return fac_roots

    # 判断是否为超越方程 - 对超越方程跳过重符号求解，直接用 nsolve
    is_transcendental = _has_transcendental_atoms(expr)

    # 快速路径 3：符号求解（对非超越方程优先）
    if not is_transcendental:
        try:
            sol = solve(expr, var, dict=True)
            if sol is not None and sol is not False and sol is not True and not isinstance(sol, ConditionSet):
                solutions = []
                for s in sol:
                    if isinstance(s, dict) and var in s:
                        solutions.append(s[var])
                if domain == S.Reals:
                    solutions = [s for s in solutions if hasattr(s, "is_real") and s.is_real is not False]
                if solutions:
                    return solutions
        except (NotImplementedError, TypeError):
            pass

    # 快速路径 4：solveset（对三角方程比 solve 更可靠，但通常更慢）
    # 仅对非超越方程使用，避免 solveset 在复杂三角方程上的高开销
    if not is_transcendental:
        try:
            sol_set = solveset(expr, var, domain)
            if isinstance(sol_set, FiniteSet):
                return list(sol_set)
            elif sol_set is S.EmptySet:
                pass
            elif sol_set == S.Reals:
                return "所有实数"
            elif sol_set == S.Complexes:
                return "所有复数"
            elif isinstance(sol_set, (Union, Interval, ImageSet)):
                return sol_set
        except (NotImplementedError, TypeError):
            pass

    # 兜底：数值寻根（对超越方程这是最快路径）
    num_roots = find_roots_nsolve(expr, var, domain)
    if num_roots:
        return num_roots
    return None


def solve_multi(expr: Any, vars_list: list[Symbol], domain: Any) -> Any:
    if isinstance(expr, Eq):
        eqs = [expr]
    elif isinstance(expr, list):
        eqs = expr
    else:
        eqs = [Eq(expr, 0, evaluate=False)]
    eqs_zero = [e.lhs - e.rhs for e in eqs]

    try:
        sol = linsolve(eqs_zero, vars_list)
        if sol and sol is not S.EmptySet:
            sol_tuple = next(iter(sol))
            return [{v: sol_tuple[i] for i, v in enumerate(vars_list)}]
    except (NotImplementedError, TypeError, ValueError):
        pass

    try:
        sol = solve(eqs, *vars_list, dict=True)
        if sol is not None and sol is not False and sol is not True:
            if isinstance(sol, dict):
                return [sol]
            if len(sol) > 0:
                return sol
    except (NotImplementedError, TypeError):
        pass

    try:
        sol = nonlinsolve(eqs, vars_list)
        if sol and not isinstance(sol, ConditionSet):
            return [{v: val for v, val in zip(vars_list, s)} for s in sol]
    except (NotImplementedError, TypeError):
        pass

    if len(vars_list) == len(eqs):
        try:
            guesses = [0.5] * len(vars_list)
            sol = nsolve(eqs_zero, vars_list, guesses, tol=1e-14, maxsteps=100, prec=50)
            return [{v: sol[i] for i, v in enumerate(vars_list)}]
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    return None


def _ics_key(item: tuple[Any, Any]) -> int:
    k = item[0]
    if isinstance(k, Subs):
        d = k.args[0]
        if isinstance(d, Derivative):
            return _get_derivative_info(d)[2]
    return 0


# ----------------------------------------------------------------------
# 主控逻辑拆分
# ----------------------------------------------------------------------
def _extract_options(inp: str) -> tuple[str, SolverOptions]:
    numeric_mode = _RE_NUMERIC_FLAG.search(inp) is not None
    match_init = _RE_INIT.search(inp)
    init_val: list[float] | None = None
    if match_init:
        init_str = match_init.group(1)
        init_val = [float(v.strip()) for v in init_str.split(",")]
    match_tspan = _RE_TSPAN.search(inp)
    t_span: tuple[float, float] = (0.0, 10.0)
    if match_tspan:
        t_span = (float(match_tspan.group(1)), float(match_tspan.group(2)))

    clean_inp = _RE_CLEAN_INPUT.sub("", inp)
    clean_inp = _RE_WHITESPACE.sub(" ", clean_inp).strip(" ,")
    return clean_inp, SolverOptions(
        numeric_mode=numeric_mode, init_val=init_val, t_span=t_span
    )


def _solve_ode_system(
    expr_list: list[Any], options: SolverOptions
) -> Any:
    ode_eq: Eq | None = None
    ics: dict[Any, Any] = {}
    for e in expr_list:
        if not isinstance(e, Eq):
            continue
        if isinstance(e.lhs, Subs):
            ics[e.lhs] = e.rhs
        elif hasattr(e.lhs, "func") and e.lhs.func.is_Function and not e.has(Derivative):
            ics[e.lhs] = e.rhs
        else:
            for d in e.find(Derivative):
                arg, _, _ = _get_derivative_info(d)
                if hasattr(arg, "func") and arg.func.is_Function:
                    ode_eq = e
                    break

    if options.numeric_mode:
        if ode_eq is None:
            return "数值求解失败，请检查方程"
        expr = ode_eq
        init_val = options.init_val
        if init_val is None and ics:
            sorted_ics = sorted(ics.items(), key=_ics_key)
            init_val = [
                float(v.evalf()) if hasattr(v, "evalf") else float(v)
                for _, v in sorted_ics
            ]
        elif init_val is None:
            return "数值求解需要初始条件 (init=...)"
        numeric_solution = solve_ivp_numerical(
            expr, t_span=options.t_span, initial_conditions=init_val
        )
        if isinstance(numeric_solution, str):
            return numeric_solution
        if numeric_solution is not None and getattr(numeric_solution, "success", False):
            return numeric_solution
        return "数值求解失败，请检查方程或初始条件"

    if ode_eq is not None and ics:
        try:
            # simplify=False 更快，后续手动简化
            sol = dsolve(ode_eq, ics=ics, simplify=False)
            return cancel(expand(sol))
        except (NotImplementedError, TypeError):
            return "ODE解析求解失败"
    if ode_eq is not None:
        try:
            sol = dsolve(ode_eq, simplify=False)
            return cancel(expand(sol))
        except (NotImplementedError, TypeError):
            return "ODE解析求解失败"
    return "ODE方程组解析求解暂不支持"


def _dispatch_ode(expr: Any, options: SolverOptions) -> Any:
    expr = normalize_ode(expr)
    if isinstance(expr, list):
        return _solve_ode_system(expr, options)
    if options.numeric_mode:
        numeric_solution = solve_ivp_numerical(
            expr, t_span=options.t_span, initial_conditions=options.init_val
        )
        if isinstance(numeric_solution, str):
            return numeric_solution
        if numeric_solution is not None and getattr(numeric_solution, "success", False):
            return numeric_solution
        return "数值求解失败，请检查方程或初始条件"
    try:
        sol = dsolve(expr, simplify=False)
        return cancel(expand(sol))
    except (NotImplementedError, TypeError):
        return "ODE解析求解失败"


def solve_problem(inp: str) -> Any:
    if not inp or not inp.strip():
        return "输入为空"

    clean_inp, options = _extract_options(inp)
    if not clean_inp:
        return "输入为空"

    parsed = preprocess(clean_inp)

    try:
        expr = parse(parsed.raw)
    except Exception as e:
        return f"解析错误: {e}"

    if isinstance(expr, bool):
        return expr

    if is_ode(expr):
        return _dispatch_ode(expr, options)

    if isinstance(expr, Relational) and not isinstance(expr, Equality):
        vars_list = extract_vars(expr)
        if vars_list:
            try:
                return reduce_inequalities(expr, vars_list)
            except Exception:
                return "不等式求解失败"
        try:
            return reduce_inequalities(expr)
        except Exception:
            return "不等式求解失败"

    vars_list = parsed.for_vars if parsed.for_vars is not None else extract_vars(expr)
    if not vars_list:
        return expr
    if len(vars_list) == 1:
        return solve_one(expr, vars_list[0], parsed.domain)
    return solve_multi(expr, vars_list, parsed.domain)


# ----------------------------------------------------------------------
# 格式化输出
# ----------------------------------------------------------------------
@lru_cache(maxsize=512)
def _evalf_cached(expr: Expr, n: int = 30) -> Any:
    return expr.evalf(n)


def approx_of(x: Any, n: int = 30) -> str:
    if isinstance(x, Expr):
        return fmt_item(_evalf_cached(x, n))
    return fmt_item(x)


def _format_list_result(result: list[Any] | tuple[Any, ...]) -> str:
    formatted: list[str] = []
    for item in result:
        if isinstance(item, dict):
            d_exact = {k: fmt_item(v) for k, v in item.items()}
            d_approx = {k: approx_of(v) for k, v in item.items()}
            formatted.append(f"精确: {d_exact}  ≈  {d_approx}")
        elif isinstance(item, (list, tuple)):
            formatted.append(_format_list_result(item))
        else:
            exact = fmt_item(item)
            approx = approx_of(item)
            if exact == approx:
                formatted.append(exact)
            else:
                formatted.append(f"{exact}  ≈  {approx}")
    return str(formatted)


def fmt_with_approx(result: Any) -> str:
    if result is None:
        return "无解"
    if isinstance(result, str):
        return result
    if hasattr(result, "success"):
        if not result.success:
            return "数值求解失败"
        t = getattr(result, "t", None)
        y = getattr(result, "y", None)
        sol = getattr(result, "sol", None)
        if t is not None and y is not None and len(y) > 0:
            t_arr = np.asarray(t)
            y_arr = np.asarray(y)
            t0 = float(t_arr[0])
            t1 = float(t_arr[-1])
            y_min = float(y_arr[0].min())
            y_max = float(y_arr[0].max())
            if sol is not None:
                t_mid = (t0 + t1) / 2
                y_mid = sol(t_mid)
                if isinstance(y_mid, np.ndarray):
                    y_mid_val = float(y_mid[0])
                else:
                    y_mid_val = float(y_mid)
                return (
                    f"数值解: t∈[{t0:.2f}, {t1:.2f}], "
                    f"y∈[{y_min:.4f}, {y_max:.4f}], "
                    f"y({t_mid:.2f})≈{y_mid_val:.6f}"
                )
            return (
                f"数值解: t∈[{t0:.2f}, {t1:.2f}], "
                f"y∈[{y_min:.4f}, {y_max:.4f}]"
            )
        return "数值求解失败"
    if isinstance(result, Eq):
        exact = fmt_item(result)
        approx_rhs = approx_of(result.rhs)
        return f"{exact}  ≈  {result.lhs} = {approx_rhs}"
    if isinstance(result, (list, tuple)):
        return _format_list_result(result)
    if isinstance(result, dict):
        d_exact = {k: fmt_item(v) for k, v in result.items()}
        d_approx = {k: approx_of(v) for k, v in result.items()}
        return f"精确: {d_exact}  ≈  {d_approx}"
    if isinstance(result, (set, frozenset)):
        formatted = []
        for v in result:
            exact = fmt_item(v)
            approx = approx_of(v)
            if exact == approx:
                formatted.append(exact)
            else:
                formatted.append(f"{exact}  ≈  {approx}")
        return str(formatted)
    if result is S.Reals:
        return "所有实数"
    if result is S.Complexes:
        return "所有复数"
    if isinstance(result, (Union, Interval, ImageSet)):
        return str(result)
    exact = fmt_item(result)
    approx = approx_of(result)
    if exact == approx:
        return exact
    return f"{exact}  ≈  {approx}"


def _mp_nstr(val: Any) -> str:
    s = cast(str, mp.nstr(val, 30, strip_zeros=True)).rstrip(".")
    return "0" if not s or s == "-0" else s


def _format_complex_pair(re_val: float, im_val: float) -> str:
    if abs(im_val) < 1e-30:
        return fmt_item(re_val)
    if abs(re_val) < 1e-30:
        if im_val < 0:
            return f"-{fmt_item(-im_val)}*I"
        return f"{fmt_item(im_val)}*I"
    re_str = fmt_item(re_val)
    if im_val < 0:
        return f"{re_str} - {fmt_item(-im_val)}*I"
    return f"{re_str} + {fmt_item(im_val)}*I"


def fmt_item(x: Any) -> str:
    if isinstance(x, complex):
        return _format_complex_pair(x.real, x.imag)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        return format(x, ".15g").rstrip(".")
    if isinstance(x, Expr) and x.is_Number:
        if x.is_real:
            try:
                f = float(x)
                if f == int(f):
                    return str(int(f))
            except Exception:
                pass
            return _mp_nstr(mp.mpf(str(_evalf_cached(x, 50))))
        else:
            re_part, im_part = x.as_real_imag()
            im_val = float(_evalf_cached(im_part, 50))
            if abs(im_val) < 1e-30:
                return fmt_item(re_part)
            re_str = fmt_item(re_part)
            if im_val < 0:
                return f"{re_str} - {fmt_item(-im_val)}*I"
            return f"{re_str} + {fmt_item(im_val)}*I"
    if isinstance(x, mp.mpf):
        return _mp_nstr(x)
    if isinstance(x, mp.mpc):
        if mp.almosteq(x.imag, 0, abs_eps=1e-30):
            return fmt_item(x.real)
        if mp.almosteq(x.real, 0, abs_eps=1e-30):
            return _format_complex_pair(0.0, float(x.imag))
        return _format_complex_pair(float(x.real), float(x.imag))
    return str(x)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    print("智能计算器 (SymPy + SciPy + mpmath) - 高精度自动输出")
    print("支持: 方程, 方程组, 不等式, 微分方程(解析/数值), 复数解自动识别")
    print("示例:")
    print("  x**3-1=0")
    print("  x**2 - 5*x + 6 = 0")
    print("  cos(x) = sin(x)")
    print("  cos(x) - x = 0")
    print("  y' + y = 0")
    print("  y' + y = 0, y(0)=1")
    print("  y'' + y = 0, y(0)=1, y'(0)=0")
    print("  y' + y = 0 numeric init=1")
    print("  y'' + y = 0 numeric init=1,0")
    print("  sin(x) > 0.5 domain=real")
    print("  x + y = 10, 2*x - y = 5")
    print("exit 退出")
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
        res = solve_problem(cmd)
        print("=>", fmt_with_approx(res))
        print()


if __name__ == "__main__":
    main()

