"""Generate CSV and standalone LaTeX tables for Round 3 results."""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from pathlib import Path

from .study_utils import environment_info, read_csv, write_csv, write_json


def make_tables() -> None:
    exact = _read_float_rows("results/exact/raw_exact.csv")
    scaling = _read_float_rows("results/scaling/raw_scaling.csv")
    stress_simple = _read_float_rows("results/stress_search/simple_bottom20.csv")
    stress_poly = _read_float_rows("results/stress_search/poly_bottom20.csv")

    table_exact = _table_publication_overall(exact)
    table_dim = _table_publication_dimension_n6(exact)
    table_scaling = _table_scaling(scaling)
    table_stress = _table_stress(stress_simple, stress_poly)

    write_csv("results/tables/table_exact.csv", table_exact)
    write_csv("results/tables/table_dimension.csv", table_dim)
    write_csv("results/tables/publication_overall.csv", table_exact)
    write_csv("results/tables/publication_dimension_n6.csv", table_dim)
    write_csv("results/tables/table_scaling.csv", table_scaling)
    write_csv("results/tables/table_stress.csv", table_stress)
    _write_latex("results/tables/table_exact.tex", "Exact Adaptive Benchmark: Overall Performance", table_exact)
    _write_latex("results/tables/table_dimension.tex", "Matched n=6 Dimension Comparison", table_dim)
    _write_latex("results/tables/publication_overall.tex", "Exact Adaptive Benchmark: Overall Performance", table_exact)
    _write_latex("results/tables/publication_dimension_n6.tex", "Matched n=6 Dimension Comparison", table_dim)
    _write_latex("results/tables/table_scaling.tex", "Scaling Runtime", table_scaling)
    _write_latex("results/tables/table_stress.tex", "Difficult-Instance Search", table_stress)

    write_json("results/configs/environment.json", environment_info())
    write_json("results/configs/code_manifest.json", _code_manifest())


def _table_publication_overall(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        _overall_row("Simple LP", rows, "simple_ratio"),
        _overall_row("Polymatroid LP", rows, "poly_ratio"),
        _overall_row("Surrogate-density greedy", rows, "surrogate_density_ratio"),
        _overall_row("Multi-D density greedy", rows, "multidim_density_ratio"),
    ]


def _overall_row(label: str, rows: list[dict[str, object]], key: str) -> dict[str, object]:
    vals = [float(r[key]) for r in rows if r[key] != ""]
    return {
        "Algorithm": label,
        "Observations": len(vals),
        "Mean": f"{statistics.mean(vals):.6f}",
        "Median": f"{statistics.median(vals):.6f}",
        "Minimum": f"{min(vals):.6f}",
    }


def _table_publication_dimension_n6(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for d in sorted({int(r["d"]) for r in rows}):
        group = [r for r in rows if int(r["d"]) == d and int(r["n"]) == 6]
        out.append(
            {
                "d": d,
                "Observations": len(group),
                "Simple mean": f"{_mean(group, 'simple_ratio'):.6f}",
                "Simple guarantee": f"{7.0 / (16.0 * (d + 1)):.6f}",
                "Poly mean": f"{_mean(group, 'poly_ratio'):.6f}",
                "Poly guarantee": f"{(1.0 - math.exp(-(d + 1))) / (2.0 * (d + 1)):.6f}",
            }
        )
    return out


def _table_scaling(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for d in sorted({int(r["d"]) for r in rows}):
        for n in sorted({int(r["n"]) for r in rows if int(r["d"]) == d}):
            group = [r for r in rows if int(r["d"]) == d and int(r["n"]) == n]
            out.append(
                {
                    "d": d,
                    "n": n,
                    "Instances": len(group),
                    "Simple construct sec": f"{_mean(group, 'simple_construct_time'):.6f}",
                    "Poly construct sec": f"{_mean(group, 'poly_construct_time'):.6f}",
                    "Baseline construct sec": f"{_mean(group, 'baseline_construct_time'):.6f}",
                    "Evaluation sec": f"{_mean(group, 'evaluation_time'):.3f}",
                }
            )
    return out


def _table_stress(simple: list[dict[str, object]], poly: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    if simple:
        s = min(simple, key=lambda r: float(r["simple_ratio"]))
        rows.append(
            {
                "Algorithm": "Simple",
                "d": int(s["d"]),
                "n": int(s["n"]),
                "Regime": s["regime"],
                "Lowest ratio found": f"{float(s['simple_ratio']):.3f}",
                "ADAPT_exact": f"{float(s['ADAPT_exact']):.3f}",
                "ALG": f"{float(s['simple_reward']):.3f}",
                "Seed": int(s["seed"]),
            }
        )
    if poly:
        p = min(poly, key=lambda r: float(r["poly_ratio"]))
        rows.append(
            {
                "Algorithm": "Polymatroid",
                "d": int(p["d"]),
                "n": int(p["n"]),
                "Regime": p["regime"],
                "Lowest ratio found": f"{float(p['poly_ratio']):.3f}",
                "ADAPT_exact": f"{float(p['ADAPT_exact']):.3f}",
                "ALG": f"{float(p['poly_reward']):.3f}",
                "Seed": int(p["seed"]),
            }
        )
    return rows


def _write_latex(path: str, caption: str, rows: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write("\\begin{tabular}{" + "l" * len(headers) + "}\n\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n\\midrule\n")
        for row in rows:
            f.write(" & ".join(str(row[h]).replace("_", "\\_") for h in headers) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def _read_float_rows(path: str) -> list[dict[str, object]]:
    rows = read_csv(path)
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if value == "":
                converted[key] = value
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
        out.append(converted)
    return out


def _mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r[key] != ""]
    return sum(vals) / len(vals) if vals else 0.0


def _code_manifest() -> dict[str, str]:
    manifest = {}
    for path in sorted(Path("experiments").glob("*.py")):
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(Path("tests").glob("*.py")):
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


if __name__ == "__main__":
    make_tables()
