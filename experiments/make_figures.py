"""Generate clean PNG and PDF figures without external plotting packages."""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from .study_utils import read_csv


COLORS = {
    "Simple": (31, 96, 145),
    "Polymatroid": (199, 88, 45),
    "Surrogate": (76, 140, 71),
    "Multi-D": (117, 87, 160),
}


def make_figures() -> None:
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    exact = _convert(read_csv("results/exact/raw_exact.csv"))
    scaling = _convert(read_csv("results/scaling/raw_scaling.csv"))
    _figure_dimension(exact)
    _figure_regime(exact)
    _figure_runtime(scaling)
    _figure_surrogate(exact)


def _figure_dimension(rows):
    groups = sorted({int(r["d"]) for r in rows if r["reward_variant"] == "independent"})
    series = {}
    for name, key in [("Simple", "simple_ratio"), ("Polymatroid", "poly_ratio"), ("Surrogate", "surrogate_density_ratio"), ("Multi-D", "multidim_density_ratio")]:
        points = []
        for d in groups:
            vals = [float(r[key]) for r in rows if int(r["d"]) == d and r["reward_variant"] == "independent"]
            mean, se = _mean_se(vals)
            points.append((d, mean, se))
        series[name] = points
    _draw_line_plot("results/figures/performance_vs_dimension", "Performance vs Dimension", "dimension d", "mean ratio to ADAPT", series, y_min=0.0, y_max=1.05)


def _figure_regime(rows):
    regimes = sorted({r["regime"] for r in rows if r["reward_variant"] == "independent"})
    bars = []
    for regime in regimes:
        group = [r for r in rows if r["regime"] == regime and r["reward_variant"] == "independent"]
        bars.append((regime, {"Simple": _mean([float(r["simple_ratio"]) for r in group]), "Polymatroid": _mean([float(r["poly_ratio"]) for r in group])}))
    _draw_grouped_bars("results/figures/regime_comparison", "Performance by Regime", bars, y_max=1.05)


def _figure_runtime(rows):
    groups = sorted({int(r["n"]) for r in rows})
    series = {}
    for name, key in [("Simple", "simple_construct_time"), ("Polymatroid", "poly_construct_time")]:
        points = []
        for n in groups:
            vals = [float(r[key]) for r in rows if int(r["n"]) == n]
            points.append((n, _mean(vals), 0.0))
        series[name] = points
    max_y = max(v for points in series.values() for _, v, _ in points) * 1.25
    _draw_line_plot("results/figures/runtime_vs_n", "Construction Runtime vs n", "n", "seconds", series, y_min=0.0, y_max=max_y)


def _figure_surrogate(rows):
    points = [(float(r["mean_surrogate_geometry_ratio"]), float(r["simple_ratio"]), float(r["poly_ratio"])) for r in rows if r["reward_variant"] == "independent"]
    _draw_scatter("results/figures/surrogate_diagnostic", "Surrogate Diagnostic vs Performance", "mean E[max S] / sum E[S]", "ratio to ADAPT", points)


def _draw_line_plot(path_prefix, title, x_label, y_label, series, y_min, y_max):
    img, draw = _canvas(title, x_label, y_label)
    left, top, right, bottom = 90, 55, 760, 480
    all_x = [x for points in series.values() for x, _, _ in points]
    x_min, x_max = min(all_x), max(all_x)
    _axes(draw, left, top, right, bottom)
    for name, points in series.items():
        color = COLORS[name]
        coords = []
        for x, y, se in points:
            px = _scale(x, x_min, x_max, left, right)
            py = _scale(y, y_min, y_max, bottom, top)
            coords.append((px, py))
            err = 1.96 * se
            ey1 = _scale(y - err, y_min, y_max, bottom, top)
            ey2 = _scale(y + err, y_min, y_max, bottom, top)
            draw.line((px, ey1, px, ey2), fill=color, width=2)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=3)
    _legend(draw, series.keys(), 580, 70)
    _ticks(draw, left, top, right, bottom, x_min, x_max, y_min, y_max)
    _save(img, path_prefix)


def _draw_grouped_bars(path_prefix, title, bars, y_max):
    img, draw = _canvas(title, "regime", "mean ratio to ADAPT")
    left, top, right, bottom = 90, 55, 760, 480
    _axes(draw, left, top, right, bottom)
    group_w = (right - left) / len(bars)
    bar_w = group_w * 0.25
    for i, (label, vals) in enumerate(bars):
        center = left + group_w * (i + 0.5)
        for j, name in enumerate(["Simple", "Polymatroid"]):
            x0 = center + (j - 1) * bar_w
            y = vals[name]
            y0 = _scale(y, 0.0, y_max, bottom, top)
            draw.rectangle((x0, y0, x0 + bar_w, bottom), fill=COLORS[name])
        draw.text((center - group_w * 0.38, bottom + 8), label.replace("_", "\n"), fill=(0, 0, 0))
    _ticks(draw, left, top, right, bottom, 0, len(bars), 0, y_max, x_ticks=False)
    _legend(draw, ["Simple", "Polymatroid"], 580, 70)
    _save(img, path_prefix)


def _draw_scatter(path_prefix, title, x_label, y_label, points):
    img, draw = _canvas(title, x_label, y_label)
    left, top, right, bottom = 90, 55, 760, 480
    _axes(draw, left, top, right, bottom)
    xs = [p[0] for p in points]
    x_min, x_max = min(xs), max(xs)
    for x, y1, y2 in points:
        px = _scale(x, x_min, x_max, left, right)
        draw.ellipse((px - 2, _scale(y1, 0, 1.05, bottom, top) - 2, px + 2, _scale(y1, 0, 1.05, bottom, top) + 2), fill=COLORS["Simple"])
        draw.rectangle((px - 2, _scale(y2, 0, 1.05, bottom, top) - 2, px + 2, _scale(y2, 0, 1.05, bottom, top) + 2), fill=COLORS["Polymatroid"])
    _ticks(draw, left, top, right, bottom, x_min, x_max, 0, 1.05)
    _legend(draw, ["Simple", "Polymatroid"], 580, 70)
    _save(img, path_prefix)


def _canvas(title, x_label, y_label):
    img = Image.new("RGB", (850, 560), "white")
    draw = ImageDraw.Draw(img)
    draw.text((90, 20), title, fill=(0, 0, 0))
    draw.text((370, 525), x_label, fill=(0, 0, 0))
    draw.text((10, 250), y_label, fill=(0, 0, 0))
    return img, draw


def _axes(draw, left, top, right, bottom):
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, top, left, bottom), fill=(0, 0, 0), width=2)


def _ticks(draw, left, top, right, bottom, x_min, x_max, y_min, y_max, x_ticks=True):
    for t in range(6):
        y = bottom - (bottom - top) * t / 5
        val = y_min + (y_max - y_min) * t / 5
        draw.line((left - 5, y, left, y), fill=(0, 0, 0))
        draw.text((30, y - 7), f"{val:.2f}", fill=(0, 0, 0))
    if x_ticks:
        for t in range(5):
            x = left + (right - left) * t / 4
            val = x_min + (x_max - x_min) * t / 4
            draw.line((x, bottom, x, bottom + 5), fill=(0, 0, 0))
            draw.text((x - 12, bottom + 10), f"{val:.1f}", fill=(0, 0, 0))


def _legend(draw, names, x, y):
    for i, name in enumerate(names):
        yy = y + i * 20
        draw.rectangle((x, yy, x + 12, yy + 12), fill=COLORS[name])
        draw.text((x + 18, yy - 1), name, fill=(0, 0, 0))


def _save(img, prefix):
    img.save(prefix + ".png")
    c = canvas.Canvas(prefix + ".pdf", pagesize=letter)
    tmp = prefix + ".png"
    c.drawImage(tmp, 36, 300, width=540, height=356)
    c.showPage()
    c.save()


def _scale(v, v_min, v_max, out_min, out_max):
    if abs(v_max - v_min) < 1e-12:
        return (out_min + out_max) / 2
    return out_min + (v - v_min) * (out_max - out_min) / (v_max - v_min)


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _mean_se(vals):
    if not vals:
        return 0.0, 0.0
    mean = statistics.mean(vals)
    se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, se


def _convert(rows):
    out = []
    for row in rows:
        converted = {}
        for k, v in row.items():
            if v == "":
                converted[k] = v
                continue
            try:
                converted[k] = float(v)
            except ValueError:
                converted[k] = v
        out.append(converted)
    return out


if __name__ == "__main__":
    make_figures()

