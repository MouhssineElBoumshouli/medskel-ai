"""The measurement table shown in the app: this eye vs FIVES eyes measured the same way.

Pure Python + NumPy (no PyTorch, no Gradio), so it can be tested on its own.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from .morphometry import percentile_of

# Tortuosity is left out on purpose: its value depends on the skeleton method
# (report.md section 6.4), so it is not meaningful to compare.
SHOWN = [
    ("vessel_density", "Vessel density", "% of the retina covered by vessels", lambda v: f"{100 * v:.1f}%"),
    ("fractal_dimension", "Fractal dimension", "how much the vessel tree fills the space (1 = a line, 2 = a filled area)", lambda v: f"{v:.3f}"),
    ("total_length", "Total vessel length", "sum of all centre lines, pixels", lambda v: f"{v:,.0f} px"),
    ("n_bifurcations", "Branch points", "places where a vessel splits", lambda v: f"{v:.0f}"),
    ("median_calibre", "Median vessel width", "pixels", lambda v: f"{v:.1f} px"),
]
OTHER_GROUPS = [("D", "Diabetic retinopathy"), ("G", "Glaucoma"), ("A", "AMD")]  # table columns show each group's median


def load_reference(path: str | Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _median(values) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def compare_text(percentile: float) -> str:
    """Plain words for a percentile among healthy eyes."""
    if not np.isfinite(percentile):
        return "n/a"
    if percentile <= 0.5:
        text = "lower than all healthy eyes"
    elif percentile >= 99.5:
        text = "higher than all healthy eyes"
    elif percentile < 50:
        text = f"lower than {100 - percentile:.0f}% of healthy eyes"
    else:
        text = f"higher than {percentile:.0f}% of healthy eyes"
    if percentile < 5 or percentile > 95:
        text += " (outside the usual healthy range)"
    return text


def table_rows(measures: dict, reference: dict) -> list[dict]:
    healthy = reference["groups"]["N"]["values"]
    rows = []
    for key, name, meaning, fmt in SHOWN:
        value = measures.get(key, float("nan"))
        ref = healthy.get(key, [])
        p = percentile_of(value, ref)
        quart = np.percentile(ref, [25, 75]) if ref else [float("nan")] * 2
        rows.append({
            "key": key, "name": name, "meaning": meaning,
            "value": value, "value_text": fmt(value) if np.isfinite(value) else "n/a",
            "healthy_median": _median(ref),
            "healthy_text": f"{fmt(_median(ref))} ({fmt(quart[0])} – {fmt(quart[1])})" if ref else "n/a",
            "percentile": p, "compare": compare_text(p),
            "others": {g: fmt(_median(reference["groups"][g]["values"].get(key, []))) for g, _ in OTHER_GROUPS},
        })
    return rows


def markdown_table(measures: dict, reference: dict | None, comparable: bool) -> str:
    if reference is None or not comparable:
        lines = ["| Measure | This eye |", "|---|---|"]
        for key, name, _, fmt in SHOWN:
            v = measures.get(key, float("nan"))
            lines.append(f"| {name} | {fmt(v) if np.isfinite(v) else 'n/a'} |")
        why = ("No reference file found (run `python -m medskel_ai.study analyze`)." if reference is None
               else "No comparison: turn on *Resize to the training scale*, so pixels mean the same as in FIVES.")
        return "\n".join(lines) + f"\n\n{why}"
    n = reference["groups"]["N"]["n"]
    head = ("| Measure | This eye | Healthy eyes: median (middle half) | Compared with healthy eyes | "
            + " | ".join(f"{label} (median)" for _, label in OTHER_GROUPS) + " |")
    lines = [head, "|---" * (4 + len(OTHER_GROUPS)) + "|"]
    for r in table_rows(measures, reference):
        lines.append(f"| **{r['name']}**<br><small>{r['meaning']}</small> | **{r['value_text']}** | {r['healthy_text']} | {r['compare']} | "
                     + " | ".join(r["others"][g] for g, _ in OTHER_GROUPS) + " |")
    lines.append("")
    lines.append(f"Reference: {n} healthy FIVES test eyes (and about 50 per disease group) measured exactly the same way "
                 "(our U-Net, then MedSkel centre lines). Pixels at the FIVES scale, so the comparison only makes sense "
                 "for FIVES-like photographs. Healthy and diseased eyes overlap a lot: **one unusual value does not mean "
                 "disease. This is not a diagnosis.**")
    return "\n".join(lines)


def csv_text(measures: dict, reference: dict | None) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["measure", "value", "unit", "healthy_median", "percentile_among_healthy"])
    units = {"vessel_density": "fraction of field of view", "fractal_dimension": "", "total_length": "px",
             "n_bifurcations": "count", "median_calibre": "px"}
    rows = table_rows(measures, reference) if reference else [{"key": k, "value": measures.get(k, float("nan")),
                                                               "healthy_median": "", "percentile": ""} for k, *_ in SHOWN]
    for r in rows:
        writer.writerow([r["key"], r["value"], units[r["key"]], r["healthy_median"], r["percentile"]])
    return out.getvalue()


# ----------------------------------------------------------------------------
# Plain-words explanations and the "where does this eye sit?" chart
# ----------------------------------------------------------------------------

# What each measure means, and what the measurement study found in FIVES
# (results/morphometry/summary.md, Q3a expert masks and Q3b CNN masks).
EXPLANATIONS = {
    "vessel_density": ("How much of the retina is covered by vessels.",
                       "In FIVES, eyes with diabetic retinopathy, glaucoma or AMD had **less** vessel area than healthy eyes, on average (clear difference)."),
    "fractal_dimension": ("How richly the vessel tree branches and fills the space. Higher = a fuller, more detailed tree.",
                          "All three disease groups had a **simpler** tree than healthy eyes, on average (clear difference)."),
    "total_length": ("The length of all vessel centre lines added together.",
                     "All three disease groups had **shorter** vessels in total, on average (clear difference)."),
    "n_bifurcations": ("How many times the vessels split into two.",
                       "All three disease groups had **fewer** splits, on average (clear difference)."),
    "median_calibre": ("The typical vessel width.",
                       "Only small differences between groups, and **no clear difference** with the AI's drawings. Read it with care."),
}
CHART_GROUPS = [("N", "Healthy", "#3a9d6e"), ("D", "Diabetic retinopathy", "#d08a2c"), ("G", "Glaucoma", "#7c6bc4"), ("A", "AMD", "#4a86c5")]


def explanations_markdown() -> str:
    lines = ["#### What the measurements mean"]
    for key, name, *_ in SHOWN:
        meaning, finding = EXPLANATIONS[key]
        lines.append(f"- **{name}**: {meaning} {finding}")
    lines.append("")
    lines.append("**How to read the chart:** each bar is the middle half of one group (the line inside is the typical value, "
                 "the thin lines reach the usual range). The red dot is this eye. The bars **overlap**: a value can be typical "
                 "of a disease group and still be normal for a healthy eye. That is why these numbers are clues for research, "
                 "**not a diagnosis**.")
    return "\n".join(lines)


def group_chart(measures: dict, reference: dict):
    """One row per measure: the 4 groups as horizontal box plots, this eye as a red dot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(SHOWN), 1, figsize=(7.2, 1.55 * len(SHOWN) + 0.4), constrained_layout=True)
    for ax, (key, name, _, fmt) in zip(np.atleast_1d(axes), SHOWN):
        data = [reference["groups"][g]["values"].get(key, []) for g, _, _ in CHART_GROUPS]
        positions = np.arange(len(CHART_GROUPS))[::-1]
        style = dict(positions=positions, widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True)
        try:  # Matplotlib >= 3.10
            box = ax.boxplot(data, orientation="horizontal", **style)
        except TypeError:  # older Matplotlib
            box = ax.boxplot(data, vert=False, **style)
        for patch, (_, _, color) in zip(box["boxes"], CHART_GROUPS):
            patch.set(facecolor=color, alpha=0.55, edgecolor=color)
        for element in ("whiskers", "caps", "medians"):
            for line, color in zip(box[element], [c for _, _, c in CHART_GROUPS for _ in range(1 if element == "medians" else 2)]):
                line.set(color=color if element != "medians" else "#222222", linewidth=1.4)
        value = measures.get(key, float("nan"))
        if np.isfinite(value):
            ax.axvline(value, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.scatter([value] * len(CHART_GROUPS), positions, color="#d62728", s=28, zorder=5)
            ax.set_title(f"{name}: this eye = {fmt(value)}", fontsize=10, loc="left")
        else:
            ax.set_title(f"{name}: not measured", fontsize=10, loc="left")
        ax.set_yticks(positions, [label for _, label, _ in CHART_GROUPS], fontsize=8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _p, f=fmt: f(x)))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # (no figure title: the app shows "Where does this eye sit?" as the label)
    return fig
