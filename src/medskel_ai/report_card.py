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
