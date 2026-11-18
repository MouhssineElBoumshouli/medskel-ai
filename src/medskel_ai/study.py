"""MedSkel AI measurement study: from a CNN mask to vessel measurements.

    python -m medskel_ai.study all          # the three steps below, in order
    python -m medskel_ai.study predict      # 1) CNN masks for FIVES test and CHASE_DB1 (GPU)
    python -m medskel_ai.study measure      # 2) measurements of every mask, both skeleton methods (CPU, parallel)
    python -m medskel_ai.study analyze      # 3) statistics, tables and figures -> results/morphometry/

Every step can be stopped and restarted: finished work is kept and skipped.

Three questions, decided before any result was seen
---------------------------------------------------
Q1 (CHASE_DB1, two human observers): when the CNN mask replaces a human mask,
    how much do the measurements move? Compare with how much they move between
    the two humans. And does the MedSkel skeleton move less than thinning?
    This repeats the original MedSkel inter-observer experiment and adds the CNN.
Q2 (FIVES test, 200 images): same question against the FIVES expert mask.
Q3 (FIVES, disease groups A/D/G/N): are the measurements different between
    healthy and diseased eyes? Can a simple model guess the group from the
    measurements alone? This REPEATS a known kind of finding (e.g. Yusufu et
    al. 2025 on UK Biobank). It is an association in one dataset, not a
    diagnosis.

Scientific care
---------------
- Pixels only. FIVES and CHASE have different cameras, so absolute numbers are
  not comparable between the two datasets; only relative differences are.
- Images left out of Q3: the two empty training masks (447_G, 448_G) and
  train/174_D, whose expert mask does not match its photograph (found in v2).
  Byte-identical photographs (6 pairs) are counted once.
- CHASE: two eyes per child are not independent; intervals resample children.
- Q3 classifier: trained on TRAINING images only, scored on the 200 TEST images,
  once with expert masks and once with CNN masks (the realistic pipeline:
  photo -> CNN -> measurements -> guess). A label-shuffling control gives the
  score expected by pure chance.
- Thresholds, methods and measures are fixed in this file before running.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from . import morphometry as mm

# The data/chase/predict modules import PyTorch. They are imported inside the
# functions that need them, so the statistics step also runs without PyTorch.

REPO = Path(__file__).resolve().parents[2]
MASK_DIR = REPO / "runs" / "morphometry" / "masks"
OUT_DIR = REPO / "results" / "morphometry"
MEASUREMENTS = OUT_DIR / "measurements.csv"
LABEL_ERRORS = frozenset({"train/174_D"})  # expert mask does not match the photograph (v2 report, section 4.7)
GROUPS = ("N", "A", "D", "G")
GROUP_NAMES = {"N": "normal", "A": "AMD", "D": "diabetic retinopathy", "G": "glaucoma"}
CSV_FIELDS = ["key", "dataset", "image_id", "source", "group", "child", "method", *mm.ALL_MEASURES, "seconds"]


# ============================================================================
# Step 1: CNN masks
# ============================================================================

def _mask_path(image_id: str) -> Path:
    return MASK_DIR / (image_id.replace("/", "__") + ".png")


def step_predict(args) -> None:
    import torch  # only this step needs PyTorch

    from .data import list_pairs, load_image
    from .chase import CHASE_FOV_DEGREES, FIVES_FOV_DEGREES, FIVES_FOV_PIXELS, field_of_view, fov_width, list_chase, predict_at_scale
    from .predict import load_model, predict_probability

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.run) / "best.pt"
    model, device, threshold = load_model(checkpoint, flips=True)
    print(f"model {checkpoint}, threshold {threshold} (chosen on validation, with flips), device {device}", flush=True)

    jobs = [(p["id"], p["image"], None) for p in list_pairs(args.data, "test")]
    if Path(args.chase).exists():
        jobs += [(c["id"], c["image"], "chase") for c in list_chase(args.chase)]
    else:
        print(f"CHASE_DB1 not found at {args.chase}: skipping it", flush=True)
    for n, (image_id, image_path, kind) in enumerate(jobs, 1):
        out = _mask_path(image_id)
        if out.exists():
            continue
        image = load_image(image_path)
        with torch.no_grad():
            if kind == "chase":  # same scale matching as the v2 CHASE evaluation ("matched" setting)
                scale = (FIVES_FOV_PIXELS / FIVES_FOV_DEGREES) / (fov_width(field_of_view(image)) / CHASE_FOV_DEGREES)
                prob = predict_at_scale(model, image, device, scale, flips=True)
            else:
                prob = predict_probability(model, image, device, flips=True)
        Image.fromarray(((prob >= threshold) * 255).astype(np.uint8)).save(out)
        print(f"predict {n}/{len(jobs)} {image_id}", flush=True)
    print("predict: done", flush=True)


# ============================================================================
# Step 2: measurements
# ============================================================================

def build_jobs(data: str, chase: str, include_train: bool) -> list[dict]:
    """One job = one mask to measure (with both skeleton methods)."""
    from .data import EMPTY_TRAIN_MASKS, list_pairs

    jobs = []
    if Path(chase).exists():
        from .chase import list_chase

        for c in list_chase(chase):
            base = {"dataset": "chase", "image_id": c["id"], "group": "", "child": c["child"], "image": c["image"]}
            jobs.append({**base, "source": "observer1", "mask": c["mask1"]})
            jobs.append({**base, "source": "observer2", "mask": c["mask2"]})
            jobs.append({**base, "source": "cnn", "mask": str(_mask_path(c["id"]))})
    for p in list_pairs(data, "test"):
        base = {"dataset": "fives_test", "image_id": p["id"], "group": p["group"], "child": "", "image": p["image"]}
        jobs.append({**base, "source": "expert", "mask": p["mask"]})
        jobs.append({**base, "source": "cnn", "mask": str(_mask_path(p["id"]))})
    if include_train:
        for p in list_pairs(data, "train"):
            if p["id"] in EMPTY_TRAIN_MASKS or p["id"] in LABEL_ERRORS:
                continue
            jobs.append({"dataset": "fives_train", "image_id": p["id"], "group": p["group"], "child": "", "image": p["image"], "source": "expert", "mask": p["mask"]})
    for job in jobs:
        job["key"] = f"{job['image_id']}|{job['source']}"
    return jobs


def _measure_job(job: dict, repo_path: str | None) -> list[dict]:
    """Runs in a worker process: load one mask, measure it with both methods."""
    from .chase import field_of_view
    from .data import load_image, load_mask

    mask = load_mask(job["mask"]).astype(bool)
    fov = field_of_view(load_image(job["image"]))
    if fov.shape != mask.shape:
        raise ValueError(f"{job['key']}: mask {mask.shape} and photo {fov.shape} sizes differ")
    rows = []
    for method in mm.METHODS:
        row = mm.measure(mask, method, fov=fov, repo_path=repo_path)
        rows.append({k: job.get(k, "") for k in ("key", "dataset", "image_id", "source", "group", "child")} | row)
    return rows


def read_measurements(path: Path | None = None) -> list[dict]:
    path = path or MEASUREMENTS
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in (*mm.ALL_MEASURES, "seconds"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
    return rows


def step_measure(args) -> None:
    repo_path = args.medskel_repo or str(mm.default_upstream_path())
    mm.import_upstream(repo_path)  # fail early with a clear message if the clone is missing
    jobs = build_jobs(args.data, args.chase, include_train=not args.skip_train)
    missing = [j["key"] for j in jobs if j["source"] == "cnn" and not Path(j["mask"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} CNN masks are missing (e.g. {missing[0]}). Run the predict step first.")
    done = {r["key"] for r in read_measurements()}
    todo = [j for j in jobs if j["key"] not in done]
    print(f"measure: {len(jobs)} masks, {len(done)} already done, {len(todo)} to do, {args.workers} workers", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not MEASUREMENTS.exists()
    with MEASUREMENTS.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        # Import the worker through the package (not __main__) so Windows worker processes can find it.
        from medskel_ai.study import _measure_job as worker

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, job, repo_path): job for job in todo}
            for n, future in enumerate(as_completed(futures), 1):
                rows = future.result()
                writer.writerows(rows)
                f.flush()  # written immediately, so a stop loses at most the masks in progress
                times = ", ".join(f"{r['method']} {r['seconds']:.0f}s" for r in rows)
                print(f"measure {n}/{len(todo)} {futures[future]['key']} ({times})", flush=True)
    print("measure: done", flush=True)


# ============================================================================
# Step 3: statistics
# ============================================================================

def cluster_bootstrap(values, clusters=None, stat=np.median, n_boot: int = 5000, seed: int = 0) -> list[float]:
    """95% interval of a statistic (default: median), resampling whole clusters (e.g. children)."""
    v = np.asarray(values, dtype=float)
    labels = np.arange(len(v)) if clusters is None else np.asarray(clusters)
    keep = ~np.isnan(v)
    v, labels = v[keep], labels[keep]
    if len(v) < 3:
        return [float("nan"), float("nan")]
    ids = np.unique(labels)
    members = [np.where(labels == i)[0] for i in ids]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), size=len(ids))
        stats.append(stat(v[np.concatenate([members[i] for i in pick])]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return [float(low), float(high)]


def holm(pvalues: dict) -> dict:
    """Holm correction: makes several p-values safe to read together (fewer false alarms)."""
    items = sorted(((p, k) for k, p in pvalues.items() if np.isfinite(p)))
    m, out, running = len(items), {}, 0.0
    for rank, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    return {k: out.get(k, float("nan")) for k in pvalues}


def _index(rows: list[dict]) -> dict:
    """(image_id, source, method) -> row"""
    return {(r["image_id"], r["source"], r["method"]): r for r in rows}


def _paired_p(a, b) -> float:
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6 or np.allclose(a[ok], b[ok]):
        return float("nan")
    try:
        return float(wilcoxon(a[ok], b[ok]).pvalue)
    except (ValueError, AttributeError):  # e.g. too few non-zero differences
        return float("nan")


def agreement_study(rows: list[dict], dataset: str, reference_sources: list[str], clusters_key: str | None) -> dict:
    """Relative difference of each measure between the CNN mask and each reference mask.

    For CHASE also: observer 2 vs observer 1 (the human baseline).
    """
    idx = _index([r for r in rows if r["dataset"] == dataset])
    images = sorted({k[0] for k in idx})
    out = {"n_images": len(images), "pairs": {}}
    pairs = [(f"cnn_vs_{ref}", "cnn", ref) for ref in reference_sources]
    if dataset == "chase":
        pairs.append(("human_observer2_vs_observer1", "observer2", "observer1"))
    for name, a_src, b_src in pairs:
        out["pairs"][name] = {}
        for method in mm.METHODS:
            per_measure = {}
            for measure in mm.ALL_MEASURES:
                values, clusters = [], []
                for image in images:
                    ra, rb = idx.get((image, a_src, method)), idx.get((image, b_src, method))
                    if ra is None or rb is None:
                        continue
                    values.append(mm.relative_difference(ra[measure], rb[measure]))
                    clusters.append(ra[clusters_key] if clusters_key else image)
                per_measure[measure] = {
                    "median": float(np.nanmedian(values)) if values else float("nan"),
                    "mean": float(np.nanmean(values)) if values else float("nan"),
                    "ci95_median": cluster_bootstrap(values, clusters),
                    "n": int(np.isfinite(values).sum()) if values else 0,
                    "values": values,
                }
            out["pairs"][name][method] = per_measure
    # Paired tests: MedSkel vs thinning, for every pair and skeleton measure.
    out["tests_medskel_vs_thinning"] = {
        name: {m: _paired_p(out["pairs"][name]["medskel"][m]["values"], out["pairs"][name]["thinning"][m]["values"]) for m in mm.SKELETON_MEASURES}
        for name in out["pairs"]
    }
    if dataset == "chase":
        # Does replacing a human by the CNN move a measure more than replacing one human by the other?
        out["tests_cnn_vs_human"] = {
            method: {m: _paired_p(out["pairs"]["cnn_vs_observer1"][method][m]["values"], out["pairs"]["human_observer2_vs_observer1"][method][m]["values"]) for m in mm.ALL_MEASURES}
            for method in mm.METHODS
        }
    for name in out["pairs"]:  # keep the JSON readable: drop the raw lists
        for method in out["pairs"][name]:
            for m in out["pairs"][name][method]:
                out["pairs"][name][method][m].pop("values")
    return out


def _deduplicate(ids: list[str]) -> set[str]:
    """Keep one copy of each byte-identical photograph (list from the v1 data audit)."""
    audit = REPO / "results" / "data_audit.json"
    drop = set()
    if audit.exists():
        for group in json.loads(audit.read_text()).get("identical_image_files", []):
            drop.update(sorted(group)[1:])
    return {i for i in ids if i not in drop}


def group_table(rows: list[dict], method: str) -> dict:
    """Per-group medians, Kruskal-Wallis (any difference?) and each disease vs normal (Mann-Whitney, Holm)."""
    from scipy.stats import kruskal, mannwhitneyu

    out = {}
    for measure in mm.ALL_MEASURES:
        by_group = {g: np.array([r[measure] for r in rows if r["group"] == g and r["method"] == method], float) for g in GROUPS}
        by_group = {g: v[np.isfinite(v)] for g, v in by_group.items()}
        entry = {"groups": {g: {"n": int(len(v)), "median": float(np.median(v)) if len(v) else float("nan"),
                                "q1": float(np.percentile(v, 25)) if len(v) else float("nan"),
                                "q3": float(np.percentile(v, 75)) if len(v) else float("nan")} for g, v in by_group.items()}}
        present = [v for v in by_group.values() if len(v) >= 3]
        entry["kruskal_p"] = float(kruskal(*present).pvalue) if len(present) >= 2 else float("nan")
        raw, effect = {}, {}
        for g in ("A", "D", "G"):
            a, b = by_group[g], by_group["N"]
            if len(a) >= 3 and len(b) >= 3:
                u = mannwhitneyu(a, b, alternative="two-sided")
                raw[g] = float(u.pvalue)
                effect[g] = float(2 * u.statistic / (len(a) * len(b)) - 1)  # rank-biserial: -1..+1, 0 = no shift
            else:
                raw[g], effect[g] = float("nan"), float("nan")
        entry["vs_normal_p_holm"] = holm(raw)
        entry["vs_normal_rank_biserial"] = effect
        out[measure] = entry
    return out


def _features(rows: list[dict], method: str, ids: list[str], source: str) -> np.ndarray:
    idx = _index(rows)
    return np.array([[idx[(i, source, method)][m] for m in mm.ALL_MEASURES] for i in ids], float)


def classifier_study(rows: list[dict], method: str, n_permutations: int = 500, seed: int = 0) -> dict:
    """Can a simple model guess the disease group from the 6 measurements?

    Model: logistic regression on standardized measurements (deliberately simple,
    so the result is about the measurements, not about a clever model).
    Train: FIVES training images (expert masks). Test: the 200 test images,
    (a) with expert masks, (b) with CNN masks. Chance level for 4 balanced
    groups is a balanced accuracy of 0.25; the shuffled-label control estimates
    it empirically.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = [r for r in rows if r["dataset"] == "fives_train" and r["method"] == method]
    test = [r for r in rows if r["dataset"] == "fives_test" and r["method"] == method]
    train_ids = sorted(_deduplicate([r["image_id"] for r in train]))
    test_ids = sorted(_deduplicate({r["image_id"] for r in test}))
    if len(train_ids) < 40 or not test_ids:
        return {"skipped": "needs at least 40 measured training images (do not use --skip-train)"}
    label = {r["image_id"]: r["group"] for r in train + test}
    x_train, y_train = _features(rows, method, train_ids, "expert"), np.array([label[i] for i in train_ids])
    y_test = np.array([label[i] for i in test_ids])

    def model():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))

    clf = model().fit(x_train, y_train)
    out = {"n_train": len(train_ids), "n_test": len(test_ids), "features": list(mm.ALL_MEASURES), "chance_balanced_accuracy": 0.25}
    for source in ("expert", "cnn"):
        x_test = _features(rows, method, test_ids, source)
        pred = clf.predict(x_test)
        prob = clf.predict_proba(x_test)
        n_idx, d_idx = list(clf.classes_).index("N"), list(clf.classes_).index("D")
        both = np.isin(y_test, ["N", "D"])
        out[f"test_{source}_masks"] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_auc_one_vs_rest": float(roc_auc_score(y_test, prob, multi_class="ovr", labels=clf.classes_)),
            # Healthy vs diabetic retinopathy only: how well does P(D) - P(N) separate the two groups?
            "auc_D_vs_N": float(roc_auc_score(y_test[both] == "D", prob[both, d_idx] - prob[both, n_idx])),
            "confusion_matrix": {"labels": list(clf.classes_), "rows_true_cols_predicted": confusion_matrix(y_test, pred, labels=clf.classes_).tolist()},
        }
    # Control: shuffle the training labels, so the measurements carry no information about the group.
    rng = np.random.default_rng(seed)
    x_test = _features(rows, method, test_ids, "expert")
    null = [balanced_accuracy_score(y_test, model().fit(x_train, rng.permutation(y_train)).predict(x_test)) for _ in range(n_permutations)]
    observed = out["test_expert_masks"]["balanced_accuracy"]
    out["shuffled_label_control"] = {
        "n_permutations": n_permutations,
        "mean_balanced_accuracy": float(np.mean(null)),
        "p_value": float((1 + np.sum(np.array(null) >= observed)) / (1 + n_permutations)),
    }
    out["standardized_coefficients"] = {c: dict(zip(mm.ALL_MEASURES, map(float, w))) for c, w in zip(clf.classes_, clf[-1].coef_)}
    return out


def _fmt(x, digits=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _pfmt(p):
    return "n/a" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def write_summary(analysis: dict, path: Path) -> None:
    """A Markdown summary. Numbers only, no conclusions beyond what the tables show."""
    lines = ["# Measurement study: results", "", "Generated by `python -m medskel_ai.study analyze`. All lengths and widths in pixels.",
             f"MedSkel code: {mm.UPSTREAM_URL} at commit {analysis['meta']['medskel_commit']}, epsilon {mm.MEDSKEL_EPSILON}.", ""]
    names = {"total_length": "total length", "n_bifurcations": "bifurcations", "median_calibre": "median width",
             "median_tortuosity": "median tortuosity", "vessel_density": "vessel density", "fractal_dimension": "fractal dimension"}

    def agreement_block(title, block, pairs):
        lines.extend([f"## {title}", "", f"{block['n_images']} images. Median relative difference |a-b|/mean(a,b), lower = measurements move less. 95% interval in brackets.", ""])
        lines.append("| Measure | " + " | ".join(f"{p} ({m})" for p in pairs for m in mm.METHODS) + " |")
        lines.append("|---" * (1 + len(pairs) * len(mm.METHODS)) + "|")
        for measure in mm.ALL_MEASURES:
            cells = []
            for p in pairs:
                for m in mm.METHODS:
                    e = block["pairs"][p][m][measure]
                    cells.append(f"{_fmt(e['median'])} [{_fmt(e['ci95_median'][0])}–{_fmt(e['ci95_median'][1])}]")
            lines.append(f"| {names[measure]} | " + " | ".join(cells) + " |")
        lines.extend(["", "Wilcoxon p, MedSkel vs thinning (same images):", ""])
        for p in pairs:
            tests = block["tests_medskel_vs_thinning"][p]
            lines.append(f"- {p}: " + ", ".join(f"{names[m]} {_pfmt(v)}" for m, v in tests.items()))
        lines.append("")

    if "chase" in analysis:
        agreement_block("Q1. CHASE_DB1: CNN mask vs human masks, and human vs human", analysis["chase"],
                        ["cnn_vs_observer1", "cnn_vs_observer2", "human_observer2_vs_observer1"])
        lines.extend(["Wilcoxon p, (CNN vs observer 1) vs (observer 2 vs observer 1), same images:", ""])
        for method, tests in analysis["chase"]["tests_cnn_vs_human"].items():
            lines.append(f"- {method}: " + ", ".join(f"{names[m]} {_pfmt(v)}" for m, v in tests.items()))
        lines.extend(["", "Note: the original MedSkel repository reported median human-vs-human total-length differences of 0.093 (thinning) and 0.032 (MedSkel, epsilon 4). The human column above should be close to those numbers; if it is not, check the setup before reading anything else.", ""])
    if "fives_test" in analysis:
        agreement_block("Q2. FIVES test: CNN mask vs expert mask", analysis["fives_test"], ["cnn_vs_expert"])
    for key, title in (("groups_expert", "Q3a. Disease groups, FIVES expert masks (train + test)"), ("groups_cnn", "Q3b. Disease groups, CNN masks (FIVES test)")):
        if key not in analysis:
            continue
        for method, table in analysis[key].items():
            lines.extend([f"## {title}, {method}", "", "Median [Q1–Q3] per group. Kruskal-Wallis: any difference between the 4 groups. Holm-corrected Mann-Whitney: each disease vs normal.", ""])
            lines.append("| Measure | " + " | ".join(f"{GROUP_NAMES[g]} (n={table['vessel_density']['groups'][g]['n']})" for g in GROUPS) + " | Kruskal p | A vs N | D vs N | G vs N |")
            lines.append("|---" * 9 + "|")
            for measure, e in table.items():
                cells = [f"{_fmt(e['groups'][g]['median'], 4)} [{_fmt(e['groups'][g]['q1'], 4)}–{_fmt(e['groups'][g]['q3'], 4)}]" for g in GROUPS]
                cells += [_pfmt(e["kruskal_p"])] + [_pfmt(e["vs_normal_p_holm"][g]) for g in ("A", "D", "G")]
                lines.append(f"| {names[measure]} | " + " | ".join(cells) + " |")
            lines.append("")
    if "classifier" in analysis:
        lines.extend(["## Q3c. Guessing the disease group from the 6 measurements", "",
                      "Logistic regression, trained on training images (expert masks), tested once on the test images (with expert masks, then with CNN masks). Chance = 0.25 balanced accuracy.", "",
                      "| Skeleton method | Test masks | Balanced accuracy | Macro AUC | AUC healthy vs DR | Shuffled-label control (mean, p) |", "|---|---|---|---|---|---|"])
        for method, res in analysis["classifier"].items():
            if "skipped" in res:
                lines.append(f"| {method} | – | skipped: {res['skipped']} | | | |")
                continue
            control = res["shuffled_label_control"]
            for source in ("expert", "cnn"):
                r = res[f"test_{source}_masks"]
                lines.append(f"| {method} | {source} | {_fmt(r['balanced_accuracy'])} | {_fmt(r['macro_auc_one_vs_rest'])} | {_fmt(r['auc_D_vs_N'])} | "
                             + (f"{_fmt(control['mean_balanced_accuracy'])}, p {_pfmt(control['p_value'])} |" if source == "expert" else "|"))
        lines.append("")
    lines.extend(["## Timing", "", "| Method | median seconds per mask | n |", "|---|---|---|"])
    for method, t in analysis["timing"].items():
        lines.append(f"| {method} | {_fmt(t['median_seconds'], 1)} | {t['n']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figures(rows: list[dict], analysis: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: CHASE relative differences, human vs CNN, per method.
    if "chase" in analysis:
        measures = list(mm.SKELETON_MEASURES)
        pairs = ["human_observer2_vs_observer1", "cnn_vs_observer1", "cnn_vs_observer2"]
        labels = {"human_observer2_vs_observer1": "human 2 vs human 1", "cnn_vs_observer1": "CNN vs human 1", "cnn_vs_observer2": "CNN vs human 2"}
        fig, axes = plt.subplots(1, len(measures), figsize=(4 * len(measures), 3.8))
        for ax, measure in zip(axes, measures):
            x = np.arange(len(pairs))
            for k, method in enumerate(mm.METHODS):
                med = [analysis["chase"]["pairs"][p][method][measure]["median"] for p in pairs]
                ci = np.array([analysis["chase"]["pairs"][p][method][measure]["ci95_median"] for p in pairs])
                ax.bar(x + (k - 0.5) * 0.38, med, 0.38, yerr=[np.array(med) - ci[:, 0], ci[:, 1] - np.array(med)], capsize=3, label=method)
            ax.set_xticks(x, [labels[p] for p in pairs], rotation=20, ha="right", fontsize=8)
            ax.set_title(measure.replace("_", " "), fontsize=10)
            ax.set_ylabel("median relative difference")
        axes[0].legend(fontsize=8)
        fig.suptitle("CHASE_DB1: how much each measurement moves when the mask changes (lower = more stable)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "chase_agreement.png", dpi=130)
        plt.close(fig)

    # Figure 2: measurements by disease group (expert masks, thinning).
    expert = [r for r in rows if r["source"] == "expert" and r["method"] == "thinning"]
    if expert:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, measure in zip(axes.ravel(), mm.ALL_MEASURES):
            data = [[r[measure] for r in expert if r["group"] == g and np.isfinite(r[measure])] for g in GROUPS]
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(GROUPS) + 1), [GROUP_NAMES[g] for g in GROUPS], fontsize=8, rotation=15)
            ax.set_title(measure.replace("_", " "), fontsize=10)
        fig.suptitle("FIVES expert masks: measurements by group (thinning skeleton, pixels)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "groups_expert.png", dpi=130)
        plt.close(fig)


REFERENCE = OUT_DIR / "reference_cnn_medskel.json"


def build_reference(rows: list[dict]) -> dict:
    """Reference values for the app: FIVES TEST images, CNN masks, MedSkel centre lines, per group.

    The app measures a CNN mask with MedSkel, so its reference must be made the
    same way (expert masks differ from CNN masks by ~6% in density). Test images
    only, because training images were seen by the CNN.
    """
    rows = [r for r in rows if r["dataset"] == "fives_test" and r["source"] == "cnn" and r["method"] == "medskel"]
    keep = _deduplicate({r["image_id"] for r in rows})
    rows = [r for r in rows if r["image_id"] in keep]
    groups = {}
    for g in GROUPS:
        members = [r for r in rows if r["group"] == g]
        groups[g] = {"name": GROUP_NAMES[g], "n": len(members),
                     "values": {m: sorted(float(r[m]) for r in members if np.isfinite(r[m])) for m in mm.ALL_MEASURES}}
    return {"description": "FIVES test images, CNN masks (final U-Net, flips, threshold 0.30), MedSkel epsilon 4, pixels at 2048 px scale",
            "groups": groups}


def step_analyze(args) -> None:
    rows = read_measurements()
    if not rows:
        raise SystemExit("No measurements yet. Run the measure step first.")
    analysis = {"meta": {"medskel_commit": mm.upstream_commit(args.medskel_repo) or "unknown", "medskel_epsilon": mm.MEDSKEL_EPSILON, "methods": list(mm.METHODS)}}
    if any(r["dataset"] == "chase" for r in rows):
        analysis["chase"] = agreement_study(rows, "chase", ["observer1", "observer2"], clusters_key="child")
    if any(r["dataset"] == "fives_test" for r in rows):
        analysis["fives_test"] = agreement_study(rows, "fives_test", ["expert"], clusters_key=None)
    expert = [r for r in rows if r["dataset"] in ("fives_train", "fives_test") and r["source"] == "expert"]
    keep = _deduplicate({r["image_id"] for r in expert})
    expert = [r for r in expert if r["image_id"] in keep]
    if expert:
        analysis["groups_expert"] = {m: group_table(expert, m) for m in mm.METHODS}
    cnn = [r for r in rows if r["dataset"] == "fives_test" and r["source"] == "cnn"]
    keep_cnn = _deduplicate({r["image_id"] for r in cnn})
    cnn = [r for r in cnn if r["image_id"] in keep_cnn]
    if cnn:
        analysis["groups_cnn"] = {m: group_table(cnn, m) for m in mm.METHODS}
    if any(r["dataset"] == "fives_train" for r in rows):
        analysis["classifier"] = {m: classifier_study(rows, m, n_permutations=args.permutations) for m in mm.METHODS}
    analysis["timing"] = {m: {"median_seconds": float(np.median([r["seconds"] for r in rows if r["method"] == m])),
                              "n": sum(r["method"] == m for r in rows)} for m in mm.METHODS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "analysis.json").write_text(json.dumps(analysis, indent=1))
    write_summary(analysis, OUT_DIR / "summary.md")
    figures(rows, analysis)
    if any(r["dataset"] == "fives_test" and r["source"] == "cnn" for r in rows):
        REFERENCE.write_text(json.dumps(build_reference(rows), indent=1))
    print(f"analyze: wrote {OUT_DIR / 'summary.md'}, analysis.json and figures", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN mask -> skeleton (thinning | MedSkel) -> vessel measurements -> statistics")
    parser.add_argument("step", choices=["predict", "measure", "analyze", "all"])
    parser.add_argument("--data", default=r"C:\data\FIVES")
    parser.add_argument("--chase", default=r"C:\data\CHASEDB1")
    parser.add_argument("--run", default=str(REPO / "runs" / "e2_long"), help="folder with the final model best.pt")
    parser.add_argument("--medskel-repo", default=None, help="clone of the original medskel repo (default: ../medskel)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--skip-train", action="store_true", help="do not measure the 600 training masks (no classifier then)")
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()
    if args.step in ("predict", "all"):
        step_predict(args)
    if args.step in ("measure", "all"):
        step_measure(args)
    if args.step in ("analyze", "all"):
        step_analyze(args)


if __name__ == "__main__":
    main()
