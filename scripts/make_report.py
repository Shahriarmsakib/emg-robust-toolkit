#!/usr/bin/env python3
"""Regenerate the paper's Table 1 and Fig. 2 from EMG-Robust sweep results.

    python scripts/make_report.py                     # reads results/*.csv, writes report/
    python scripts/make_report.py --readme README.md  # also refreshes the results table in the README
    python scripts/make_report.py --metric fingertip  # same report for fingertip error

Needs only pandas, NumPy and Matplotlib -- no dataset, checkpoints or GPU.

Percent change is computed for every run against the clean baseline of the same model and user
session, then averaged over seeds:

    pct_change = (perturbed_error - clean_error) / clean_error * 100

Negative values (a perturbation that slightly lowers the error) are kept, not clipped.

Outputs (in --out):
    runs_with_pct.csv   every run, with clean baseline and % change for landmark and fingertip error
    summary.csv         one row per model x user x perturbation x level (mean / SD in mm and %)
    table1.tex/.md      clean + severe-level error in mm, mean +- SD over seeds for stochastic perturbations
    fig2_heatmap.pdf/.png  % change for every condition, one annotated grayscale heatmap
    fig2.tex            LaTeX figure block (caption + \\Description) for the heatmap
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

REQUIRED = {"model", "user_session", "perturbation", "level", "value", "seed", "landmark_mm", "fingertip_mm"}
METRICS = ("landmark", "fingertip")

PERT_ORDER = ["noise", "dropout", "amplitude", "temporal_shift"]
PERT_NAME = {"noise": "Additive Noise", "dropout": "Channel Dropout",
             "amplitude": "Amplitude Scaling", "temporal_shift": "Temporal Shift"}
PERT_SHORT = {"noise": "Noise", "dropout": "Dropout", "amplitude": "Amp.", "temporal_shift": "Shift"}
PERT_PROSE = {"noise": "noise", "dropout": "dropout", "amplitude": "amplitude scaling",
              "temporal_shift": "temporal shift"}
LEVEL_ORDER = ["mild", "moderate", "severe"]
LEVEL_SHORT = {"mild": "Mild", "moderate": "Mod.", "severe": "Sev."}
MODEL_NAME = {"neuropose": "NeuroPose", "vemg2pose": "vemg2pose"}
WINDOW_S = 5  # evaluation window length (s); sigma_c is computed within each window

FONTS = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]  # first one installed is used


# --------------------------------------------------------------------------- helpers
def ordered(items, preferred):
    items = list(dict.fromkeys(items))
    return [x for x in preferred if x in items] + sorted(x for x in items if x not in preferred)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def tex_escape(s):
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def param_label(pert, values, fs):
    """Swept parameter for each severity level, in the units used in the paper."""
    v = [float(x) for x in values]
    j = " / ".join
    if pert == "noise":
        return "\u03b1 " + j(f"{x:g}" for x in v)                # noise std = alpha * sigma_c
    if pert == "dropout":
        return j(f"{x:g}" for x in v) + " ch."
    if pert == "amplitude":
        return "\u00d7" + j(f"{x:g}" for x in v)
    if pert == "temporal_shift":
        return j(f"{1000 * x / fs:g}" for x in v) + " ms"         # CSV stores the shift in samples
    return j(f"{x:g}" for x in v)


# --------------------------------------------------------------------------- data
def load_runs(results_dir):
    """Read every sweep CSV in results_dir. User labels (U1, U2, ...) come from 'user<N>' in the file name."""
    frames = []
    for f in sorted(Path(results_dir).glob("*.csv")):
        df = pd.read_csv(f, dtype={"dropped_channels": str, "user_session": str})
        missing = REQUIRED - set(df.columns)
        if missing:
            print(f"  skipping {f.name}: missing columns {sorted(missing)}")
            continue
        m = re.search(r"user[_-]?(\d+)", f.stem, re.IGNORECASE)
        df["user"] = f"U{m.group(1)}" if m else None
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        sys.exit(f"No sweep CSVs with columns {sorted(REQUIRED)} found in {results_dir}")
    runs = pd.concat(frames, ignore_index=True)

    # Sessions whose file name has no user number get the next free label, in order of appearance.
    taken = set(runs.user.dropna())
    nxt = 1
    for sess in runs.loc[runs.user.isna(), "user_session"].unique():
        while f"U{nxt}" in taken:
            nxt += 1
        runs.loc[runs.user_session == sess, "user"] = f"U{nxt}"
        taken.add(f"U{nxt}")
    per_session = runs.groupby("user_session").user.nunique()
    per_label = runs.groupby("user").user_session.nunique()
    if (per_session > 1).any() or (per_label > 1).any():
        sys.exit("User labels and user_session IDs don't match one-to-one; check the 'user<N>' file names.")

    runs["model_name"] = runs.model.map(lambda m: MODEL_NAME.get(str(m).lower(), str(m)))
    return runs


def add_pct_change(runs):
    """Attach each run's clean baseline (same model + session) and its % change, for both metrics."""
    base = runs[runs.perturbation == "baseline"]
    counts = base.groupby(["model", "user_session"]).size()
    if (counts != 1).any():
        sys.exit(f"Expected exactly one baseline row per model x session:\n{counts[counts != 1]}")
    b = base.set_index(["model", "user_session"])
    pert = runs[runs.perturbation != "baseline"].copy()
    key = pd.MultiIndex.from_frame(pert[["model", "user_session"]])
    if (~key.isin(b.index)).any():
        missing = sorted(set(key[~key.isin(b.index)]))
        sys.exit(f"No clean baseline for: {missing}")
    for m in METRICS:
        clean = b[f"{m}_mm"].reindex(key).to_numpy()
        pert[f"{m}_clean_mm"] = clean
        pert[f"{m}_pct"] = 100.0 * (pert[f"{m}_mm"].to_numpy() - clean) / clean
    return pert


def summarize(pert):
    keys = ["model", "model_name", "user", "user_session", "perturbation", "level"]
    rows = []
    for k, d in pert.groupby(keys, sort=False):
        r = dict(zip(keys, k), value=d.value.iloc[0], n_runs=len(d))
        if "dropped_channels" in d and k[4] == "dropout":
            r["n_distinct_channel_sets"] = d.dropped_channels.nunique()
        for m in METRICS:
            r[f"{m}_clean_mm"] = d[f"{m}_clean_mm"].iloc[0]
            r[f"{m}_mm_mean"] = d[f"{m}_mm"].mean()
            r[f"{m}_mm_sd"] = d[f"{m}_mm"].std(ddof=1) if len(d) > 1 else np.nan
            r[f"{m}_pct_mean"] = d[f"{m}_pct"].mean()
            r[f"{m}_pct_sd"] = d[f"{m}_pct"].std(ddof=1) if len(d) > 1 else np.nan
        rows.append(r)
    S = pd.DataFrame(rows)
    rank = lambda col, pref: S[col].map({x: i for i, x in enumerate(ordered(S[col], pref))})
    S = S.assign(_p=rank("perturbation", PERT_ORDER), _l=rank("level", LEVEL_ORDER),
                 _m=rank("model_name", list(MODEL_NAME.values())),
                 _u=S.user.map(lambda u: natural_key(u)[1] if len(natural_key(u)) > 1 else 0))
    return S.sort_values(["_p", "_u", "_m", "_l"]).drop(columns=["_p", "_l", "_m", "_u"]).reset_index(drop=True)


def sanity_notes(pert, S):
    """Things worth stating in the paper or README; printed, never fatal."""
    notes = []
    for p, d in S.groupby("perturbation"):
        n = sorted(d.n_runs.unique())
        if len(n) > 1:
            notes.append(f"{p}: cells have different numbers of runs {n}")
    dd = pert[pert.perturbation == "dropout"] if "dropped_channels" in pert else pert.iloc[:0]
    msgs = set()
    for (lvl, _, _), d in dd.groupby(["level", "model", "user_session"]):
        dup = d.groupby("dropped_channels").seed.apply(lambda s: sorted(int(x) for x in s))
        dup = dup[dup.map(len) > 1]
        for chans, seeds in dup.items():
            msgs.add(f"dropout {lvl}: seeds {', '.join(map(str, seeds))} drop the same channel(s) {chans} "
                     f"-> {d.dropped_channels.nunique()} distinct channel sets in {len(d)} runs")
    notes += sorted(msgs)
    return notes


# --------------------------------------------------------------------------- Table 1
def table1(S, metric):
    sev = S[S.level == "severe"]
    perts = ordered(sev.perturbation, PERT_ORDER)
    stoch = {p: sev[sev.perturbation == p].n_runs.max() > 1 for p in perts}
    n_seeds = int(sev[sev.perturbation.isin([p for p in perts if stoch[p]])].n_runs.max() or 1)
    models = ordered(sev.model_name, list(MODEL_NAME.values()))
    users = sorted(sev.user.unique(), key=natural_key)

    def cell(r, p, tex):
        mean, sd = r.loc[p, f"{metric}_mm_mean"], r.loc[p, f"{metric}_mm_sd"]
        if not stoch[p]:
            return f"{mean:.2f}"
        return f"{mean:.2f} & {sd:.2f}" if tex else f"{mean:.2f} ± {sd:.2f}"

    st = [PERT_PROSE.get(p, p) for p in perts if stoch[p]]
    de = [PERT_PROSE.get(p, p) for p in perts if not stoch[p]]
    cap = (f"{metric.capitalize()} error (mm) on clean input and at the \\emph{{severe}} level of each perturbation.")
    if st:
        cap += f" {' and '.join(st).capitalize()}: mean\\,$\\pm$\\,SD over {n_seeds} seeds"
        cap += f"; {' and '.join(de)} {'is' if len(de) == 1 else 'are'} deterministic." if de else "."

    ncol = sum(2 if stoch[p] else 1 for p in perts)
    spec = "@{}l c r " + " ".join("r@{\\,$\\pm$\\,}l" if stoch[p] else "r" for p in perts) + "@{}"
    hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{{PERT_SHORT.get(p, p)}}}" if stoch[p] else PERT_SHORT.get(p, p)
                     for p in perts)
    tex = ["% Generated by scripts/make_report.py -- edit the script, not this file.",
           "% Needs \\usepackage{multirow}; acmart already loads booktabs.",
           "\\begin{table}[t]",
           f"\\caption{{{cap}}}",
           f"\\label{{tab:robustness}}",
           "\\small\\centering",
           "\\setlength{\\tabcolsep}{3.3pt}",
           f"\\begin{{tabular}}{{{spec}}}",
           "\\toprule",
           f" & & & \\multicolumn{{{ncol}}}{{c}}{{Severe perturbation}} \\\\",
           f"\\cmidrule(l{{2pt}}){{4-{3 + ncol}}}",
           f"Model & User & Clean & {hdr} \\\\",
           "\\midrule"]
    md = [f"{metric.capitalize()} error (mm), clean vs. **severe** level"
          + (f" (± = SD over {n_seeds} seeds)" if st else "") + ":", "",
          "| Model | User | Clean | " + " | ".join(PERT_SHORT.get(p, p) for p in perts) + " |",
          "|---|---|--:|" + "--:|" * len(perts)]
    for mi, m in enumerate(models):
        for ui, u in enumerate(users):
            r = sev[(sev.model_name == m) & (sev.user == u)].set_index("perturbation")
            if r.empty:
                continue
            clean = f"{r[f'{metric}_clean_mm'].iloc[0]:.2f}"
            name = f"\\multirow{{{len(users)}}}{{*}}{{{tex_escape(m)}}}" if ui == 0 else ""
            tex.append(f"{name} & {u} & {clean} & " + " & ".join(cell(r, p, True) for p in perts) + " \\\\")
            md.append(f"| {m if ui == 0 else ''} | {u} | {clean} | " + " | ".join(cell(r, p, False) for p in perts) + " |")
        if mi < len(models) - 1:
            tex.append("\\addlinespace[2pt]")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(tex) + "\n", "\n".join(md) + "\n"


# --------------------------------------------------------------------------- Fig. 2
def heatmap(S, metric, fs, pdf_path, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONTS,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    INK, SUB, RULE = "#1a1a1a", "#666666", "#3c3c3c"
    FS_CELL, FS_LAB, FS_HEAD, FS_SUB = 7.0, 7.3, 7.5, 6.2
    cw, ch, mgap, ggap, top, COL_W = 0.335, 0.185, 0.09, 0.06, 0.40, 3.33  # inches; ACM column = 3.33 in

    perts = ordered(S.perturbation, PERT_ORDER)
    levels = ordered(S.level, LEVEL_ORDER)
    models = ordered(S.model_name, list(MODEL_NAME.values()))
    users = sorted(S.user.unique(), key=natural_key)
    col = f"{metric}_pct_mean"
    val = {(r.perturbation, r.user, r.model_name, r.level): getattr(r, col) for r in S.itertuples()}
    params = {p: param_label(p, [S[(S.perturbation == p) & (S.level == lv)].value.iloc[0]
                                 for lv in levels if ((S.perturbation == p) & (S.level == lv)).any()], fs)
              for p in perts}
    unit = [f"Increase in {metric}", "error vs. clean (%)"]

    fig = plt.figure(figsize=(COL_W, 1))
    rend = fig.canvas.get_renderer()

    def tw(s, size):
        t = fig.text(0, 0, s, fontsize=size)
        w = t.get_window_extent(renderer=rend).width / fig.dpi
        t.remove()
        return w

    lab_w = max([tw(PERT_NAME.get(p, p), FS_LAB) for p in perts] + [tw(params[p], FS_SUB) for p in perts]
                + [tw(s, FS_SUB) for s in unit])
    user_w = max(tw(u, FS_SUB + 0.4) for u in users)
    grid_w = len(models) * len(levels) * cw + (len(models) - 1) * mgap
    W = max(COL_W, lab_w + 0.12 + user_w + 0.05 + grid_w + 0.01)
    H = top + len(perts) * len(users) * ch + (len(perts) - 1) * ggap + 0.02
    fig.set_size_inches(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    x0 = W - grid_w - 0.01

    finite = [v for v in val.values() if np.isfinite(v)]
    vmax = max(5.0, 5 * math.ceil(max(finite) / 5))
    cmap = colors.LinearSegmentedColormap.from_list("g", [cm.Greys(0.02), cm.Greys(0.82)])
    norm = colors.Normalize(0, vmax, clip=True)

    def lum(rgb):
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    ink_l = lum(colors.to_rgb(INK))
    cx = lambda mi, li: x0 + mi * (len(levels) * cw + mgap) + li * cw

    ax.text(0, 0.10, unit[0], ha="left", va="center", fontsize=FS_SUB, color=SUB)
    ax.text(0, 0.21, unit[1], ha="left", va="center", fontsize=FS_SUB, color=SUB)
    for mi, m in enumerate(models):
        xa, xb = cx(mi, 0), cx(mi, len(levels) - 1) + cw
        ax.text((xa + xb) / 2, 0.13, m, ha="center", va="center", fontsize=FS_HEAD, color=INK)
        ax.plot([xa + 0.02, xb - 0.02], [0.22, 0.22], color=RULE, lw=0.6, solid_capstyle="butt")
        for li, lv in enumerate(levels):
            ax.text(cx(mi, li) + cw / 2, 0.31, LEVEL_SHORT.get(lv, lv.title()), ha="center", va="center",
                    fontsize=FS_SUB + 0.4, color=SUB)

    y = top
    for p in perts:
        yg = y
        for u in users:
            for mi, m in enumerate(models):
                for li, lv in enumerate(levels):
                    v = val.get((p, u, m, lv), np.nan)
                    fc = cmap(norm(v)) if np.isfinite(v) else (1, 1, 1, 1)
                    ax.add_patch(Rectangle((cx(mi, li), y), cw, ch, facecolor=fc, edgecolor="white", lw=0.8))
                    # dark text unless its contrast falls below 3.5:1, then white
                    txt = INK if (lum(fc[:3]) + 0.05) / (ink_l + 0.05) >= 3.5 else "white"
                    s = f"{v:.1f}".replace("-", "\u2212") if np.isfinite(v) else "\u2013"
                    ax.text(cx(mi, li) + cw / 2, y + ch / 2 + 0.004, s, ha="center", va="center",
                            fontsize=FS_CELL, color=txt)
            ax.text(x0 - 0.05, y + ch / 2 + 0.004, u, ha="right", va="center", fontsize=FS_SUB + 0.4, color=SUB)
            y += ch
        yc = (yg + y) / 2
        ax.text(0, yc - 0.045, PERT_NAME.get(p, p), ha="left", va="center", fontsize=FS_LAB, color=INK)
        ax.text(0, yc + 0.065, params[p], ha="left", va="center", fontsize=FS_SUB, color=SUB)
        y += ggap

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    k = max(val, key=lambda kk: val[kk] if np.isfinite(val[kk]) else -np.inf)
    return dict(perts=perts, users=users, models=models, levels=levels, max_key=k, max_val=val[k], width_in=W)


def fig2_tex(info, S, metric):
    stoch = [PERT_PROSE.get(p, p) for p in info["perts"] if S[S.perturbation == p].n_runs.max() > 1]
    n = int(S.n_runs.max())
    cap = (f"Increase in {metric} error (\\%) relative to each user and model clean baseline (darker = larger). "
           "Rows: perturbation, with its swept parameter, and held-out user; columns: model and severity.")
    if "noise" in info["perts"]:
        cap += (f" Noise std is $\\alpha\\,\\sigma_c$, where $\\sigma_c$ is channel $c$'s SD within the "
                f"{WINDOW_S} s window.")
    if stoch:
        cap += f" {' and '.join(stoch).capitalize()} cells average {n} seeds."
    if "n_distinct_channel_sets" in S.columns:
        drp = S[(S.perturbation == "dropout") & S.n_distinct_channel_sets.notna()]
        coll = drp[drp.n_distinct_channel_sets < drp.n_runs]
        if not coll.empty:
            lvls = ordered(coll.level, LEVEL_ORDER)
            n_sets, n_runs = int(coll.n_distinct_channel_sets.min()), int(coll.n_runs.max())
            cap += (f" At the {' and '.join(lvls)} level{'s' if len(lvls) > 1 else ''}, a seed collision leaves "
                    f"only {n_sets} distinct channel set{'s' if n_sets != 1 else ''} across the {n_runs} runs "
                    "(see repository README).")
    p, u, m, lv = info["max_key"]
    desc = (f"Grayscale heatmap of the percent increase in {metric} error. Rows are "
            f"{len(info['perts'])} perturbations (" + ", ".join(PERT_NAME.get(x, x).lower() for x in info["perts"])
            + f"), each split by {len(info['users'])} users; columns are {len(info['models'])} models ("
            + ", ".join(info["models"]) + f"), each at {len(info['levels'])} severity levels. Darker cells are "
            f"larger increases; the largest is {info['max_val']:.1f}\\% ({m}, {u}, {PERT_NAME.get(p, p).lower()}, {lv}).")
    return ("% Generated by scripts/make_report.py -- edit the script, not this file.\n"
            "% acmart requires \\Description for every figure (accessibility).\n"
            "\\begin{figure}[t]\n  \\centering\n"
            f"  \\includegraphics[width=\\columnwidth]{{fig2_heatmap{'' if metric == 'landmark' else '_' + metric}.pdf}}\n"
            f"  \\caption{{{cap}}}\n  \\Description{{{desc}}}\n  \\label{{fig:heatmap}}\n\\end{{figure}}\n")


def update_readme(path, md):
    start, end = "<!-- report:table1:start -->", "<!-- report:table1:end -->"
    text = Path(path).read_text(encoding="utf-8")
    if start not in text or end not in text:
        print(f"  {path}: markers {start} / {end} not found, README left unchanged")
        return
    pre, rest = text.split(start, 1)
    post = rest.split(end, 1)[1]
    Path(path).write_text(f"{pre}{start}\n{md}{end}{post}", encoding="utf-8")
    print(f"  updated results table in {path}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=REPO / "results", type=Path, help="folder with sweep CSVs")
    ap.add_argument("--out", default=REPO / "report", type=Path, help="output folder")
    ap.add_argument("--metric", default="landmark", choices=METRICS, help="error used for Table 1 and Fig. 2")
    ap.add_argument("--fs", default=2000.0, type=float, help="EMG sampling rate (Hz), to show shifts in ms")
    ap.add_argument("--readme", type=Path, help="README to refresh between the report:table1 markers")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    sfx = "" if a.metric == "landmark" else f"_{a.metric}"
    runs = load_runs(a.results)
    pert = add_pct_change(runs)
    S = summarize(pert)

    pert.drop(columns=["model_name"]).to_csv(a.out / "runs_with_pct.csv", index=False, float_format="%.4f")
    S.to_csv(a.out / "summary.csv", index=False, float_format="%.4f")
    tex, md = table1(S, a.metric)
    (a.out / f"table1{sfx}.tex").write_text(tex, encoding="utf-8")
    (a.out / f"table1{sfx}.md").write_text(md, encoding="utf-8")
    info = heatmap(S, a.metric, a.fs, a.out / f"fig2_heatmap{sfx}.pdf", a.out / f"fig2_heatmap{sfx}.png")
    (a.out / f"fig2{sfx}.tex").write_text(fig2_tex(info, S, a.metric), encoding="utf-8")

    n_cells = len(S)
    print(f"Read {len(runs)} runs ({runs.model_name.nunique()} models x {runs.user.nunique()} users) "
          f"-> {n_cells} conditions. Wrote to {a.out}/:")
    for f in ["runs_with_pct.csv", "summary.csv", f"table1{sfx}.tex", f"table1{sfx}.md",
              f"fig2_heatmap{sfx}.pdf", f"fig2_heatmap{sfx}.png", f"fig2{sfx}.tex"]:
        print(f"  {f}")
    print("Users: " + ", ".join(f"{u} = user {s}" for u, s in
                                sorted(runs[["user", "user_session"]].drop_duplicates().itertuples(index=False),
                                       key=lambda t: natural_key(t[0]))))
    for n in sanity_notes(pert, S):
        print(f"Note: {n}")
    if a.readme:
        update_readme(a.readme, md)


if __name__ == "__main__":
    main()
