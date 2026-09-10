"""
Emit manuscript LaTeX table bodies from the number ledger CSV (numbers.csv).

  --what table1      Table 1 data rows (accuracy + step cost, FP32 only)
  --what primitives  Supplementary cost-primitives table (full environment)

Only data rows are produced for table1; its environment, caption and notes
live in the manuscript. Bold = best per column, underline = second best,
matching the round-2 table convention. Accuracy columns are constants here
(they come from the accuracy experiments, not the Orin harness).

Usage:
  python benchmarks/make_table1_tex.py --ledger "<folder>/numbers.csv" --what table1
"""
import argparse

import pandas as pd

# (ledger row id, method label, system cell, quantization, acc 1-shot, acc 25-shot)
ROWS = [
    ("T1.0", "CLP-SNN", r"Loihi 2\tnote1", "INT8", (55.4, 2.5), (90.0, 0.4)),
    ("T1.1", "CLP", r"GPU\tnote2", "FP32", (57.0, 2.4), (93.0, 0.2)),
    ("T1.2", "CLP", r"CPU\tnote3", "FP32", (57.0, 2.4), (93.0, 0.2)),
    ("T1.3", "NCM", "GPU", "FP32", (53.1, 2.9), (84.5, 0.8)),
    ("T1.4", "NCM", "CPU", "FP32", (53.1, 2.9), (84.5, 0.8)),
    ("T1.5", "Replay", "GPU", "FP32", (55.1, 2.5), (91.6, 0.4)),
    ("T1.6", "Replay", "CPU", "FP32", (55.1, 2.5), (91.6, 0.4)),
    ("T1.7", r"SLDA\tnote4", "GPU", "FP32", (54.8, 2.5), (96.2, 0.1)),
    ("T1.8", r"SLDA\tnote4", "CPU", "FP32", (54.8, 2.5), (96.2, 0.1)),
]
GROUPS = [["T1.0"], ["T1.1", "T1.2"], ["T1.3", "T1.4"], ["T1.5", "T1.6"], ["T1.7", "T1.8"]]

# Supplementary primitives table: (ledger id, method, device, precision, step rule)
# Ledger section E is keyed by row order in reports_table1.csv; the ids below
# mirror that order (E.1 ... E.13).
PRIM = [
    ("E.1", "CLP", "GPU", "FP32", "update"),
    ("E.2", "CLP", "CPU", "FP32", "update"),
    ("E.3", "NCM", "GPU", "FP32", "update\\,+\\,query"),
    ("E.4", "NCM", "GPU", "FP16", "update\\,+\\,query"),
    ("E.5", "NCM", "CPU", "FP32", "update\\,+\\,query"),
    ("E.6", "Replay", "GPU", "FP32", "update"),
    ("E.7", "Replay", "GPU", "FP16", "update"),
    ("E.8", "Replay", "CPU", "FP32", "update"),
    ("E.9", "SLDA, explicit inversion", "GPU", "FP32", "update\\,+\\,query"),
    ("E.10", "SLDA, explicit inversion", "GPU", "FP16", "update\\,+\\,query"),
    ("E.11", "SLDA, explicit inversion", "CPU", "FP32", "update\\,+\\,query"),
    ("E.12", "SLDA, rank-one", "GPU", "FP32", "update\\,+\\,query"),
    ("E.13", "SLDA, rank-one", "CPU", "FP32", "update\\,+\\,query"),
]
SINGLE_PASS_PRIM = {"CLP", "Replay"}


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def mark(cells, best_is_max):
    order = sorted(cells.items(), key=lambda kv: kv[1], reverse=best_is_max)
    best = order[0][1]
    second = next((v for _, v in order if v != best), None)
    return {rid: ("bf" if v == best else ("ul" if v == second else "")) for rid, v in cells.items()}


def wrap(text, m):
    return {"bf": r"\textbf{" + text + "}", "ul": r"\underline{" + text + "}"}.get(m, text)


def emit_table1(led):
    cost = {rid: {k: float(led[f"{rid}.{k}"]) for k in ("lat_ms", "tot_mJ", "dyn_mJ")}
            for rid, *_ in ROWS}
    m_acc1 = mark({rid: a1[0] for rid, _, _, _, a1, _ in ROWS}, True)
    m_acc25 = mark({rid: a25[0] for rid, _, _, _, _, a25 in ROWS}, True)
    m_lat = mark({r: c["lat_ms"] for r, c in cost.items()}, False)
    m_tot = mark({r: c["tot_mJ"] for r, c in cost.items()}, False)
    m_dyn = mark({r: c["dyn_mJ"] for r, c in cost.items()}, False)

    by_id = {row[0]: row for row in ROWS}
    out = []
    for gi, group in enumerate(GROUPS):
        if gi:
            out.append(r"\midrule")
        for j, rid in enumerate(group):
            _, method, system, quant, a1, a25 = by_id[rid]
            c = cost[rid]
            method_cell = (r"\multirow{%d}{*}{%s}" % (len(group), method)) if j == 0 else ""
            acc1_cell = wrap(f"{a1[0]:.1f}", m_acc1[rid]) + r"{\scriptsize$\pm$" + f"{a1[1]:.1f}" + "}"
            acc25_cell = wrap(f"{a25[0]:.1f}", m_acc25[rid]) + r"{\scriptsize$\pm$" + f"{a25[1]:.1f}" + "}"
            out.append(" & ".join([
                method_cell, system, quant, acc1_cell, acc25_cell,
                wrap(fmt(c["lat_ms"]), m_lat[rid]),
                wrap(fmt(c["tot_mJ"]), m_tot[rid]),
                wrap(fmt(c["dyn_mJ"]), m_dyn[rid]),
            ]) + r" \\" + f"  % {rid}")
    return "\n".join(out)


def emit_primitives(led):
    def val(rid, key):
        k = f"{rid}.{key}"
        return float(led[k]) if k in led.index else None

    L = []
    L.append(r"\begin{table}[h]")
    L.append(r"\centering")
    L.append(r"\caption{Cost primitives of the online continual learning step on the")
    L.append(r"         NVIDIA Jetson Orin Nano (batch size 1, feature dimensionality")
    L.append(r"         $d = 1280$, 40 classes, unified harness). The \emph{update}")
    L.append(r"         primitive is the model update on a labelled sample; the")
    L.append(r"         \emph{query} primitive is scoring one incoming sample. For CLP and")
    L.append(r"         Replay the update pass already computes the prediction, so no")
    L.append(r"         separate query enters the step (Table~\ref{table:1}). Latency is")
    L.append(r"         the raw mean over the stream, which is the amortized per-sample")
    L.append(r"         cost; the 99th percentile is given per primitive and is not")
    L.append(r"         additive across primitives. The explicit-inversion SLDA rows are a")
    L.append(r"         reference implementation that inverts the $1280 \times 1280$")
    L.append(r"         regularized covariance at every sample ($\approx 2.1 \times 10^{9}$")
    L.append(r"         multiply-accumulates, against $\approx 3.4 \times 10^{6}$ for the")
    L.append(r"         rank-one update used in Table~\ref{table:1}); they are reported to")
    L.append(r"         quantify that gap, not as the paper's SLDA baseline. In the FP16")
    L.append(r"         rows the statistics are maintained in FP32 and half precision is")
    L.append(r"         applied to the matrix products only (Methods).}")
    L.append(r"\label{tab:cost_primitives}")
    L.append(r"\resizebox{\columnwidth}{!}{%")
    L.append(r"\begin{tabular}{@{}llc rr rr rr l@{}}")
    L.append(r"\toprule")
    L.append(r" & & & \multicolumn{2}{c}{\textbf{Latency (ms)}} &"
             r" \multicolumn{2}{c}{\textbf{p99 latency (ms)}} &"
             r" \multicolumn{2}{c}{\textbf{Dyn.\ energy (mJ)}} & \\")
    L.append(r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    L.append(r"\textbf{Method} & \textbf{Device} & \textbf{Quant.} &"
             r" \textbf{upd.} & \textbf{qry.} & \textbf{upd.} & \textbf{qry.} &"
             r" \textbf{upd.} & \textbf{qry.} & \textbf{Step} \\")
    L.append(r"\midrule")
    prev = None
    for rid, method, device, prec, rule in PRIM:
        if prev is not None and method != prev:
            L.append(r"\midrule")
        prev = method
        single = method in SINGLE_PASS_PRIM
        cells = [method if method != prev or True else "", device, prec]
        for key in ("qry_lat_ms", "qry_p99_ms", "qry_dyn_mJ"):
            pass
        upd_lat, qry_lat = val(rid, "upd_lat_ms"), val(rid, "qry_lat_ms")
        upd_p99, qry_p99 = val(rid, "upd_p99_ms"), val(rid, "qry_p99_ms")
        upd_dyn, qry_dyn = val(rid, "upd_dyn_mJ"), val(rid, "qry_dyn_mJ")
        dash = "---"
        row = [method, device, prec,
               fmt(upd_lat), dash if single else fmt(qry_lat),
               fmt(upd_p99), dash if single else fmt(qry_p99),
               fmt(upd_dyn), dash if single else fmt(qry_dyn),
               rule]
        L.append(" & ".join(row) + r" \\" + f"  % {rid}")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--what", choices=["table1", "primitives"], default="table1")
    args = ap.parse_args()
    led = pd.read_csv(args.ledger).set_index("key")["value"]
    print(emit_table1(led) if args.what == "table1" else emit_primitives(led))


if __name__ == "__main__":
    main()
