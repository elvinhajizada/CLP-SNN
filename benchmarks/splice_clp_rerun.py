"""
Splice the CLP re-run (2026-09-12, single-sync implementation, commit e4dd384)
into the Phase 3 benchmark tables, normalising static power.

The Phase 3 suite and the CLP re-run were measured in different sessions with
different idle baselines (5.75 W vs 4.34 W: peripherals, not the algorithm).
Latency and dynamic power are session-independent and taken from the re-run.
Static power is a board property, so the Phase 3 CLP static baseline of the
same device is kept, and total power, total energy and both EDPs are
recomputed from it:

    total_W  = static_W(Phase 3, same device) + dynamic_W(re-run)
    total_mJ = latency_ms * total_W
    EDP      = energy * latency

Per-op rows (reports.csv) and the Table 1 view (reports_table1.csv) are both
rewritten; the originals are kept as *_phase3.csv. The source column of the
CLP rows records the re-run experiment ids and the normalisation.

Usage:
  python benchmarks/splice_clp_rerun.py            # uses the default paths below
"""
import argparse
import os

import pandas as pd

_BM = os.path.dirname(os.path.abspath(__file__))

METHOD = "CLP"



def splice_long(base, rerun):
    base = base.copy()
    for _, r in rerun.iterrows():
        m = (base["Method"] == METHOD) & (base["Op"] == r["Op"]) & (base["Device"] == r["Device"]) \
            & (base["Precision"] == r["Precision"])
        assert m.sum() == 1, (r["Op"], r["Device"])
        i = base.index[m][0]
        static_w = float(base.at[i, "Static Power (W)"])
        lat = float(r["Latency mean (ms)"])
        dyn_w = float(r["Dynamic Power (W)"])
        tot_w = static_w + dyn_w
        tot_mj = lat * tot_w
        dyn_mj = lat * dyn_w
        base.at[i, "Exp"] = f"clp2/{r['Exp']}"
        for col in ["Latency mean (ms)", "Latency median (ms)", "Latency p99 (ms)", "GPU Usage (%)"]:
            base.at[i, col] = round(float(r[col]), 3)
        base.at[i, "Dynamic Power (W)"] = round(dyn_w, 3)
        base.at[i, "Total Power (W)"] = round(tot_w, 3)
        base.at[i, "Total Energy (mJ)"] = round(tot_mj, 3)
        base.at[i, "Dynamic Energy (mJ)"] = round(dyn_mj, 3)
        base.at[i, "Total EDP (uJs)"] = round(tot_mj * lat, 3)
        base.at[i, "Dynamic EDP (uJs)"] = round(dyn_mj * lat, 3)
    return base


def rebuild_table1(base_t1, long_df):
    """Recompute the CLP rows of the Table 1 view from the spliced per-op rows."""
    t1 = base_t1.copy()
    for dev in long_df.loc[long_df["Method"] == METHOD, "Device"].unique():
        upd = long_df[(long_df["Method"] == METHOD) & (long_df["Device"] == dev) & (long_df["Op"] == "C_upd")].iloc[0]
        qry = long_df[(long_df["Method"] == METHOD) & (long_df["Device"] == dev) & (long_df["Op"] == "C_qry")].iloc[0]
        m = (t1["Method"] == METHOD) & (t1["Device"] == dev)
        assert m.sum() == 1
        i = t1.index[m][0]
        vals = {
            "C_upd Latency (ms)": upd["Latency mean (ms)"],
            "C_qry Latency (ms)": qry["Latency mean (ms)"],
            "C_upd p99 (ms)": upd["Latency p99 (ms)"],
            "C_qry p99 (ms)": qry["Latency p99 (ms)"],
            "C_upd Total E (mJ)": upd["Total Energy (mJ)"],
            "C_qry Total E (mJ)": qry["Total Energy (mJ)"],
            "C_upd Dyn E (mJ)": upd["Dynamic Energy (mJ)"],
            "C_qry Dyn E (mJ)": qry["Dynamic Energy (mJ)"],
            "Static Power (W)": upd["Static Power (W)"],
        }
        vals["C_upd+C_qry Latency (ms)"] = vals["C_upd Latency (ms)"] + vals["C_qry Latency (ms)"]
        vals["C_upd+C_qry Total E (mJ)"] = vals["C_upd Total E (mJ)"] + vals["C_qry Total E (mJ)"]
        vals["C_upd+C_qry Dyn E (mJ)"] = vals["C_upd Dyn E (mJ)"] + vals["C_qry Dyn E (mJ)"]
        for k, v in vals.items():
            t1.at[i, k] = round(float(v), 3)
        t1.at[i, "Exp (upd, qry)"] = (f"{upd['Exp']}, {qry['Exp']} (re-run 2026-09-12; "
                                      f"static {upd['Static Power (W)']:.3f} W from Phase 3)")
    return t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_long", default=os.path.join(_BM, "reports.csv"))
    ap.add_argument("--base_table1", default=os.path.join(_BM, "reports_table1.csv"))
    ap.add_argument("--rerun_long", default=os.path.join(_BM, "reports_clp2.csv"))
    args = ap.parse_args()

    base_long = pd.read_csv(args.base_long)
    base_t1 = pd.read_csv(args.base_table1)
    rerun = pd.read_csv(args.rerun_long)

    for path, df in [(args.base_long, base_long), (args.base_table1, base_t1)]:
        keep = path.replace(".csv", "_phase3.csv")
        if not os.path.exists(keep):
            df.to_csv(keep, index=False)
            print("kept original:", keep)

    long_new = splice_long(base_long, rerun)
    t1_new = rebuild_table1(base_t1, long_new)
    long_new.to_csv(args.base_long, index=False)
    t1_new.to_csv(args.base_table1, index=False)
    print("wrote:", args.base_long)
    print("wrote:", args.base_table1)
    print(t1_new[t1_new["Method"] == METHOD].to_string(index=False))


if __name__ == "__main__":
    main()
