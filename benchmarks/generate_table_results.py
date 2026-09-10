"""
Build Table 1 source tables from the benchmark reports written by benchmark.py.

Two CSVs are written next to the reports folder (named after it):

  <reports>.csv        one row per timed op (fit = C_upd, predict = C_qry),
                       with precision, latency in ms (mean / median / p99),
                       power and energy. Long format, traceable to exp_N.
  <reports>_table1.csv one row per (method, device, precision) with C_upd and
                       C_qry side by side and their sum, i.e. the per-sample
                       cost at rho = 1 that Table 1 reports.

Usage:
  python benchmarks/generate_table_results.py --logs_path benchmarks/reports
"""
import argparse
import os
import pickle
import re

import pandas as pd

EVALUATION_TYPES = ['fp32', 'fp16', 'fake_quant', 'tensorrt']
QRY_SUFFIX = '-qry'


def exp_index(name):
    m = re.fullmatch(r'exp_(\d+)', name)
    return int(m.group(1)) if m else None


def load_rows(benchmark_path):
    rows = []
    for exp in sorted(os.listdir(benchmark_path), key=lambda n: (exp_index(n) is None, exp_index(n) or 0, n)):
        exp_path = os.path.join(benchmark_path, exp)
        report_file = os.path.join(exp_path, 'report.pkl')
        if not os.path.isdir(exp_path) or not os.path.isfile(report_file):
            continue
        with open(report_file, 'rb') as f:
            data = pickle.load(f)

        meta = {k: v for k, v in data.items() if k not in EVALUATION_TYPES}
        for eval_type in EVALUATION_TYPES:
            if eval_type not in data:
                continue
            row = dict(data[eval_type])
            row.update(meta)
            row['evaluation_type'] = eval_type
            row['exp'] = exp
            full_name = row['model_name']
            row['op'] = 'C_qry' if full_name.endswith(QRY_SUFFIX) else 'C_upd'
            row['method'] = full_name[:-len(QRY_SUFFIX)] if full_name.endswith(QRY_SUFFIX) else full_name
            rows.append(row)
    return rows


def long_table(df):
    us_to_ms = 1e-3
    out = pd.DataFrame({
        'Exp': df['exp'],
        'Method': df['method'],
        'Op': df['op'],
        'Device': df['device'],
        'Precision': df['evaluation_type'],
        'Batch Size': df['batch_size'],
        'Latency mean (ms)': df['mean_inference_time_us'] * us_to_ms,
        'Latency median (ms)': df['median_inference_time_us'] * us_to_ms,
        'Latency p99 (ms)': df['p99_inference_time_us'] * us_to_ms,
        'Total Power (W)': df['total_power_watt'],
        'Static Power (W)': df['static_total_power_watt'],
        'Dynamic Power (W)': df['dynamic_power_watt'],
        'Total Energy (mJ)': df['total_energy_per_frame_milliJ'],
        'Dynamic Energy (mJ)': df['dynamic_energy_per_frame_milliJ'],
        'Total EDP (uJs)': df['total_edp_per_frame_uJ_s'],
        'Dynamic EDP (uJs)': df['dynamic_edp_per_frame_uJ_s'],
        'GPU Usage (%)': df['total_gpu_usage_percentage'],
    })
    return out


def table1(df):
    """One row per (method, device, precision): C_upd, C_qry, and their sum."""
    us_to_ms = 1e-3
    keys = ['method', 'device', 'evaluation_type']
    metrics = {
        'mean_inference_time_us': ('Latency (ms)', us_to_ms),
        'p99_inference_time_us': ('p99 (ms)', us_to_ms),
        'total_energy_per_frame_milliJ': ('Total E (mJ)', 1.0),
        'dynamic_energy_per_frame_milliJ': ('Dyn E (mJ)', 1.0),
    }
    records = []
    for (method, device, prec), g in df.groupby(keys, sort=False):
        ops = {r['op']: r for _, r in g.iterrows()}
        rec = {'Method': method, 'Device': device, 'Precision': prec}
        for col, (label, scale) in metrics.items():
            upd = ops['C_upd'][col] * scale if 'C_upd' in ops else None
            qry = ops['C_qry'][col] * scale if 'C_qry' in ops else None
            rec[f'C_upd {label}'] = upd
            rec[f'C_qry {label}'] = qry
            rec[f'C_upd+C_qry {label}'] = (upd + qry) if upd is not None and qry is not None else None
        rec['Static Power (W)'] = ops.get('C_upd', ops.get('C_qry'))['static_total_power_watt']
        rec['Exp (upd, qry)'] = f"{ops['C_upd']['exp'] if 'C_upd' in ops else '-'}, {ops['C_qry']['exp'] if 'C_qry' in ops else '-'}"
        records.append(rec)
    out = pd.DataFrame(records)
    # p99 of a sum is not the sum of p99s; keep the per-op p99 and drop the summed one
    return out.drop(columns=['C_upd+C_qry p99 (ms)'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Benchmark Results Table')
    parser.add_argument('--logs_path', type=str, required=True,
                        help='Path to the reports folder written by benchmark.py.')
    args = parser.parse_args()

    benchmark_path = os.path.normpath(args.logs_path)
    output_name = os.path.basename(benchmark_path)
    output_dir = os.path.dirname(benchmark_path) or '.'
    print(f'Benchmark results path: {benchmark_path}')

    rows = load_rows(benchmark_path)
    if not rows:
        raise SystemExit(f'No report.pkl found under {benchmark_path}')
    df = pd.DataFrame(rows)

    long_df = long_table(df).round(3)
    t1_df = table1(df).round(3)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', None)
    print('\nPer-op results:')
    print(long_df.to_string(index=False))
    print('\nTable 1 view (C_upd + C_qry at rho = 1):')
    print(t1_df.to_string(index=False))

    for name, frame in ((output_name, long_df), (f'{output_name}_table1', t1_df)):
        csv_path = os.path.join(output_dir, f'{name}.csv')
        frame.to_csv(csv_path, index=False)
        frame.to_pickle(os.path.join(output_dir, f'{name}.pkl'))
        print(f'Saved: {csv_path}')
