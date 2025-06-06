import os
import sys
import subprocess
import argparse
from sort_and_cal import load_file_list
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run build_correlations for runs in a list")
    parser.add_argument('--run-list', default='file_list_correlations.txt', help='Text file of run names')
    parser.add_argument('--base-dir', default='long_run_4mbar_500V',
                        help='Directory under processed_data containing per-run folders')
    args = parser.parse_args()

    runs = load_file_list(args.run_list)
    for run in runs:
        env = os.environ.copy()
        env['RUN_DIR'] = os.path.join(args.base_dir, run)
        print(f"Processing run {run}...")
        subprocess.run([sys.executable, 'build_correlations.py'], check=True, env=env)

    # Merge per-run pickle files into combined dataframes
    imp_dfs = []
    decay_dfs = []
    corr_dfs = []
    for run in runs:
        run_out = os.path.join('correlations', args.base_dir, run)
        try:
            imp_dfs.append(pd.read_pickle(os.path.join(run_out, 'coincident_imp.pkl')))
            decay_dfs.append(pd.read_pickle(os.path.join(run_out, 'decay_candidates.pkl')))
            corr_dfs.append(pd.read_pickle(os.path.join(run_out, 'final_correlated.pkl')))
        except FileNotFoundError:
            print(f"Warning: missing correlation files for run {run}")

    out_dir = os.path.join('correlations', args.base_dir)
    os.makedirs(out_dir, exist_ok=True)
    if imp_dfs:
        pd.concat(imp_dfs, ignore_index=True).to_pickle(os.path.join(out_dir, 'coincident_imp.pkl'))
    if decay_dfs:
        pd.concat(decay_dfs, ignore_index=True).to_pickle(os.path.join(out_dir, 'decay_candidates.pkl'))
    if corr_dfs:
        pd.concat(corr_dfs, ignore_index=True).to_pickle(os.path.join(out_dir, 'final_correlated.pkl'))


if __name__ == '__main__':
    main()
