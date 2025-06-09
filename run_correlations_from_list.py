import os
import sys
import subprocess
import argparse
import shutil
from sort_and_cal import load_file_list
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run build_correlations for runs in a list")
    parser.add_argument('--run-list', default='file_list_correlations.txt', help='Text file of run names')
    parser.add_argument('--base-dir', default='',
                        help='Directory used to store correlation outputs')
    parser.add_argument('--max-memory-mb', type=float, default=None,
                        help='Optional memory limit in MB for each subprocess')
    args = parser.parse_args()

    runs = load_file_list(args.run_list)
    for run in runs:
        env = os.environ.copy()
        env['RUN_DIR'] = run
        print(f"Processing run {run}...")
        cmd = [sys.executable, 'build_correlations.py',
               '--base-dir', args.base_dir]
        if args.max_memory_mb is not None:
            cmd += ['--max-memory-mb', str(args.max_memory_mb)]
        subprocess.run(cmd, check=True, env=env)

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

    # Remove per-run outputs now that they are merged
    for run in runs:
        run_out = os.path.join('correlations', args.base_dir, run)
        if os.path.isdir(run_out):
            shutil.rmtree(run_out)


if __name__ == '__main__':
    main()
