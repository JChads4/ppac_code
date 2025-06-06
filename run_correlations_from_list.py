import os
import sys
import subprocess
import argparse
from sort_and_cal import load_file_list


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


if __name__ == '__main__':
    main()
