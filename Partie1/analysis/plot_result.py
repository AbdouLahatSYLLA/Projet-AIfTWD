import argparse
import pathlib
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob


def read_pkl(path):
    """Read all pickle objects from a file."""
    events = []
    with open(path, "rb") as openfile:
        while True:
            try:
                events.append(pickle.load(openfile))
            except EOFError:
                break
    return events


def parse_log_file(filepath, group_name=None):
    """
    Read log file and normalize data.
    If group_name is provided (e.g. 'Server', 'Client1'), it replaces the filename
    to allow aggregation of multiple runs.
    """
    raw_data = read_pkl(filepath)

    # Default curve name is filename (without extension)
    filename = os.path.basename(filepath)
    run_name = os.path.splitext(filename)[0]

    # Use generic group_name for aggregation if provided
    # e.g. rename 'server_stats' from run 0 and run 1 to 'Server'
    if group_name:
        run_name = group_name

    normalized_data = []
    server_buffer = {}

    for entry in raw_data:
        # --- CASE 1: SERVER ---
        if 'round' in entry and ('global_accuracy' in entry or 'global_loss' in entry):
            r = entry['round']
            if r not in server_buffer:
                server_buffer[r] = {"epoch": r, "accuracy": None, "loss": None}

            if 'global_accuracy' in entry:
                server_buffer[r]['accuracy'] = entry['global_accuracy'] * 100
            if 'global_loss' in entry:
                server_buffer[r]['loss'] = entry['global_loss']

        # --- CASE 2: FEDERATED CLIENT ---
        elif 'round' in entry:
            acc = entry.get('accuracy', 0)
            if acc < 1.0: acc *= 100

            normalized_data.append({
                "epoch": entry['round'],
                "accuracy": acc,
                "loss": entry.get('loss', 0),
                "agent": run_name
            })

        # --- CASE 3: CENTRALIZED ---
        elif 'epoch' in entry:
            acc = entry.get('accuracy', 0)
            if acc < 1.0: acc *= 100

            normalized_data.append({
                "epoch": entry['epoch'],
                "accuracy": acc,
                "loss": entry.get('loss', 0),
                "agent": run_name
            })

    if server_buffer:
        for r, values in server_buffer.items():
            normalized_data.append({
                "epoch": values['epoch'],
                "accuracy": values['accuracy'],
                "loss": values['loss'],
                "agent": run_name
            })

    return pd.DataFrame(normalized_data)


def load_from_files(file_paths):
    """Load specific list of files."""
    all_dfs = []
    for path in file_paths:
        if os.path.exists(path):
            print(f"Loading file: {path}")
            df = parse_log_file(path)
            all_dfs.append(df)
    return all_dfs


def load_from_logdir(logdirs):
    """
    Scan log directories to aggregate runs.
    Expected structure: logdir/run_id/file.pkl
    """
    all_dfs = []

    for logdir in logdirs:
        root_path = pathlib.Path(logdir)
        # Find all .pkl files recursively
        all_files = list(root_path.glob("**/*.pkl"))

        if not all_files:
            print(f"Warning: No .pkl files found in {logdir}")
            continue

        print(f"Scanning directory: {logdir} ({len(all_files)} files found)")

        for filepath in all_files:
            # File name serves as category (e.g. 'server_stats', 'centralized_stats')
            file_name = filepath.stem

            # Prettify name if needed
            pretty_name = file_name.replace('_stats', '').replace('stats_', '').capitalize()

            df = parse_log_file(str(filepath), group_name=pretty_name)
            if not df.empty:
                all_dfs.append(df)

    return all_dfs


def plot_comparison(dataframes, metric_mode='both', output_name="comparison_plot"):
    if not dataframes:
        print("Error: No data to plot.")
        return

    full_df = pd.concat(dataframes, ignore_index=True)

    sns.set_theme(style="darkgrid")

    if metric_mode == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_list = [('accuracy', axes[0]), ('loss', axes[1])]
    elif metric_mode == 'accuracy':
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        plot_list = [('accuracy', axes)]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        plot_list = [('loss', axes)]

    for metric, ax in plot_list:
        # Seaborn automatically calculates mean and standard deviation
        # when there are multiple values for the same x and hue.
        sns.lineplot(
            data=full_df,
            x="epoch",
            y=metric,
            hue="agent",
            marker="o",
            errorbar='sd',
            ax=ax
        )
        ax.set_title(f"Comparison of {metric.capitalize()}")
        ax.set_xlabel("Epochs / Rounds")
        ax.set_ylabel(metric.capitalize())
        if metric == 'accuracy':
            ax.set_ylabel("Accuracy (%)")

    plt.tight_layout()
    filename = f"{output_name}_{metric_mode}.png"
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Training Results")

    group = parser.add_mutually_exclusive_group(required=True)

    # Option 1: Specific files
    group.add_argument(
        "--files",
        nargs='+',
        help="List specific .pkl files."
    )

    # Option 2: Full directories (for averaging)
    group.add_argument(
        "--logdir",
        nargs='+',
        help="Path to folder(s) containing runs (e.g. Partie1/logs/federated)."
    )

    parser.add_argument(
        "--metric",
        choices=['accuracy', 'loss', 'both'],
        default='both',
        help="Metric to plot."
    )

    args = parser.parse_args()

    dfs = []
    if args.files:
        dfs = load_from_files(args.files)
    elif args.logdir:
        dfs = load_from_logdir(args.logdir)

    plot_comparison(dfs, args.metric)