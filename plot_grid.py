#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILENAME_RE = re.compile(
    r'alpha(?P<alpha>\d+(?:\.\d+)?)_beta(?P<beta>\d+(?:\.\d+)?)_gamma_rel(?P<gamma_rel>\d+(?:\.\d+)?)_gamma_red(?P<gamma_red>\d+(?:\.\d+)?)',
    re.IGNORECASE,
)

def parse_from_filename(name: str) -> Optional[Dict[str, float]]:
    m = FILENAME_RE.search(name)
    if not m:
        return None
    return {k: float(v) for k, v in m.groupdict().items()}

def load_results(tsv_path: str) -> pd.DataFrame:
    rows = []
    header = None
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('//') or line.startswith('TOP '):
                continue
            parts = line.split('\t')
            if not parts:
                continue
            if parts[0] == 'file':
                header = parts
                continue
            if header is None or len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    if not rows:
        raise RuntimeError(f"No valid rows read from {tsv_path}")
    df = pd.DataFrame(rows)

    # Avoid collisions with parsed columns
    rename_map = {}
    for c in ['alpha', 'beta', 'gamma']:
        if c in df.columns:
            rename_map[c] = f'{c}_tsv'
    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce metrics and fallbacks
    for col in ['rouge1_f', 'rouge2_f', 'rouge_lsum_f', 'alpha_tsv', 'beta_tsv', 'gamma_tsv']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse parameters from filename
    parsed = df['file'].apply(parse_from_filename)
    parsed_df = parsed.apply(lambda d: d if d is not None else {})
    df = pd.concat([df, pd.json_normalize(parsed_df)], axis=1)

    # Build final params
    if 'alpha' not in df.columns and 'alpha_tsv' not in df.columns:
        raise RuntimeError("Missing 'alpha'.")
    if 'beta' not in df.columns and 'beta_tsv' not in df.columns:
        raise RuntimeError("Missing 'beta'.")
    if 'gamma_rel' not in df.columns or 'gamma_red' not in df.columns:
        raise RuntimeError("Missing 'gamma_rel'/'gamma_red' parsed from filenames.")

    df['alpha'] = pd.to_numeric(df.get('alpha', df.get('alpha_tsv')), errors='coerce')
    df['beta'] = pd.to_numeric(df.get('beta', df.get('beta_tsv')), errors='coerce')
    df['gamma_rel'] = pd.to_numeric(df['gamma_rel'], errors='coerce')
    df['gamma_red'] = pd.to_numeric(df['gamma_red'], errors='coerce')

    if 'alpha_tsv' in df.columns:
        df['alpha'] = df['alpha'].fillna(df['alpha_tsv'])
    if 'beta_tsv' in df.columns:
        df['beta'] = df['beta'].fillna(df['beta_tsv'])

    df = df.drop(columns=[c for c in ['alpha_tsv', 'beta_tsv', 'gamma_tsv'] if c in df.columns])

    # Final types
    for c in ['alpha', 'beta', 'gamma_rel', 'gamma_red']:
        df[c] = df[c].astype(float)

    return df

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_best_metric_vs_gamma_red_by_beta(df: pd.DataFrame, out_dir: str, metric: str):
    # Best over alpha and gamma_rel for each (beta, gamma_red)
    agg = (
        df.groupby(['beta', 'gamma_red'], as_index=False)[metric]
          .max()
          .sort_values(['beta', 'gamma_red'])
    )

    plt.figure(figsize=(7, 5))
    for b, gdf in agg.groupby('beta'):
        gdf = gdf.sort_values('gamma_red')
        plt.plot(gdf['gamma_red'], gdf[metric], marker='o', label=f'beta={b:.1f}')
    plt.xlabel('gamma_red (redundancy weight)')
    plt.ylabel(metric)
    plt.title(f'Best {metric} vs gamma_red (max over alpha, gamma_rel) per beta')
    plt.legend(title='beta')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'best_{metric}_vs_gamma_red_by_beta.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot best ROUGE-LSUM vs gamma_red, separated by beta.")
    parser.add_argument('--tsv', type=str,
                        default='/teamspace/studios/this_studio/data_after_bpe/rouge_grid_results.tsv',
                        help='Path to TSV results file')
    parser.add_argument('--out', type=str,
                        default='/teamspace/studios/this_studio/HiBERT_code/plots',
                        help='Output directory for plots')
    parser.add_argument('--metric', type=str, default='rouge_lsum_f',
                        choices=['rouge1_f', 'rouge2_f', 'rouge_lsum_f'],
                        help='Metric to plot')
    args = parser.parse_args()

    ensure_dir(args.out)
    df = load_results(args.tsv)
    df = df.dropna(subset=[args.metric])

    # Optional: save parsed data
    df.to_csv(os.path.join(args.out, 'parsed_results.csv'), index=False)

    # Single requested plot
    plot_best_metric_vs_gamma_red_by_beta(df, args.out, args.metric)

if __name__ == '__main__':
    main()