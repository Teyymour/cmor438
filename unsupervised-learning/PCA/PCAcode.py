#!/usr/bin/env python3
print("🔥 PCAcode.py is running 🔥")   # ← smoke test

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    csv_path = '/Users/pbat/Projects/cmor438/data/meteorite-landings.csv'
    print("→ Working dir:", os.getcwd())
    print("→ File exists:", os.path.exists(csv_path))
    if not os.path.exists(csv_path):
        print(f"  ERROR: cannot find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("→ Raw data shape:", df.shape)
    print("→ Columns:", df.columns.tolist())

    numeric_cols = ['mass', 'reclat', 'reclong']
    df_num = df[numeric_cols].dropna()
    print(f"→ Rows after dropping NA in {numeric_cols}: {df_num.shape[0]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)
    print("→ Means (should be ~0):", X_scaled.mean(axis=0))
    print("→ Vars  (should be 1):",  X_scaled.var(axis=0))

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    plt.plot(
        range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_,
        marker='o'
    )
    plt.savefig('scree_plot.png')
    print("→ Scree plot saved as scree_plot.png")

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.savefig('pc1_vs_pc2.png')
    print("→ PC1 vs PC2 plot saved as pc1_vs_pc2.png")

    print("\nExplained-variance ratio per PC:")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio:.4f}")

if __name__ == '__main__':
    main()
