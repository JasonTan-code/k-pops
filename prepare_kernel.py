import os
import glob
import argparse
import numpy as np
import pandas as pd

def linear_kernel(X1, X2):
    """Compute the linear kernel (dot product) between two matrices."""
    return X1 @ X2.T

def main(prefix_path):
    prefix_dir = os.path.dirname(prefix_path)
    prefix_base = os.path.basename(prefix_path)

    # Input file patterns
    npy_pattern = os.path.join(prefix_dir, f"{prefix_base}.mat.*.npy")
    cols_pattern = os.path.join(prefix_dir, f"{prefix_base}.cols*.txt")
    rows_file = os.path.join(prefix_dir, f"{prefix_base}.rows.txt")

    # Load and concatenate .npy feature files
    feature_npy_files = sorted(glob.glob(npy_pattern))
    if not feature_npy_files:
        raise FileNotFoundError(f"No .npy files found matching {npy_pattern}")
    features_npy = [np.load(f) for f in feature_npy_files]
    features_npy = np.concatenate(features_npy, axis=1)

    # Load and concatenate column name files
    feature_cols_files = sorted(glob.glob(cols_pattern))
    if not feature_cols_files:
        raise FileNotFoundError(f"No column files found matching {cols_pattern}")
    feature_cols = [pd.read_csv(f, header=None, names=["feature_name"]) for f in feature_cols_files]
    feature_cols = pd.concat(feature_cols, axis=0)

    # Load row (gene) names
    if not os.path.isfile(rows_file):
        raise FileNotFoundError(f"Row file not found: {rows_file}")
    feature_rows = pd.read_csv(rows_file, header=None, names=["gene_name"])

    # Assemble feature matrix
    features = pd.DataFrame(features_npy, index=feature_rows.gene_name, columns=feature_cols.feature_name)
    X = np.array(features).astype("float32")
    X_centered = X - X.mean(axis=0)

    # Compute linear kernel
    print("Computing linear kernel...")
    K = linear_kernel(X_centered, X_centered)

    # Output file paths
    kernel_file = os.path.join(prefix_dir, "kernel.bin")
    genes_file = os.path.join(prefix_dir, "kernel.genes")

    # Save outputs
    print(f"Saving kernel to: {kernel_file}")
    K.tofile(kernel_file)

    print(f"Saving gene list to: {genes_file}")
    feature_rows.to_csv(genes_file, index=False, header=False)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a linear kernel matrix from PoPS feature files.")
    parser.add_argument("--prefix", required=True, help="Full prefix path to pops_features (e.g., /path/to/pops_features)")
    args = parser.parse_args()
    main(args.prefix)
