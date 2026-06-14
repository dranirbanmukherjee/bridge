"""
Data loading and cleaning module for BRIDGE pipeline.

Provides domain-agnostic data loading and preprocessing utilities.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def load_data(
    path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load data from CSV or parquet file.

    Args:
        path: Path to data file (.csv, .csv.gz, or .parquet).
        verbose: Whether to print progress messages.

    Returns:
        Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if verbose:
        print(f"Loading data from {path}...")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in [".csv", ".gz"]:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if verbose:
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    return df


def clean_data(
    df: pd.DataFrame,
    description_col: str = "description",
    attribute_cols: list[str] | None = None,
    remove_duplicates: bool = True,
    drop_missing: bool = True,
    create_id: bool = True,
    id_prefix: str = "W",
    verbose: bool = True,
) -> pd.DataFrame:
    """Clean and preprocess data for BRIDGE pipeline.

    Performs common data cleaning operations including removing duplicates,
    dropping missing values, and creating unique identifiers.

    Args:
        df: Raw input DataFrame.
        description_col: Column containing text descriptions.
        attribute_cols: Columns containing attribute labels. If None, no
            attribute-based filtering is applied.
        remove_duplicates: Whether to remove rows with duplicate descriptions.
        drop_missing: Whether to drop rows with missing descriptions or attributes.
        create_id: Whether to create a unique ID column.
        id_prefix: Prefix for ID values (e.g., "ID" creates "ID1", "ID2", ...).
        verbose: Whether to print progress messages.

    Returns:
        Cleaned DataFrame with reset index.
    """
    df = df.copy()
    initial_rows = len(df)

    if verbose:
        print(f"Cleaning data (initial rows: {initial_rows})...")

    # Remove index column if present
    for col in ["X", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            if verbose:
                print(f"  Removed '{col}' column")

    # Drop missing descriptions
    if drop_missing and description_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[description_col])
        if verbose and len(df) < before:
            print(f"  Dropped {before - len(df)} rows with missing {description_col}")

    # Drop missing attributes
    if drop_missing and attribute_cols:
        before = len(df)
        df = df.dropna(subset=attribute_cols)
        if verbose and len(df) < before:
            print(f"  Dropped {before - len(df)} rows with missing attributes")

    # Remove duplicate descriptions
    if remove_duplicates and description_col in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[description_col], keep="first")
        if verbose and len(df) < before:
            print(f"  Removed {before - len(df)} duplicate descriptions")

    # Create unique ID
    if create_id:
        df["id"] = [f"{id_prefix}{i}" for i in range(1, len(df) + 1)]
        if verbose:
            print("  Created 'id' column")

    # Reset index
    df = df.reset_index(drop=True)

    if verbose:
        print(f"Cleaning complete. Final rows: {len(df)} ({len(df)/initial_rows*100:.1f}%)")

    return df


def filter_by_attribute_frequency(
    df: pd.DataFrame,
    attribute_col: str,
    min_count: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter data to keep only attributes with minimum frequency.

    Removes rows where the attribute value appears fewer than min_count times.
    Useful for ensuring sufficient training samples per class.

    Args:
        df: Data to filter.
        attribute_col: Column containing attribute values.
        min_count: Minimum number of occurrences required to keep an attribute.
        verbose: Whether to print progress messages.

    Returns:
        Filtered DataFrame with reset index, containing only rows where the
        attribute value appears at least min_count times.
    """
    df = df.copy()
    initial_rows = len(df)

    # Get value counts
    counts = df[attribute_col].value_counts()
    keep_values = counts[counts >= min_count].index

    # Filter
    df = df[df[attribute_col].isin(keep_values)]

    if verbose:
        n_removed = initial_rows - len(df)
        n_attrs_removed = len(counts) - len(keep_values)
        print(f"Filtered {attribute_col}: kept {len(keep_values)} values "
              f"(removed {n_attrs_removed} with <{min_count} occurrences)")
        print(f"  Rows: {initial_rows} → {len(df)} ({n_removed} removed)")

    return df.reset_index(drop=True)


def create_composite_field(
    df: pd.DataFrame,
    columns: list[str],
    new_col: str,
    separator: str = " ",
    verbose: bool = True,
) -> pd.DataFrame:
    """Create composite field by concatenating multiple columns.

    Combines values from multiple columns into a single string field.
    NaN values are treated as empty strings.

    Args:
        df: Input DataFrame.
        columns: List of column names to concatenate, in order.
        new_col: Name for the new composite column.
        separator: String separator between concatenated values.
        verbose: Whether to print progress messages.

    Returns:
        DataFrame with the new composite column added.
    """
    df = df.copy()

    # Vectorized concatenation using pandas str.cat (more efficient than loop)
    base_col = df[columns[0]].fillna("").astype(str)
    if len(columns) > 1:
        other_cols = [df[col].fillna("").astype(str) for col in columns[1:]]
        df[new_col] = base_col.str.cat(other_cols, sep=separator)
    else:
        df[new_col] = base_col

    if verbose:
        print(f"Created '{new_col}' from {columns}")

    return df


def get_data_summary(df: pd.DataFrame, attribute_cols: list[str] | None = None) -> dict[str, Any]:
    """Get summary statistics for data.

    Computes basic statistics about the DataFrame including row/column counts
    and optionally per-attribute statistics.

    Args:
        df: Data to summarize.
        attribute_cols: Attribute columns to include in summary. If provided,
            includes unique value counts and missing value counts per attribute.

    Returns:
        Dictionary containing summary statistics with keys:
        - 'n_rows': Number of rows
        - 'n_columns': Number of columns
        - 'columns': List of column names
        - 'n_unique_{col}': Unique values per attribute (if attribute_cols provided)
        - 'n_missing_{col}': Missing values per attribute (if attribute_cols provided)
    """
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
    }

    if attribute_cols:
        for col in attribute_cols:
            if col in df.columns:
                summary[f"n_unique_{col}"] = df[col].nunique()
                summary[f"n_missing_{col}"] = df[col].isna().sum()

    return summary
