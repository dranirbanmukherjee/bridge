"""
Command-line interface for BRIDGE pipeline.

Provides CLI commands for running the pipeline, training, and exporting.
"""

import argparse
import sys


def main():
    """Main CLI entry point.

    Parses command-line arguments and dispatches to the appropriate subcommand
    handler (run, version, etc.).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="BRIDGE: Behavioral Research Through Interpretable, Dimensionality-reduced "
                    "Generative AI Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on wine data
  bridge run --data wine_data.csv --description-col full_description \\
             --attributes region:province varietal:variety --output ./output

  # Run with pre-computed embeddings
  bridge run --data wine_data.csv --embeddings embeddings.npy \\
             --attributes region:province varietal:variety

  # Skip hyperparameter tuning (faster)
  bridge run --data wine_data.csv --attributes region varietal --skip-tuning
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run BRIDGE pipeline")
    run_parser.add_argument(
        "--data", "-d", required=True,
        help="Path to data file (CSV or parquet)"
    )
    run_parser.add_argument(
        "--description-col", default="description",
        help="Column containing text descriptions (default: description)"
    )
    run_parser.add_argument(
        "--attributes", "-a", nargs="+", required=True,
        help="Attributes to learn, format: attr_name:col_name or just attr_name"
    )
    run_parser.add_argument(
        "--embeddings", "-e",
        help="Path to pre-computed embeddings (.npy)"
    )
    run_parser.add_argument(
        "--output", "-o", default="./bridge_output",
        help="Output directory (default: ./bridge_output)"
    )
    run_parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Skip hyperparameter tuning"
    )
    run_parser.add_argument(
        "--n-trials", type=int, default=75,
        help="Number of tuning trials (default: 75)"
    )
    run_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )

    # Version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "version":
        from bridge import __version__
        print(f"bridge {__version__}")
        return 0

    if args.command == "run":
        return run_pipeline(args)

    parser.print_help()
    return 1


def run_pipeline(args) -> int:
    """Execute the run command.

    Loads data, initializes the BRIDGE pipeline, fits the model, and exports
    results based on the provided command-line arguments.

    Args:
        args: Parsed argparse namespace containing command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    import numpy as np

    from bridge import BRIDGEPipeline
    from bridge.data import clean_data, load_data

    verbose = not args.quiet

    # Parse attributes
    attributes = []
    column_mapping = {}
    for attr_spec in args.attributes:
        if ":" in attr_spec:
            attr_name, col_name = attr_spec.split(":", 1)
        else:
            attr_name = col_name = attr_spec
        attributes.append(attr_name)
        column_mapping[attr_name] = col_name

    if verbose:
        print(f"Attributes: {attributes}")
        print(f"Column mapping: {column_mapping}")

    # Load data
    df = load_data(args.data, verbose=verbose)

    # Clean data
    df = clean_data(
        df,
        description_col=args.description_col,
        attribute_cols=list(column_mapping.values()),
        verbose=verbose,
    )

    # Load embeddings if provided
    embeddings = None
    if args.embeddings:
        embeddings = np.load(args.embeddings)
        if verbose:
            print(f"Loaded embeddings: {embeddings.shape}")

    # Initialize pipeline
    pipeline = BRIDGEPipeline(
        attributes=attributes,
        output_dir=args.output,
        verbose=verbose,
    )

    # Prepare labels
    labels = {
        attr_name: df[col_name].values
        for attr_name, col_name in column_mapping.items()
    }

    # Fit
    pipeline.fit(
        descriptions=df[args.description_col].tolist(),
        labels=labels,
        embeddings=embeddings,
        tune=not args.skip_tuning,
        n_trials=args.n_trials,
    )

    # Export
    pipeline.export()

    if verbose:
        print("\nPipeline complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
