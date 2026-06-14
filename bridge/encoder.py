"""
Attribute encoding module for BRIDGE pipeline.

Provides domain-agnostic label encoding for any categorical attributes.
Converts text labels to 0-based integer indices suitable for neural network training.
"""

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class AttributeEncoder:
    """Generic label encoder for categorical attributes.

    Converts text labels to 0-based integer indices for any number of attributes.
    This is a domain-agnostic replacement for WineLabelEncoder.

    Attributes:
        attribute_names: Names of the attributes being encoded.
        encoders: Dictionary mapping attribute names to sklearn LabelEncoder instances.
        is_fitted: Whether the encoder has been fitted on data.

    Example:
        >>> encoder = AttributeEncoder(["region", "varietal"])
        >>> encoder.fit(df, {"region": "province", "varietal": "variety"})
        >>> labels = encoder.transform(df, {"region": "province", "varietal": "variety"})
        >>> labels["region"]  # 0-based integer array
    """

    def __init__(self, attribute_names: list[str]):
        """Initialize encoder with attribute names.

        Args:
            attribute_names: Names of the attributes to encode
                (e.g., ["region", "varietal"]).
        """
        self.attribute_names = list(attribute_names)
        self.encoders: dict[str, LabelEncoder] = {
            name: LabelEncoder() for name in self.attribute_names
        }
        self.is_fitted = False

    def fit(
        self,
        data: pd.DataFrame | dict[str, np.ndarray],
        column_mapping: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> "AttributeEncoder":
        """Fit the encoder on data.

        Learns the mapping from text labels to integer indices for each attribute.

        Args:
            data: Data containing attribute values. Can be a pandas DataFrame
                or a dictionary mapping column names to numpy arrays.
            column_mapping: Maps attribute names to column names in data.
                If None, assumes column names match attribute names.
            verbose: Whether to print progress messages.

        Returns:
            The fitted encoder instance (self).
        """
        if column_mapping is None:
            column_mapping = {name: name for name in self.attribute_names}

        for attr_name in self.attribute_names:
            col_name = column_mapping.get(attr_name, attr_name)

            if isinstance(data, pd.DataFrame):
                values = data[col_name].fillna("").astype(str).values
            else:
                values = np.array(data[col_name]).astype(str)

            self.encoders[attr_name].fit(values)

        self.is_fitted = True

        if verbose:
            print("AttributeEncoder fitted:")
            for attr_name in self.attribute_names:
                n_classes = len(self.encoders[attr_name].classes_)
                print(f"  {attr_name}: {n_classes} unique values")

        return self

    def transform(
        self,
        data: pd.DataFrame | dict[str, np.ndarray],
        column_mapping: dict[str, str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Transform attribute values to 0-based integer labels.

        Args:
            data: Data containing attribute values. Can be a pandas DataFrame
                or a dictionary mapping column names to numpy arrays.
            column_mapping: Maps attribute names to column names in data.
                If None, assumes column names match attribute names.

        Returns:
            Dictionary mapping attribute names to integer label arrays.

        Raises:
            RuntimeError: If encoder has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transform")

        if column_mapping is None:
            column_mapping = {name: name for name in self.attribute_names}

        result = {}
        for attr_name in self.attribute_names:
            col_name = column_mapping.get(attr_name, attr_name)

            if isinstance(data, pd.DataFrame):
                values = data[col_name].fillna("").astype(str).values
            else:
                values = np.array(data[col_name]).astype(str)

            result[attr_name] = self.encoders[attr_name].transform(values)

        return result

    def fit_transform(
        self,
        data: pd.DataFrame | dict[str, np.ndarray],
        column_mapping: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> dict[str, np.ndarray]:
        """Fit the encoder and transform data in one step.

        Convenience method equivalent to calling fit() followed by transform().

        Args:
            data: Data containing attribute values.
            column_mapping: Maps attribute names to column names in data.
            verbose: Whether to print progress messages.

        Returns:
            Dictionary mapping attribute names to integer label arrays.
        """
        self.fit(data, column_mapping, verbose=verbose)
        return self.transform(data, column_mapping)

    def num_classes(self, attribute_name: str) -> int:
        """Get number of classes for an attribute.

        Args:
            attribute_name: Name of the attribute.

        Returns:
            Number of unique classes for the attribute, or 0 if not fitted.
        """
        if not self.is_fitted:
            return 0
        return len(self.encoders[attribute_name].classes_)

    def get_class_name(self, attribute_name: str, index: int) -> str:
        """Get class name from integer index.

        Args:
            attribute_name: Name of the attribute.
            index: Integer index of the class.

        Returns:
            Original string label corresponding to the index.

        Raises:
            RuntimeError: If encoder has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted first")
        return self.encoders[attribute_name].classes_[index]

    def get_mappings(self) -> dict[str, dict[str, int]]:
        """Get label-to-index mappings for all attributes.

        Returns:
            Nested dictionary mapping attribute names to dictionaries that
            map string labels to integer indices.
            Structure: {attribute_name: {label: index, ...}, ...}

        Raises:
            RuntimeError: If encoder has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted first")

        return {
            attr_name: {
                label: int(idx)
                for idx, label in enumerate(self.encoders[attr_name].classes_)
            }
            for attr_name in self.attribute_names
        }

    def save(self, path: str) -> None:
        """Save encoder mappings to JSON file.

        Args:
            path: Path to the output JSON file.

        Raises:
            RuntimeError: If encoder has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before saving")

        data = {
            "attribute_names": self.attribute_names,
            "mappings": self.get_mappings(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Encoder saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AttributeEncoder":
        """Load encoder from JSON file.

        Args:
            path: Path to the JSON file containing encoder mappings.

        Returns:
            AttributeEncoder instance with restored mappings.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        encoder = cls(data["attribute_names"])

        for attr_name, mapping in data["mappings"].items():
            classes = sorted(mapping.keys(), key=lambda x, m=mapping: m[x])
            encoder.encoders[attr_name].classes_ = np.array(classes)

        encoder.is_fitted = True
        print(f"Encoder loaded from {path}")
        return encoder

    def __repr__(self) -> str:
        """Return string representation of the encoder."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"AttributeEncoder(attributes={self.attribute_names}, {fitted_str})"
