"""Stable serialization module for the calibrated Isolation Forest artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline


@dataclass
class IsolationForestArtifact:
    """Calibrated artifact with two-value iteration for existing app consumers."""

    preprocessor: Pipeline
    model: IsolationForest
    threshold: float
    alpha: float
    variant: str

    def __iter__(self) -> Iterator[object]:
        yield self.preprocessor
        yield self.model
