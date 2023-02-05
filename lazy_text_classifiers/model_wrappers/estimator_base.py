#!/usr/bin/env python

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from typing_extensions import Self

###############################################################################

class EstimatorBase(ABC):
    """Base class for estimators."""

    @abstractmethod
    def fit(
        self: Self,
        x: Iterable[str],
        y: Iterable[str],
    ) -> Self:
        """
        Fit the estimator.

        Parameters
        ----------
        x: Iterable[str]
            The training data.
        y: Iterable[str]
            The testing data.

        Returns
        -------
        Any
            The estimator.
        """
        pass

    @abstractmethod
    def predict(
        self: Self,
        x: Iterable[str],
    ) -> Iterable[str]:
        """
        Predict the values using the fitted estimator.

        Parameters
        ----------
        x: Iterable[str]
            The data to predict.

        Returns
        -------
        Iterable[str]
            The predictions.
        """
        pass