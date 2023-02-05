#!/usr/bin/env python

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

###############################################################################


class EstimatorBase(ABC):
    """Base class for estimators."""

    @abstractmethod
    def fit(
        self: "EstimatorBase",
        x: Iterable[str],
        y: Iterable[str],
    ) -> "EstimatorBase":
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
        "EstimatorBase"
            The estimator.
        """
        pass

    @abstractmethod
    def predict(
        self: "EstimatorBase",
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

    # TODO: save function