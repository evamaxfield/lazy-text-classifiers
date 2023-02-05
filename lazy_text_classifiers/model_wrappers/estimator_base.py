#!/usr/bin/env python

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection

###############################################################################


class EstimatorBase(ABC):
    """Base class for estimators."""

    @abstractmethod
    def fit(
        self: "EstimatorBase",
        x: Collection[str],
        y: Collection[str],
    ) -> "EstimatorBase":
        """
        Fit the estimator.

        Parameters
        ----------
        x: Collection[str]
            The training data.
        y: Collection[str]
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
        x: Collection[str],
    ) -> Collection[str]:
        """
        Predict the values using the fitted estimator.

        Parameters
        ----------
        x: Collection[str]
            The data to predict.

        Returns
        -------
        Collection[str]
            The predictions.
        """
        pass

    # TODO: save function
