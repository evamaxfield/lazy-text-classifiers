#!/usr/bin/env python

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from typing_extensions import Self

from .constants import MODEL_NAME_WRAPPER_LUT

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from .model_wrappers.estimator_base import EstimatorBase

###############################################################################


class LazyTextClassifiers:

    # TODO: optuna??

    def __init__(
        self: Self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        random_state: int | None = None,
    ) -> Self:
        """
        Class for managing fitting all classifiers.

        Parameters
        ----------
        verbose: int
            Logging verbosity level.
            Default: 0 (do log anything)
        ignore_warnings: bool
            Should all warnings be ignores.
        random_state: int | None
            A seed to initialize random state.
            Default: None (no pre-set random seed)
        """
        self.verbose = verbose

        # Set up warnings
        if ignore_warnings:
            import warnings
            warnings.filterwarnings("ignore")

        # Handle seed
        if random_state:
            torch.manual_seed(random_state)
            random.seed(random_state)
            np.random.seed(random_state)
    
    def fit(
        self: Self,
        x_train: Iterable[str],
        x_test: Iterable[str],
        y_train: Iterable[str],
        y_test: Iterable[str],
        model_kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Fit all models with the provided data.

        Parameters
        ----------
        x_train: Iterable[str]
            The training data, an iterable object where each item is a string.
        x_test: Iterable[str]
            The testing data, an iterable object where each item is a string.
        y_train: Iterable[str]
            The training labels, an iterable object where each item is a class.
        y_test: Iterable[str]
            The testing labels, an iterable object where each item is a class.
        
        model_kwargs: dict[str, Any] | None
            Any specific model kwargs to pass through.
            Default None (use default parameters and settings for all models)

        Returns
        -------
        pd.DataFrame
            The results and metrics returned from fitting all models.
        """
        # Handle kwargs
        if model_kwargs is None:
            model_kwargs = {}

        # Iterate model fitting and preds
        self.fit_models: dict[str, "EstimatorBase" | "Pipeline"] = {}
        results_rows: list[dict[str, str | float]] = []
        for model_name, model_wrapper in MODEL_NAME_WRAPPER_LUT.items():
            # Start perf time
            start_time = time.perf_counter()

            # Instantiate estimate with any extra kwargs
            print(f"Initializing model: '{model_name}'")
            estimator = model_wrapper(
                **model_kwargs.get(model_name, {}),
                verbose=self.verbose > 0,
            )

            # Run the fit method passing in x_train and y_train
            print(f"Fitting model: '{model_name}'")
            estimator = estimator.fit(x_train, y_train)
            self.fit_models[model_name] = estimator

            # Record training time
            duration = time.perf_counter() - start_time

            # Run the eval
            print(f"Evaluating model: '{model_name}'")
            preds = estimator.predict(x_test)

            # Calc metrics
            acc = accuracy_score(
                y_test,
                preds,
            )
            bal_acc = balanced_accuracy_score(
                y_test,
                preds,
            )
            pre, rec, f1, _ = precision_recall_fscore_support(
                y_test,
                preds,
                average="weighted",
            )
            results_rows.append({
                "model": model_name,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "precision": pre,
                "recall": rec,
                "f1": f1,
                "time": duration,
            })
        
        # Create dataframe of results
        self.results_df = pd.DataFrame(
            results_rows,
        ).sort_values(by="f1", ascending=False).reset_index(drop=True)

        return self.results_df