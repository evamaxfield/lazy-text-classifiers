#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection

import pandas as pd
from datasets import Dataset, load_metric
from setfit import SetFitModel, SetFitTrainer

from .estimator_base import EstimatorBase

###############################################################################


class TransformerEstimator(EstimatorBase):

    # Properties
    trainer: SetFitTrainer | None = None

    def __init__(
        self: "TransformerEstimator",
        base_model: str = "distilbert-base-uncased",
        training_args: dict[str, Any] | None = None,
        eval_size: float = 0.2,
        output_dir: Path | str | None = None,
        verbose: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        self.base_model = base_model
        self.eval_size = eval_size
        self.verbose = verbose

        # Handle output dir
        if output_dir:
            self.model_dir = Path(output_dir).resolve()
        else:
            self.model_dir = Path("lazy-text-setfit-transformer/").resolve()

        # Handle training arguments
        if training_args:
            self.training_args = training_args
        else:
            # Determine logging steps
            self.training_args = {
                "batch_size": 4,
                "num_epochs": 1,
            }

    def fit(
        self: "TransformerEstimator",
        x: Collection[str],
        y: Collection[str],
    ) -> "TransformerEstimator":
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
        "TransformerEstimator"
            The estimator.
        """
        # Remove printing
        if not self.verbose:
            from ..logging_utils import set_global_logging_level

            set_global_logging_level()

        # Create dataframes and split to eval
        df_all = pd.DataFrame(
            {
                "text": x,
                "label": y,
            }
        )

        # Make the label luts
        label_names = df_all.label.unique()
        label2id, id2label = {}, {}
        for i, label in enumerate(label_names):
            label2id[label] = str(i)
            id2label[str(i)] = label

        self.label2id = label2id
        self.id2label = id2label

        # Make the model
        model = SetFitModel.from_pretrained(self.base_model)

        # Create datasets
        train_dataset = Dataset.from_pandas(df_all)
        train_dataset = train_dataset.class_encode_column("label")

        # Load metrics and create metric compute func
        f1_metric = load_metric("f1")

        def compute_metrics(
            y_pred: Collection[str], y_test: Collection[str]
        ) -> dict | None:
            f1_score = f1_metric.compute(
                predictions=y_pred,
                references=y_test,
                average="weighted",
            )
            return f1_score

        # Determine number of training iterations
        if "num_iterations" not in self.training_args:
            # This is sort of a hack, but as you increase in dataset size,
            # you don't need as many training iterations
            self.training_args["num_iterations"] = max(200 // len(df_all), 4)

        # Create trainer
        self.trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            metric=compute_metrics,
            **self.training_args,
        )

        # Train
        self.trainer.train()
        self.trainer.model.save_pretrained(str(self.model_dir))

        # TODO: save label2id and id2label

        return self

    def predict(
        self: "TransformerEstimator",
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
        if self.trainer is not None:
            preds = self.trainer.model.predict(x)
            return [self.id2label[str(pred)] for pred in preds]

        raise ValueError("SetFit model has not been trained yet.")


def _make_pipeline(
    **kwargs: Any,
) -> TransformerEstimator:
    return TransformerEstimator(**kwargs)
