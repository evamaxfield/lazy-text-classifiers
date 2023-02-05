#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    pipeline,
)

from .estimator_base import EstimatorBase

if TYPE_CHECKING:
    from datasets.arrow_dataset import Batch
    from transformers.tokenization_utils_base import BatchEncoding

###############################################################################


class TransformerEstimator(EstimatorBase):

    # Properties
    trainer: Trainer | None = None

    def __init__(
        self: "TransformerEstimator",
        base_model: str = "distilbert-base-uncased",
        training_args: TrainingArguments | None = None,
        eval_size: float = 0.2,
        output_dir: Path | str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        self.base_model = base_model
        self.eval_size = eval_size

        # Handle output dir
        if output_dir:
            self.model_dir = Path(output_dir).resolve()
        else:
            self.model_dir = Path("lazy-text-fine-tuned-transformer/").resolve()

        # Handle training arguments
        if training_args:
            self.training_args = training_args
        else:
            self.training_args = TrainingArguments(
                output_dir=self.model_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=3e-5,
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                per_device_eval_batch_size=4,
                per_device_train_batch_size=4,
                num_train_epochs=5,
                weight_decay=0.01,
            )

    def fit(
        self: "TransformerEstimator",
        x: Iterable[str],
        y: Iterable[str],
    ) -> "TransformerEstimator":
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
        "TransformerEstimator"
            The estimator.
        """
        # Create dataframes and split to eval
        df_all = pd.DataFrame(
            {
                "text": x,
                "label": y,
            }
        )
        train_df, eval_df = train_test_split(
            df_all,
            test_size=self.eval_size,
            stratify=df_all["label"],
        )

        # Make the label luts
        label_names = df_all.label.unique()
        label2id, id2label = {}, {}
        for i, label in enumerate(label_names):
            label2id[label] = str(i)
            id2label[str(i)] = label

        # Make the model
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.base_model,
            num_labels=len(id2label),
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )

        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        train_dataset = train_dataset.class_encode_column("label")
        eval_dataset = eval_dataset.class_encode_column("label")

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        def preprocess_function(examples: "BatchEncoding") -> "Batch":
            return tokenizer(examples["text"], truncation=True)

        # Preprocess data
        tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

        # Load metrics and create metric compute func
        f1_metric = load_metric("f1")

        def compute_metrics(eval_pred: EvalPrediction) -> dict | None:
            predictions = np.argmax(eval_pred.predictions, axis=-1)
            f1_score = f1_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average="weighted",
            )
            return f1_score

        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Create trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train
        self.trainer.train()
        self.trainer.save_model()

        return self

    def predict(
        self: "TransformerEstimator",
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
        pipe = pipeline(
            "text-classification",
            model=str(self.model_dir),
            tokenizer=str(self.model_dir),
        )
        return [pred[0]["label"] for pred in pipe(x, truncation=True, top_k=1)]


def _make_pipeline(
    **kwargs: Any,
) -> TransformerEstimator:
    return TransformerEstimator(**kwargs)