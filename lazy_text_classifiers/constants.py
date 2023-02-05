#!/usr/bin/env python

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

from .model_wrappers import (
    fine_tuned_transformer,
    semantic_logit,
    setfit_transformer,
    tfidf_logit,
)

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from .model_wrappers.estimator_base import EstimatorBase


class ModelNames:
    tfidf_logit = "tfidf-logit"
    semantic_logit = "semantic-logit"
    setfit_transformer = "setfit-transformer"
    fine_tuned_transformer = "fine-tuned-transformer"


# Use a varienty of base models for the semantic models
SEMANTIC_BASE_MODELS = {
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert-emotions": "joeddav/distilbert-base-uncased-go-emotions-student",
}


SEMANTIC_MODEL_VARIANTS = {}
for short_base_model_name, full_base_model_name in SEMANTIC_BASE_MODELS.items():
    SEMANTIC_MODEL_VARIANTS[f"semantic-logit-{short_base_model_name}"] = partial(
        semantic_logit._make_pipeline,
        sentence_encoder_kwargs={
            "name": full_base_model_name,
        },
    )
    SEMANTIC_MODEL_VARIANTS[f"fine-tuned-{short_base_model_name}"] = partial(
        fine_tuned_transformer._make_pipeline,
        base_model=full_base_model_name,
    )


MODEL_NAME_WRAPPER_LUT: dict[str, Callable[..., "Pipeline" | "EstimatorBase"]] = {
    ModelNames.tfidf_logit: tfidf_logit._make_pipeline,
    **SEMANTIC_MODEL_VARIANTS,
    ModelNames.setfit_transformer: setfit_transformer._make_pipeline,
}
