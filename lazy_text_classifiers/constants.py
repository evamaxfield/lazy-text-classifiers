#!/usr/bin/env python

from __future__ import annotations

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


MODEL_NAME_WRAPPER_LUT: dict[str, Callable[..., "Pipeline" | "EstimatorBase"]] = {
    ModelNames.tfidf_logit: tfidf_logit._make_pipeline,
    ModelNames.semantic_logit: semantic_logit._make_pipeline,
    ModelNames.fine_tuned_transformer: fine_tuned_transformer._make_pipeline,
    ModelNames.setfit_transformer: setfit_transformer._make_pipeline,
}
