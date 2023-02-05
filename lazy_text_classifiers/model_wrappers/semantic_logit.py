#!/usr/bin/env python

from __future__ import annotations

from typing import Any

from embetter.text import SentenceEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline, make_pipeline

###############################################################################


def _make_pipeline(
    sentence_encoder_kwargs: dict[str, Any] | None = None,
    logit_regression_cv_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Pipeline:
    # Make sentence encoder
    if sentence_encoder_kwargs:
        enc = SentenceEncoder(
            **sentence_encoder_kwargs,
        )
    else:
        enc = SentenceEncoder()

    if logit_regression_cv_kwargs:
        logit = LogisticRegressionCV(
            **logit_regression_cv_kwargs,
        )
    else:
        logit = LogisticRegressionCV()

    return make_pipeline(
        enc,
        logit,
        verbose=verbose,
    )
