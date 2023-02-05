#!/usr/bin/env python

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline, make_pipeline

###############################################################################


def _make_pipeline(
    vectorizer_kwargs: dict[str, Any] | None = None,
    logit_regression_cv_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Pipeline:
    # Make vectorizer
    if vectorizer_kwargs:
        vect = TfidfVectorizer(
            **vectorizer_kwargs,
        )
    else:
        vect = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
        )

    if logit_regression_cv_kwargs:
        logit = LogisticRegressionCV(
            **logit_regression_cv_kwargs,
        )
    else:
        logit = LogisticRegressionCV()

    return make_pipeline(
        vect,
        logit,
        verbose=verbose,
    )
