#!/usr/bin/env python

from __future__ import annotations

from typing import Collection

import altair as alt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from .constants import SEMANTIC_BASE_MODELS

###############################################################################

def lazy_pca(
    x: Collection[str],
    n_components: int = 2,
    base_model: str = SEMANTIC_BASE_MODELS["bge-base-en-v1dot5"],
    **kwargs: dict[str, int | str | bool],
) -> tuple[PCA, np.ndarray]:
    """
    Lazy PCA.

    Parameters
    ----------
    x: Collection[str]
        Collection of strings to embed.
    n_components: int
        Number of components to keep.
        Default: 2
    base_model: str
        Name of the base model to use for sentence embeddings.
        Default: "BAAI/bge-base-en-v1.5"
    kwargs: dict[str, int | str | bool]
        Additional keyword arguments to pass to the SentenceTransformer
        constructor.

    Returns
    -------
    pca: PCA
        Fitted PCA model.
    transformed_components: np.ndarray
        PCA'ed embeddings of shape: (len(X), n_components).
    """
    # Get sentence embeddings
    model = SentenceTransformer(base_model, **kwargs)
    embeddings = model.encode(x, normalize_embeddings=True)

    # Fit PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(embeddings)

    return pca, components


def viz_components(
    components: np.ndarray,
    labels: Collection[str],
    width: int = 200,
    height: int = 200,
) -> None:
    # Create a dataframe with the label and each individual PCA value for that example
    rows = []
    for example_i in range(len(components)):
        this_row = {"label": labels[example_i]}
        for component_i in range(len(components[example_i])):
            this_row[f"pc_{component_i}"] = components[example_i][component_i]
        
        rows.append(this_row)

    # Convert to DF
    components_df = pd.DataFrame(rows)
    
    # return render
    return alt.Chart(components_df).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color='label:N',
    ).properties(
        width=width,
        height=height,
    ).repeat(
        row=[f"pc_{i}" for i in range(components.shape[1])],
        column=[f"pc_{i}" for i in range(components.shape[1])],
    )
