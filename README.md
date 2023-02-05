# lazy-text-classifiers

[![Build Status](https://github.com/evamaxfield/lazy-text-classifiers/workflows/CI/badge.svg)](https://github.com/evamaxfield/lazy-text-classifiers/actions)
[![Documentation](https://github.com/evamaxfield/lazy-text-classifiers/workflows/Documentation/badge.svg)](https://evamaxfield.github.io/lazy-text-classifiers)

Build and test a variety of text multi-class classification models.

---

## Installation

**Stable Release:** `pip install lazy-text-classifiers`<br>
**Development Head:** `pip install git+https://github.com/evamaxfield/lazy-text-classifiers.git`

## Quickstart

```python
from lazy_text_classifiers import LazyTextClassifiers
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Example data from sklearn
# `x` should be an iterable of strings
# `y` should be an iterable of string labels
data = fetch_20newsgroups(subset="all", remove=("header", "footers", "quotes"))
x = data.data[:1000]
y = data.target[:1000]
y = [data.target_names[id_] for id_ in y]

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.4,
    random_state=12,
)

# Init and fit all models
ltc = LazyTextClassifiers(random_state=12)
results = ltc.fit(x_train, x_test, y_train, y_test)

# Results is a dataframe
# | model                  |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |    time |
# |:-----------------------|-----------:|--------------------:|------------:|---------:|---------:|--------:|
# | semantic-logit         |    0.73    |            0.725162 |    0.734887 |  0.73    | 0.728247 |  13.742 |
# | tfidf-logit            |    0.70625 |            0.700126 |    0.709781 |  0.70625 | 0.702073 | 187.217 |
# | fine-tuned-transformer |    0.11125 |            0.1118   |    0.10998  |  0.11125 | 0.109288 | 220.105 |

# Get a specific model
semantic_logit = ltc.fit_models["semantic-logit"]
# either an scikit-learn Pipeline or a custom Transformer wrapper class

# All models have a `save` function which will store into the normal format
# * pickle for scikit-learn pipelines
# * torch model directory for Transformers
```

## Documentation

For full package documentation please visit [evamaxfield.github.io/lazy-text-classifiers](https://evamaxfield.github.io/lazy-text-classifiers).

## Acknowledgements

This package was heavily inspired by [lazypredict](https://github.com/shankarpandala/lazypredict).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
