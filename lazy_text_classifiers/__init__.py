"""Top-level package for lazy_text_classifiers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lazy-text-classifiers")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"

from .model_selection import LazyTextClassifiers  # noqa: F401