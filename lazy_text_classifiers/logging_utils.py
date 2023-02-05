#!/usr/bin/env python

import logging
import re


def set_global_logging_level(
    level: int = logging.ERROR,
    prefices: list[str] | None = None,
) -> None:
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their
    loggers have been initialized.

    Parameters
    ----------
    level: int
        Desired level. e.g. logging.INFO.
        Default is logging.ERROR
    prefices: list[str] | None
        One or more string prefices to match (e.g. ["transformers", "torch"])
        Default of None will match all loggers.
        The match is a case-sensitive `module_name.startswith(prefix)`

    Notes
    -----
    Credit:
    https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
    """
    if prefices is None:
        prefices = [""]

    prefix_re = re.compile(rf"^(?:{ '|'.join(prefices) })")
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
