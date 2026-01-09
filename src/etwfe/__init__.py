"""
ETWFE: Extended Two-Way Fixed Effects Estimator

Python implementation of the Wooldridge (2021, 2023) ETWFE methodology
for difference-in-differences estimation with heterogeneous treatment effects.

Example
-------
>>> from etwfe import etwfe, ETWFE
>>> model = etwfe("y ~ 0", tvar="year", gvar="first_treat", data=df, ivar="id")
>>> model.summary()
>>> model.emfx(type="event")
"""

from etwfe.core import ETWFE, etwfe

__version__ = "0.1.0"
__author__ = "Armand Kapllani"
__all__ = ["ETWFE", "etwfe", "__version__"]
