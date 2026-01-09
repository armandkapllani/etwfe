# ETWFE: Extended Two-Way Fixed Effects

[![PyPI version](https://badge.fury.io/py/etwfe.svg)](https://badge.fury.io/py/etwfe)
[![CI](https://github.com/YOUR_USERNAME/etwfe/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/etwfe/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of the Extended Two-Way Fixed Effects (ETWFE) estimator from Wooldridge (2021, 2023) for difference-in-differences estimation with heterogeneous treatment effects.

## Installation

```bash
pip install etwfe
```

For development:
```bash
pip install etwfe[dev,full]
```

## Quick Start

```python
import pandas as pd
from etwfe import etwfe

# Load your panel data
df = pd.read_csv("your_data.csv")

# Fit ETWFE model
model = etwfe(
    fml="outcome ~ 0",           # outcome variable (no controls)
    tvar="year",                  # time period variable
    gvar="first_treat_year",      # treatment cohort variable
    data=df,
    ivar="unit_id"                # unit identifier for FE
)

# View summary
model.summary()

# Compute marginal effects
att = model.emfx(type="simple")       # Overall ATT
event = model.emfx(type="event")      # Event study
cohort = model.emfx(type="group")     # By treatment cohort
calendar = model.emfx(type="calendar") # By calendar time

# Plot event study
model.plot(type="event")
```

## Features

- **Wooldridge (2021, 2023) ETWFE methodology**: Properly handles heterogeneous treatment effects in staggered DiD settings
- **Multiple control group options**: "notyet" (not-yet-treated) or "never" (never-treated)
- **GLM support**: Poisson, logit, probit, and Gaussian families
- **Heterogeneous effects**: Support for treatment effect heterogeneity via `xvar`
- **Flexible fixed effects**: Unit and time fixed effects
- **Standard errors**: Heteroskedasticity-robust and clustered standard errors via pyfixest
- **Marginal effects**: Simple ATT, event study, cohort-specific, and calendar-time effects
- **Visualization**: Built-in plotting for event studies

## API Reference

### ETWFE Class

```python
from etwfe import ETWFE

model = ETWFE(
    fml="y ~ x1 + x2",     # Formula: outcome ~ controls
    tvar="year",            # Time variable
    gvar="first_treat",     # Treatment cohort variable
    data=df,
    ivar="id",              # Unit ID (optional, for unit FE)
    xvar="group",           # Heterogeneity variable (optional)
    tref=2000,              # Reference time period (optional)
    gref=9999,              # Reference cohort (optional)
    cgroup="notyet",        # Control group: "notyet" or "never"
    family=None,            # GLM family: None, "poisson", "logit", "probit"
    vcov="hetero",          # Variance-covariance estimator
)
model.fit()
```

### Convenience Function

```python
from etwfe import etwfe

# Creates and fits in one step
model = etwfe("y ~ 0", tvar="year", gvar="first_treat", data=df)
```

### Marginal Effects

```python
# Simple overall ATT
model.emfx(type="simple")

# Event study (by event time)
model.emfx(type="event")

# By treatment cohort
model.emfx(type="group")

# By calendar time
model.emfx(type="calendar")

# Options
model.emfx(
    type="event",
    by_xvar=True,           # Separate effects by xvar
    compress=True,          # Compress data for speed
    predict="response",     # "response" or "link" for GLM
    post_only=True,         # Post-treatment only
    vcov=False,             # Skip SE computation
    window=5,               # Event window [-5, +5]
)
```

### Plotting

```python
# Event study plot
model.plot(type="event")

# Cohort effects
model.plot(type="group")

# Calendar time effects
model.plot(type="calendar")
```

## Requirements

- Python >= 3.9
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- pyfixest >= 0.18.0

Optional:
- patsy >= 0.5.3 (for model matrix construction)
- scipy >= 1.7.0 (for probit link function)

## References

- Wooldridge, J. M. (2021). "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and Difference-in-Differences Estimators." *Working Paper*.
- Wooldridge, J. M. (2023). "Simple Approaches to Nonlinear Difference-in-Differences with Panel Data." *The Econometrics Journal*.

## License

MIT License
