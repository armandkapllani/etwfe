# ETWFE: Extended Two-Way Fixed Effects

[![PyPI version](https://badge.fury.io/py/etwfe.svg)](https://badge.fury.io/py/etwfe)
[![CI](https://github.com/armandkapllani/etwfe/actions/workflows/ci.yml/badge.svg)](https://github.com/armandkapllani/etwfe/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of the Extended Two-Way Fixed Effects (ETWFE) estimator for difference-in-differences designs with staggered treatment adoption.

## Overview

Two-way fixed effects (TWFE) estimation is the dominant approach for causal inference with panel data. However, recent research shows that standard TWFE can produce biased estimates when treatment timing varies across units. Wooldridge (2023, 2025) clarifies that the problem is not TWFE itself, but applying it to an insufficiently flexible model.

The solution is to saturate the regression with cohort×time interactions—what Wooldridge calls **extended TWFE (ETWFE)**. This package automates the approach: constructing the saturated specification, estimating via fixed effects, and recovering interpretable treatment effects through marginal effect aggregation.

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
from etwfe import etwfe
import pyreadr

# Load the minimum wage dataset
url = "https://github.com/bcallaway11/did/raw/master/data/mpdta.rda"
pyreadr.download_file(url, "mpdta.rda")
mpdta = pyreadr.read_r("mpdta.rda")["mpdta"]

# Prepare variables
mpdta["first_treat"] = mpdta["first.treat"].astype(int)
mpdta["year"] = mpdta["year"].astype(int)
mpdta["countyreal"] = mpdta["countyreal"].astype(int)

# Fit ETWFE model
mod = etwfe(
    fml="lemp ~ lpop",
    tvar="year",
    gvar="first_treat",
    data=mpdta,
    vcov={"CRV1": "countyreal"}
)

# Overall ATT
mod.emfx()
```

```
   .Dtreat   estimate  std.error  statistic    p.value   conf.low  conf.high
0     True  -0.050551   0.012523  -4.036608   0.000055  -0.075096  -0.026006
```

## Example: Event Study

```python
# Event study estimates
mod.emfx(type="event")
```

```
   event   estimate  std.error  statistic   p.value   conf.low  conf.high
0      0  -0.033139   0.013406  -2.472082  0.013440  -0.059414  -0.006864
1      1  -0.057312   0.017049  -3.361577  0.000774  -0.090728  -0.023895
2      2  -0.137531   0.034484  -3.988596  0.000066  -0.205119  -0.069943
3      3  -0.109435   0.032164  -3.402209  0.000668  -0.172476  -0.046394
```

```python
# Plot event study
mod.plot(type="event")
```

## Features

### Estimation Methods

- **Wooldridge ETWFE methodology**: Properly handles heterogeneous treatment effects in staggered DiD
- **Multiple control groups**: `"notyet"` (not-yet-treated) or `"never"` (never-treated)
- **Non-linear models**: Poisson, logit, and probit via the `family` argument
- **Heterogeneous effects**: Treatment effect heterogeneity via `xvar`

### Aggregation Options

- **Simple ATT**: Overall average treatment effect on the treated
- **Event study**: Effects by time relative to treatment
- **Cohort effects**: Effects by treatment cohort
- **Calendar effects**: Effects by calendar time

### Performance

- **Compression**: `compress=True` reduces data volume for large datasets
- **Skip standard errors**: `vcov=False` for faster iteration on point estimates

## Control Groups

By default, ETWFE uses not-yet-treated units as controls. Pre-treatment effects are mechanistically zero in this case:

```python
mod = etwfe(fml="lemp ~ lpop", tvar="year", gvar="first_treat", 
            data=mpdta, vcov={"CRV1": "countyreal"})
mod.emfx(type="event")  # Post-treatment only
```

To examine pre-treatment trends, use never-treated units as controls:

```python
mod_never = etwfe(fml="lemp ~ lpop", tvar="year", gvar="first_treat",
                  data=mpdta, vcov={"CRV1": "countyreal"}, cgroup="never")
mod_never.emfx(type="event", post_only=False)  # Includes pre-treatment
mod_never.plot(type="event", post_only=False)
```

## Heterogeneous Treatment Effects

Estimate effects separately by subgroup using `xvar`:

```python
# Create Great Lakes indicator
gls_fips = [17, 18, 26, 27, 36, 39, 42, 55]
mpdta["gls"] = mpdta["countyreal"].apply(lambda x: int(str(x)[:2]) in gls_fips)

# Fit model with heterogeneity
mod_het = etwfe(fml="lemp ~ lpop", tvar="year", gvar="first_treat",
                data=mpdta, vcov={"CRV1": "countyreal"}, 
                cgroup="never", xvar="gls")

# ATT by subgroup
mod_het.emfx(by_xvar=True)

# Event study by subgroup
mod_het.plot(type="event", by_xvar=True, post_only=False, style="ribbon")
```

## Non-Linear Models

For count outcomes, binary variables, or fractional responses, linear parallel trends may be implausible. Use the `family` argument for non-linear estimation:

```python
import numpy as np

# Create employment count
mpdta["emp"] = np.exp(mpdta["lemp"])

# Poisson ETWFE
mod_pois = etwfe(fml="emp ~ lpop", tvar="year", gvar="first_treat",
                 data=mpdta, vcov={"CRV1": "countyreal"}, family="poisson")
mod_pois.emfx(type="event")
```

## API Reference

### etwfe()

```python
etwfe(
    fml="y ~ x1 + x2",     # Formula: outcome ~ controls
    tvar="year",            # Time variable
    gvar="first_treat",     # Treatment cohort variable (0 = never treated)
    data=df,
    ivar="id",              # Unit ID (optional)
    xvar="group",           # Heterogeneity variable (optional)
    tref=None,              # Reference time period (optional)
    gref=None,              # Reference cohort (optional)
    cgroup="notyet",        # Control group: "notyet" or "never"
    family=None,            # GLM family: None, "poisson", "logit", "probit"
    vcov="hetero",          # Variance-covariance estimator
)
```

### emfx()

```python
model.emfx(
    type="simple",          # "simple", "event", "group", or "calendar"
    by_xvar=False,          # Separate effects by xvar
    compress=False,         # Compress data for speed
    predict="response",     # "response" or "link" for GLM
    post_only=True,         # Post-treatment only
    vcov=None,              # Override vcov (False to skip SEs)
)
```

### plot()

```python
model.plot(
    type="event",           # "event", "group", or "calendar"
    by_xvar=False,          # Separate lines by xvar
    post_only=True,         # Post-treatment only
    style="errorbar",       # "errorbar" or "ribbon"
)
```

## Requirements

- Python >= 3.9
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- pyfixest >= 0.18.0

## References

- Wooldridge, J. M. (2025). [Two-way fixed effects, the two-way Mundlak regression, and difference-in-differences estimators.](https://link.springer.com/article/10.1007/s00181-024-02681-z) *Empirical Economics*, 69(5), 2545-2587.

- Wooldridge, J. M. (2023). [Simple approaches to nonlinear difference-in-differences with panel data.](https://academic.oup.com/ectj/article/26/3/C31/7039335) *The Econometrics Journal*, 26(3), C31-C66.

- Callaway, B. and Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.

- Wong, J., Forsell, E., Lewis, R., Mao, T., and Wardrop, M. (2021). [You only compress once: Optimal data compression for estimating linear models.](https://arxiv.org/abs/2102.11297) arXiv:2102.11297.

## Acknowledgments

This package builds on several excellent open-source projects:

- **[pyfixest](https://github.com/py-econometrics/pyfixest)** — High-dimensional fixed effects regression in Python
- **[marginaleffects](https://github.com/vincentarelbundock/pymarginaleffects)** — Marginal effects, predictions, and comparisons
- **[etwfe (R)](https://grantmcdermott.com/etwfe/)** — The original R implementation by Grant McDermott

We are grateful to the developers of these packages for their contributions to the open-source econometrics ecosystem.

## License

MIT License
