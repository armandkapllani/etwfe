"""
Extended Two-Way Fixed Effects (ETWFE) estimator.

Implementation of Wooldridge (2021, 2023) ETWFE methodology for
difference-in-differences with heterogeneous treatment effects.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

try:
    import patsy
except ImportError:
    patsy = None  # type: ignore[assignment]

__all__ = ["ETWFE", "etwfe"]

SUPPORTED_FAMILIES = {"poisson", "logit", "probit", "gaussian", None}


@dataclass
class ETWFE:
    """
    Extended Two-Way Fixed Effects estimator.

    Implements the Wooldridge (2021, 2023) ETWFE methodology for
    difference-in-differences estimation with heterogeneous treatment effects.

    Parameters
    ----------
    fml : str
        Formula in the form "y ~ x1 + x2" or "y ~ 0" for no controls.
    tvar : str
        Name of the time period variable.
    gvar : str
        Name of the treatment cohort (first-treated period) variable.
    data : pd.DataFrame
        Panel data with observations.
    ivar : str, optional
        Name of the unit identifier for unit fixed effects.
    xvar : str, optional
        Name of a variable for heterogeneous treatment effects.
    tref : int, optional
        Reference time period. Defaults to minimum time.
    gref : int, optional
        Reference cohort (control group). Defaults to never-treated or latest cohort.
    cgroup : {"notyet", "never"}
        Control group type. "notyet" uses not-yet-treated, "never" uses never-treated.
    fe : {"vs", "feo", "none"}, optional
        Fixed effects specification. Defaults to "feo" for linear, "none" for GLM.
    family : {"poisson", "logit", "probit", "gaussian"}, optional
        Distribution family for GLM. None defaults to Gaussian (OLS).
    vcov : str or dict, optional
        Variance-covariance estimator. Defaults to "hetero".
    fit_kwargs : dict
        Additional keyword arguments passed to pyfixest estimators.

    Examples
    --------
    >>> import pandas as pd
    >>> from etwfe import ETWFE
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'id': [1, 1, 2, 2, 3, 3],
    ...     'year': [2000, 2001, 2000, 2001, 2000, 2001],
    ...     'first_treat': [2001, 2001, 2001, 2001, 9999, 9999],
    ...     'y': [1.0, 2.5, 1.2, 2.8, 1.1, 1.3]
    ... })
    >>> model = ETWFE(
    ...     fml="y ~ 0",
    ...     tvar="year",
    ...     gvar="first_treat",
    ...     data=data,
    ...     ivar="id"
    ... )
    >>> model.fit()
    >>> model.summary()
    """

    fml: str
    tvar: str
    gvar: str
    data: pd.DataFrame
    ivar: Optional[str] = None
    xvar: Optional[str] = None
    tref: Optional[int] = None
    gref: Optional[int] = None
    cgroup: Literal["notyet", "never"] = "notyet"
    fe: Optional[Literal["vs", "feo", "none"]] = None
    family: Optional[Literal["poisson", "logit", "probit", "gaussian"]] = None
    vcov: Optional[Union[str, Dict[str, Any]]] = None
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Internal state
    model_: Any = field(default=None, repr=False)
    formula_: Optional[str] = field(default=None, repr=False)
    _fit_data: Optional[pd.DataFrame] = field(default=None, repr=False)
    _gref_min_flag: bool = field(default=False, repr=False)
    _yvar: Optional[str] = field(default=None, repr=False)
    _ctrls: List[str] = field(default_factory=list, repr=False)

    # xvar internals
    _xvar_dm_cols: List[str] = field(default_factory=list, repr=False)
    _xvar_time_dummies: List[str] = field(default_factory=list, repr=False)

    # Internal categorical columns
    _gcat_col: str = field(default="__etwfe_gcat", repr=False)
    _tcat_col: str = field(default="__etwfe_tcat", repr=False)

    def __post_init__(self) -> None:
        self.data = self.data.copy()

        if self.family is not None and self.family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"Unsupported family '{self.family}'. " f"Supported: {SUPPORTED_FAMILIES - {None}}"
            )

        # Non-linear non-gaussian: mirror etwfe behavior (no absorbed FE for SEs)
        if self.family is not None and self.family != "gaussian":
            if self.ivar is not None:
                warnings.warn(
                    f"Non-linear family '{self.family}' detected. "
                    "Setting ivar=None (unit FE not supported for GLM here)."
                )
                self.ivar = None
            if self.fe is None:
                self.fe = "none"

        if self.fe is None:
            self.fe = "feo"

        self._parse_formula()
        self._set_references()

    def _parse_formula(self) -> None:
        """Parse the formula and validate columns."""
        parts = self.fml.replace(" ", "").split("~")
        if len(parts) != 2:
            raise ValueError(f"Invalid formula: {self.fml}")

        self._yvar = parts[0]
        rhs = parts[1]
        self._ctrls = [] if rhs in ("0", "1") else [c.strip() for c in rhs.split("+")]

        required = [self._yvar, self.tvar, self.gvar] + self._ctrls
        if self.ivar:
            required.append(self.ivar)
        if self.xvar:
            required.append(self.xvar)

        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _set_references(self) -> None:
        """Set reference levels for time and cohort."""
        ug = sorted(pd.unique(self.data[self.gvar].dropna()))
        ut = sorted(pd.unique(self.data[self.tvar].dropna()))

        if self.tref is None:
            self.tref = int(min(ut))

        if self.gref is None:
            gref_cands = [g for g in ug if g > max(ut)]
            if len(gref_cands) == 0:
                gref_cands = [g for g in ug if g < min(ut)]
            if len(gref_cands) == 0 and self.cgroup == "notyet":
                gref_cands = [max(ug)]
            if len(gref_cands) == 0:
                raise ValueError(f"Could not identify '{self.cgroup}' control group.")
            self.gref = int(min(gref_cands))

        self._gref_min_flag = self.gref < min(ut)

    def _ensure_ref_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ordered categoricals so patsy/pyfixest uses correct reference levels.

        This matches fixest's i(g, i.t, ref=gref, ref2=tref) baseline behavior.
        """
        gvals = df[self.gvar].dropna().astype(int)
        tvals = df[self.tvar].dropna().astype(int)

        g_levels = sorted(pd.unique(gvals))
        t_levels = sorted(pd.unique(tvals))

        if int(self.gref) not in g_levels:
            g_levels = [int(self.gref)] + g_levels
        if int(self.tref) not in t_levels:
            t_levels = [int(self.tref)] + t_levels

        g_cats = [int(self.gref)] + [g for g in g_levels if int(g) != int(self.gref)]
        t_cats = [int(self.tref)] + [t for t in t_levels if int(t) != int(self.tref)]

        df[self._gcat_col] = pd.Categorical(
            df[self.gvar].astype("Int64").astype(float).astype("Int64"),
            categories=g_cats,
            ordered=True,
        )
        df[self._tcat_col] = pd.Categorical(
            df[self.tvar].astype("Int64").astype(float).astype("Int64"),
            categories=t_cats,
            ordered=True,
        )
        return df

    def _ensure_str_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add convenience string columns for printing."""
        df[f"{self.gvar}_str"] = df[self.gvar].astype(int).astype(str)
        df[f"{self.tvar}_str"] = df[self.tvar].astype(int).astype(str)
        if self.ivar:
            df[f"{self.ivar}_str"] = df[self.ivar].astype(str)
        return df

    def _build_xvar_dm_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Build demeaned xvar columns matching R etwfe logic.

        Creates .Dtreated_cohort = 1 for cohorts != gref, else 0,
        then demeans xvar by cohort using those weights.
        """
        assert self.xvar is not None

        w = (
            ((df[self.gvar].notna()) & (df[self.gvar].astype(int) != int(self.gref)))
            .astype(float)
            .values
        )
        df["_Dtreated_cohort"] = w

        x = df[self.xvar]
        dm_cols: List[str] = []

        # Numeric / bool
        if (x.dtype == bool) or np.issubdtype(x.dtype, np.number):
            x_num = x.astype(float).values
            g = df[self.gvar].astype(int).values
            sumw = pd.Series(w).groupby(g).sum()
            sumwx = pd.Series(w * x_num).groupby(g).sum()
            mean_g = (sumwx / sumw).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            dm = x_num - mean_g.reindex(g).values
            col = f"{self.xvar}_dm"
            df[col] = dm.astype(float)
            dm_cols.append(col)
            return dm_cols

        # Categorical / object => dummy expansion (k-1) then demean each dummy
        x_cat = x.astype("category")
        dummies = pd.get_dummies(x_cat, prefix=self.xvar, drop_first=True, dtype=float)
        g = df[self.gvar].astype(int).values

        for c in dummies.columns:
            v = dummies[c].values
            sumw = pd.Series(w).groupby(g).sum()
            sumwv = pd.Series(w * v).groupby(g).sum()
            mean_g = (sumwv / sumw).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            dm = v - mean_g.reindex(g).values
            dm_col = f"{c}_dm"
            df[dm_col] = dm.astype(float)
            dm_cols.append(dm_col)

        return dm_cols

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for estimation."""
        df = self.data.copy()
        df["_g"] = df[self.gvar].astype(float)
        df["_t"] = df[self.tvar].astype(float)

        # Treatment indicator
        if self.cgroup == "notyet":
            df["_Dtreat"] = ((df["_t"] >= df["_g"]) & (df["_g"] != self.gref)).astype(float)
            if not self._gref_min_flag:
                df.loc[df["_t"] >= self.gref, "_Dtreat"] = np.nan
            else:
                df.loc[df["_t"] <= self.gref, "_Dtreat"] = np.nan
        else:
            df["_Dtreat"] = (df["_t"] != (df["_g"] - 1)).astype(float)
            df.loc[df["_g"] == self.gref, "_Dtreat"] = 0.0

        # Demean controls by cohort
        for ctrl in self._ctrls:
            df[f"{ctrl}_dm"] = (
                df.groupby(self.gvar)[ctrl].transform(lambda x: x - x.mean()).astype(float)
            )

        # xvar handling
        self._xvar_dm_cols = []
        self._xvar_time_dummies = []

        if self.xvar:
            self._xvar_dm_cols = self._build_xvar_dm_columns(df)
            all_times = sorted(pd.unique(df[self.tvar].dropna()))

            for dm_col in self._xvar_dm_cols:
                for t in all_times:
                    if int(t) == int(self.tref):
                        continue
                    name = f"_t{int(t)}_{dm_col}"
                    df[name] = ((df[self.tvar] == t).astype(float) * df[dm_col]).astype(float)
                    self._xvar_time_dummies.append(name)

        # Enforce reference levels
        df = self._ensure_ref_categoricals(df)
        df = self._ensure_str_cols(df)
        df["_event"] = (df["_t"] - df["_g"]).astype(float)

        return df

    def _build_formula(self) -> str:
        """Build the pyfixest formula."""
        gcat = self._gcat_col
        tcat = self._tcat_col
        main_int = f"C({gcat}):C({tcat})"

        parts: List[str] = [f"_Dtreat:{main_int}"]

        if self._ctrls:
            for ctrl in self._ctrls:
                parts.append(f"_Dtreat:{main_int}:{ctrl}_dm")
            if self.fe != "vs":
                for ctrl in self._ctrls:
                    parts.extend([ctrl, f"C({gcat}):{ctrl}", f"C({tcat}):{ctrl}"])

        if self.xvar:
            for dm_col in self._xvar_dm_cols:
                parts.append(f"_Dtreat:{main_int}:{dm_col}")
            if self._xvar_time_dummies:
                parts.extend(self._xvar_time_dummies)

        rhs = " + ".join(parts)

        if self.fe != "none":
            fe_var = self.ivar if self.ivar else self.gvar
            formula = f"{self._yvar} ~ {rhs} | {fe_var} + {self.tvar}"
        else:
            parts.extend([f"C({gcat})", f"C({tcat})"])
            formula = f"{self._yvar} ~ {' + '.join(parts)}"

        self.formula_ = formula
        return formula

    def fit(self) -> "ETWFE":
        """
        Fit the ETWFE model.

        Returns
        -------
        ETWFE
            The fitted model (self).
        """
        df = self._prepare_data()
        df_fit = df.dropna(subset=["_Dtreat", self._yvar]).copy()
        df_fit["_Dtreat"] = df_fit["_Dtreat"].astype(float)
        formula = self._build_formula()

        vcov = self.vcov if self.vcov else "hetero"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            if self.family is None or self.family == "gaussian":
                self.model_ = pf.feols(formula, data=df_fit, vcov=vcov, **self.fit_kwargs)

            elif self.family == "poisson":
                if not hasattr(pf, "fepois"):
                    raise NotImplementedError(
                        "pyfixest.fepois is not available in this pyfixest version. "
                        "Please upgrade pyfixest or use a supported GLM backend."
                    )
                self.model_ = pf.fepois(formula, data=df_fit, vcov=vcov, **self.fit_kwargs)

            elif self.family in ("logit", "probit"):
                if not hasattr(pf, "feglm"):
                    raise NotImplementedError(
                        "pyfixest.feglm is not available in your pyfixest version."
                    )
                self.model_ = pf.feglm(
                    formula, data=df_fit, vcov=vcov, family=self.family, **self.fit_kwargs
                )
            else:
                raise ValueError(f"Unsupported family: {self.family}")

        self._fit_data = df_fit
        return self

    def _compress_data(
        self, df: pd.DataFrame, by_xvar: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Compress data by period-cohort groups for faster marginal effects."""
        group_cols = [self.gvar, self.tvar]
        if by_xvar and self.xvar:
            group_cols.append(self.xvar)

        weights_df = df.groupby(group_cols).size().reset_index(name="_N")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_keep = group_cols + [c for c in numeric_cols if c not in group_cols]
        df_compressed = df[cols_to_keep].groupby(group_cols).mean().reset_index()
        df_compressed = df_compressed.merge(weights_df, on=group_cols, how="left")

        df_compressed = self._ensure_ref_categoricals(df_compressed)
        df_compressed = self._ensure_str_cols(df_compressed)

        return df_compressed, df_compressed["_N"].values

    def _rhs_formula_only(self) -> str:
        """Extract RHS of formula (before fixed effects)."""
        if self.formula_ is None:
            raise ValueError("Model formula not built.")
        return self.formula_.split("|")[0].strip()

    def _get_rhs_design_matrix(self, df_new: pd.DataFrame) -> np.ndarray:
        """Get design matrix for new data."""
        if patsy is None:
            raise ImportError(
                "patsy is required for model-matrix based emfx(). " "Install via: pip install patsy"
            )

        coef_names = self.model_.coef().index.tolist()
        fml_main = self._rhs_formula_only()

        _, X = patsy.dmatrices(fml_main, df_new, return_type="dataframe")
        X = X.reindex(columns=coef_names, fill_value=0.0)

        return X.to_numpy()

    def _invlink_and_deriv(self, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inverse link function and its derivative."""
        if self.family is None or self.family == "gaussian":
            return eta, np.ones_like(eta)

        if self.family == "poisson":
            mu = np.exp(eta)
            return mu, mu

        if self.family == "logit":
            mu = 1.0 / (1.0 + np.exp(-eta))
            return mu, mu * (1.0 - mu)

        if self.family == "probit":
            from scipy.stats import norm

            mu = norm.cdf(eta)
            return mu, norm.pdf(eta)

        raise ValueError(f"Unsupported family: {self.family}")

    def _compute_slopes_and_jacobians(
        self,
        df: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        predict: Literal["response", "link"] = "response",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute treatment effect slopes and Jacobians for delta method."""
        if weights is None:
            weights = np.ones(len(df), dtype=float)

        beta = self.model_.coef().values

        df1 = df.copy()
        df0 = df.copy()
        df1["_Dtreat"] = 1.0
        df0["_Dtreat"] = 0.0

        X1 = self._get_rhs_design_matrix(df1)
        X0 = self._get_rhs_design_matrix(df0)

        eta1 = X1 @ beta
        eta0 = X0 @ beta

        if predict == "link" or (self.family is None or self.family == "gaussian"):
            slopes = eta1 - eta0
            jac = X1 - X0
            return slopes, jac, weights

        mu1, d1 = self._invlink_and_deriv(eta1)
        mu0, d0 = self._invlink_and_deriv(eta0)

        slopes = mu1 - mu0
        jac = (d1[:, None] * X1) - (d0[:, None] * X0)

        return slopes, jac, weights

    def emfx(
        self,
        type: Literal["simple", "group", "calendar", "event"] = "simple",
        by_xvar: Union[bool, str] = "auto",
        compress: Union[bool, str] = "auto",
        predict: Literal["response", "link"] = "response",
        post_only: bool = True,
        vcov: Union[bool, str, Dict[str, Any], None] = None,
        window: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> pd.DataFrame:
        """
        Compute marginal effects (ATT estimates).

        Parameters
        ----------
        type : {"simple", "group", "calendar", "event"}
            Type of marginal effect to compute:
            - "simple": Overall ATT
            - "group": ATT by treatment cohort
            - "calendar": ATT by calendar time
            - "event": ATT by event time (periods since treatment)
        by_xvar : bool or "auto"
            Whether to compute effects by xvar levels.
        compress : bool or "auto"
            Whether to compress data for faster computation.
        predict : {"response", "link"}
            Prediction scale for GLM models.
        post_only : bool
            Whether to include only post-treatment observations.
        vcov : bool, str, dict, or None
            Variance-covariance specification. False to skip SE computation.
        window : int or tuple, optional
            Event window for filtering (for event study).

        Returns
        -------
        pd.DataFrame
            DataFrame with estimates, standard errors, and confidence intervals.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if by_xvar == "auto":
            by_xvar = self.xvar is not None

        nrows = len(self._fit_data)
        if compress == "auto":
            compress = nrows >= 500_000

        if compress and self.ivar is not None:
            warnings.warn("ivar is not None. Marginal effects calculated without compression.")
            compress = False

        if compress and nrows >= 500_000:
            warnings.warn(
                f"Dataset has {nrows:,} rows. Compressing by period-cohort groups "
                "to reduce estimation time. This may slightly reduce accuracy. "
                "Override with compress=False."
            )

        df = self._fit_data.copy()

        # R-matching filtering
        if self.cgroup == "never":
            df = df[df["_g"] != self.gref].copy()
            if type != "event":
                df = df[(df["_Dtreat"] == 1.0) & (df["_t"] >= df["_g"])].copy()
            elif type == "event" and not post_only:
                df = df[df["_g"] != self.gref].copy()
            else:
                df = df[df["_Dtreat"] == 1.0].copy()
        else:
            df = df[df["_Dtreat"] == 1.0].copy()

        if window is not None:
            w = (window, window) if isinstance(window, int) else window
            df = df[(df["_t"] >= df["_g"] - w[0]) & (df["_t"] <= df["_g"] + w[1])].copy()

        if type == "event":
            df["event"] = (df["_t"] - df["_g"]).astype(int)

        weights = None
        if compress:
            df, weights = self._compress_data(df, by_xvar=by_xvar)

        skip_vcov = vcov is False
        if (
            not skip_vcov
            and self.family is not None
            and self.family != "gaussian"
            and self.fe != "none"
        ):
            warnings.warn(
                f"Cannot estimate standard errors for non-Gaussian family "
                f"'{self.family}' with fixed effects. Re-estimate with fe='none' "
                "for SEs, or use vcov=False."
            )
            skip_vcov = True

        slopes, jacobians, weights = self._compute_slopes_and_jacobians(
            df, weights=weights, predict=predict
        )

        vcov_matrix = None if skip_vcov else getattr(self.model_, "_vcov", None)

        def compute_aggregated(
            mask: np.ndarray, w_subset: Optional[np.ndarray]
        ) -> Tuple[float, float, float, float]:
            if w_subset is None:
                w_subset = np.ones(mask.sum(), dtype=float)

            w_sum = w_subset.sum()
            est = np.average(slopes[mask], weights=w_subset)

            if skip_vcov or vcov_matrix is None:
                return est, np.nan, np.nan, np.nan

            WJ = jacobians[mask] * w_subset[:, None]
            gbar = WJ.sum(axis=0, keepdims=True) / w_sum
            se = float(np.sqrt(gbar @ vcov_matrix @ gbar.T)[0, 0])

            return est, se, est - 1.96 * se, est + 1.96 * se

        if type == "simple":
            df["_Dtreat_agg"] = 1.0
            group_var = "_Dtreat_agg"
            primary_var = "_Dtreat"
        elif type == "event":
            group_var = "event"
            primary_var = "event"
        elif type == "group":
            group_var = self.gvar
            primary_var = self.gvar
        else:
            group_var = self.tvar
            primary_var = self.tvar

        group_vars: List[str] = [group_var]
        if by_xvar and self.xvar:
            group_vars.append(self.xvar)

        results: List[Dict[str, Any]] = []

        if len(group_vars) == 1:
            for val in sorted(pd.unique(df[group_var])):
                mask = df[group_var].values == val
                if mask.sum() == 0:
                    continue

                w_subset = weights[mask] if weights is not None else None
                est, se, lo, hi = compute_aggregated(mask, w_subset)

                results.append(
                    {
                        primary_var: (1.0 if type == "simple" else val),
                        "estimate": est,
                        "std.error": se,
                        "conf.low": lo,
                        "conf.high": hi,
                    }
                )
        else:
            combos = df[group_vars].drop_duplicates()
            for _, combo in combos.iterrows():
                mask = np.ones(len(df), dtype=bool)
                for gv in group_vars:
                    mask &= df[gv].values == combo[gv]

                if mask.sum() == 0:
                    continue

                w_subset = weights[mask] if weights is not None else None
                est, se, lo, hi = compute_aggregated(mask, w_subset)

                row = {
                    "estimate": est,
                    "std.error": se,
                    "conf.low": lo,
                    "conf.high": hi,
                }
                if type == "simple":
                    row["_Dtreat"] = 1.0
                else:
                    row[primary_var] = combo[group_var]
                row[self.xvar] = combo[self.xvar]
                results.append(row)

        out = pd.DataFrame(results)

        sort_cols = [primary_var]
        if by_xvar and self.xvar and self.xvar in out.columns:
            sort_cols.append(self.xvar)

        out = out.sort_values(sort_cols).reset_index(drop=True)

        # Compatibility columns
        if type == "group" and self.gvar in out.columns:
            out[f"{self.gvar}_str"] = out[self.gvar].astype(int).astype(str)
        if type == "calendar" and self.tvar in out.columns:
            out[f"{self.tvar}_str"] = out[self.tvar].astype(int).astype(str)

        return out

    def summary(self) -> None:
        """Print a summary of the fitted model."""
        if self.model_ is None:
            print("Model not fitted. Call fit() first.")
            return

        print("=" * 70)
        print("ETWFE: Extended Two-Way Fixed Effects (Wooldridge 2021, 2023)")
        print("=" * 70)
        print(f"Outcome: {self._yvar}")
        print(f"Time: {self.tvar} (ref: {self.tref})")
        print(f"Cohort: {self.gvar} (ref: {self.gref})")
        print(f"Control group: {self.cgroup}")

        if self.family:
            print(f"Family: {self.family}")
        if self._ctrls:
            print(f"Controls: {', '.join(self._ctrls)}")
        if self.ivar:
            print(f"Unit FE: {self.ivar}")
        if self.xvar:
            print(f"Heterogeneity: {self.xvar}")

        print(f"\nObservations: {len(self._fit_data):,}")
        print(f"Treated obs: {(self._fit_data['_Dtreat'] == 1).sum():,}")

        att = self.emfx(type="simple")
        se_val = att["std.error"].values[0]
        print(f"\nSimple ATT: {att['estimate'].values[0]:.6f} (SE: {se_val:.6f})")
        if not np.isnan(se_val):
            print(
                f"95% CI: [{att['conf.low'].values[0]:.6f}, " f"{att['conf.high'].values[0]:.6f}]"
            )

    def plot(
        self,
        type: Literal["event", "group", "calendar"] = "event",
        post_only: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        color: str = "darkcyan",
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """
        Plot marginal effects.

        Parameters
        ----------
        type : {"event", "group", "calendar"}
            Type of plot to create.
        post_only : bool
            Whether to include only post-treatment observations.
        figsize : tuple
            Figure size.
        title : str, optional
            Plot title.
        color : str
            Color for the plot.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs
            Additional arguments passed to emfx().

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        mfx = self.emfx(type=type, post_only=post_only, **kwargs)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if type == "event":
            x_var = "event"
        elif type == "group":
            x_var = self.gvar
        else:
            x_var = self.tvar

        x = mfx[x_var].astype(float).values
        y = mfx["estimate"].values

        if (
            "conf.low" in mfx.columns
            and "conf.high" in mfx.columns
            and not np.isnan(mfx["conf.low"].values).all()
        ):
            yerr = [y - mfx["conf.low"].values, mfx["conf.high"].values - y]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                capsize=4,
                color=color,
                markersize=8,
                linewidth=2,
                capthick=1.5,
            )
        else:
            ax.plot(x, y, "o", color=color, markersize=8)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

        if type == "event":
            ax.axvline(-0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        xlabel = {
            "event": "Event Time (periods relative to treatment)",
            "group": "Treatment Cohort",
            "calendar": "Calendar Time",
        }[type]

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Average Treatment Effect", fontsize=11)

        family_str = f" ({self.family})" if self.family else ""
        ax.set_title(title or f"ETWFE {type.capitalize()} Study{family_str}", fontsize=13)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return ax


def etwfe(
    fml: str,
    tvar: str,
    gvar: str,
    data: pd.DataFrame,
    ivar: Optional[str] = None,
    xvar: Optional[str] = None,
    cgroup: Literal["notyet", "never"] = "notyet",
    family: Optional[Literal["poisson", "logit", "probit", "gaussian"]] = None,
    vcov: Optional[Union[str, Dict[str, Any]]] = None,
    **fit_kwargs: Any,
) -> ETWFE:
    """
    Convenience function to create and fit an ETWFE model.

    Parameters
    ----------
    fml : str
        Formula in the form "y ~ x1 + x2" or "y ~ 0" for no controls.
    tvar : str
        Name of the time period variable.
    gvar : str
        Name of the treatment cohort (first-treated period) variable.
    data : pd.DataFrame
        Panel data with observations.
    ivar : str, optional
        Name of the unit identifier for unit fixed effects.
    xvar : str, optional
        Name of a variable for heterogeneous treatment effects.
    cgroup : {"notyet", "never"}
        Control group type.
    family : {"poisson", "logit", "probit", "gaussian"}, optional
        Distribution family for GLM.
    vcov : str or dict, optional
        Variance-covariance estimator.
    **fit_kwargs
        Additional arguments passed to the estimator.

    Returns
    -------
    ETWFE
        The fitted ETWFE model.

    Examples
    --------
    >>> from etwfe import etwfe
    >>> model = etwfe("y ~ 0", tvar="year", gvar="first_treat", data=df, ivar="id")
    >>> model.summary()
    >>> model.emfx(type="event")
    """
    model = ETWFE(
        fml=fml,
        tvar=tvar,
        gvar=gvar,
        data=data,
        ivar=ivar,
        xvar=xvar,
        cgroup=cgroup,
        family=family,
        vcov=vcov,
        fit_kwargs=fit_kwargs,
    )
    model.fit()
    return model
