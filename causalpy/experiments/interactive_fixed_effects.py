#   Copyright 2026 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Bayesian interactive fixed-effects synthetic control.

The untreated path is Xu's linear factor mean: an additive unit intercept,
an additive time intercept, and a loading-factor product. Rank is not
selected. A multiplicative gamma process shrinks unused factor scales toward
zero, and posterior intervals marginalize that prior.

Shared factors and shared time intercepts are updated from the control panel
only. The control likelihood uses the one shared positive residual scale, not
a unit observation scale. That is one random variable, bounded below by a
fraction of the control-panel outcome spread. A noise-free panel would
otherwise collapse the factor posterior onto a ridge of exact fits. Treated
pre-period outcomes may update that scale's draws after the factor and
time-intercept sample, and must not resample those draws. Loadings use the
updated draws. The factor posterior is the control-panel posterior under that
scale as informed by the control panel. It is not recomputed under the
treated-pre update. Treated post-intervention outcomes never enter the
likelihood.

The control unit intercepts and loadings are integrated out of the control likelihood, which is exact for this Gaussian model. NUTS samples only the shared terms, and the integrated terms are drawn exactly from their Gaussian conditional after sampling. Treated intercepts and loadings are drawn the same way, at the updated scale, one draw per control-panel draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from matplotlib import pyplot as plt
from pytensor.tensor.linalg import solve_triangular

from causalpy.constants import HDI_PROB
from causalpy.date_utils import validate_treatment_time_against_index
from causalpy.experiments._panel_counterfactual import WidePanel, resolve_treated_unit
from causalpy.experiments._results import CausalResult
from causalpy.plot_utils import _PosteriorPlotStyle, format_r2_score
from causalpy.utils import _bayesian_r2_score

_DRAW_DIMS = ("chain", "draw")
_FACTORS = "factors"
_DEFAULT_MAX_FACTORS = 4
_MGP_A1 = 2.0
_MGP_A2 = 3.0
_RESIDUAL_FLOOR_FRACTION = 0.05
_MIN_OUTCOME_SPREAD = 1e-6
_SCALE_UPDATE_STEPS = 12
_SCALE_FD_STEP = 1e-4
_INTERCEPT_SD = 5.0


@dataclass(frozen=True)
class InteractiveFixedEffectsResult(CausalResult):
    """Result bundle for interactive fixed-effects synthetic control.

    ``predictions_*`` are the untreated counterfactual. Posterior intervals on
    those fields marginalize the shrinkage prior. They are not conditional on
    a selected rank. ``score`` is the pre-period prediction error, fit to
    pre-period treated outcomes only.

    ``shrinkage_state`` holds posterior draws of one positive scale per factor,
    up to ``max_factors``. It is not a selected integer rank and not a derived
    effective-rank count.
    """

    shrinkage_state: xr.DataArray = field(kw_only=True)


class InteractiveFixedEffectsSyntheticControl:
    """Interactive fixed-effects synthetic control on a :class:`WidePanel`.

    Not a :class:`~causalpy.experiments.synthetic_control.SyntheticControl`
    subclass. Several treated units share one ``treatment_time``. Control units
    are donors. There is no ``n_factors`` argument: ``max_factors`` is an upper
    bound, not a selected rank.

    ``build`` constructs the control-panel graph. Its likelihood uses the one
    shared positive residual scale, not a unit observation scale. That scale
    is bounded below by a fraction of the control-panel outcome spread.
    ``fit`` samples the control-panel posterior of the shared factors, shared
    time intercepts, per-factor shrinkage scales, and that one scale. Treated
    pre-period outcomes may update the scale's draws after that sample. They
    must not resample the factor or time-intercept draws. Loadings use the
    updated draws.

    The untreated counterfactual is not a donor-row prediction. Raw factors and
    raw loadings are not identifiers: a rotation or a sign flip leaves the
    product unchanged. This sampler does not impose orthonormality.

    Parameters
    ----------
    data : pandas.DataFrame
        Balanced wide panel indexed by time. Used columns must have no
        missing values.
    treatment_time : int, float, or pandas.Timestamp
        Shared first post-period time. Rows at this time are post-period.
    control_units : sequence of str
        Donor columns.
    treated_units : sequence of str
        Treated columns. One or more.
    max_factors : int, default 4
        Upper bound on the number of factors. Not a selected rank.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_time: int | float | pd.Timestamp,
        control_units: list[str] | tuple[str, ...],
        treated_units: list[str] | tuple[str, ...],
        *,
        max_factors: int = _DEFAULT_MAX_FACTORS,
    ) -> None:
        if isinstance(max_factors, bool) or not isinstance(max_factors, int):
            raise TypeError("max_factors must be an int, not a selected rank.")
        if max_factors < 1:
            raise ValueError("max_factors must be at least 1.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        validate_treatment_time_against_index(data.index, treatment_time)
        control = tuple(control_units)
        treated = tuple(treated_units)
        if not control or not treated:
            raise ValueError(
                "v1 requires one or more treated units and control units as donors."
            )
        overlap = set(control) & set(treated)
        if overlap:
            raise ValueError(
                f"control_units and treated_units overlap: {sorted(overlap)}."
            )
        missing_columns = [
            name for name in control + treated if name not in data.columns
        ]
        if missing_columns:
            raise KeyError(f"panel is missing columns: {missing_columns}.")
        used = data.loc[:, list(control + treated)]
        if used.isna().any().any():
            raise ValueError(
                "missing values in the used columns are out of bounds for v1."
            )
        panel = WidePanel.from_frame(data, treatment_time, control, treated)
        if panel.pre.empty or panel.post.empty:
            raise ValueError("the panel must have both a pre-period and a post-period.")
        self.panel = panel
        self.max_factors = max_factors
        self._control_model: pm.Model | None = None
        self.result: InteractiveFixedEffectsResult | None = None

    @property
    def control_units(self) -> tuple[str, ...]:
        """Donor names, in the order supplied at construction."""
        return self.panel.control_units

    @property
    def treated_units(self) -> tuple[str, ...]:
        """Treated names, in the order supplied at construction."""
        return self.panel.treated_units

    def build(self) -> Self:
        """Construct the control-panel graph without sampling.

        Idempotent. The graph's observed outcomes are control units at every
        time. Treated outcomes, including pre-period outcomes, are not in this
        graph: they must not update factors or time intercepts.
        """
        if self._control_model is None:
            self._control_model = _control_model(
                self._control_outcomes(), self.max_factors
            )
        return self

    def fit(
        self,
        *,
        draws: int = 40,
        tune: int = 80,
        chains: int = 1,
        cores: int = 1,
        random_seed: int | None = None,
        progressbar: bool = False,
        **sample_kwargs: Any,
    ) -> Self:
        """Sample the joint posterior and populate :attr:`result`.

        Keyword arguments are forwarded to the control-panel :func:`pymc.sample`
        call. That sample integrates out the control intercepts and loadings; they are then drawn exactly, one per control-panel draw. The treated conditional draws are aligned the same way. Before they are drawn, the one shared residual scale's draws are updated from treated pre-period outcomes. The treated intercepts and loadings are drawn exactly from their Gaussian conditional at those updated draws. This does not resample the scale, the factors, or the time intercepts, and it does not take caller-supplied factor draws.

        Treated post-intervention outcomes are read only when impacts are
        aligned, after sampling. Changing them does not change the untreated
        trajectory.

        Parameters
        ----------
        draws : int, default 40
            Control-panel posterior draws per chain.
        tune : int, default 80
            Control-panel tuning steps.
        chains : int, default 1
            Control-panel chains.
        cores : int, default 1
            Cores for the control-panel sample.
        random_seed : int, optional
            Seed for the control-panel sample. The conditional sample uses the
            next integer when this is set.
        progressbar : bool, default False
            Whether PyMC shows a progress bar.
        **sample_kwargs
            Extra arguments for the control-panel :func:`pymc.sample` call.

        Returns
        -------
        InteractiveFixedEffectsSyntheticControl
            The fitted experiment.
        """
        self.build()
        assert self._control_model is not None
        control_idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            progressbar=progressbar,
            model=self._control_model,
            initvals=_control_initvals(self._control_model, self._control_outcomes()),
            **sample_kwargs,
        )
        rng = np.random.default_rng(None if random_seed is None else random_seed + 1)
        posterior = _with_control_draws(
            _as_dataset(control_idata), self._control_outcomes(), rng
        )
        conditional = _sample_treated_conditional(
            self._control_outcomes(), self.panel.treated_pre, posterior, rng=rng
        )
        self.result = self.compose_result(
            unit_intercept=conditional["unit_intercept"],
            time_intercept=posterior["time_intercept"],
            loadings=conditional["loadings"],
            factors=posterior["shared_factors"],
            shrinkage_state=posterior["shrinkage_state"],
        )
        return self

    def compose_result(
        self,
        *,
        unit_intercept: xr.DataArray,
        time_intercept: xr.DataArray,
        loadings: xr.DataArray,
        factors: xr.DataArray,
        shrinkage_state: xr.DataArray,
    ) -> InteractiveFixedEffectsResult:
        """Compose the frozen result from component draws.

        The untreated counterfactual at each time is the additive unit
        intercept plus the additive time intercept plus the loading-factor
        product. Impact uses :meth:`WidePanel.impacts`. ``score`` uses
        pre-period treated outcomes only. This method does not update
        shrinkage from treated post-intervention outcomes and does not select
        a rank.

        Parameters
        ----------
        unit_intercept : xarray.DataArray
            Draws with dimensions ``chain``, ``draw``, and ``treated_units``.
        time_intercept : xarray.DataArray
            Draws with dimensions ``chain``, ``draw``, and ``obs_ind``.
        loadings : xarray.DataArray
            Per-treated-unit loadings. The ``factors`` axis has length
            ``max_factors``.
        factors : xarray.DataArray
            Shared factors with dimensions ``chain``, ``draw``, ``obs_ind``,
            and ``factors``.
        shrinkage_state : xarray.DataArray
            One positive scale per factor, dimensions ``chain``, ``draw``,
            and ``factors``.

        Returns
        -------
        InteractiveFixedEffectsResult
            Predictions, impacts, pre-period score, and shrinkage state.
        """
        self._require_shrinkage_state(shrinkage_state)
        self._require_factor_axis(loadings, "loadings")
        self._require_factor_axis(factors, "factors")
        counterfactual = untreated_counterfactual(
            unit_intercept, time_intercept, loadings, factors
        )
        predictions_pre = counterfactual.sel(obs_ind=self.panel.pre.index)
        predictions_post = counterfactual.sel(obs_ind=self.panel.post.index)
        impact_pre, impact_post, impact_post_cumulative = self.panel.impacts(
            predictions_pre, predictions_post
        )
        return InteractiveFixedEffectsResult(
            predictions_pre=predictions_pre,
            predictions_post=predictions_post,
            impact_pre=impact_pre,
            impact_post=impact_post,
            impact_post_cumulative=impact_post_cumulative,
            score=pre_period_score(predictions_pre, self.panel.treated_pre),
            shrinkage_state=shrinkage_state,
        )

    def plot(
        self,
        *,
        round_to: int | None = None,
        treated_unit: str | None = None,
        ci_prob: float = HDI_PROB,
        kind: Literal["ribbon", "histogram", "spaghetti"] = "ribbon",
        ci_kind: Literal["hdi", "eti"] = "hdi",
        num_samples: int = 50,
        plot_predictors: bool = False,
        figsize: tuple[float, float] = (7, 8),
        show: bool = True,
    ) -> tuple[plt.Figure, list[plt.Axes] | np.ndarray]:
        """Plot the reconstructed untreated trajectory and impact panels.

        Requires :meth:`fit`. Bands are posterior intervals for the untreated
        trajectory and the impacts. They marginalize the shrinkage prior and
        do not claim a selected-rank interval or nominal coverage. Raw factor
        series are not drawn: they are not uniquely identified.

        Parameters
        ----------
        round_to : int, optional
            Significant figures for the pre-period :math:`R^2` in the title.
        treated_unit : str, optional
            Treated unit to draw. Defaults to the first treated name.
        ci_prob : float, default 0.94
            Interval mass. Must be in ``(0, 1]``.
        kind : {"ribbon", "histogram", "spaghetti"}, default "ribbon"
            How posterior uncertainty is rendered.
        ci_kind : {"hdi", "eti"}, default "hdi"
            Interval type when ``kind="ribbon"``.
        num_samples : int, default 50
            Draw lines when ``kind="spaghetti"``.
        plot_predictors : bool, default False
            Overlay donor trajectories on the counterfactual panel.
        figsize : tuple of float, default (7, 8)
            Figure size in inches.
        show : bool, default True
            Whether to display the figure.

        Returns
        -------
        tuple of matplotlib.figure.Figure and numpy.ndarray
            Three panels: untreated trajectory, point impact, cumulative impact.
        """
        if self.result is None:
            raise RuntimeError("Call fit() before plot().")
        style: _PosteriorPlotStyle = {
            "ci_prob": ci_prob,
            "kind": kind,
            "ci_kind": ci_kind,
            "num_samples": num_samples,
        }
        unit = resolve_treated_unit(self.treated_units, treated_unit)
        unit_index = self.treated_units.index(unit)
        title = format_r2_score(
            self.result.score,
            unit_index=unit_index,
            round_to=round_to,
            context="on pre-period treated outcomes",
        )
        figure, axes = self.panel.plot(
            self.result,
            group="posterior",
            treated_unit=unit,
            title=title,
            style=style,
            figsize=figsize,
            plot_predictors=plot_predictors,
        )
        if show:
            plt.show()
        return figure, axes

    def _control_outcomes(self) -> xr.DataArray:
        """Control outcomes at every time. Treated columns are not included."""
        pre = self.panel.control("pre")
        post = self.panel.control("post")
        return xr.concat([pre, post], dim="obs_ind")

    def _require_shrinkage_state(self, shrinkage_state: xr.DataArray) -> None:
        if not isinstance(shrinkage_state, xr.DataArray):
            raise TypeError("shrinkage_state must be an xarray.DataArray of scales.")
        if shrinkage_state.dtype.kind in {"i", "u", "b"}:
            raise TypeError("shrinkage_state must be scale draws, not an integer rank.")
        dims = tuple(shrinkage_state.dims)
        if dims != (*_DRAW_DIMS, _FACTORS):
            raise ValueError(
                "shrinkage_state dims must be ('chain', 'draw', 'factors'), "
                f"got {dims}."
            )
        if shrinkage_state.sizes[_FACTORS] != self.max_factors:
            raise ValueError(
                "shrinkage_state must have one scale per factor up to "
                f"max_factors={self.max_factors}, got {shrinkage_state.sizes[_FACTORS]}."
            )

    def _require_factor_axis(self, array: xr.DataArray, name: str) -> None:
        if name == "factors":
            _require_shared_factors(array)
        if _FACTORS not in array.dims:
            raise ValueError(f"{name} must have a {_FACTORS!r} dimension.")
        if array.sizes[_FACTORS] != self.max_factors:
            raise ValueError(
                f"{name} factor axis must have length max_factors="
                f"{self.max_factors}, got {array.sizes[_FACTORS]}."
            )


def untreated_counterfactual(
    unit_intercept: xr.DataArray,
    time_intercept: xr.DataArray,
    loadings: xr.DataArray,
    factors: xr.DataArray,
) -> xr.DataArray:
    """Return unit intercept + time intercept + loading-factor product.

    Raw factors and raw loadings are not unique. The returned trajectory is
    the identified product, not a selected-factor reconstruction.

    Parameters
    ----------
    unit_intercept : xarray.DataArray
        Additive unit intercept. Must include ``treated_units``.
    time_intercept : xarray.DataArray
        Additive time intercept. Must include ``obs_ind``.
    loadings : xarray.DataArray
        Loadings with a ``factors`` dimension.
    factors : xarray.DataArray
        Shared factors. Dimensions must be ``chain``, ``draw``, ``obs_ind``,
        and ``factors``.

    Returns
    -------
    xarray.DataArray
        Untreated counterfactual.
    """
    _require_shared_factors(factors)
    product = (loadings * factors).sum(dim=_FACTORS)
    return unit_intercept + time_intercept + product


_SHARED_FACTOR_DIMS = frozenset((*_DRAW_DIMS, "obs_ind", _FACTORS))


def _require_shared_factors(factors: xr.DataArray) -> None:
    """Reject a factor array that is not shared across treated units."""
    dims = set(factors.dims)
    if dims != _SHARED_FACTOR_DIMS:
        raise ValueError(
            "factors must be the shared fit with dimensions "
            "('chain', 'draw', 'obs_ind', 'factors'), "
            f"got {tuple(factors.dims)}."
        )


def pre_period_score(
    predictions_pre: xr.DataArray,
    treated_pre: xr.DataArray,
) -> pd.Series:
    """Bayesian :math:`R^2` of pre-period predictions against pre-period outcomes.

    Post-period outcomes are not an argument.

    Parameters
    ----------
    predictions_pre : xarray.DataArray
        Counterfactual draws on the pre-period.
    treated_pre : xarray.DataArray
        Observed treated outcomes strictly before treatment.

    Returns
    -------
    pandas.Series
        ``unit_{i}_r2`` and ``unit_{i}_r2_std`` for each treated unit.
    """
    stacked = predictions_pre.stack(sample=_DRAW_DIMS)
    scores: dict[str, float] = {}
    for i, unit in enumerate(treated_pre.coords["treated_units"].values):
        unit_mu = stacked.sel(treated_units=unit).transpose("sample", "obs_ind")
        unit_y = treated_pre.sel(treated_units=unit)
        unit_score = _bayesian_r2_score(unit_y.to_numpy(), unit_mu.to_numpy())
        scores[f"unit_{i}_r2"] = float(unit_score["r2"])
        scores[f"unit_{i}_r2_std"] = float(unit_score["r2_std"])
    return pd.Series(scores)


def _factor_names(max_factors: int) -> list[str]:
    return [f"f{i}" for i in range(max_factors)]


def _control_outcome_spread(control_y: xr.DataArray) -> float:
    """Column-centered RMS of the control panel.

    The residual-scale floor is a fraction of this spread. A constant panel
    still gets a positive floor, so the factor posterior cannot sit on an
    exact-fit ridge.
    """
    values = np.asarray(
        control_y.transpose("obs_ind", "control_units").values, dtype=float
    )
    centered = values - values.mean(axis=0, keepdims=True)
    return float(max(np.sqrt(np.mean(centered**2)), _MIN_OUTCOME_SPREAD))


def _marginal_unit_logp(
    value: pt.TensorVariable,
    time_intercept: pt.TensorVariable,
    factors: pt.TensorVariable,
    shrinkage: pt.TensorVariable,
    sigma: pt.TensorVariable,
) -> pt.TensorVariable:
    """Log density of one unit's series with its intercept and loadings integrated out.

    Given the time intercepts, factors, factor scales, and residual scale, a unit's series is Gaussian: ``y = xi + W b + e`` with ``W = [1, F]``, ``b ~ N(0, D)`` and ``D = diag(_INTERCEPT_SD**2, shrinkage**2)``. The covariance ``sigma**2 I + W D W'`` is handled with the Woodbury identity and the matrix determinant lemma, so each unit costs one ``(max_factors + 1)``-square Cholesky. Leading dimensions of ``value`` are units.
    """
    centered = value - time_intercept
    batch = tuple(centered.shape[:-1])
    n_time = centered.shape[-1]
    n_factors = factors.shape[-1]
    factor_b = pt.broadcast_to(factors, (*batch, *factors.shape[-2:]))
    scale_b = pt.broadcast_to(shrinkage, (*batch, n_factors))
    variance = pt.broadcast_to(sigma, batch) ** 2
    design = pt.concatenate([pt.ones((*batch, n_time, 1)), factor_b], axis=-1)
    prior_var = pt.concatenate(
        [pt.full((*batch, 1), _INTERCEPT_SD**2), scale_b**2], axis=-1
    )
    design_t = pt.matrix_transpose(design)
    precision = (
        pt.eye(n_factors + 1) / prior_var[..., None, :]
        + design_t @ design / variance[..., None, None]
    )
    chol = pt.linalg.cholesky(precision)
    whitened = solve_triangular(
        chol, design_t @ centered[..., None], lower=True, b_ndim=2
    )[..., 0]
    quadratic = (
        pt.sum(centered**2, axis=-1) / variance
        - pt.sum(whitened**2, axis=-1) / variance**2
    )
    logdet = (
        n_time * pt.log(variance)
        + pt.sum(pt.log(prior_var), axis=-1)
        + 2.0 * pt.sum(pt.log(pt.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    )
    return -0.5 * (n_time * np.log(2.0 * np.pi) + logdet + quadratic)


def _marginal_unit_random(
    time_intercept: np.ndarray,
    factors: np.ndarray,
    shrinkage: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator | None = None,
    size: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Draw unit series from the generative model behind :func:`_marginal_unit_logp`."""
    rng = np.random.default_rng() if rng is None else rng
    time_intercept, factors, shrinkage, sigma = (
        np.asarray(a, dtype=float) for a in (time_intercept, factors, shrinkage, sigma)
    )
    shape = (
        tuple(size)
        if size is not None
        else np.broadcast_shapes(
            time_intercept.shape[:-1],
            factors.shape[:-2],
            shrinkage.shape[:-1],
            sigma.shape,
        )
    )
    n_time, n_factors = factors.shape[-2:]
    intercept = rng.normal(0.0, _INTERCEPT_SD, size=(*shape, 1))
    loadings = rng.normal(size=(*shape, n_factors)) * np.broadcast_to(
        shrinkage, (*shape, n_factors)
    )
    product = np.einsum(
        "...tk,...k->...t",
        np.broadcast_to(factors, (*shape, n_time, n_factors)),
        loadings,
    )
    noise = rng.normal(size=(*shape, n_time)) * np.broadcast_to(sigma, shape)[..., None]
    return time_intercept + intercept + product + noise


def _conjugate_unit_draws(
    centered: np.ndarray,
    factors: np.ndarray,
    shrinkage: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact draws of each unit's intercept and loadings from their Gaussian conditional.

    ``centered`` is ``(sample, time, unit)`` outcomes minus time intercepts, ``factors`` is ``(sample, time, factor)``, ``shrinkage`` is ``(sample, factor)``, and ``scale`` is ``(sample,)``. Returns ``(sample, unit, factor + 1)``; column 0 is the intercept. The conditional is ``N(V W' y / s**2, V)`` with ``V = (D^-1 + W'W / s**2)^-1``.
    """
    n_sample, n_time, n_unit = centered.shape
    design = np.concatenate([np.ones((n_sample, n_time, 1)), factors], axis=-1)
    prior_var = np.concatenate(
        [np.full((n_sample, 1), _INTERCEPT_SD**2), shrinkage**2], axis=-1
    )
    variance = scale**2
    design_t = np.swapaxes(design, -1, -2)
    precision = (
        np.eye(design.shape[-1]) / prior_var[:, None, :]
        + design_t @ design / variance[:, None, None]
    )
    chol = np.linalg.cholesky(precision)
    mean = np.linalg.solve(precision, design_t @ centered) / variance[:, None, None]
    noise = rng.standard_normal(mean.shape)
    draws = mean + np.linalg.solve(np.swapaxes(chol, -1, -2), noise)
    return np.swapaxes(draws, -1, -2)


def _control_model(control_y: xr.DataArray, max_factors: int) -> pm.Model:
    """Gaussian factor graph on control cells only.

    Control unit intercepts and loadings are integrated out of the likelihood (:func:`_marginal_unit_logp`), so NUTS samples only the shared factors, time intercepts, factor scales, and residual scale. :func:`_with_control_draws` draws the integrated terms exactly after sampling.

    The likelihood uses the one shared positive residual scale. That scale is
    bounded below by a fraction of the control-panel outcome spread, so a
    noise-free panel does not collapse the factor posterior onto a ridge of
    exact fits. A unit observation scale is not this model. Treated outcomes
    are not in this graph, so they do not update these factor or time-intercept
    draws.
    """
    factor_names = _factor_names(max_factors)
    alpha = np.array([_MGP_A1 if i == 0 else _MGP_A2 for i in range(max_factors)])
    with pm.Model(
        coords={
            "obs_ind": control_y.coords["obs_ind"].values,
            "control_units": control_y.coords["control_units"].values,
            "factors": factor_names,
        }
    ) as model:
        delta = pm.Gamma("mgp_delta", alpha=alpha, beta=1.0, dims="factors")
        precision = pt.cumprod(delta)
        pm.Deterministic("shrinkage_state", 1.0 / pt.sqrt(precision), dims="factors")
        spread = _control_outcome_spread(control_y)
        sigma_raw = pm.HalfNormal("sigma_raw", sigma=1.0)
        sigma = pm.Deterministic(
            "sigma", spread * (_RESIDUAL_FLOOR_FRACTION + sigma_raw)
        )
        time_intercept = pm.Normal("time_intercept", 0.0, _INTERCEPT_SD, dims="obs_ind")
        shared_factors = pm.Normal(
            "shared_factors", 0.0, 1.0, dims=("obs_ind", "factors")
        )
        observed = control_y.transpose("control_units", "obs_ind")
        pm.CustomDist(
            "control_y",
            time_intercept,
            shared_factors,
            model["shrinkage_state"],
            sigma,
            logp=_marginal_unit_logp,
            random=_marginal_unit_random,
            signature="(t),(t,k),(k),()->(t)",
            observed=observed.values,
            dims=("control_units", "obs_ind"),
        )
    return model


def _control_initvals(
    model: pm.Model, control_y: xr.DataArray
) -> dict[str, np.ndarray]:
    """SVD start for the factor product. Not a rank selector."""
    values = control_y.transpose("obs_ind", "control_units").values
    centered = values - values.mean(axis=0, keepdims=True)
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    n_factors = int(model.dim_lengths["factors"].eval())
    rank = min(n_factors, left.shape[1])
    factors = np.zeros((values.shape[0], n_factors))
    if rank:
        scale = np.sqrt(np.maximum(singular[:rank], 1e-8))
        factors[:, :rank] = left[:, :rank] * scale
    return {
        "shared_factors": factors,
        "mgp_delta": np.array([1.0 if i == 0 else 1.5 for i in range(n_factors)]),
        "sigma_raw": np.array(1.0),
    }


def _scale_log_posterior(
    log_scale: float,
    *,
    gram: np.ndarray,
    centered: np.ndarray,
    control_rss: float,
    n_control: int,
    spread: float,
) -> float:
    """Profile the one scale given factors, integrating intercepts and loadings."""
    scale = np.exp(log_scale)
    raw = scale / spread - _RESIDUAL_FLOOR_FRACTION
    if not np.isfinite(raw) or raw <= 0.0:
        return -1e300
    scale_sq = scale**2
    covariance = scale_sq * np.eye(gram.shape[0]) + gram
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0 or not np.isfinite(logdet):
        return -1e300
    solved = np.linalg.solve(covariance, centered)
    quadratic = float(np.sum(centered * solved))
    n_units = centered.shape[1]
    prior = -0.5 * raw**2
    control = -n_control * log_scale - control_rss / (2.0 * scale_sq)
    return -0.5 * n_units * logdet - 0.5 * quadratic + control + prior


def _updated_residual_scale(
    sigma: np.ndarray,
    control_rss: np.ndarray,
    n_control: int,
    factors_pre: np.ndarray,
    shrinkage: np.ndarray,
    treated_centered: np.ndarray,
    *,
    spread: float,
) -> np.ndarray:
    """Move each control-panel scale draw using treated pre-period outcomes.

    The target conditions on the sampled factors. It uses the control residuals
    at those draws and the treated pre-period after integrating the unit
    intercept and loadings under their priors. Draws below the control-panel
    floor are raised to that floor. Factors are not resampled. This is not a
    second residual-scale random variable.
    """
    updated = np.empty(sigma.shape, dtype=float)
    n_time = treated_centered.shape[1]
    floor = spread * _RESIDUAL_FLOOR_FRACTION
    for index in range(sigma.shape[0]):
        design = np.column_stack((np.ones(n_time), factors_pre[index]))
        prior_sd = np.concatenate(([_INTERCEPT_SD], shrinkage[index]))
        gram = (design * prior_sd**2) @ design.T
        centered = treated_centered[index]
        rss = float(control_rss[index])
        log_s = float(np.log(max(float(sigma[index]), floor)))
        for _ in range(_SCALE_UPDATE_STEPS):
            current = _scale_log_posterior(
                log_s,
                gram=gram,
                centered=centered,
                control_rss=rss,
                n_control=n_control,
                spread=spread,
            )
            above = _scale_log_posterior(
                log_s + _SCALE_FD_STEP,
                gram=gram,
                centered=centered,
                control_rss=rss,
                n_control=n_control,
                spread=spread,
            )
            below = _scale_log_posterior(
                log_s - _SCALE_FD_STEP,
                gram=gram,
                centered=centered,
                control_rss=rss,
                n_control=n_control,
                spread=spread,
            )
            if below <= -1e299:
                gradient = (above - current) / _SCALE_FD_STEP
                curvature = 0.0
            else:
                gradient = (above - below) / (2.0 * _SCALE_FD_STEP)
                curvature = (above - 2.0 * current + below) / _SCALE_FD_STEP**2
            if not np.isfinite(gradient) or not np.isfinite(curvature):
                break
            if curvature < -1e-6:
                step = -gradient / curvature
            else:
                step = 0.5 if gradient > 0.0 else -0.5
            step = float(np.clip(step, -1.5, 1.5))
            accepted = log_s
            for fraction in (1.0, 0.5, 0.25):
                trial = log_s + fraction * step
                if np.exp(trial) < floor:
                    continue
                trial_value = _scale_log_posterior(
                    trial,
                    gram=gram,
                    centered=centered,
                    control_rss=rss,
                    n_control=n_control,
                    spread=spread,
                )
                if trial_value >= current - 1e-6:
                    accepted = trial
                    break
            log_s = accepted
        updated[index] = float(np.exp(log_s))
    return updated


def _with_control_draws(
    posterior: xr.Dataset, control_y: xr.DataArray, rng: np.random.Generator
) -> xr.Dataset:
    """Add exact draws of the integrated control intercepts and loadings.

    The control graph integrates these terms out. One draw per posterior draw comes from their Gaussian conditional given that draw's factors, time intercepts, factor scales, and residual scale. They feed the control residual sum of squares used by the treated-pre scale update.
    """
    stacked = posterior.stack(sample=_DRAW_DIMS)
    factors = stacked["shared_factors"].transpose("sample", "obs_ind", "factors")
    centered = control_y - stacked["time_intercept"]
    draws = _conjugate_unit_draws(
        centered.transpose("sample", "obs_ind", "control_units").values,
        factors.values,
        stacked["shrinkage_state"].transpose("sample", "factors").values,
        np.asarray(stacked["sigma"].transpose("sample").values, dtype=float),
        rng,
    )
    sample = stacked.indexes["sample"]
    units = control_y.coords["control_units"].values
    intercept = xr.DataArray(
        draws[..., 0],
        dims=("sample", "control_units"),
        coords={"control_units": units},
    )
    loadings = xr.DataArray(
        draws[..., 1:],
        dims=("sample", "control_units", "factors"),
        coords={"control_units": units, "factors": factors.coords["factors"].values},
    )
    return posterior.assign(
        control_unit_intercept=_restore_draws(intercept, sample),
        control_loadings=_restore_draws(loadings, sample),
    )


def _sample_treated_conditional(
    control_y: xr.DataArray,
    treated_pre: xr.DataArray,
    control_posterior: xr.Dataset,
    *,
    rng: np.random.Generator,
) -> dict[str, xr.DataArray]:
    """Draw treated intercepts and loadings exactly at the updated residual scale.

    Factors and time intercepts are data from the control-panel posterior. The one shared residual scale is not a random variable here: its draws are updated from treated pre-period outcomes, then held fixed. Given those, the treated intercepts and loadings have a Gaussian conditional, drawn exactly with one draw per control-panel draw (:func:`_conjugate_unit_draws`). Treated pre-period outcomes cannot move the factor or time-intercept draws.
    """
    posterior = _as_dataset(control_posterior)
    stacked = posterior.stack(sample=_DRAW_DIMS)
    draw_index = stacked.indexes["sample"]
    stacked = stacked.drop_vars(["sample", "chain", "draw"]).assign_coords(
        sample=np.arange(len(draw_index))
    )
    factors = stacked["shared_factors"].transpose("sample", "obs_ind", "factors")
    time_intercept = stacked["time_intercept"].transpose("sample", "obs_ind")
    shrinkage = stacked["shrinkage_state"].transpose("sample", "factors")
    control_mu = _control_mean(stacked)
    residuals = control_y - control_mu
    control_rss = (residuals**2).sum(("obs_ind", "control_units")).values
    n_control = int(control_y.sizes["obs_ind"] * control_y.sizes["control_units"])
    pre_times = treated_pre.coords["obs_ind"]
    factor_pre = (
        factors.sel(obs_ind=pre_times).transpose("sample", "obs_ind", "factors").values
    )
    xi_pre = time_intercept.sel(obs_ind=pre_times).transpose("sample", "obs_ind").values
    treated_values = treated_pre.transpose("obs_ind", "treated_units").values
    treated_centered = treated_values[None, :, :] - xi_pre[:, :, None]
    shrinkage_values = np.asarray(shrinkage.values, dtype=float)
    updated_scale = _updated_residual_scale(
        np.asarray(stacked["sigma"].values, dtype=float),
        np.asarray(control_rss, dtype=float),
        n_control,
        factor_pre,
        shrinkage_values,
        treated_centered,
        spread=_control_outcome_spread(control_y),
    )
    draws = _conjugate_unit_draws(
        treated_centered, factor_pre, shrinkage_values, updated_scale, rng
    )
    coords = {
        "sample": stacked.coords["sample"].values,
        "treated_units": treated_pre.coords["treated_units"].values,
    }
    intercept = xr.DataArray(
        draws[..., 0], dims=("sample", "treated_units"), coords=coords
    )
    loadings = xr.DataArray(
        draws[..., 1:],
        dims=("sample", "treated_units", "factors"),
        coords={**coords, "factors": factors.coords["factors"].values},
    )
    scale = xr.DataArray(
        updated_scale, dims=("sample",), coords={"sample": coords["sample"]}
    )
    return {
        "unit_intercept": _restore_draws(intercept, draw_index),
        "loadings": _restore_draws(loadings, draw_index),
        "residual_scale": _restore_draws(scale, draw_index),
    }


def _control_mean(stacked: xr.Dataset) -> xr.DataArray:
    loadings = stacked["control_loadings"].transpose(
        "sample", "control_units", "factors"
    )
    factors = stacked["shared_factors"].transpose("sample", "obs_ind", "factors")
    product = (loadings * factors).sum("factors")
    return stacked["control_unit_intercept"] + stacked["time_intercept"] + product


def _as_dataset(posterior: xr.Dataset | Any) -> xr.Dataset:
    """Return a posterior dataset from a Dataset, DataTree, or InferenceData."""
    if isinstance(posterior, xr.Dataset):
        return posterior
    children = getattr(posterior, "children", None)
    if children and "posterior" in children:
        return _as_dataset(posterior["posterior"])
    dataset = getattr(posterior, "dataset", None)
    if dataset is not None and hasattr(dataset, "data_vars"):
        if isinstance(dataset, xr.Dataset):
            return dataset
        return xr.Dataset(dataset)
    raise TypeError("control posterior must be an xarray Dataset or a PyMC DataTree.")


def _restore_draws(array: xr.DataArray, draw_index: pd.Index) -> xr.DataArray:
    """Restore ``chain`` and ``draw`` from the integer conditional index."""
    coords = xr.Coordinates.from_pandas_multiindex(draw_index, "sample")
    restored = array.assign_coords(coords).unstack("sample")
    return restored.transpose("chain", "draw", ...)
