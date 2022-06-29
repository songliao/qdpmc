"""
This modules provides an API for specification of market dynamics.
Currently, supported market models are

* Black-Scholes market model
"""
import numpy as np
from qdpmc.structures.base import ProcessCoordinator, OptionABC
import math
from scipy.stats import norm
from numba import float64
import numba as nb
import functools


__all__ = ['BlackScholes', 'Heston']

_cache_keys = ['S plus', 'S minus', 'V plus', 'V minus',
               'R plus', 'R minus', 'DF plus', 'DF minus',
               'DF next day', 'Paths next day']


def _mk_dict(keys, values):
    return dict(zip(keys, values))


class BlackScholes:
    """A Black-Scholes process. A Black-Scholes market has two securities: a
    risky asset and a risk-free bond.

    Dynamics of the asset price is driven by a geometric Brownian motion:

    .. math::

        \\mathrm{d}S_t=r S_t\\mathrm{d}t + \\sigma S_t \\mathrm{d}W_t

    and the log-return follows

    .. math::

        \\mathrm{d}\\left(\\mathrm{log}{S_t}\\right)=
        (r-q-\\frac{\\sigma^2}{2})\\mathrm{d}t+\\sigma\\mathrm{d}W_t

    where the drift (under the risk-neutral measure) is the risk-free rate.

    Parameters regarding market dynamics are set here before implementing
    Monte Carlo simulation.

    Parameters
    ----------
    r : scalar
        The instantaenous risk-free rate.
    q : scalar
        The continuous yield.
    v : scalar
        The diffusion parameter.
    day_counter : int
        An integer that controls the numder of trading days in a year. Default is 252."""

    def __init__(self, r, q, v, day_counter=252):
        self.r = r / day_counter
        self.q = q / day_counter
        self.v = v / (day_counter ** 0.5)
        self.day_counter = day_counter

        self._cache = {}
        self._cached_drift = {}
        self._cached_diffusion = {}

    @staticmethod
    def _project_dd(drift, diffusion, eps):

        exp_ds = drift + np.multiply(eps, diffusion)
        log_paths = exp_ds.cumsum(axis=1)
        return log_paths

    def _logs_drift_diffusion(self, dt):
        """Return the drift and diffusion of the logarithm of stock price.

        This function only returns array of drifts and diffusions. Along with
        which a random number generator these can generate simulated realizations
        of paths of the asset price."""
        dt = np.array(dt)
        dt[dt < 0] = 0
        drift = (self.r - self.q - 0.5 * self.v * self.v) * dt
        diffusion = self.v * np.sqrt(dt)
        return drift, diffusion

    @property
    def coordinator(self):
        return _BSCoordinator


class _BSCoordinator(ProcessCoordinator):
    def __init__(self, option: OptionABC, bs: BlackScholes):
        self.option = option
        self.bs = bs

        self.t = option.sim_t_array
        self.dt = np.diff(self.t)
        self.df = np.exp(-bs.r * self.t[1:])
        self.drift, self.diffusion = bs._logs_drift_diffusion(self.dt)

        self._points_per_path = len(self.dt)
        self._CACHE = {}

    def generate_eps(self, seed, batch_size):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, (batch_size, self._points_per_path))

    def paths_given_eps(self, eps):
        return self.bs._project_dd(drift=self.drift, diffusion=self.diffusion,
                                   eps=eps)

    def shift(self, paths, ds, dr, dv, eps):

        _key_inputs = str(ds) + str(dr) + str(dv)

        try:
            val = self._CACHE[_key_inputs]
        except KeyError:
            drift = self.drift
            diffusion = self.diffusion
            day_counter = self.bs.day_counter
            t = self.t
            dt = self.dt
            r, v = self.bs.r, self.bs.v

            # the following needs to be cached

            s_shift_plus = np.log(1.0 + ds)
            s_shift_minus = np.log(1.0 - ds)

            r_shift = dr / self.bs.day_counter * t[1:]
            df_plus = np.exp(-(r + dr / day_counter) * t[1:])
            df_minus = np.exp(-(r - dr / day_counter) * t[1:])

            # This is to calculate new paths w/r/t shift in volatility
            _sq = (dv * dv / 2) / day_counter
            _inter = v * dv / (day_counter ** 0.5)
            _v_shift_diffusion = dv / (day_counter ** 0.5) * np.sqrt(dt)
            v_drift_plus = -(_sq + _inter) * dt + drift
            v_drift_minus = -(_sq - _inter) * dt + drift
            v_diffusion_plus = diffusion + _v_shift_diffusion
            v_diffusion_minus = diffusion - _v_shift_diffusion

            dt_next_day = dt.copy()
            dt_next_day[0] -= 1
            dt_next_day[dt_next_day < 0] = 0
            df_next_day = np.exp(-r * (t[1:] - 1))
            drift_next_day, diffusion_next_day = \
                self.bs._logs_drift_diffusion(dt_next_day)

            val = _mk_dict(
                keys=_cache_keys,
                values=[
                    s_shift_plus, s_shift_minus,
                    (v_drift_plus, v_diffusion_plus),
                    (v_drift_minus, v_diffusion_minus),
                    r_shift, -r_shift, df_plus, df_minus,
                    df_next_day,
                    (drift_next_day, diffusion_next_day)
                ]
            )

            self._CACHE.update({_key_inputs: val})

        shifted_paths = {
            'S plus': paths + val['S plus'],
            'S minus': paths + val['S minus'],
            'R plus': paths + val['R plus'],
            'R minus': paths + val['R minus'],
            'V plus': self.bs._project_dd(drift=val['V plus'][0],
                                          diffusion=val['V plus'][1], eps=eps),
            'V minus': self.bs._project_dd(drift=val['V minus'][0],
                                           diffusion=val['V minus'][1],
                                           eps=eps),
            'DF plus': val['DF plus'],
            'DF minus': val['DF minus'],
            'DF next day': val['DF next day'],
            'Paths next day': self.bs._project_dd(
                drift=val['Paths next day'][0],
                diffusion=val['Paths next day'][1], eps=eps)
        }

        return shifted_paths


