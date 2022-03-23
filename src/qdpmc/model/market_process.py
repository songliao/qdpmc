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


# ===============================Heston Model============================
@nb.vectorize(
    [float64(float64, float64,
             float64, float64)]
)
def _qe_one_path_one_step(psi, u1, m, z1):
    if psi <= 2:
        psiinv = 1 / psi
        b2 = 2*psiinv - 1 + math.sqrt(2*psiinv)*math.sqrt(2*psiinv-1)
        a = m / (1 + b2)
        return a * (math.sqrt(b2) + z1) ** 2
    else:
        p = (psi - 1) / (psi + 1)
        if u1 <= p:
            return 0
        beta = (1 - p) / m
        return math.log((1 - p) / (1 - u1)) / beta


@functools.lru_cache(maxsize=128)
def _get_kvkr(kappa, theta, volvol, dt, rho, gamma1, gamma2):
    k1 = np.e ** (-kappa * dt)
    k0 = -theta * k1
    k2 = (1 - k1) / kappa * volvol ** 2 * k1
    k3 = theta * volvol ** 2 / (2 * kappa) * (1 - k1) ** 2
    kv = [k0, k1, k2, k3]

    del k0, k1, k2, k3
    k0 = -rho * kappa * theta / volvol * dt
    k1 = gamma1 * dt * (kappa * rho / volvol - 0.5) - \
        rho / volvol
    k2 = gamma2 * dt * (kappa * rho / volvol - 0.5) + \
        rho / volvol
    k3 = gamma1 * dt * (1 - rho ** 2)
    k4 = gamma2 / gamma1 * k3
    kr = [k0, k1, k2, k3, k4]
    return kv, kr


_spec = ['''
UniTuple(float64[:, :], 2)(
    float64[:], float64[:], float64, float64,
    float64, float64, float64[:, :],
    float64[:, :], float64[:, :], int32, int32
)
''']


@nb.jit(_spec, nopython=True, cache=True, error_model='numpy')
def _jitable_heston(
        kv, kr, mu, theta, v0, dt, u, z_v, z,
        batch_size=100, grid_points_in_time=100
):
    v0 = np.full(batch_size, v0)
    V = np.zeros((batch_size, grid_points_in_time))
    X = np.zeros(V.shape)

    last_v = v0
    last_x = np.zeros(v0.shape)
    for i in range(grid_points_in_time):
        k0, k1, k2, k3 = kv
        m = theta + last_v * k1 + k0
        s2 = last_v * k2 + k3
        p = s2 / (m ** 2)
        v = _qe_one_path_one_step(p, u[:, i], m, z_v[:, i])
        k0r, k1r, k2r, k3r, k4r = kr
        rt = np.sqrt(k3r * last_v + k4r * v)
        x = last_x + mu * dt + k0r + k1r * last_v + k2r * v + \
            rt * z[:, i]
        V[:, i] = v
        X[:, i] = x
        last_v, last_x = v, x

    assert V.shape == (batch_size, grid_points_in_time)
    assert V.shape == X.shape
    return V, X


class Heston:

    def __init__(
            self, r, q, rho, theta, kappa, xi, default_v0, day_counter=252
    ):
        """A stochastic-volatility model due to Heston (1993).

        .. math::
            \\begin{align*}
            \\mathrm{d}S_t&=(r - q)S_t\\mathrm{d}t + \\sqrt{v_t} S_t \\mathrm{d}W_t \\\\
            \\mathrm{d}v_t&=\\kappa(\\theta-v_t)\\mathrm{d}t +
                \\xi \\sqrt{v_t} \\mathrm{d}Z_t
            \\end{align*}

        When passing a Heston process into the Monte Carlo engine, products will be valued
        by discounting the payoff at the risk-free (parameter *r*) rate.

        Parameters
        ----------
        r : scalar
            The risk-free rate. It is used as the continuous discount rate.
        q : scalar
            The continuous yield of the underlying asset. Notice that *(r-q)* is the
            drift of the price of the underlying asset under a risk-neutral measure.
        rho : scalar
            The correlation between the two standard Brownian motions. This must be
            greater than -1 and less than 1.
        theta : scalar
            The long-term mean of *v*.
        kappa : scalar
            The mean-reverting intensity. The larger it is, the quicker *v* reverts to
            *theta*.
        xi : scalar
            The volatility of *v*.
        default_v0 : scalar
            The default starting value of the variance.
        day_counter : int
            Number of days per year. This affects the discount factor."""
        self.r = r
        self.q = q
        self.mu = r - q
        self.rho = rho
        self.theta = theta
        self.kappa = kappa
        self.xi = xi
        self.volvol = xi
        self.default_v0 = default_v0
        self.day_counter = day_counter

    def generate_path(
            self, t, v0=None, gamma1=0.5, gamma2=0.5,
            batch_size=100, seed=None, grid_points_in_time=None
    ):
        """Generate a set of projections.


        Parameters
        ----------
        t : scalar
            The time index through which the path are generated. Note that *t=1*
            represents a 1-year horizon. Notice that *t * self.day_counter* should be an
            integer.
        v0 : scalar
            The initial value of *v*. If is None, it is set to *default_v0*. Default is
            None.
        gamma1, gamma2 : scalar
            Controls the finite-difference scheme when simulating the log return. Central
            scheme corresponds to *gamma1=gamma2=0.5*.
        batch_size : int
            How many paths to generate at a time.
        seed : int
            The random seed. If None, it will be chosen randomly. Default is None.
        grid_points_in_time : int
            Number of grid points in time. If None, daily simulation is assumed. Default
            is None

        Returns
        -------
        v : ndarray
            Projections of variance.
        x : ndarray
            Projections of log return."""
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, size=(batch_size, grid_points_in_time))
        z = rng.normal(0, 1, size=(batch_size, grid_points_in_time))
        v, x = self.generate_path_given_uz(
            v0, t, u, z, gamma1, gamma2,
            batch_size, grid_points_in_time
        )
        return v, x

    def generate_path_given_uz(
            self, t, u, z, v0=None, gamma1=0.5, gamma2=0.5,
            batch_size=100, grid_points_in_time=None
    ):
        """Project variance and log return given random numbers."""
        if v0 is None:
            v0 = self.default_v0
        if grid_points_in_time is None:
            grid_points_in_time = int(self.day_counter * t)
        try:
            assert self.day_counter * t == grid_points_in_time
        except AssertionError:
            raise ValueError(
                "generated time grid isn't made of integers. "
                "make sure that (t*day_counter) is an integer"
            )

        z_v = norm.ppf(u)
        dt = t / grid_points_in_time

        kv, kr = _get_kvkr(
            self.kappa, self.theta, self.volvol, dt,
            self.rho, gamma1, gamma2
        )

        kv = np.array(kv)
        kr = np.array(kr)
        return _jitable_heston(
            kv, kr, self.mu, self.theta, v0, dt, u, z_v, z,
            batch_size, grid_points_in_time
        )

    @property
    def coordinator(self):
        return _HestonCoordinator


# ====================Purely for compatiblity with MC engine=======================
class _HestonCoordinator(ProcessCoordinator):
    def __init__(self, option: OptionABC, hst: Heston):
        self.option = option
        self.hst = hst
        self.t = option.sim_t_array[-1] / hst.day_counter

        self.df = np.exp(-hst.r * option.sim_t_array[1:] / hst.day_counter)

        self._batch_size = None
        self._points_per_path = int(option.sim_t_array[-1])
        self._points_for_valuation = option.sim_t_array[1:] - 1

    def generate_eps(self, seed, batch_size):
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, (batch_size, self._points_per_path))
        z = rng.normal(0, 1, (batch_size, self._points_per_path))
        self._batch_size = batch_size
        return u, z

    def paths_given_eps(self, eps):
        u, z = eps
        paths = self.hst.generate_path_given_uz(
            t=self.t, u=u, z=z, v0=None,
            batch_size=self._batch_size)[1]
        return paths[:, self._points_for_valuation]

    def shift(self, paths, ds, dr, dv, eps):
        raise NotImplementedError(
            "does not support calculating greeks or shifting paths"
            "under Heston model at this point"
        )
