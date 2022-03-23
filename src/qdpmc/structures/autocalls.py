# Todo: tying for ob_days
# Todo: test UpOutDownIn
# Todo: add documentation to UpOutDownIn
# Todo: Update documentation for Payoff

"""
This module implements standard Monte Carlo method
for valuation of typical autocall structures:

* Standard snowball structure

This module provides a flexible and intuitive API for
Monte Carlo pricing of barrier options. It allows for

* Time-varying level for both knock-out barrier and knock-in barrier.
"""


import numpy as np
from qdpmc.tools.helper import (
    up_ko_t_and_surviving_paths,
    down_ki_paths,
    arr_scalar_converter,
    merge_days,
)
from qdpmc.tools.payoffs import plain_vanilla
from qdpmc.structures.base import StructureMC
from qdpmc.structures._docs import _pv_log_paths_docs
from qdpmc._decorators import DocstringWriter


__all__ = ['StandardSnowball', 'UpOutDownIn']


class StandardSnowball(StructureMC):
    def __init__(
            self, spot, barrier_out, barrier_in, ob_days_in,
            ob_days_out, ko_coupon, full_coupon
    ):
        """A standard snowball structure.


        A standard snowball structure gives its holder a large payoff if
        the price of the underlying asset stays between a certain range.
        If the price of the underlying asset exceeds the knock-out barrier
        on any observation day, the contract ends immediately and coupon
        is paid to the holder with the amount depending on the day of knock-out.
        Major reason for large loss to the holder could be a plummet in the
        price of the underlying asset.

        Parameters
        ----------
        spot : scalar
            The spot (i.e. on the valuation day) of the price of the underlying asset.
        barrier_out : scalar or array_like
            The knock-out barrier level. Can be either a scalar
            or an array. If a scalar is passed, it will be treated as the time-
            invariant level of barrier. If an array is passed, it must match
            the length of *ob_days_out*.
        barrier_in : scalar or array_like
            The knock-in barrier level. Similar to *barrier-out*.
        ob_days_in : array_like
            The observation day for knock-in. Must be an array of
            integers with each of its elements indicating the number of days that
            an observation day is away from the valuation day.
        ob_days_out : array_like
            The observation day for knock-out. Similar to *ob_days_in*.
        ko_coupon : array_like
            Coupon paid to the holder in a knock-out event.
            Must match the length of *ob_days_out*. Note that this should be specified
            in absolute amounts, not in percentages.
        full_coupon : scalar
            Coupon paid to the holder if the contract survives to maturity day without
            knock-out or knock-in. Note that this should be specified in absolute
            amounts, not in percentages.

        Examples
        --------

        .. ipython:: python

            option = qm.StandardSnowball(
                spot=100,
                barrier_out=105,
                barrier_in=80,
                ob_days_in=np.linspace(1, 252, 252),
                ob_days_out=np.linspace(1, 252, 12),
                ko_coupon=np.linspace(1, 15, 12),
                full_coupon=15
            )
            mc = qm.MonteCarlo(125, 800)
            bs = qm.BlackScholes(0.03, 0, 0.25, 252)
            option.calc_value(mc, bs, request_greeks=True)"""

        self._spot = spot
        self._strike = spot

        self.barrier_out = arr_scalar_converter(barrier_out, ob_days_out)
        self.barrier_in = arr_scalar_converter(barrier_in, ob_days_in)
        self.ob_days_in = ob_days_in
        self.ob_days_out = ob_days_out
        self.ko_coupon = arr_scalar_converter(ko_coupon, ob_days_out)
        self.full_coupon = full_coupon

        self.log_barrier_in = np.log(self.barrier_in / spot)
        self.log_barrier_out = np.log(self.barrier_out / spot)
        _t, self._idx_in, self._idx_out = merge_days(ob_days_in, ob_days_out)
        self._sim_t_array = np.append([0], _t)

    def _set_spot(self, val):
        if val <= 0:
            raise ValueError("Spot price should be positive.")
        self._spot = val
        # do not forget to reset log barriers
        self.log_barrier_in = np.log(self.barrier_in / val)
        self.log_barrier_out = np.log(self.barrier_out / val)

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        df_ko_obs = df[self._idx_out]
        _df = df[-1]
        # find out KO time and indices of NKO paths
        ko_t_idx, _, nko_idx = up_ko_t_and_surviving_paths(log_paths[:, self._idx_out],
                                                           self.log_barrier_out,
                                                           return_idx=True)
        # vector of present value of KO paths
        pv_out = self.ko_coupon[ko_t_idx] * df_ko_obs[ko_t_idx]
        # NKO paths
        paths_nko = log_paths[nko_idx]
        # KI paths
        ki_paths = down_ki_paths(paths_nko[:, self._idx_in], self.log_barrier_in,
                                 return_idx=False)
        # vector of present value of KO paths
        pv_in = -plain_vanilla(
            np.exp(ki_paths[:, -1])*self.spot, self._strike, option_type='put'
        ) * _df
        # present value of paths NKI and NKO
        # this is a scalar
        pv_full_c = (len(log_paths) - len(pv_out) - len(pv_in)) * self.full_coupon * _df
        return (pv_out.sum() + pv_in.sum() + pv_full_c) / len(log_paths)


class UpOutDownIn(StructureMC):
    def __init__(
            self, spot, upper_barrier_out, ob_days_out,
            rebate_out, lower_barrier_in, ob_days_in,
            payoff_in, payoff_nk
    ):
        """A structured products with a high barrier and a low barrier. The high barrier
        dominates the lower one in the sense that, when both a "knock-out" and a
        "knock-in" occur during the life of the product, the status is determined as
        "knock-out".

        Parameters
        ----------
        spot : scalar
            The spot (i.e. on the valuation day) of the price of the underlying asset.
        upper_barrier_out : scalar or array_like
            The knock-out barrier level. Can be either a scalar
            or an array. If a scalar is passed, it will be treated as the time-
            invariant level of barrier. If an array is passed, it must match
            the length of *ob_days_out*.
        ob_days_out : array_like
            A 1-D array of integers specifying observation days. Each of its elements
            represents the number of days that an observation day is from the valuation
            day.
        rebate_out : scalar or array_like
            The rebate of the option. If a constant is passed, then it will be
            treated as the *time-invariant* rebate paid to the option holder. If an array
            is passed, then it must match the length of *ob_days*.
        lower_barrier_in : scalar or array_like
            Similar to *upper_barrier_out*.
        ob_days_in : array_like
            Similar to *ob_days_out*.
        payoff_in : Payoff
            Applies when there is a "knock-in" but no "knock-out".
        payoff_nk : Payoff
            Applies when there is neither "knock-in" nor "knock-out".
        """
        # Taken as is
        self._spot = spot
        self.ob_days_out = ob_days_out
        self.ob_days_in = ob_days_in
        # Convert to NumPy arrays
        self.upper_barrier_out = arr_scalar_converter(upper_barrier_out, ob_days_out)
        self.lower_barrier_in = arr_scalar_converter(lower_barrier_in, ob_days_in)
        self.rebate_out = arr_scalar_converter(rebate_out, ob_days_out)
        # Wrapped payoff function
        self.payoff_in = payoff_in
        self.payoff_nk = payoff_nk
        # Union of the observation day arrays
        # Together with the spot day it consists the "simulation day" array
        _t, self._idx_in, self._idx_out = merge_days(ob_days_in, ob_days_out)
        # Simulation day array
        self._sim_t_array = np.append([0], _t)
        # Log barriers relative to the spot price
        # Note that they change every time self.spot is reset
        # and thus should be appended to self._set_spot
        self.log_barrier_out = np.log(self.upper_barrier_out / self.spot)
        self.log_barrier_in = np.log(self.lower_barrier_in / self.spot)

    def _set_spot(self, val):
        if val <= 0:
            raise ValueError("Spot price should be positive.")
        self._spot = val
        # Do not forget to reset log barriers
        self.log_barrier_out = np.log(self.upper_barrier_out / self.spot)
        self.log_barrier_in = np.log(self.lower_barrier_in / self.spot)

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        df_ko_ob = df[self._idx_out]
        df_terminal = df[-1]
        # Identify ko paths: ko time idx and nko path idx
        ko_t_idx, _, nko_paths_idx = up_ko_t_and_surviving_paths(
            paths=log_paths[:, self._idx_out], barrier=self.log_barrier_out,
            return_idx=True)
        # nko paths
        nko_paths = log_paths[nko_paths_idx]
        # Identify ki paths from nko paths
        ki_paths_idx = down_ki_paths(paths=nko_paths[:, self._idx_in],
                                     barrier=self.log_barrier_in, return_idx=True)
        # ki paths and nk paths
        ki_paths = nko_paths[ki_paths_idx]
        nk_paths = nko_paths[np.logical_not(ki_paths_idx)]
        # PV of payoff from three sets of paths
        pv_out = self.rebate_out[ko_t_idx] * df_ko_ob[ko_t_idx]
        pv_in = self.payoff_in(np.exp(ki_paths[:, -1]) * self.spot) * df_terminal
        pv_nk = self.payoff_nk(np.exp(nk_paths[:, -1]) * self.spot) * df_terminal
        # Average three PVs
        return (pv_out.sum() + pv_in.sum() + pv_nk.sum()) / len(log_paths)
