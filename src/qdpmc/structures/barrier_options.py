# Todo: typing for ob_days is now array_like
"""
This module implements standard Monte Carlo method
for valuation of typical barrier options, which include

* Single-barrier options
    * Up-and-out options
    * Down-and-out options
    * Up-and-in options
    * Down-and-in options

* Double-barrier options
    * Double-out options
    * Double-in options

This module provides a flexible and intuitive API for
Monte Carlo pricing of barrier options because it allows for

* Customized payoff function, and
* Time-varying barrier level and rebate
"""


import numpy as np
from qdpmc.tools.helper import (
    arr_scalar_converter,
    up_ki_paths,
    down_ki_paths,
    double_ko_t_and_surviving_paths,
    double_ki_paths,
    up_ko_t_and_surviving_paths,
    down_ko_t_and_surviving_paths,
    merge_days,
    fill_arr
)
from qdpmc.structures.base import StructureMC
from qdpmc.structures._docs import (
    _single_barrier_out_param_docs,
    _single_barrier_in_param_docs,
    _pv_log_paths_docs,
    _payoff_docs,
)
from qdpmc._decorators import DocstringWriter


__all__ = ['SingleBarrierOption', 'UpOut', 'UpIn', 'DownOut', 'DownIn',
           'DoubleBarrierOption', 'DoubleOut', 'DoubleIn']


class SingleBarrierOption(StructureMC):
    """Single-barrier options. Intended to be subclassed not used."""

    def __init__(self, spot, barrier, rebate, ob_days, payoff):
        _out = True
        # rebates of knock-in contracts should be scalars since they
        # are paid at expiry.
        if isinstance(self, DownIn) or isinstance(self, UpIn):
            if hasattr(rebate, "__iter__"):
                raise ValueError(
                    "Rebates of knock-in options should be scalars"
                )
            _out = False

        self._spot = spot
        self.barrier = arr_scalar_converter(barrier, ob_days)

        if _out:
            self.rebate = arr_scalar_converter(rebate, ob_days)
        else:
            self.rebate = rebate

        self.ob_days = ob_days
        self._sim_t_array = np.append([0], ob_days)
        self.log_barrier = np.log(self.barrier / spot)
        self.payoff = payoff

    def _set_spot(self, val):
        if val <= 0:
            raise ValueError("Spot price should be positive.")
        self._spot = val
        # do not forget to reset log barriers
        self.log_barrier = np.log(self.barrier / val)


class UpOut(SingleBarrierOption):
    __doc__ = """An up-and-out option.
    
    
    An up-and-out option is knocked out if, during its life, the price of the 
    underlying asset exceeds the barrier level on any observation day.
    When the option is knocked out, a rebate is immediately paid to the
    option holder.
    
    %(param_docs)s
    
    Examples
    --------
    
    .. ipython:: python
    
        option = qm.UpOut(
            spot=100,
            barrier=120,
            rebate=0,
            ob_days=np.linspace(1, 252, 252),
            payoff=qm.Payoff(
                qm.plain_vanilla,
                strike=100,
                option_type="call"
            )
        )
        mc = qm.MonteCarlo(125, 800)
        bs = qm.BlackScholes(0.03, 0, 0.25, 252)
        option.calc_value(mc, bs, request_greeks=True)
        
    For time-varying barrier and rebate, pass in an array to *barrier*:
    
    .. ipython:: python
    
        option_tvb = qm.UpOut(
            spot=100,
            barrier=np.linspace(110, 120, 252),
            rebate=np.linspace(0, 3, 252),
            ob_days=np.linspace(1, 252, 252),
            payoff=qm.Payoff(
                qm.plain_vanilla,
                strike=100,
                option_type="call"
            )
        )
        option_tvb.calc_value(mc, bs)
        
    """ % {'param_docs': _single_barrier_out_param_docs}

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):

        ko_t, _, nko_paths = up_ko_t_and_surviving_paths(
            log_paths, self.log_barrier, return_idx=False
        )
        terminal = nko_paths[:, -1]
        surviving = self.payoff(np.exp(terminal)*self.spot) * df[-1]
        knocked_out = self.rebate[ko_t] * df[ko_t]
        pv_terminal = pv_knocked_out = 0

        if surviving.size > 0:
            pv_terminal = np.sum(surviving)
        if knocked_out.size > 0:
            pv_knocked_out = np.sum(knocked_out)

        return (pv_terminal + pv_knocked_out) / len(log_paths)


class DownOut(SingleBarrierOption):
    __doc__ = """ A down-and-out option.


    A down-and-out barrier option is knocked out if, during its life, 
    the price of the underlying asset is below the barrier level on any 
    observation day. When the option is knocked out, a rebate is 
    immediately paid to the option holder.
    
    %(param_docs)s
    
    Examples
    --------
    
    .. ipython:: python
        
        option = qm.DownOut(
            spot=100,
            barrier=80,
            rebate=0,
            ob_days=np.linspace(1, 252, 252),
            payoff=qm.Payoff(
                qm.plain_vanilla,
                strike=100,
                option_type="call"
            )
        )
        mc = qm.MonteCarlo(125, 800)
        bs = qm.BlackScholes(0.03, 0, 0.25, 252)
        option.calc_value(mc, bs, request_greeks=True)
        
    """ % {'param_docs': _single_barrier_out_param_docs}

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        ko_t, _, nko_paths = down_ko_t_and_surviving_paths(
            log_paths, self.log_barrier, return_idx=False
        )
        terminal = nko_paths[:, -1]
        surviving = self.payoff(np.exp(terminal)*self.spot) * df[-1]
        knocked_out = self.rebate[ko_t] * df[ko_t]
        pv_terminal = pv_knocked_out = 0

        if surviving.size > 0:
            pv_terminal = np.sum(surviving)
        if knocked_out.size > 0:
            pv_knocked_out = np.sum(knocked_out)

        return (pv_terminal + pv_knocked_out) / len(log_paths)


class DownIn(SingleBarrierOption):
    __doc__ = """ A down-and-in option.
    
    A down-and-in option begins to function as a normal option once
    the price of the underlying asset is below the barrier level
    on any observation day. If during its life the barrier is not
    hit, a rebate will be paid to the option holder at maturity.
    
    %(param_docs)s
    
    Examples
    --------
    
    .. ipython:: python
    
        option = qm.DownIn(
            spot=100,
            barrier=80,
            rebate=0,
            ob_days=np.linspace(1, 252, 252),
            payoff=qm.Payoff(
                qm.plain_vanilla,
                strike=100,
                option_type="call"
            )
        )
        mc = qm.MonteCarlo(125, 800)
        bs = qm.BlackScholes(0.03, 0, 0.25, 252)
        option.calc_value(mc, bs, request_greeks=True)
        
    """ % {'param_docs': _single_barrier_in_param_docs}

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        ki_paths = down_ki_paths(log_paths, self.log_barrier, False)
        terminal = ki_paths[:, -1]
        num_in = len(terminal)
        num_voided = len(log_paths) - num_in
        in_payoff = self.payoff(np.exp(terminal) * self.spot) * df[-1]
        pv_in = 0

        if in_payoff.size > 0:
            pv_in = np.sum(in_payoff)

        return (pv_in + self.rebate * num_voided * df[-1]) / len(log_paths)


class UpIn(SingleBarrierOption):
    __doc__ = """ An up-and-in option.
    
    An up-and-in option begins to function as a normal option once
    the price of the underlying asset is above the barrier level
    on any observation day. If during its life the barrier is not
    hit, a rebate will be paid to the option holder at maturity.

    %(param_docs)s
    
    Examples
    --------
    
    .. ipython:: python
    
        option = qm.UpIn(
            spot=100,
            barrier=120,
            rebate=0,
            ob_days=np.linspace(1, 252, 252),
            payoff=qm.Payoff(
                qm.plain_vanilla,
                strike=100,
                option_type="call"
            )
        )
        mc = qm.MonteCarlo(125, 800)
        bs = qm.BlackScholes(0.03, 0, 0.25, 252)
        option.calc_value(mc, bs, request_greeks=True)
        
    """ % {'param_docs': _single_barrier_in_param_docs}

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        ki_paths = up_ki_paths(log_paths, self.log_barrier, False)
        terminal = ki_paths[:, -1]
        num_in = len(terminal)
        num_voided = len(log_paths) - num_in
        in_payoff = self.payoff(np.exp(terminal) * self.spot) * df[-1]
        pv_in = 0
        if in_payoff.size > 0:
            pv_in = np.sum(in_payoff)
        return (pv_in + self.rebate * num_voided * df[-1]) / len(log_paths)


class DoubleBarrierOption(StructureMC):
    """Double-barrier options. Intended to be subclassed not used."""

    def __init__(self, spot, barrier_up, barrier_down,
                 ob_days_up, ob_days_down, payoff):
        self._spot = spot
        self.barrier_up = arr_scalar_converter(barrier_up, ob_days_up)
        self.barrier_down = arr_scalar_converter(barrier_down, ob_days_down)
        self.ob_days_up = ob_days_up
        self.ob_days_down = ob_days_down

        self.log_barrier_up = np.log(self.barrier_up / spot)
        self.log_barrier_down = np.log(self.barrier_down / spot)

        self._t, _, _ = merge_days(ob_days_up, ob_days_down)
        self._filled_up = fill_arr(
            self.log_barrier_up, ob_days_up, self._t, np.inf
        )
        # need to fill the array with minus infinity since log returns can go negative
        self._filled_down = fill_arr(
            self.log_barrier_down, ob_days_down, self._t, -np.inf
        )
        self._sim_t_array = np.append([0], self._t)
        self.payoff = payoff

    def _set_spot(self, val):
        if val <= 0:
            raise ValueError('Spot price should be positive.')
        self._spot = val
        # do not forget to reset log barriers
        self.log_barrier_up = np.log(self.barrier_up / val)
        self.log_barrier_down = np.log(self.barrier_down / val)
        self._filled_up = fill_arr(
            self.log_barrier_up, self.ob_days_up, self._t, np.inf
        )
        self._filled_down = fill_arr(
            self.log_barrier_down, self.ob_days_down, self._t, -np.inf
        )


class DoubleOut(DoubleBarrierOption):
    def __init__(
            self, spot, barrier_up, barrier_down,
            ob_days_up, ob_days_down, payoff,
            rebate=None, rebate_up=None, rebate_down=None
    ):
        super(DoubleOut, self).__init__(
            spot, barrier_up, barrier_down, ob_days_up, ob_days_down, payoff
        )

        # If rebate is passed, then up out and down out will not be distinguished.
        # In this case, identical rebate is applied to up out and down out,
        # and rebate_up and rebate_down are ignored
        if rebate is None:
            if (rebate_up is None) or (rebate_down is None):
                raise AttributeError(
                    "Both rebate_up and rebate_down must be specified when rebate is None"
                )
            self.rebate_up = arr_scalar_converter(rebate_up, ob_days_up)
            self.rebate_down = arr_scalar_converter(rebate_down, ob_days_down)
            # fill the array with nan so we will know if something is wrong
            self._filled_rebate_up = fill_arr(self.rebate_up, ob_days_up, self._t, np.nan)
            self._filled_rebate_down = fill_arr(self.rebate_down, ob_days_down, self._t,
                                                np.nan)
            self._identical_rebate = False
        else:
            self.rebate = arr_scalar_converter(rebate, self._t)
            self._identical_rebate = True

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        if self._identical_rebate:
            # KO time and NKO paths
            ko_t, _, nko_paths = double_ko_t_and_surviving_paths(
                log_paths, self._filled_up, self._filled_down, False
            )

            knocked_out = self.rebate[ko_t] * df[ko_t]
            pv_knocked_out = knocked_out.sum()

        else:
            ko_t, ko_paths, nko_paths = double_ko_t_and_surviving_paths(
                log_paths, self._filled_up, self._filled_down, False
            )

            # Price of underlying asset @ ko time
            ko_price = ko_paths[range(len(ko_paths)), ko_t]
            up_barrier_when_ko = self._filled_up[ko_t]
            down_barrier_when_ko = self._filled_down[ko_t]
            up_out_t = ko_t[ko_price >= up_barrier_when_ko]
            down_out_t = ko_t[ko_price <= down_barrier_when_ko]

            pv_knocked_out = (
                (self._filled_rebate_up[up_out_t] * df[up_out_t]).sum() +
                (self._filled_rebate_down[down_out_t] * df[down_out_t]).sum()
            )

        surviving = self.payoff(np.exp(nko_paths[:, -1]) * self.spot) * df[-1]
        pv_terminal = np.sum(surviving)

        return (pv_terminal + pv_knocked_out) / len(log_paths)

    __init__.__doc__ = """ A double-out option.

    A double-out option is knocked out if, during its life, the price of the
    underlying asset is above the upper barrier or below the lower barrier.
    A rebate is paid to the option holder once the option is knocked out.

    Parameters
    ----------
    spot : scalar 
        The spot price of the underlying asset.
    barrier_up : scalar or array_like 
        The upper barrier of the option. This can be either
        a scalar or an array. If a scalar is passed, it will be treated as
        the time-invariant level of barrier. If an array is passed, it must
        match the length of *ob_days_up*.
    barrier_down : scalar or array_like
        The lower barrier of the option. This can be either
        a scalar or an array. If a scalar is passed, it will be treated as
        the time-invariant level of barrier. If an array is passed, it must
        match the length of *ob_days_down*".
    ob_days_up : array_like
        The array of observation days for the upper barrier.
        This must be an array of integers with each element representing
        the number of days that an observation day is from the valuation day.
        The last element of the union of *ob_days_up* and *ob_days_down* is
        assumed to be the maturity of the double-barrier option.
    ob_days_down:
        Similar to *ob_days_up*.
    payoff : Payoff
        %(payoff_docs)s
    rebate : scalar or array_like
        If a scalar or an array is passed, then identical rebates will
        apply to knock-outs from above and below. If *rebate* is specified, both
        *rebate_up* and *rebate_down* will be ignored. If an array is passed,
        it must match the length of the union of *ob_days_up* and *ob_days_down*
    rebate_up : scalar or array_like
        The rebate paid to the holder if the option is knocked
        out from above. Can be either a scalar or an array.
    rebate_down : scalar or array_like
        The rebate paid to the holder if the option is knocked
        out from below. Can be either a scalar or an array.


    Examples
    --------

    .. ipython:: python

        option = qm.DoubleOut(
            spot=100,
            barrier_up=120,
            barrier_down=80,
            ob_days_up=np.linspace(1, 252, 252),
            ob_days_down=np.linspace(1, 252, 252),
            payoff=qm.Payoff(
                qm.plain_vanilla,
                strike=100,
                option_type="call"
            ),
            rebate_up=1,
            rebate_down=2
        )
        mc = qm.MonteCarlo(125, 800)
        bs = qm.BlackScholes(0.03, 0, 0.25, 252)
        option.calc_value(mc, bs, request_greeks=True)""" % {
        'payoff_docs': _payoff_docs,
    }


class DoubleIn(DoubleBarrierOption):
    def __init__(
            self, spot, barrier_up, barrier_down,
            ob_days_up, ob_days_down, rebate, payoff
    ):
        if hasattr(rebate, "__iter__"):
            raise ValueError("Rebates of knock-in options should be a scalar")
        super(DoubleIn, self).__init__(
            spot, barrier_up, barrier_down, ob_days_up, ob_days_down, payoff
        )
        self.rebate = rebate

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        ki_paths = double_ki_paths(log_paths, self._filled_up, self._filled_down, False)
        terminal = ki_paths[:, -1]
        num_in = len(terminal)
        num_voided = len(log_paths) - num_in
        in_payoff = self.payoff(np.exp(terminal) * self.spot) * df[-1]
        pv_in = 0

        if in_payoff.size > 0:
            pv_in = np.sum(in_payoff)

        return (pv_in + self.rebate * num_voided * df[-1]) / len(log_paths)

    __init__.__doc__ = """ A double-in option.

        A double-in option begins to function as a normal function (i.e., knocks
        in) if on any observation day the price of the underlying asset is above
        the upper barrier or the lower barrier. A rebate is paid at the maturity
        of the option if the option does not knock in during its life.

        Parameters
        ----------
        spot : scalar
            The spot price of the underlying asset.
        barrier_up : scalar or array_like
            The upper barrier of the option. This can be either
            a scalar or an array. If a scalar is passed, it will be treated as
            the time-invariant level of barrier. If an array is passed, it must
            match the length of *ob_days_up*.
        barrier_down : scalar or array_like
            The lower barrier of the option. This can be either
            a scalar or an array. If a scalar is passed, it will be treated as
            the time-invariant level of barrier. If an array is passed, it must
            match the length of *ob_days_down*".
        ob_days_up : array_like
            The array of observation days for the upper barrier.
            This must be an array of integers with each element representing
            the number of days that an observation day is from the valuation day.
            The last element of the union of *ob_days_up* and *ob_days_down* is
            assumed to be the maturity of the double-barrier option.
        ob_days_down : array_like 
            Similar to *ob_days_up*.
        rebate : scalar
            The rebate of the option. Must be a constant for knock-in options
        payoff : Payoff
            %(payoff_docs)s

        Examples
        --------

        .. ipython:: python

            option = qm.DoubleIn(
                spot=100,
                barrier_up=120,
                barrier_down=80,
                ob_days_up=np.linspace(1, 252, 21),
                ob_days_down=np.linspace(1, 252, 252),
                rebate=2,
                payoff=qm.Payoff(
                    qm.plain_vanilla,
                    strike=100,
                    option_type="call"
                )
            )
            mc = qm.MonteCarlo(125, 800)
            bs = qm.BlackScholes(0.03, 0, 0.25, 252)
            option.calc_value(mc, bs, request_greeks=True)""" % {
        'payoff_docs': _payoff_docs,
    }
