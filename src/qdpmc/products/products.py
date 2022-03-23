import datetime
import qdpmc.structures as structures
import qdpmc.tools.payoffs as pay
from functools import partial
from qdpmc.tools.helper import arr_scalar_converter
from qdpmc.dateutil import Calendar
from scipy.optimize import fsolve
from numpy import array, any, argmax

__all__ = ['SnowballProd']


def _interval_coupon(
        coupon_rate, principal,
        last_payment_date: datetime.date,
        next_payment_date: datetime.date,
        day_counter=365
):
    """Returns the amount of interval coupon between two payment dates.
    Principle, coupon rate, and day counter are mandatory."""
    td = (next_payment_date - last_payment_date).days
    return coupon_rate * principal * td / day_counter


def _update_day_arr(arr, offset, *more):
    """Here, arr is an ascending array of scalars and offset is a scalar. The function
    returns the positive part of arr - offset. If more is passed in, it also returns
    more[arr > offset]"""

    if not offset:
        return arr, *more
    nn = []
    for v in arr:
        d = v - offset
        if d >= 0:
            nn.append(d)

    if more:
        return nn, *(m[len(arr) - len(nn):] for m in more)
    return nn


def _check_ob_dates(ob_dates, calendar):
    """Check if all dates in ob_dates are trading days. If True, return ob_dates
    as-is. Otherwise raise ValueError."""
    for date in ob_dates:
        date = _check_is_trading(date, calendar)
        if not calendar.is_trading(date):
            raise ValueError("%s does not trade" % str(date))
    return ob_dates


def _check_is_trading(date, calendar):
    """Check if *start* is trading."""
    if not isinstance(date, datetime.date):
        raise TypeError("{} is not a datetime.date object".format(date))
    if not calendar.is_trading(date):
        raise ValueError("given non-trading day: {}".format(date))
    return date


def _check_payoff(payoff):
    """Check if payoff is a *Payoff* object."""
    if not isinstance(payoff, pay.Payoff):
        raise TypeError("payoff must be a Payoff object")
    return payoff


def _check_calendar(calendar):
    """Check if calendar is a Calendar object."""
    if not isinstance(calendar, Calendar):
        raise TypeError("calendar must be Calendar object")
    return calendar


class SnowballProd:
    """A snowball structure is an autocallable structured product with snowballing
    coupon payments.

    Parameters
    ----------
    start_date : datetime.date
        A datetime.date object indicating the starting day of the
        structured product. It must be a trading as determined by *calendar*
    initial_price : scalar
        The price of the underlying asset on *start_date*.
    ko_barriers : scalar or array_like
        The knock-out level of the structure. It can either be
        a scalar or be an array of numbers. If a scalar is passed in, it will be
        treated as the time-invariant barrier level. If an array is passed in,
        it must match the length of *ko_ob_dates*.
    ko_ob_dates : array_like
        The observation dates for knock-out. It must be an array
        of datetime.date objects. All of these dates must be trading dates as
        determined by *calendar*.
    ki_barriers : scalar or array_like
        Similar to *ko_barriers*. It controls the level of knock-in barrier.
    ki_ob_dates : array_like or "daily"
        similar to *ko_ob_dates*. *"daily"* indicates daily observation for the
        knock-in event.
    ki_payoff : Payoff
        Controls payoff which applies when, during the life of the contract,
        a knock-in event occurs while a knock-out does not.
    ko_coupon_rate : scalar
        the coupon rate that applies in the event of a knock-out.
    maturity_coupon_rate : scalar
        this rate applies when there is neither knock-out nor knock-in during
        the entire life of the contract.
    calendar : Calendar
        *Calendar* object. If *None* a default calendar will be used.

    Note
    ----
    The day count convention for coupon payment is *ACT/365*.
    The maturity is ``ko_ob_dates[-1]``.
    Notional principal is equal to ``initial_price``.

    Examples
    --------
    .. ipython:: python

        import datetime
        # instantiate a Calendar object so we can use it to generate periodic
        # datetime.date array
        calendar = qm.Calendar()
        # this should be a trading day
        start = datetime.date(2019, 1, 31)
        assert calendar.is_trading(start)
        # monthly trading dates excluding the start
        monthly_dates = calendar.periodic(start, "1m", 13)[1:]
        # a short put
        short_put = - qm.Payoff(qm.plain_vanilla, option_type="put",
                                strike=100)
        # instantiate the structured product
        option = qm.SnowballProd(
            start_date=start, initial_price=100, ko_barriers=105,
            ko_ob_dates=monthly_dates, ki_barriers=80, ki_ob_dates="daily",
            ki_payoff=short_put, ko_coupon_rate=0.15,
            maturity_coupon_rate=0.15
        )
        mc = qm.MonteCarlo(100, 1000)
        bs = qm.BlackScholes(0.03, 0, 0.25, 252)
        # value the contract given day and spot price
        option.value(datetime.date(2019, 5, 7), 102, False, mc, bs)"""

    def __init__(
            self, start_date, initial_price, ko_barriers,
            ko_ob_dates, ki_barriers, ki_ob_dates, ki_payoff,
            ko_coupon_rate, maturity_coupon_rate,
            calendar: Calendar = None
    ):
        _inputs = locals()
        _inputs.pop("self")
        self._inputs = _inputs

        if calendar is None:
            calendar = Calendar()
        # check values
        self.calendar = _check_calendar(calendar)
        self.start_date = _check_is_trading(start_date, calendar)
        self.ki_payoff = _check_payoff(ki_payoff)
        self.ko_ob_dates = _check_ob_dates(ko_ob_dates, calendar)
        try:
            # check if ki_ob_dates are trading
            self.ki_ob_dates = _check_ob_dates(ki_ob_dates, calendar)
        except TypeError:
            # try "daily" if ki_ob_dates are not datetime.date objects
            if ki_ob_dates != "daily":
                raise ValueError(
                    "ki_ob_dates must either be an array of "
                    "trading days or 'daily, got {}".format(ki_ob_dates)
                )
            else:
                end = ko_ob_dates[-1]
                ki_ob_dates = calendar.trading_days_between(
                    start=start_date, end=end, endpoints=True
                )[1:]
        self.ki_ob_dates = ki_ob_dates
        # those that need not be checked
        self.initial_price = initial_price
        self.ko_barriers = arr_scalar_converter(ko_barriers, ko_ob_dates)
        self.ki_barriers = arr_scalar_converter(ki_barriers, ki_ob_dates)
        self.ko_coupon_rate = ko_coupon_rate
        self.maturity_coupon_rate = maturity_coupon_rate
        # inferred and calculated values
        # these values and arrays can be directly passed to the structure
        # constructor
        self.maturity_date = ko_ob_dates[-1]
        self.ob_days_out = calendar.to_scalar(ko_ob_dates, start_date)
        self.ob_days_in = calendar.to_scalar(ki_ob_dates, start_date)
        _frozen = partial(
            _interval_coupon, principal=initial_price,
            last_payment_date=start_date
        )
        self._maturity_coupon_pmt = _frozen(
            coupon_rate=maturity_coupon_rate,
            next_payment_date=self.maturity_date
        )
        # convert a constant into a Payoff object, so it can be passed to the
        # structure constructor
        self.nk_payoff = pay.Payoff(
            pay.constant_payoff, self._maturity_coupon_pmt
        )
        # infer from given values the rebate array that can be passed to the
        # structure constructor
        self.ko_rebate = [_frozen(
            coupon_rate=ko_coupon_rate, next_payment_date=d
        ) for d in ko_ob_dates]

        self._frozen = _frozen

    def to_structure(self, valuation_date, spot, ki_flag):
        """Return the structure used to value the product."""
        valuation_date = _check_is_trading(valuation_date, self.calendar)
        td = self.calendar.num_trading_days_between(
            start=self.start_date, end=valuation_date, count_end=True
        )
        ob_days_out, rebate_out, barrier_out = _update_day_arr(
            self.ob_days_out, td, self.ko_rebate, self.ko_barriers
        )
        ob_days_in, barrier_in = _update_day_arr(
            self.ob_days_in, td, self.ki_barriers
        )
        if not ki_flag:
            obj = structures.UpOutDownIn(
                spot=spot, ob_days_out=ob_days_out, rebate_out=rebate_out,
                ob_days_in=ob_days_in, payoff_in=self.ki_payoff,
                upper_barrier_out=barrier_out, lower_barrier_in=barrier_in,
                payoff_nk=self.nk_payoff
            )
        else:
            obj = structures.UpOut(
                spot=spot, ob_days=ob_days_out, rebate=rebate_out,
                payoff=self.ki_payoff, barrier=barrier_out
            )
        return obj

    def value(self, valuation_date, spot, ki_flag, *args, **kwargs):
        """Value the product given a date and a spot price. *args* and *kwargs*
        are positional and keyword arguments forwarded to
        :meth:`qdpmc.engine.monte_carlo.MonteCarlo.calc`

        Parameters
        ----------
        valuation_date : datetime.date
            the valuate date. It must be a trading day.
        spot : scalar
            the spot price.
        ki_flag: bool
            whether to mark the product as knock-in. If *True*, the
            structure is an up-and-out option.
        """
        return self.to_structure(valuation_date, spot, ki_flag).calc_value(
            *args, **kwargs)

    def find_coup_rate(self, engine, process, target_pv,
                       entropy=None, caller=None):
        """Give a target PV, find the coupon rate.

        *entropy* and *caller* are forwarded to
        :meth:`qdpmc.engine.monte_carlo.MonteCarlo.calc`
        """
        e = entropy

        def _call(c):
            nonlocal e
            if e is None:
                e = engine.most_recent_entropy
            inputs = self._inputs
            inputs['ko_coupon_rate'] = c[0]
            inputs['maturity_coupon_rate'] = c[0]
            s = self.__class__(**inputs)
            diff = s.value(self.start_date, self.initial_price, False,
                           engine, process, entropy=e,
                           caller=caller) - target_pv
            return diff

        rate = fsolve(_call, array([self.ko_coupon_rate]))[0]
        return dict(result=rate, diff=_call([rate]))

    def backtest(self, daily_underlying_asset_prices):
        """backtest the performance of the product in the option holders' view
        with given underlying asset price series, find out the actual holding period and its final payoff

        Parameters
        ----------
        daily_underlying_asset_prices: array_like
            the simulated or historical daily underlying asset price series for performance backtest, the length of the
            price series must be greater than the option's life

        Returns
        -------
        end_date : datetime.date
            the actual end date of the trade, could possibly be earlier than the maturity date
        payout: double
            the actual payout of the product to the option holder
        """
        strct = self.to_structure(self.start_date, self.initial_price, False)
        ko_ob_days = strct.ob_days_out
        ki_ob_days = strct.ob_days_in
        ko_barriers = strct.upper_barrier_out
        ki_barriers = strct.log_barrier_in
        n_days = ko_ob_days[-1]
        rebates = strct.rebate_out

        prices = daily_underlying_asset_prices / daily_underlying_asset_prices[0] * self.initial_price
        knocked_out = any(prices[ko_ob_days] > ko_barriers)
        if knocked_out:
            ko_index = argmax(prices[ko_ob_days] > ko_barriers)
            return self.ko_ob_dates[ko_index], rebates[ko_index]
        knocked_in = any(prices[ki_ob_days] < ki_barriers)
        if knocked_in:
            return self.maturity_date, self.ki_payoff(prices[n_days])
        else:
            return self.maturity_date, rebates[-1]


class SingleBarrierOption:
    _structure = structures.SingleBarrierOption
    _out = True

    def __init__(self, start, barrier, rebate, ob_dates, payoff, calendar):
        self.calendar = _check_calendar(calendar)
        self.start = _check_is_trading(start, calendar)
        self.barrier = arr_scalar_converter(barrier, ob_dates)

        if self.__class__._out:
            self.rebate = arr_scalar_converter(rebate, ob_dates)
        else:
            if hasattr(rebate, "__iter__"):
                raise TypeError(
                    "rebate of knock-in options should be a scalar"
                )
            self.rebate = rebate

        self.ob_dates = ob_dates
        self.payoff = _check_payoff(payoff)
        # scalars
        self.ob_days = calendar.to_scalar(ob_dates, start)

    def to_structure(self, valuation_date=None, spot=None):
        td = self.calendar.num_trading_days_between(self.start, valuation_date)
        ob_days, rebate, barrier = _update_day_arr(
            valuation_date, td, self.rebate, self.barrier
        )
        return self.__class__._structure(
            spot=spot, barrier=barrier, rebate=rebate,
            ob_days=ob_days, payoff=-self.payoff
        )

    def value(self, valuation_date, spot, *args, **kwargs):
        return self.to_structure(valuation_date, spot
                                 ).calc_value(*args, **kwargs)


class UpOut(SingleBarrierOption):
    _structure = structures.UpOut


class DownOut(SingleBarrierOption):
    _structure = structures.DownOut


class UpIn(SingleBarrierOption):
    _structure = structures.UpIn
    _out = False


class DownIn(SingleBarrierOption):
    _structure = structures.DownIn
    _out = False


if __name__ == "__main__":
    pass
