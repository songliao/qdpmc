# Todo: make it convenient to add, reverse, etc., payoff functions
# Todo: add documentation to Payoff

import numpy as np


def plain_vanilla(underlying_price, strike, option_type='call'):
    """A plain-vanilla payoff function.

    Parameters
    ----------
    underlying_price : ndarray
        An array of price of the underlying asset.
    strike : scalar
        The strike of the option.
    option_type : str
        "put" or "call".

    Returns
    -------
    ndarray
        The payoff array."""
    underlying_price = np.array(underlying_price)
    if option_type == 'put':
        return -np.minimum(underlying_price - strike, 0)
    elif option_type == 'call':
        return np.maximum(underlying_price - strike, 0)
    else:
        raise ValueError(
            "Option type should be 'call' or 'put', got %s" % option_type
        )


def cash_or_nothing(underlying_price, strike, cash_amount):
    """A cash-or-nothing payoff function.

    Parameters
    ----------
    underlying_price : ndarray
        An array of price of the underlying asset.
    strike : scalar
        The strike of the option.
    cash_amount : scalar
        Amount of cash paid.

    Returns
    -------
    ndarray
        The payoff array."""
    payoff = np.full(len(underlying_price), float(cash_amount))
    payoff[underlying_price <= strike] = 0
    return payoff


def asset_or_nothing(underlying_price, strike):
    """An asset-or-nothing payoff function

    Parameters
    ----------
    underlying_price : ndarray
        An array of price of the underlying asset.
    strike : scalar
        The strike of the option.

    Returns
    -------
    ndarray
        The payoff array.
    """
    payoff = np.full(len(underlying_price), underlying_price)
    payoff[underlying_price <= strike] = 0
    return payoff


def constant_payoff(underlying_price, amount):
    return np.full(len(underlying_price), float(amount))


class Payoff:
    """A class that allows payoff functions to be added, negated, and scalar-multiplied.


    The first argument *func* must be callable, and the first argument of it must be
    the array of underlying asset prices.
    *args* and *keywords* contain arguments other than the price array of *func*.
    Calls to a *Payoff* object will be forwarded to *func* with the price array. Make
    sure that all other necessary arguments are specified, or an error will be raised.

    Examples
    --------

    Here are some examples.

    .. ipython:: python

        def my_payoff_func(asset_price, param1, param2):
            return param1 * np.array(asset_price) + param2

        my_payoff = qm.Payoff(
            func=my_payoff_func,
            param1 = 3.0,
            param2 = 2.0
        )

        my_payoff([1, 2, 3, 4, 5])
        -my_payoff([1, 2, 3, 4, 5])
        payoff25 = 2.5 * my_payoff
        payoff25([1, 2, 3, 4, 5])

    The user function should implement vectorization whenever possible to
    achieve maximum speed of computation."""

    def __init__(self, func, *args, **keywords):

        if not callable(func):
            raise TypeError("The first argument must be callable")

        # Raise an error if args and keywords are already passed and func is a
        # Payoff instance since arguments of func.func get multiple values.
        if isinstance(func, Payoff):
            if args or keywords:
                raise TypeError(
                    "No arguments should be passed when func "
                    "is a Payoff instance"
                )
            func = func.func
            args = func.args
            keywords = func.keywords

        self.func = func
        self.args = args
        self.keywords = keywords

    def to_log(self, spot):
        """Convert the payoff function to log.
        """

        def _func(log_return, *args, **kwargs):
            return self.func(np.exp(log_return) * spot, *args, **kwargs)

        return Payoff(_func, *self.args, **self.keywords)

    def _partial(self, asset_price):
        return self.func(asset_price, *self.args, **self.keywords)

    def __call__(self, asset_price):
        return self._partial(asset_price)

    def __add__(self, other):

        if not isinstance(other, Payoff):
            raise TypeError("Payoff can only be added to another Payoff")

        def f(asset_price):
            return self._partial(asset_price) + other(asset_price)
        return Payoff(f)

    def __sub__(self, other):

        if not isinstance(other, Payoff):
            raise TypeError("Payoff can only be subtracted from another Payoff")

        def f(asset_price):
            return self._partial(asset_price) - other(asset_price)
        return Payoff(f)

    def __neg__(self):
        def f(asset_price):
            return -self._partial(asset_price)
        return Payoff(f)

    def __rmul__(self, scalar):
        def f(asset_price):
            return scalar * (self._partial(asset_price))
        return Payoff(f)

    __mul__ = __rmul__
