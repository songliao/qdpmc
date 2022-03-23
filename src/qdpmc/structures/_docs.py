_single_barrier_in_rebate_docs = """
    rebate : scalar
        The rebate of the option. Must be a constant for knock-in options
"""

_single_barrier_out_rebate_docs = """
    rebate : scalar or array_like
        The rebate of the option. If a constant is passed, then it will be
        treated as the *time-invariant* rebate paid to the option holder. If an array
        is passed, then it must match the length of *ob_days*.
"""

_payoff_docs = """A *Payoff* instance that controls the payoff
"""

_single_barrier_param_docs = """
    Parameters
    ----------
    spot : scalar
        Spot (ie.e, on the valuation day) price of the underlying asset.
    barrier : scalar or array_like
        The barrier of the option. If a constant is passed, then it will be treated as the
        *time-invariant* barrier level of the option. If an array is passed, then it must
        match the length of *ob_days*.
    %(rebate_docs)s
    ob_days : array_like
        A 1-D array of integers specifying observation days. Each of its elements
        represents the number of days that an observation day is from the valuation day.
    payoff : Payoff
        %(payoff_docs)s"""
###
_single_barrier_out_param_docs = _single_barrier_param_docs % \
                                 {'payoff_docs': _payoff_docs,
                                  'rebate_docs': _single_barrier_out_rebate_docs}

###
_single_barrier_in_param_docs = _single_barrier_param_docs % \
                                {'payoff_docs': _payoff_docs,
                                 'rebate_docs': _single_barrier_in_rebate_docs}

###
_calc_value_docs = """Calculates the present value and Greeks of the option.

    Parameters
    ----------
    engine : Engine
        An instance of Engine which determines the number of iterations and the batch
        size.
    process : Heston or BlackScholes
        Market process.
    args, kwargs :
        Forwarded to %(calc)s""" % \
                   {'calc': ':meth:`qdpmc.engine.monte_carlo.MonteCarlo.calc`'}

###
_pv_log_paths_docs = """Calculate the present value given a set of paths
    and an array of discount factor.
    
    Parameters
    ----------
    log_paths : array_like
        A 2-D array containing the set of projections of the
        price of the underlying asset.
    df : array_like
        A 1-D array specifying the discount factors.
        
    Returns
    -------
    scalar
        The present value of the option."""

###
_spot_docs = """The spot price of the underlying asset.
"""

###
_sim_t_array_docs = """The array of days that the price of the underlying asset
must be simulated to calculate the present value of the option.
"""
