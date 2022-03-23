import numpy as np
from qdpmc.structures.base import StructureMC
from qdpmc.model.market_process import BlackScholes
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from multiprocessing import cpu_count


__all__ = ['MonteCarlo']


def _run_one_time_caller(
        batch_size: int,
        option: StructureMC, process: BlackScholes,
        request_greeks=False, fd_steps=None, fd_scheme=None,
):
    """Return a function that takes *seed* as its sole parameter. Given a seed,
    the returned function generates random numbers, calculate requested fields,
    and return the values."""
    _coordinator = process.coordinator(option, process)

    df = _coordinator.df

    if not request_greeks:
        # if calculation of Greek letters is not requested,
        # then this function simply simulates random paths
        # and computes the present value.
        def _calc(seed):
            eps = _coordinator.generate_eps(seed, batch_size)
            path = _coordinator.paths_given_eps(eps)
            return option.pv_log_paths(path, df)
    else:
        def _str_to_int(s):
            if s == 'central' or s == 0:
                return 0
            elif s == 'forward' or s == 1:
                return 1
            elif s == 'backward' or s == -1:
                return -1
            else:
                raise ValueError("Unknown finite-different scheme: %s" % str(s))

        if fd_steps is None:
            fd_steps = dict(ds=0.01, dr=0.0001, dv=0.01)
        if fd_scheme is None:
            fd_scheme = dict(Delta=0, Rho=0, Vega=0)
        else:
            for key in ['Delta', 'Rho', 'Vega']:
                try:
                    fd_scheme[key] = _str_to_int(fd_scheme[key])
                except KeyError:
                    fd_scheme[key] = 0

        ds, dr, dv = fd_steps['ds'], fd_steps['dr'], fd_steps['dv']
        dss, drs, dvs = fd_scheme['Delta'], fd_scheme['Rho'], fd_scheme['Vega']
        ds_sq = ds * ds

        def _calc(seed):
            eps = _coordinator.generate_eps(seed, batch_size)
            base_path = _coordinator.paths_given_eps(eps)
            shifted_path = _coordinator.shift(
                paths=base_path,
                ds=ds, dr=dr, dv=dv, eps=eps
            )

            # pv, delta and gamma
            pv = option.pv_log_paths(base_path, df)
            pv_s_plus = option.pv_log_paths(shifted_path['S plus'], df)
            pv_s_minus = option.pv_log_paths(shifted_path['S minus'], df)

            if dss == 0:  # central difference
                delta = (pv_s_plus - pv_s_minus) / (2 * ds) / option.spot
            elif dss == 1:  # forward difference
                delta = (pv_s_plus - pv) / ds / option.spot
            else:  # backward difference
                delta = (pv - pv_s_minus) / ds / option.spot

            # gamma can only be calculated with central difference
            gamma = (pv_s_plus + pv_s_minus - 2 * pv) / (
                    ds_sq * option.spot * option.spot
            )

            # rho
            if drs == 0:  # central difference
                pv_r_plus = option.pv_log_paths(
                    shifted_path['R plus'], shifted_path['DF plus']
                )
                pv_r_minus = option.pv_log_paths(
                    shifted_path['R minus'], shifted_path['DF minus']
                )
                rho = (pv_r_plus - pv_r_minus) / (2 * dr)
            elif drs == 1:  # forward difference
                pv_r_plus = option.pv_log_paths(
                    shifted_path['R plus'], shifted_path['DF plus']
                )
                rho = (pv_r_plus - pv) / dr
            else:  # backward difference
                pv_r_minus = option.pv_log_paths(
                    shifted_path['R minus'], shifted_path['DF minus']
                )
                rho = (pv - pv_r_minus) / dr

            # vega
            if dvs == 0:  # central difference
                pv_v_plus = option.pv_log_paths(shifted_path['V plus'], df)
                pv_v_minus = option.pv_log_paths(shifted_path['V minus'], df)
                vega = (pv_v_plus - pv_v_minus) / (2 * dv)
            elif dvs == 1:  # forward difference
                pv_v_plus = option.pv_log_paths(shifted_path['V plus'], df)
                vega = (pv_v_plus - pv) / dv
            else:  # backward difference
                pv_v_minus = option.pv_log_paths(shifted_path['V minus'], df)
                vega = (pv - pv_v_minus) / dv

            pv_next_day = option.pv_log_paths(
                shifted_path['Paths next day'], shifted_path['DF next day']
            )
            theta = (pv_next_day - pv) / process.day_counter

            return pv, delta, gamma, rho, vega, theta

    _calc.__doc__ = """Run 1 time of Monte Carlo simulation given a random seed.
    Parameters: \n""" + str(locals())
    return _calc


def joblib_caller(func, iterator, **kwargs):
    func = delayed(wrap_non_picklable_objects(func))
    with Parallel(**kwargs) as p:
        res = p(func(s) for s in iterator)
    return res


class MonteCarlo:
    most_recent_entropy = property(
        lambda self: self._most_recent_entropy,
        lambda self, v: None, lambda self: None,
        "The most recently used entropy."
    )

    @property
    def caller(self):
        return self._caller

    @caller.setter
    def caller(self, val):
        self._caller = val

    @caller.deleter
    def caller(self):
        self._caller = None

    caller.__doc__ = \
        """Default caller used to implement Monte Carlo simulation.
        if set to None, *joblib.Parallel* will be used."""

    def __init__(self, batch_size: int, num_iter: int, caller=None):
        """A Monte Carlo engine for valuing path-dependent options.

        Parameters regarding the simulation are specified here. This engine implements
        vectorization, which significantly enhances algorithm speed.

        Parameters
        ----------
        batch_size : int
            An integer telling the engine how many paths should be generated in each
            iteration.
        num_iter : int
            Number of iterations. The product of *batch_size* and *num_iter* is the total
            number of paths generated."""
        self.batch_size = batch_size
        self.num_iter = num_iter
        self._most_recent_entropy = None
        self._caller = caller

    def calc(self, option: StructureMC, process: BlackScholes,
             request_greeks=False, fd_steps=None, fd_scheme=None,
             entropy=None, caller=None, caller_args=None):
        """Calculates the present value of an option given a market process.

        Parameters
        ----------
        option :
            The option structure that needs to be calculated.
        process :
            An object containing information about the process
            driving the dynamics of the underlying asset. Currently this value
            must be a BlackScholes object.
        request_greeks : bool
            Whether to calculate and return Greek letters.
        fd_steps : dict
            A dictionary controlling steps of finite difference.
            Default is None, corresponding to
            *{"ds": 0.01, dr: "0.0001", dv: "0.01"}*. Note that *ds* is in log
            scale.
        fd_scheme : dict
            Scheme of finite difference for calculating Greek
            letters. Default is
            *{"Delta": "central", "Rho": "central", "Vega": "central"}*. This
            does not affect Gamma, which is always calculated using the central
            difference scheme.
        entropy : int or None
            Controls random numbers. Must be an integer.
        caller : callable
            caller used to implement computation. Default is None,
            corresponding to *self.caller*. *caller* must take at least two
            arguments:
            *caller(calc_once_give_seed, seed list, ...)*. Moreover, caller
            should return a list of results, not a scalar. See below for more
            details.
        caller_args : dict
            parameters passed to caller. Default is None, equivalent to
            *dict(n_jobs=cpu_counts)*.

        Returns
        -------
        scalar or dict
            PV or dict of Greek letters.

        Examples
        --------

        The default caller is roughly equivalent to:

        .. code-block:: python

            def caller(calc, seedsequence, /, **kwargs):
                calc = joblib.delayed(calc)
                if "n_jobs" not in kwargs:
                    kwargs["n_jobs"] = cpu_counts()
                with joblib.Parallel(**kwargs) as parallel:
                    res = parallel(calc(seed) for seed in seedsequence)
                return res

        To implement Monte Carlo with your own caller, for example one with
        no parallel computation, define the caller as follows:

        .. code-block:: python

            def mycaller(calc, seedsequence):
                return [calc(s) for s in seedsequence]

        You may also set the default caller so you do not need to specify
        ``caller`` every time ``calc`` is called.

        .. code-block:: python

            mc.caller = mycaller

        To revert to the joblib caller, delete the caller or set it to ``None``:

        .. code-block:: python

            del mc.caller
            mc.caller == None  # True"""

        ss = np.random.SeedSequence(entropy)
        # update entropy
        self._most_recent_entropy = ss.entropy
        # probability of colliding pairs = n^2/(2^128)
        subs = ss.spawn(self.num_iter)
        # _calc is the function that is called each iteration
        _calc = _run_one_time_caller(self.batch_size, option, process,
                                     request_greeks, fd_steps, fd_scheme)
        # implement computation
        if caller_args is None:
            caller_args = dict()
        if caller is None:
            caller = self._caller
        if caller is None:
            if "n_jobs" not in caller_args:
                caller_args["n_jobs"] = cpu_count()
            res = joblib_caller(_calc, subs, **caller_args)
        else:
            if not callable(caller):
                raise TypeError("caller must be callable or None")
            res = caller(_calc, subs, **caller_args)

        if not request_greeks:
            return np.mean(res, axis=0)

        _pv, _delta, _gamma, _rho, _vega, _theta = np.mean(res, axis=0)

        return dict(PV=_pv, Delta=_delta, Gamma=_gamma,
                    Rho=_rho, Vega=_vega, Theta=_theta)

    def single_iter_caller(
            self, option: StructureMC, process: BlackScholes,
            request_greeks=False, fd_steps=None, fd_scheme=None,
    ):
        """This returns a function that takes *seed* as its sole parameter.
        Given a random seed, it generates random numbers, computes requested
        values, and returns them."""
        return _run_one_time_caller(
            batch_size=self.batch_size, option=option,
            process=process,
            request_greeks=request_greeks,
            fd_steps=fd_steps, fd_scheme=fd_scheme
        )
