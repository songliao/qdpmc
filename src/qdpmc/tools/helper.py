import numpy as np
from qdpmc.tools.payoffs import Payoff


def double_ko_t_and_surviving_paths(paths, u, d, return_idx):
    """Given a set of projections of prices, find out the ones that
    reach at some point a double barrier. For these paths, return
    the indices of the first reach, and for those that do not touch
    the barrier, return a copy of them.


    :param paths: An 2D array containing the paths to be evaluated.
    :param u: Upper barrier level. It must match the length of
        each individual path.
    :param d: Lower barrier level. It must match the length of
        each individual path
    :param return_idx: If True, return the indices of knock-outs and
        survivors, rather than the paths themselves.
    :return: ko-time-index, ko-paths (or indices), nko-paths
        (or indices)
    """
    ko_idx = np.any([paths >= u, paths <= d], axis=(2, 0))
    nko_idx = np.logical_not(ko_idx)
    ko_paths = paths[ko_idx]
    ko_t = np.argmax(np.any([ko_paths >= u, ko_paths <= d], axis=0), axis=1)
    if return_idx:
        return ko_t, ko_idx, nko_idx
    nko_paths = paths[nko_idx]
    return ko_t, ko_paths, nko_paths


def double_ki_paths(paths, u, d, return_idx):
    """Returns the third element of function _double_ko_t_and_surviving_paths
    but with a higher speed.
    """
    ki_idx = np.any([paths >= u, paths <= d], axis=(2, 0))
    if return_idx:
        return ki_idx
    ki_paths = paths[ki_idx]
    return ki_paths


def payoff_wrapper(payoff_func, payoff_args):
    """Wrap the payoff function by freezing its arguments specified in
    payoff_args. An error will be raised if the wrap function cannot
    handle a numpy.array.
    """
    # if the payoff_func is passed in as an instance of Payoff
    # this means that arguments are already specified
    # In this case, raise an error
    if payoff_args is None:
        payoff_args = {}

    wrap = Payoff(payoff_func, **payoff_args)

    try:
        wrap(np.array([1, 2, 3]))
    except Exception as e:
        raise TypeError("It seems there is something wrong with the payoff "
                        "object. \n Make sure that is callable and can correctly "
                        "handle an array. \n" + str(e))
    return wrap


def merge_days(d1, d2):
    """
    Given two arrays of days, return the union of them and
    the where they are in the new array.
    """
    merged = sorted(set(np.concatenate([d1, d2])))
    d1_idx = [d in d1 for d in merged]
    d2_idx = [d in d2 for d in merged]
    return merged, d1_idx, d2_idx


def arr_scalar_converter(val, ob_days):
    """Convert a scalar to an array that matches the length of ob_dates.
    If val is already an array, check if it has the correct length.
    """
    if hasattr(val, "__iter__"):
        if len(val) != len(ob_days):
            raise ValueError("Lengths of passed arrays do not match")
        return np.array(val)
    return np.full(len(ob_days), val)


def up_ki_paths(paths, barrier, return_idx):
    ki_idx = np.any(paths >= barrier, axis=1)
    ki = paths[ki_idx]
    if return_idx:
        return ki_idx
    return ki


def down_ki_paths(paths, barrier, return_idx):
    ki_idx = np.any(paths <= barrier, axis=1)
    ki = paths[ki_idx]
    if return_idx:
        return ki_idx
    return ki


def up_ko_t_and_surviving_paths(paths, barrier, return_idx):
    ko_path_idx = np.any(paths >= barrier, axis=1)
    nko_idx = np.logical_not(ko_path_idx)
    ko_paths = paths[ko_path_idx]
    ko_t = np.argmax(ko_paths >= barrier, axis=1)
    if return_idx:
        return ko_t, ko_path_idx, nko_idx
    nko_paths = paths[nko_idx]
    return ko_t, ko_paths, nko_paths


def down_ko_t_and_surviving_paths(paths, barrier, return_idx):
    ko_path_idx = np.any(paths <= barrier, axis=1)
    nko_idx = np.logical_not(ko_path_idx)
    ko_paths = paths[ko_path_idx]
    ko_t = np.argmax(ko_paths >= barrier, axis=1)
    if return_idx:
        return ko_t, ko_path_idx, nko_idx
    nko_paths = paths[nko_idx]
    return ko_t, ko_paths, nko_paths


def fill_arr(arr, ob_days, all_days, val_with):
    # check whether ob_days is a subset of all_days
    for d in ob_days:
        if d not in all_days:
            raise ValueError("Observation days must be a subset of all days")
    # boolean array of whether total_days is in ob_days
    pos_ob_days = [d in ob_days for d in all_days]
    # array full of val_with; matches length of all_days
    filled = np.full(len(all_days), val_with)
    # set the full array's elements to barrier
    filled[pos_ob_days] = arr
    return filled
