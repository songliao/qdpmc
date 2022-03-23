# Todo: decide what to do with barriers with equal priority
#       In fact, you can never include two single barrier with
#       identical priority and direction.
#       Therefore, the answer has to be a double barrier,
#       one up, one down
# Todo: add documentation to SingleBarrier
# Todo: StructuredProduct class
# Todo: We should remove parameter *rebate* in SingleBarrier
#       and capture it with a path-dependent payoff class

from qdpmc.tools.helper import (
    arr_scalar_converter,
    up_ko_t_and_surviving_paths,
    down_ko_t_and_surviving_paths,
    fill_arr
)
from qdpmc.tools.payoffs import Payoff
from numpy import log
from qdpmc._decorators import _param_freezer


class SingleBarrier:
    # Do not distinguish barriers by "in" and "out". Instead, they are
    # characterized by whether the payoff is a terminal payoff or a rebate.

    def __init__(self, level, ob_days, up_or_down, rebate=0, payoff=None):

        self.level = arr_scalar_converter(level, ob_days)
        self.ob_days = ob_days

        if (payoff is not None) and (not isinstance(payoff, Payoff)):
            raise TypeError("Parameter payoff must be Payoff or None")
        self.payoff = payoff

        self.rebate = arr_scalar_converter(rebate, ob_days)

        if up_or_down not in ('up', 'down'):
            raise ValueError("Barrier type must be either 'up' or 'down'")
        self.up_or_down = up_or_down

        # Freeze the parameters
        _freezer = _param_freezer(barrier=self.level, return_idx=False)
        if up_or_down == "up":
            self.helper = _freezer(up_ko_t_and_surviving_paths)
        else:
            self.helper = _freezer(down_ko_t_and_surviving_paths)

    def filter(self, paths, df):
        """Payoff of hits and paths that do not hit the barrier
        """

        terminal_df = df[-1]

        hit_t, hits, nhits = self.helper(paths)
        pv_rebate = (self.rebate[hit_t] * df[hit_t]).sum() / len(paths)

        if self.payoff is not None:
            pv_payoff = self.payoff(hits[:, -1]).sum() * terminal_df / len(paths)
        else:
            pv_payoff = 0

        return {
            "PV": pv_rebate + pv_payoff,
            "Hitting": hits,
            "Day first hit": hit_t,
            "Non-hitting": nhits
        }

    def fill(self, all_ob_days, val_level, val_rebate=0):

        level = fill_arr(self.level, self.ob_days, all_ob_days, val_level)
        rebate = fill_arr(self.rebate, self.ob_days, all_ob_days, val_rebate)

        obj = SingleBarrier(
            level=level,
            ob_days=all_ob_days,
            up_or_down=self.up_or_down,
            rebate=rebate,
            payoff=self.payoff
        )

        return obj

    def to_log(self, spot):

        if self.payoff is None:
            payoff = None
        elif isinstance(self.payoff, Payoff):
            payoff = self.payoff.to_log(spot)
        else:
            raise TypeError("Unknown payoff")

        obj = SingleBarrier(
            level=log(self.level / spot),
            ob_days=self.ob_days,
            up_or_down=self.up_or_down,
            rebate=self.rebate,
            payoff=payoff
        )

        return obj
