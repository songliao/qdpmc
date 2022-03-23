import numpy as np
from qdpmc.structures.base import StructureMC


class FixedStrike(StructureMC):
    def __init__(self, spot, ob_days, payoff, avgfunc):
        self._spot = spot
        self.ob_days = ob_days
        self._sim_t_array = np.append([0], ob_days)
        self.payoff = payoff
        self.avgfunc = avgfunc

    def pv_log_paths(self, log_paths, df):
        pass