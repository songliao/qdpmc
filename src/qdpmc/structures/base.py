from abc import ABC, abstractmethod
from qdpmc.structures._docs import (
    _calc_value_docs,
    _spot_docs,
    _sim_t_array_docs
)
from qdpmc._decorators import DocstringWriter


class OptionABC(ABC):
    @abstractmethod
    def pv_log_paths(self, log_paths, df):
        pass

    @property
    @abstractmethod
    def spot(self):
        pass

    @property
    @abstractmethod
    def sim_t_array(self):
        pass


class StructureMC(OptionABC):

    @DocstringWriter(_calc_value_docs)
    def calc_value(self, engine, process, *args, **kwargs):
        return engine.calc(self, process, *args, **kwargs)

    def pv_log_paths(self, log_paths, df):
        pass

    def _set_spot(self, val):
        pass

    spot = property(lambda self: self._spot, _set_spot, lambda self: None, _spot_docs)
    sim_t_array = property(lambda self: self._sim_t_array, lambda self, v: None,
                           lambda self: None, _sim_t_array_docs)


class ProcessCoordinator(ABC):
    @abstractmethod
    def generate_eps(self, seed, batch_size):
        pass

    @abstractmethod
    def paths_given_eps(self, eps):
        pass

    @abstractmethod
    def shift(self, paths, ds, dr, dv, eps):
        pass
