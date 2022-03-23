from qdpmc import *
import numpy as np
import unittest
from qdpmc.tools.payoffs import Payoff
from qdpmc.structures.creator.creator import SingleBarrier
from qdpmc.model.market_process import Heston

# General parameters
entropy = 12345678
mc = MonteCarlo(100, 10000)
bs = BlackScholes(0.03, 0, 0.25, 252)
hst = Heston(.03, 0, 0, .0625, 1, .2, .0625, 252)
# To make life easy
dense_d_arr = list(range(1, 253))
sparse_d_arr = list(range(21, 253, 21))
snowballing_rebate = np.linspace(15/12, 15, 12)


# ============================Test Cases===========================

# A snowball
sb = UpOutDownIn(
    spot=100,
    upper_barrier_out=105,
    ob_days_out=sparse_d_arr,
    rebate_out=snowballing_rebate,
    lower_barrier_in=80,
    ob_days_in=dense_d_arr,
    payoff_in=-Payoff(plain_vanilla, strike=100, option_type="put"),
    payoff_nk=Payoff(constant_payoff, amount=15),
)

b = UpOut(
    spot=100, rebate=0, barrier=120, ob_days=sparse_d_arr,
    payoff=Payoff(plain_vanilla, strike=100)
)

sb2 = StandardSnowball(
    spot=100, barrier_in=80, barrier_out=105, ko_coupon=snowballing_rebate,
    full_coupon=15, ob_days_out=sparse_d_arr, ob_days_in=dense_d_arr,
)


def neg_vanilla(spot, strike, option_type):
    return -plain_vanilla(spot, strike, option_type)


double_out = DoubleOut(
    spot=100,
    barrier_up=120,
    barrier_down=80,
    ob_days_up=sparse_d_arr,
    ob_days_down=sparse_d_arr,
    payoff=Payoff(plain_vanilla, strike=100, option_type="call"),
    rebate_up=3,
    rebate_down=2
)


class TestFunctions(unittest.TestCase):
    def test_sb_value(self):
        print(sb.calc_value(mc, hst, entropy=entropy))
        print(sb.calc_value(mc, bs, entropy=entropy))

    def test_heston(self):
        print(b.calc_value(mc, hst))

    def test_double_out(self):
        print(double_out.calc_value(mc, bs, entropy=entropy))

    def test_payoff(self):
        payoff = Payoff(plain_vanilla, strike=50, option_type="put")
        prices = np.linspace(5, 100, 21)
        self.assertTrue(np.all(payoff(prices) == plain_vanilla(prices, 50, "put")))

    def test_payoff_neg(self):
        payoff = -Payoff(plain_vanilla, strike=50, option_type="put")
        prices = np.linspace(5, 100, 21)
        self.assertTrue(np.all(payoff(prices) == -plain_vanilla(prices, 50, "put")))

    def test_payoff_add(self):
        added = Payoff(plain_vanilla, strike=50, option_type="put") + \
            Payoff(plain_vanilla, strike=50, option_type="call")
        prices = np.linspace(5, 100, 21)
        self.assertTrue(np.all(added(prices) == plain_vanilla(prices, 50, "call") +
                               plain_vanilla(prices, 50, "put")))


barrier = SingleBarrier(
    level=104.99,
    ob_days=sparse_d_arr,
    up_or_down="up",
    rebate=0,
    payoff=Payoff(plain_vanilla, 100, "call")
)

paths = np.array(
      [[95, 101,  99, 104,  99,  95,  97,  96,  95, 106,  97,  98],
       [99, 103,  98, 101, 101,  98,  99,  97,  99,  98,  99, 101],
       [106, 103, 105, 102, 100, 101, 104, 102,  96,  95,  99,  96],
       [101, 100, 103, 105, 104,  99, 101,  97, 105, 105, 100, 106],
       [97,  96,  97, 100,  97,  97,  95,  96, 105,  98,  95, 104],
       [99,  99, 105,  98,  99, 105,  96,  99, 105,  98,  96, 102],
       [101,  95,  96, 102, 102, 101,  95, 105, 101, 106, 100, 102],
       [102, 103,  97, 106, 104, 105,  97,  98,  95,  98, 100,  96],
       [96, 102, 103,  97, 100, 104, 104,  95, 101,  96, 100, 102],
       [96,  95, 104,  98,  99, 102, 104, 106, 105,  95, 101,  95]]
)

df = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


class TestCreator(unittest.TestCase):
    def test_out_barrier(self):
        print(barrier.filter(paths, df))

    def test_fill(self):
        print(barrier.fill(dense_d_arr, np.infty).level)

    def test_to_log(self):
        log_barrier = barrier.to_log(100)
        log_paths = np.log(paths / 100)
        self.assertTrue(
            barrier.filter(paths, df)['PV'] == log_barrier.filter(log_paths, df)['PV']
        )


class TestTools(unittest.TestCase):
    def test_vectorize(self):

        def pyfunc(s, k):
            return s - k if s > k else 0

        func = np.vectorize(pyfunc, otypes=[float])
        print(func(np.linspace(1, 100, 100), 50))
