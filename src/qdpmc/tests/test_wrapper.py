import numpy as np

from qdpmc.products import *
from qdpmc.dateutil.date import Calendar
from qdpmc import MonteCarlo, BlackScholes
from qdpmc import Payoff, plain_vanilla
from unittest import TestCase
import datetime
from qdpmc.model.market_process import Heston

hst = Heston(.03, 0, -.3, .0625, 1, .2, .0625, 252)
calendar = Calendar()

start = datetime.date(2019, 1, 31)
ko_ob_dates = calendar.periodic(start, '1M', 13, "next")[1:]


mc = MonteCarlo(100, 1000)
bs = BlackScholes(0.03, 0, 0.265, 252)

option = Snowball(
    start_date=start, ko_ob_dates=ko_ob_dates, ko_barriers=105,
    ki_ob_dates="daily",  ki_barriers=70, ko_coupon_rate=0.09202428093544388,
    ki_payoff=-Payoff(plain_vanilla, 100, "put"), maturity_coupon_rate=0.09202428093544388,
    initial_price=100
)

from functools import partial

value1 = partial(option.value, spot=100, ki_flag=False, engine=mc, process=bs)


class test(TestCase):
    def test_calc(self):
        print(option.value(start, 100, False, mc, bs))

    def test_assert_trading_day(self):
        with self.assertRaises(ValueError):
            value1(datetime.date(2019, 5, 1))

    def test_ki(self):
        from qdpmc.structures import (
            UpOut as UO,
            UpOutDownIn as UODI,
        )
        self.assertIsInstance(
            option.to_structure(datetime.date(2019, 5, 7), 100, True), UO
        )
        self.assertIsInstance(
            option.to_structure(datetime.date(2019, 5, 7), 100, False), UODI
        )

    def test_find_coup(self):

        print(option.find_coup_rate(mc, bs, 0))

    def test_mc_caller(self):

        def nonsense(calc, seeds):
            return [1]*len(seeds)

        mc.caller = nonsense
        print(option.value(datetime.date(2019, 5, 7), 100, False, mc, bs))
        del mc.caller
        print(option.value(datetime.date(2019, 5, 7), 100, False, mc, bs))

    def test_calendar(self):


        calendar = Calendar()
        # start date of the contract
        start_date = datetime.date(2019, 1, 31)
        assert calendar.is_trading(start_date)
        # knock-out observation days
        ko_ob_dates = calendar.periodic(start=start_date, period="2m", count=13,
                                        if_close="next")[1:]
        import numpy as np
        print(np.array(ko_ob_dates))