import unittest
from qdpmc.dateutil.date import Calendar
import datetime

calendar = Calendar(market="china")
class Calendar_test(unittest.TestCase):
    '''
    Test calendar

    '''
    def test_trading(self):
        self.assertAlmostEqual(calendar.is_trading(datetime.date(2021,1,4)), True)

    def test_period(self):
        # start date of the contract
        start_date = datetime.date(2019, 1, 31)
        assert calendar.is_trading(start_date)
        # knock-out observation days
        ko_ob_dates = calendar.periodic(start=start_date, period="2m", count=13,
                                        if_close="next")[1:]
        import numpy as np
        print(np.array(ko_ob_dates))
        self.assertAlmostEqual(ko_ob_dates[0], datetime.date(2019, 4, 1))


if __name__ == '__main__':
    unittest.main()