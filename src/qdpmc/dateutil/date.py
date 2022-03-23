# Todo: test Calendar.add_holidays and Calendar.add_holiday_rule

import datetime
from qdpmc.dateutil._china_holidays import _is_china_holidays

__all__ = ['Calendar', 'CHINA_HOLIDAYS']

CHINA_HOLIDAYS = _is_china_holidays

_DAYS_IN_MONTH = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_BEFORE_MONTH = [-1]
dbm = 0
for dim in _DAYS_IN_MONTH[1:]:
    _DAYS_BEFORE_MONTH.append(dbm)
    dbm += dim
del dbm, dim


def _is_leap(year):
    """Return 1 if *year* is leap year else 0."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _next_month_same_day(date: datetime.date) -> datetime.date:
    """Return the date of same day next month. If next month does not have
    same day, return the last day of next month.

    For example:
    datetime.date(2021, 1, 31) -> datetime.date(2021, 2, 28)
    datetime.date(2021, 4, 30) -> datetime.date(2021, 5, 30)"""
    if date.month == 12:
        month = 1
        year = date.year + 1
    else:
        month = date.month + 1
        year = date.year
    _last_day = _DAYS_IN_MONTH[month] + (_is_leap(year) and month == 2)
    return datetime.date(year, month, min(date.day, _last_day))


class Calendar:
    def __init__(self, holiday_rule=CHINA_HOLIDAYS, other_holidays=None):
        # holiday rule taken as is
        self._holiday_rule = holiday_rule
        if other_holidays is None:
            self.holiday_rule = holiday_rule
            self._other_holidays = []
        else:
            def _holiday_rule(date):
                return holiday_rule(date) or date in other_holidays
            self.holiday_rule = _holiday_rule
            self._other_holidays = other_holidays

    def add_holidays(self, holidays):
        """Add holidays to holiday rules. *holidays* should be iterable"""
        if not hasattr(holidays, "__iter__"):
            raise TypeError("holidays must be iterable")
        other_holidays = list(self._other_holidays) + list(holidays)
        Calendar.__init__(self, self._holiday_rule, other_holidays)

    def add_holiday_rule(self, holiday_rule):
        """Add holiday rules.

        Parameters
        ----------
        holiday_rule : callable
            A function that converts datetime.date to bool.
        """
        if not callable(holiday_rule):
            raise TypeError("holiday_rule must be callable")

        def _holiday_rule(date):
            return self._holiday_rule(date) or holiday_rule(date)
        Calendar.__init__(self, _holiday_rule, self._other_holidays)

    def is_trading(self, date: datetime.date):
        """Return if *date* trades.

        Parameters
        ----------
        date : datetime.date
        """
        return date.weekday() < 5 and not self.holiday_rule(date)

    def trading_days_between(
            self,
            start: datetime.date,
            end: datetime.date,
            endpoints: bool = True
    ) -> list:
        """Return a list of trading days between *start* and *end*. Endpoints
        are counted only if they are trading.

        Parameters
        ----------
        start : datetime.date
        end : datetime.date
        endpoints : bool
            Whether to count *start* and *end* if they are trading dates.

        Returns
        -------
        list
            A list of trading dates.

        Note
        ----
        ``start`` and ``end`` are counted _only_ if they are trading dates.
        """
        if start > end:
            raise ValueError("start date must be prior to end date")
        if start == end:
            return [] if endpoints else [start]
        d_list = [start] if self.is_trading(start) and endpoints else []
        date = start
        while date < end + datetime.timedelta(days=-1 * (not endpoints)):
            date += datetime.timedelta(days=1)
            if self.is_trading(date):
                d_list.append(date)
        return d_list

    def offset(self, date: datetime.date, n: int) -> datetime.date:
        """Return date of trading day *n* days after *date*. *n* can be
        negative but must be an integer.

        Parameters
        ----------
        date : datetime.date
        n : int

        Returns
        -------
        datetime.date
            The *n*-th trading date after *date*.

        """
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        count = 0
        d = n > 0
        while count < abs(n):
            date += datetime.timedelta(days=d)
            count += self.is_trading(date)
        return date

    def periodic(
            self,
            start: datetime.date,
            period: str,
            count: int,
            if_close: str = "next",
            force: bool = False
    ) -> list:
        """Generate periodical trading dates. Duplicates will be dropped so
        the output list may be shorter than *count*. To force the length of
        output to *count*, set *force* to *True*

        Parameters
        ----------
        start : datetime.date
            The first date of the array of periodic trading dates.
        period : str
            String like 'Xm', 'Xw', and 'Xd', where X
        count : int
        if_close : bool
        force : bool

        Examples
        --------
        ..ipython:: python

            import datetime
            calendar = qm.Calendar()
            start_date = datetime.date(2019, 1, 31)
            calendar.periodic(start, '1m', 13)
        """
        try:
            n, unit = int(period[:-1]), period[-1]
            if n <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("period must be a positive integer")
        if if_close not in ("next", "prev"):
            raise ValueError(r'if_close must be one of "next" or "prev"')
        if not self.is_trading(start):
            raise ValueError("given start date is not trading")
        if unit not in ['W', 'w', 'M', 'm', 'D', 'd']:
            raise ValueError("Unit not understood")

        if unit in ['W', 'w']:
            def _next_period(_date):
                return _date + datetime.timedelta(days=7*n)
        elif unit in ['M', 'm']:
            def _next_period(_date):
                _day = _date.day
                for _ in range(n):
                    _date = _next_month_same_day(_date)
                try:
                    _date = datetime.date(_date.year, _date.month, _day)
                except ValueError:
                    pass
                return _date
        else:
            def _next_period(_date):
                return _date + datetime.timedelta(days=n)

        res = [start]
        date = start
        i = 1
        while i < count:
            date = _next_period(date)
            # set date.day to start.day if period is given in month
            if unit in ['M', 'm']:
                try:
                    date = datetime.date(date.year, date.month, start.day)
                except ValueError:
                    pass

            if not self.is_trading(date):
                if if_close == "next":
                    o = self.offset(date, 1)
                else:
                    o = self.offset(date, -1)
            else:
                o = date

            if o not in res:
                res.append(o)

            i = len(res) if force else i + 1

        return res

    def to_scalar(self, date_arr, start) -> list:
        """Convert dates into integers given a start date.

        Parameters
        ----------
        date_arr : array_like
            An array of datetime.date objects.
        start : datetime.date
            The start date.

        Returns
        -------
        list
            A list of integers.
        """
        return [self.num_trading_days_between(start, end) for end in date_arr]

    def num_trading_days_between(
            self,
            start: datetime.date,
            end: datetime.date,
            count_end: bool = True
    ) -> int:
        """Return number of trading days between two dates. If these dates
        are identical, return 0. *count_end* controls whether to count the
        end date."""
        if start == end:
            return 0
        trading_dates = self.trading_days_between(start, end, True)
        return len(trading_dates) - count_end
