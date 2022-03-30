#raw script of phoenix option
import time
import numpy as np
from scipy.stats import norm
import multiprocessing as mp
from qdpmc.dateutil.date import Calendar
import datetime

# market calendar and day count conventions
calendar = Calendar(market="china")

# define the phoenix
spot = 100
ko_barrier = spot * 1.00
coupon_rate = 0.025
coupon_barrier = spot * 0.8
ki_barrier = spot * 0.75
ki_strike = spot
r = 0.02
q = 0.0
v = 0.35
notional_principal = 100
dt = 1/252
valuation_date = datetime.date(2021, 3, 15)
is_already_knockedin = False

start_date = datetime.date(2021, 3, 15)
maturity_date = datetime.date(2022, 3, 15)
ko_ob_dates = calendar.periodic(start=start_date, period="1m", count=13,
                                        if_close="next")[1:]

# monte carlo simulation
# prepare mc simulation parameters
# ttm: 243
days_to_maturity = calendar.num_trading_days_between(start_date,maturity_date)
ko_ob_days = []
coupons = []
discount_factor = []

for date in ko_ob_dates:
    if date > valuation_date:
        ko_ob_days.append(calendar.num_trading_days_between(valuation_date, date))
        coupons.append(coupon_rate * notional_principal)
        discount_factor.append(np.exp(-r * calendar.num_trading_days_between(valuation_date, date)*dt))

ki_ob_days = np.arange(1, days_to_maturity + 1)

def mc_simulation(n):
    payoff = 0.0
    knock_in = is_already_knockedin

    eps = norm.rvs(size=int(days_to_maturity))
    rt = np.array([0])
    rt = np.append(rt, (r - q - 0.5 * v * v) * dt + v * np.sqrt(dt) * eps)
    st = spot * np.exp(rt.cumsum())

    for i in range(len(ko_ob_days)):
        od = int(ko_ob_days[i])
        # coupon is accumulated and paid immediately when st > coupon barrier
        if st[od] >= coupon_barrier:
            payoff += coupons[i] * discount_factor[i]
        # option is terminated at knocking out
        if st[od] >= ko_barrier:
            return payoff

    if not knock_in:
        # check if knock in
        for i in range(len(ki_ob_days)):
            od = int(ki_ob_days[i])
            if st[od] < ki_barrier:
                knock_in = True
                break

    # terminal
    if knock_in:
        # if ever knocked in
        return payoff - max(ki_strike - st[-1], 0) / spot * notional_principal * discount_factor[-1]
    else:
        # if never knocked in
        return payoff


if __name__ == "__main__":
    t_start = time.time()
    inputs = list(range(1000000))
    pool = mp.Pool(processes=mp.cpu_count())
    pool_outputs = pool.map(mc_simulation, inputs)
    pool.close()
    pool.join()
    mc_price = np.mean(pool_outputs, axis=0)
    t_end = time.time()

    print('mc Result - TIME: {}, MC Price: {}'.format(t_end - t_start, mc_price))
