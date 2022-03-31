# raw script of phoenix option
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
dt = 1 / 243
valuation_date = datetime.date(2021, 3, 15)
is_already_knockedin = False

start_date = datetime.date(2021, 3, 15)
maturity_date = datetime.date(2022, 3, 15)
ko_ob_dates = calendar.periodic(start=start_date, period="1m", count=13,
                                if_close="next")[1:]

# monte carlo simulation
# prepare mc simulation parameters
# ttm: 243
days_to_maturity = calendar.num_trading_days_between(start_date, maturity_date)
ko_ob_days = []
coupons = []
discount_factor = []

for date in ko_ob_dates:
    if date > valuation_date:
        ko_ob_days.append(calendar.num_trading_days_between(valuation_date, date))
        coupons.append(coupon_rate * notional_principal)
        discount_factor.append(np.exp(-r * calendar.num_trading_days_between(valuation_date, date) * dt))

ki_ob_days = np.arange(1, days_to_maturity + 1)
ki_ob_days = np.arange(1, days_to_maturity + 1)

vol_sqrt_dt = v * np.sqrt(dt)
drift = (r - q - 0.5 * v * v) * dt

log_coupon_barrier = np.log(coupon_barrier)
log_ko_barriers = np.log(ko_barrier)
log_ki_barriers = np.log(ki_barrier)


def check_if_knock_in(logpath, logbarrier):
    comp = logpath < logbarrier
    if np.any(comp):
        return True, np.argmax(comp)
    else:
        return False, 0


def mc_simulation(n):
    payoff = 0.0
    knock_in = is_already_knockedin

    eps = np.random.normal(0, 1, int(days_to_maturity))
    logst = (drift + vol_sqrt_dt * eps).cumsum() + np.log(spot)

    # coupon is accumulated and paid immediately when st > coupon barrier
    for i in range(len(ko_ob_days)):
        od = int(ko_ob_days[i]) - 1
        # coupon is accumulated and paid immediately when st > coupon barrier
        if logst[od] >= log_coupon_barrier:
            payoff += coupons[i] * discount_factor[i]
        # option is terminated at knocking out
        if logst[od] >= log_ko_barriers:
            return payoff

    knock_in, knock_in_t = check_if_knock_in(logst, log_ki_barriers)

    # terminal
    if knock_in:
        # if ever knocked in
        return payoff - max(ki_strike - np.exp(logst[-1]), 0) / spot * notional_principal * discount_factor[-1]
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
