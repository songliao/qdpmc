from qdpmc import *
import yuanrong


option = UpOut(
    100, 120, rebate=0, ob_days=list(range(1, 253)),
    payoff=Payoff(plain_vanilla, 100)
)


mc = MonteCarlo(125, 8000)
bs = BlackScholes(0.03, 0, 0.25, 252)


def yuanrong_caller(calc, seeds):
    yuanrong.init(package_ref=..., cluster_server_addr=...)
    calc = yuanrong.ship()(calc)
    res_id = [calc.ship(seed) for seed in seeds]
    res = yuanrong.get(res_id)
    yuanrong.shutdown()
    return res


print(
    option.calc_value(
        mc, bs, True,
        caller=yuanrong_caller,
    )
)
