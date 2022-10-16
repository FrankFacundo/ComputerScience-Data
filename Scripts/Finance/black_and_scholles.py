# -*- coding: utf-8 -*-
# @Author: boyac, frank
# @Last Modified by:   Frank

from __future__ import division
from math import exp, sqrt, log

from scipy.stats import norm


class BlackScholes():
    """
    Black&Scholes for European options
    """
    CALL = "c"
    PUT = "p"

    # Cumulative normal distribution
    def cnd(self, X):
        return norm.cdf(X)

    # Black Sholes Function
    def compute(self, call_put_flag, S, K, t, r, s):
        """
        call_put_flag = European option type
        S = Current stock price
        t = Time until option exercise (years to maturity)
        K = Option striking price - price of stock for buying/selling when maturity arrives 
        r = Risk-free interest rate
        N = Cumulative standard normal distribution
        e = Exponential term
        s = St. Deviation (volatility)
        Ln = NaturalLog
        """
        d1 = (log(S/K) + (r + (s ** 2)/2) * t)/(s * sqrt(t))
        d2 = d1 - s * sqrt(t)

        if call_put_flag == 'c':
            # call option
            return S * self.cnd(d1) - K * exp(-r * t) * self.cnd(d2)
        elif call_put_flag == 'p':
            # put option
            return K * exp(-r * t) * self.cnd(-d2) - S * self.cnd(-d1)


if __name__ == "__main__":
    black_scholles = BlackScholes()
    option_price_over_one_subyacent = black_scholles.compute(
        call_put_flag=BlackScholes.CALL, S=49.0, K=50.0, t=0.3846, r=0.05, s=0.2)
    
    number_subyacents = 100000
    option_price = option_price_over_one_subyacent * number_subyacents
    print(option_price)
    # 240046.10869656666