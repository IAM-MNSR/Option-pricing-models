# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:07:22 2025

@author: MNSR
"""

from numpy import exp, sqrt, log
from scipy.stats import norm


class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.Cbep = None  # Will be calculated in run()
        self.Pbep = None

    def run(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate
        Cbep = self.Cbep
        Pbep = self.Pbep
        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            strike * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma
        
        #call BEP formula BEP= BEPcall = K + C*e**rT  where K = option strike, C = price of the call option, euler const e = 2.718, r = interest rate, T = time
        
        #put bep = BEPput = K-C*e**rT
        
        self.Cbep = strike + call_price*(2.718**(interest_rate*time_to_maturity))
        
        self.Pbep = strike - put_price*(2.718**(interest_rate*time_to_maturity))


if __name__ == "__main__":
    time_to_maturity = 2
    strike = 90
    current_price = 100
    volatility = 0.2
    interest_rate = 0.05

    # Black Scholes
    BS = BlackScholes(
        time_to_maturity=time_to_maturity,
        strike=strike,
        current_price=current_price,
        volatility=volatility,
        interest_rate=interest_rate)
    BS.run()
    print(f"Call Break-Even Point (Cbep): {BS.Cbep:.2f}")
    print(f"Put Break-Even Point (Pbep): {BS.Pbep:.2f}")
    