# Option-pricing-models

https://optionpricingmodelsmnsr2.streamlit.app/

Option Pricing Models (OPM) is a project that I have created to improve my skills. I want others to share their ideas and different approaches so that we all can learn.
I have used Black Scholes ( created by Prudhvi Reddy, Muppala ) as a base or inspiration. Others include Monte Carlo simulation and my own pricing model (shortly).

These models are also used to trade options with minimized risk while targeting profits.
Common methods include:
1. Black-Scholes Model.
2. Monte Carlo Simulation.
3. Binomial Model.
4. Stochastic Volatility Models (e.g., Heston).

1. Black-Scholes Model
  - Developed in 1973 by Fischer Black and Myron Scholes.
  - Used to price European options (cannot be exercised before expiry).
  - Assumes log-normal distribution of stock prices and constant volatility.

Assumptions:

  - No arbitrage.
  - Markets are frictionless (no taxes, no transaction costs).
  - Constant interest rate and volatility.
  - Asset follows a Geometric Brownian Motion (GBM).

2. Monte Carlo Simulation (MCS)
  - A numerical method that simulates a large number of possible future asset price paths.
  - Useful for pricing path-dependent or exotic options (e.g., barrier, Asian options).

Basic Steps:

  - Generate many stock price paths using random sampling from normal distributions.
  - For each path, compute the payoff of the option.
  - Discount the average payoff back to present value.


If anyone want to do some improvements to it, here is a thing you can do:
  - I made the app in a way that it takes real inputs and map those values to the previous model that I have edited a bit, and calculates heatmap values for it.
  - But it isn't completed yet, the flaw here is in sidebar of the app as well as the values assigning method (The heatmap is associated with previous values than that of the stock)
  - So, if you want to improve something, do it. Make it in a way that the model takes stock value directly right when we specify it, rather than manually adjusting them from seeing below and writing above.
  - I will also do other models in the future.


 Dependencies:
  - yfinance: To fetch current asset prices.
  - numpy: For numerical operations.
  - matplotlib: For heatmap visualization.
