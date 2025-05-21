# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:51:28 2025

@author: MNSR
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp  # Make sure to import these
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import yfinance as yf
from datetime import datetime

#######################
# Page configuration
st.set_page_config(
    page_title="Option Pricing Models",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 5px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 8px; /* Spacing between CALL and PUT */
    border-radius: 8px; /* Rounded corners */
}

.metric-put {
    background-color: #ff355e; /* Light red (radical red) background */
    color: black; /* Black font color */
    border-radius: 8px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 0.8rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)


# sidebar to select models
with st.sidebar:
    
    selected = option_menu("OPMs",
                           ['Option Pricing Models','Black Scholes', 'Monte Carlo Simulation'],
                           icons=['cpu','graph-up-arrow','dice-5-fill'],
                           default_index=0)
    
if (selected == 'Option Pricing Models'):
    
    st.markdown("### 🧠 Option Pricing Models")
    st.markdown("""
                - **Option pricing models** help determine the **fair value** of an option contract.
                -  These models are also used to **trade options with minimized risk** while targeting profits.
                -  **Common methods include**:
                    - Black-Scholes Model  
                    - Monte Carlo Simulation  
                    - Binomial Model  
                    - Stochastic Volatility Models (e.g., Heston)
                    """)

    st.markdown("""
                ###  1. Black-Scholes Model

        - Developed in 1973 by Fischer Black and Myron Scholes
        - Used to price **European options** (cannot be exercised before expiry)
        - Assumes **log-normal distribution** of stock prices and **constant volatility**

         Assumptions: 
        - No arbitrage
        - Markets are frictionless (no taxes, no transaction costs)
        - Constant interest rate and volatility
        - Asset follows a Geometric Brownian Motion (GBM)
        """)

# Monte Carlo Simulation
    st.markdown("""
                ###  2. Monte Carlo Simulation (MCS)
                
        - A numerical method that simulates a large number of possible future asset price paths
        - Useful for pricing path-dependent or exotic options (e.g., barrier, Asian options)

        Basic Steps:
        - Generate many stock price paths using random sampling from normal distributions
        - For each path, compute the payoff of the option
        - Discount the average payoff back to present value

    """)
    
    
if (selected == 'Black Scholes'):
    
    #st.title("Black Scholes Pricing Model")
    st.markdown("### 🧮 Black Scholes Pricing Model")
    
    # (Include the BlackScholes class definition here)

    class BlackScholes:
        def __init__(
            self,
            time_to_maturity: float,
            strike: float,
            current_price: float,
            volatility: float,
            interest_rate: float,
        ):
            self.time_to_maturity = time_to_maturity
            self.strike = strike
            self.current_price = current_price
            self.volatility = volatility
            self.interest_rate = interest_rate
            

        def calculate_prices(
            self,
        ):
            time_to_maturity = self.time_to_maturity
            strike = self.strike
            current_price = self.current_price
            volatility = self.volatility
            interest_rate = self.interest_rate
        

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
            
            Cbep = strike + call_price*(2.718**(interest_rate*time_to_maturity))
            
            Pbep = strike - put_price*(2.718**(interest_rate*time_to_maturity))
            
            self.Cbep = Cbep
            self.Pbep = Pbep

            return call_price, put_price
        
        @staticmethod
        def get_top_profitable_spot_prices(call_prices, put_prices, spot_range, vol_range, bs_model, top_n=5):
            call_profitable_spots = []
            put_profitable_spots = [] 
            
            for i, vol in enumerate(vol_range):
                for j, spot in enumerate(spot_range):
                    call_price = call_prices[i, j]
                    put_price = put_prices[i, j]

                    if spot > bs_model.Cbep and call_price > 0:
                        call_profitable_spots.append(round(spot, 2))
                    if spot < bs_model.Pbep and put_price > 0:
                        put_profitable_spots.append(round(spot, 2))

            call_counts = Counter(call_profitable_spots)
            put_counts = Counter(put_profitable_spots)

            top_call = call_counts.most_common(top_n)
            top_put = put_counts.most_common(top_n)

            return top_call, top_put

    # Function to generate heatmaps
    # ... your existing imports and BlackScholes class definition ...


    # Sidebar for User Inputs
    with st.sidebar:
        st.write("`Created by:`")
        linkedin_url = "https://www.linkedin.com/in/mallunagasandeepreddy/"
        st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mallu Naga Sandeep Reddy`</a>', unsafe_allow_html=True)
        st.write("`Credits for BlackScholes:`")
        linkedin_url = "https://www.linkedin.com/in/mprudhvi/"
        st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Prudhvi Reddy, Muppala`</a>', unsafe_allow_html=True)

        current_price = st.number_input("Current Asset Price", value=100.0)
        strike = st.number_input("Strike Price", value=100.0)
        time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
        volatility = st.number_input("Volatility (σ)", value=0.2)
        interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

        st.markdown("---")
        calculate_btn = st.button('Heatmap Parameters')
        spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
        spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
        vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
        vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
        
        spot_range = np.linspace(spot_min, spot_max, 10)
        vol_range = np.linspace(vol_min, vol_max, 10)
    
    def plot_heatmap(bs_model, spot_range, vol_range, strike):
        call_prices = np.zeros((len(vol_range), len(spot_range)))
        put_prices = np.zeros((len(vol_range), len(spot_range)))
        
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                bs_temp = BlackScholes(
                    time_to_maturity=bs_model.time_to_maturity,
                    strike=strike,
                    current_price=spot,
                    volatility=vol,
                    interest_rate=bs_model.interest_rate
                )
                bs_temp.calculate_prices()
                call_prices[i, j] = bs_temp.call_price
                put_prices[i, j] = bs_temp.put_price
        
        # Plotting Call Price Heatmap
        fig_call, ax_call = plt.subplots(figsize=(10, 8))
        sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_call)
        ax_call.set_title('CALL')
        ax_call.set_xlabel('Spot Price')
        ax_call.set_ylabel('Volatility')
        
        # Plotting Put Price Heatmap
        fig_put, ax_put = plt.subplots(figsize=(10, 8))
        sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_put)
        ax_put.set_title('PUT')
        ax_put.set_xlabel('Spot Price')
        ax_put.set_ylabel('Volatility')
        
        return fig_call, fig_put
    
    # Table of Inputs
    input_data = {
        "Current Asset Price": [current_price],
        "Option Strike Price": [strike],
        "Time to Maturity (Years)": [time_to_maturity],
        "Volatility (σ) avg IV = 0.16": [volatility],
        "Risk-Free Interest Rate": [interest_rate]
    }
    input_df = pd.DataFrame(input_data)
    st.table(input_df)

    # Calculate Call and Put values
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    call_price, put_price = bs_model.calculate_prices()

    # Display Call and Put Values in colored tables
    col1, col2 = st.columns([1,1], gap="small")

    with col1:
        # Using the custom class for CALL value
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">CALL Value</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Using the custom class for PUT value
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">PUT Value</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### Options Price - Interactive Heatmap")
    #st.title("Options Price - Interactive Heatmap")
    st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")
    st.info("Volatility = vix/100. When vix increase the option prices inflates a lot and when it drops same happens with those option prices, they will just evaporate. That being said, BSM isn't that great of an option to use in VOLATILE markets.")
    st.markdown("#### Understanding Profit and loss using Break Even Points:")
    st.markdown(" -  You gain profit if stock price > Cbep, otherwise you loose.")
    st.markdown(" - You gain profit if stock price < Pbep, otherwise you loose.")

    st.markdown(
        f"""
        <div style='display:flex; gap:15px;'>
            <div style='background-color:#ffe0b2; padding:10px 15px; border-radius:10px; color:black;'>
                <b>Call Break-Even Point (Cbep):</b> ${bs_model.Cbep:.2f}
            </div>
            <div style='background-color:#ffe0b2; padding:10px 15px; border-radius:10px; color:black;'>
                <b>Put Break-Even Point (Pbep):</b> ${bs_model.Pbep:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



    # Interactive Sliders and Heatmaps for Call and Put Options
    col1, col2 = st.columns([1,1], gap="small")

    with col1:
        st.subheader("Call Price Heatmap")
        heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
        st.pyplot(heatmap_fig_call)

    with col2:
        st.subheader("Put Price Heatmap")
        _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
        st.pyplot(heatmap_fig_put)
   
    #st.markdown("### Stock Price")
    st.sidebar.markdown("---")
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="MSFT")
    period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"], index=1)
   
    @st.cache_data
    def fetch_stock_data(ticker, period="3mo"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return None

    # Fetch and process stock data
    hist_data = fetch_stock_data(stock_symbol, period)
    
    if hist_data is not None and not hist_data.empty:
        st.subheader(f"{stock_symbol} Stock Price Chart")
        
        # Plot line chart
        fig, ax = plt.subplots()
        ax.plot(hist_data.index, hist_data["Close"], label="Close Price", color='blue')
        ax.set_title(f"{stock_symbol} Price over {period}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
        
        # Calculate estimated volatility using std of log returns
        hist_data["log_return"] = np.log(hist_data["Close"] / hist_data["Close"].shift(1))
        estimated_volatility = hist_data["log_return"].std() * np.sqrt(252)  # annualized
        
        current_price = hist_data["Close"].iloc[-1]
        st.sidebar.markdown(f"**Detected Price:** ${current_price:.2f}")
        st.sidebar.markdown(f"**Estimated Volatility:** {estimated_volatility:.2%}")
        
        # Use fetched values as default for BS inputs
        strike = st.sidebar.number_input("Strike Price", value=float(round(current_price, 2)), key="strike_price")
        time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=0.5, key="time_to_maturity")
        volatility = st.sidebar.number_input("Volatility (σ)", value=float(round(estimated_volatility, 4)), key="volatility")
        interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, key="interest_rate")

        spot_min = st.sidebar.number_input('Min Spot Price', min_value=0.01, value=current_price * 0.8, step=0.01, key="spot_min")
        spot_max = st.sidebar.number_input('Max Spot Price', min_value=0.01, value=current_price * 1.2, step=0.01, key="spot_max")

        vol_min = st.sidebar.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0,
                            value=max(0.01, volatility * 0.5), step=0.01, key="vol_min")
        vol_max = st.sidebar.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0,
                            value=min(1.0, volatility * 1.5), step=0.01, key="vol_max")

        
        spot_range = np.linspace(spot_min, spot_max, 10)
        vol_range = np.linspace(vol_min, vol_max, 10)
        
    else:
        st.error("Failed to load stock data.")
            



    st.info("Any options (strike price) you have bought, your profit will be what you are left with after substracting it's current value (spot price from heatmap) from the cost value(price you have paid when buying) and also you need to remove charges from the profit to get net profit.")

if (selected == "Monte Carlo Simulation"):
    
    st.markdown("### 🎲 Monte Carlo Simulation Model")
    with st.sidebar:
        st.write("`Created by:`")
        linkedin_url = "https://www.linkedin.com/in/mallunagasandeepreddy/"
        st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mallu Naga Sandeep Reddy`</a>', unsafe_allow_html=True)
        Github_url = "https://github.com/mnsr2"
        st.markdown(f'<a href="{Github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://th.bing.com/th/id/ODLS.b2099a11-ca12-45ce-bede-5df940e38a48?w=32&h=32&o=6&pid=1.2" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mallu Naga Sandeep Reddy`</a>', unsafe_allow_html=True)
        
        # User Inputs
        st.sidebar.header("📈 Stock Ticker Lookup")
        ticker = st.sidebar.text_input("Enter Stock Symbol")


        # Fetch data from Yahoo Finance
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info['regularMarketPrice']
            st.sidebar.write(f"**Current Price:** ${current_price:.2f}")
            
            # Fetch available expiration dates
            exp_dates = stock.options
            expiry = st.sidebar.selectbox("Choose Expiry Date", exp_dates)
            
            # Fetch option chain for selected expiry
            opt_chain = stock.option_chain(expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Choose Call or Put
            option_type = st.sidebar.radio("Option Type", ("Call", "Put"))
            data = calls if option_type == "Call" else puts
            
            # Choose Strike
            strike = st.sidebar.selectbox("Choose Strike", data['strike'])
            
            # Filter row for selected strike
            opt_row = data[data['strike'] == strike].iloc[0]
            
            # Implied Volatility
            iv = opt_row['impliedVolatility']
            st.sidebar.write(f"**Implied Volatility (IV):** {iv:.2%}")
            
            # Time to Maturity in Years
            T = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days / 365
            T = max(T, 1/365)  # Avoid 0 division
            
            st.sidebar.write(f"**Time to Maturity:** {T:.2f} years") 
            S0 = st.number_input("Initial Stock Price (S₀)", value=float(current_price))
            K = st.number_input("Strike Price (K)", value=float(strike))
            T = st.number_input("Time to Maturity (T, in years)", value=T)
            r = st.number_input("Risk-Free Rate (r)", value=0.05)
            sigma = st.number_input("Volatility (σ)", value=float(iv))
            simulations = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)
            #option_type = st.radio("Option Type (Override)", ("Call", "Put"), index=0 if option_type == "Call" else 1)
            
            
            # Monte Carlo Simulation for Option Price
            Z = np.random.standard_normal(simulations)
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            if option_type == "Call":
                payoffs = np.maximum(ST - K, 0)
            else:
                payoffs = np.maximum(K - ST, 0)
            price = np.exp(-r * T) * np.mean(payoffs)

        
        except Exception as e:
            st.sidebar.error("Invalid ticker or data not available.")
            st.stop()
    st.subheader(f"🎯 Estimated {option_type} Option Price: ${price:.2f}")
    
    
    # Histogram of Payoffs
    fig1, ax1 = plt.subplots()
    ax1.hist(payoffs, bins=50, color="lightblue", edgecolor="black")
    ax1.set_title("Payoff Distribution")
    ax1.set_xlabel("Payoff")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    # Simulated Asset Paths (sampled)
    steps = 100
    dt = T / steps
    paths = np.zeros((steps, 10))
    for j in range(10):
        Z_path = np.random.standard_normal(steps)
        paths[0, j] = S0
        for t in range(1, steps):
            paths[t, j] = paths[t-1, j] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_path[t])

    fig2, ax2 = plt.subplots()
    ax2.plot(paths)
    ax2.set_title("Sample Simulated Asset Price Paths")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Asset Price")
    st.pyplot(fig2)

    # Convergence Plot
    running_avg = np.zeros(simulations)
    cumsum = 0
    for i in range(simulations):
        ST_i = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.normal())
        if option_type == "Call":
            payoff = max(ST_i - K, 0)
        else:
            payoff = max(K - ST_i, 0)
        cumsum += np.exp(-r * T) * payoff
        running_avg[i] = cumsum / (i + 1)

    fig3, ax3 = plt.subplots()
    ax3.plot(running_avg, color='purple')
    ax3.set_title("Convergence of Option Price Estimate")
    ax3.set_xlabel("Simulations")
    ax3.set_ylabel("Estimated Price")
    st.pyplot(fig3)
    


    # Greeks Estimation (Finite Difference Method)
    epsilon = 0.01
        
    # Delta
    S_up = S0 + epsilon
    S_down = S0 - epsilon
    ST_up = S_up * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(simulations))
    ST_down = S_down * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(simulations))
    if option_type == "Call":
        payoff_up = np.exp(-r * T) * np.maximum(ST_up - K, 0)
        payoff_down = np.exp(-r * T) * np.maximum(ST_down - K, 0)
    else:
        payoff_up = np.exp(-r * T) * np.maximum(K - ST_up, 0)
        payoff_down = np.exp(-r * T) * np.maximum(K - ST_down, 0)
    delta = (np.mean(payoff_up) - np.mean(payoff_down)) / (2 * epsilon)

        # Vega
    sigma_up = sigma + epsilon
    ST_vega = S0 * np.exp((r - 0.5 * sigma_up**2) * T + sigma_up * np.sqrt(T) * np.random.standard_normal(simulations))
    if option_type == "Call":
        payoff_vega = np.exp(-r * T) * np.maximum(ST_vega - K, 0)
    else:
        payoff_vega = np.exp(-r * T) * np.maximum(K - ST_vega, 0)
    vega = (np.mean(payoff_vega) - price) / epsilon

    # Rho
    r_up = r + epsilon
    ST_rho = S0 * np.exp((r_up - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(simulations))
    if option_type == "Call":
        payoff_rho = np.exp(-r_up * T) * np.maximum(ST_rho - K, 0)
    else:
        payoff_rho = np.exp(-r_up * T) * np.maximum(K - ST_rho, 0)
    rho = (np.mean(payoff_rho) - price) / epsilon

    # Theta
    T_down = T - epsilon
    ST_theta = S0 * np.exp((r - 0.5 * sigma**2) * T_down + sigma * np.sqrt(T_down) * np.random.standard_normal(simulations))
    if option_type == "Call":
        payoff_theta = np.exp(-r * T_down) * np.maximum(ST_theta - K, 0)
    else:
        payoff_theta = np.exp(-r * T_down) * np.maximum(K - ST_theta, 0)
    theta = (np.mean(payoff_theta) - price) / (-epsilon)
    
    # Gamma
    gamma = (np.mean(payoff_up) - 2 * price + np.mean(payoff_down)) / (epsilon ** 2)

    # Display Greeks
    st.subheader("Greeks (Numerical Estimates)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Delta", f"{delta:.4f}")
    col2.metric("Gamma", f"{gamma:.4f}")
    col3.metric("Vega", f"{vega:.4f}")
    col1.metric("Theta", f"{theta:.4f}")
    col2.metric("Rho", f"{rho:.4f}")
    


    # Define grid values
    strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 10)  # 80% to 120% of S0
    T_range = np.linspace(0.01, 1, 10)  # 0.01 to 1 year (approx. 1 to 365 days)
    sigma_range = np.linspace(0.05, 1.0, 10) # Volatility from 5% to 100%
    # Initialize matrix to store option prices
    price_matrix_call = np.zeros((len(T_range), len(strike_range)))
    price_matrix_put = np.zeros((len(T_range), len(strike_range)))
    
    
    for i, T_val in enumerate(T_range):
        for j, K_val in enumerate(strike_range):
            Z = np.random.standard_normal(simulations)
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T_val + sigma * np.sqrt(T_val) * Z)

            call_payoff = np.exp(-r * T_val) * np.maximum(ST - K_val, 0)
            put_payoff = np.exp(-r * T_val) * np.maximum(K_val - ST, 0)
            
            price_matrix_call[i, j] = np.mean(call_payoff)
            price_matrix_put[i, j] = np.mean(put_payoff)
            
    # Plot heatmaps
    # Add a dropdown to select heatmap type
    heatmap_type = st.selectbox("📊 Select Heatmap Type", ["Strike vs Time to Maturity (T)", "Strike vs Volatility (σ)"])
    
    # Define ranges
    strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 10)  # Example strike range
    if heatmap_type == "Strike vs Time to Maturity (T)":
        T_range = np.linspace(1/365, 1, 10)  # Time from 1 day to 1 year
        
        # Initialize matrices
        price_matrix_call = np.zeros((len(T_range), len(strike_range)))
        price_matrix_put = np.zeros((len(T_range), len(strike_range)))
        
        # Populate matrices
        for i, T_ in enumerate(T_range):
            for j, K_ in enumerate(strike_range):
                Z = np.random.standard_normal(10000)
                ST = S0 * np.exp((r - 0.5 * sigma**2) * T_ + sigma * np.sqrt(T_) * Z)
                price_matrix_call[i, j] = np.exp(-r * T_) * np.mean(np.maximum(ST - K_, 0))
                price_matrix_put[i, j] = np.exp(-r * T_) * np.mean(np.maximum(K_ - ST, 0))

    # Show side-by-side heatmaps
        col1, col2 = st.columns(2)
        with col1:
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            sns.heatmap(price_matrix_call, xticklabels=np.round(strike_range, 2), yticklabels=np.round(T_range, 2),
                   annot = True, fmt = ".2f",     cmap="RdYlGn", ax=ax4)
            ax4.set_title("Call Option Price")
            ax4.set_xlabel("Strike Price (K)")
            ax4.set_ylabel("Time to Maturity (T)")
            st.pyplot(fig4)
            
        with col2:
            fig5, ax5 = plt.subplots(figsize=(10, 8))
            sns.heatmap(price_matrix_put, xticklabels=np.round(strike_range, 2), yticklabels=np.round(T_range, 2),
                        annot = True, fmt = ".2f",    cmap="RdYlGn", ax=ax5)
            ax5.set_title("Put Option Price")
            ax5.set_xlabel("Strike Price (K)")
            ax5.set_ylabel("Time to Maturity (T)")
            st.pyplot(fig5)
            
    elif heatmap_type == "Strike vs Volatility (σ)":
        sigma_range  # Volatility from 5% to 100%

    # Initialize matrices
        price_matrix_call = np.zeros((len(sigma_range), len(strike_range)))
        price_matrix_put = np.zeros((len(sigma_range), len(strike_range)))

    # Populate matrices
    for i, sigma in enumerate(sigma_range):
        for j, K_ in enumerate(strike_range):
            Z = np.random.standard_normal(10000)
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            price_matrix_call[i, j] = np.exp(-r * T) * np.mean(np.maximum(ST - K_, 0))
            price_matrix_put[i, j] = np.exp(-r * T) * np.mean(np.maximum(K_ - ST, 0))

    # Show side-by-side heatmaps
    col1, col2 = st.columns(2)
    with col1:
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        sns.heatmap(price_matrix_call, xticklabels=np.round(strike_range, 2), yticklabels=np.round(sigma_range, 2),
                   annot = True, fmt = ".2f", cmap="RdYlGn", ax=ax6)
        ax6.set_title("Call Option Price")
        ax6.set_xlabel("Strike Price (K)")
        ax6.set_ylabel("Volatility (σ)")
        st.pyplot(fig6)

    with col2:
        fig7, ax7 = plt.subplots(figsize=(10, 8))
        sns.heatmap(price_matrix_put, xticklabels=np.round(strike_range, 2), yticklabels=np.round(sigma_range, 2),
                  annot = True, fmt = ".2f",  cmap="RdYlGn", ax=ax7)
        ax7.set_title("Put Option Price")
        ax7.set_xlabel("Strike Price (K)")
        ax7.set_ylabel("Volatility (σ)")
        st.pyplot(fig7)

    
    
    st.info("Any options (strike price) you have bought, your profit will be what you are left with after substracting it's current value (spot price from heatmap) from the cost value(price you have paid when buying) and also you need to remove charges from the profit to get net profit.")
    
        
 

   


    
