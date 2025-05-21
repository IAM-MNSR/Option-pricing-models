# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:49:30 2025

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
        option_type = st.radio("Option Type (Override)", ("Call", "Put"), index=0 if option_type == "Call" else 1)
        
        
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
            annot = True, fmt = ".2f",        cmap="RdYlGn", ax=ax4)
        ax4.set_title("Call Option Price")
        ax4.set_xlabel("Strike Price (K)")
        ax4.set_ylabel("Time to Maturity (T)")
        st.pyplot(fig4)
        
    with col2:
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        sns.heatmap(price_matrix_put, xticklabels=np.round(strike_range, 2), yticklabels=np.round(T_range, 2),
                    annot = True, fmt = ".2f",     cmap="RdYlGn", ax=ax5)
        ax5.set_title("Put Option Price")
        ax5.set_xlabel("Strike Price (K)")
        ax5.set_ylabel("Time to Maturity (T)")
        st.pyplot(fig5)
        
elif heatmap_type == "Strike vs Volatility (σ)":
    sigma_range # Volatility from 5% to 100%

# Initialize matrices
    price_matrix_call = np.zeros((len(sigma_range), len(strike_range)))
    price_matrix_put = np.zeros((len(sigma_range), len(strike_range)))

# Populate matrices
for i, sigma_ in enumerate(sigma_range):
    for j, K_ in enumerate(strike_range):
        Z = np.random.standard_normal(10000)
        ST = S0 * np.exp((r - 0.5 * sigma_**2) * T + sigma_ * np.sqrt(T) * Z)
        price_matrix_call[i, j] = np.exp(-r * T) * np.mean(np.maximum(ST - K_, 0))
        price_matrix_put[i, j] = np.exp(-r * T) * np.mean(np.maximum(K_ - ST, 0))

# Show side-by-side heatmaps
col1, col2 = st.columns(2)
with col1:
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    sns.heatmap(price_matrix_call, xticklabels=np.round(strike_range, 2), yticklabels=np.round(sigma_range, 2),
            annot = True, fmt = ".2f",     cmap="RdYlGn", ax=ax6)
    ax6.set_title("Call Option Price")
    ax6.set_xlabel("Strike Price (K)")
    ax6.set_ylabel("Volatility (σ)")
    st.pyplot(fig6)

with col2:
    fig7, ax7 = plt.subplots(figsize=(6, 5))
    sns.heatmap(price_matrix_put, xticklabels=np.round(strike_range, 2), yticklabels=np.round(sigma_range, 2),
             annot = True, fmt = ".2f",    cmap="RdYlGn", ax=ax7)
    ax7.set_title("Put Option Price")
    ax7.set_xlabel("Strike Price (K)")
    ax7.set_ylabel("Volatility (σ)")
    st.pyplot(fig7)