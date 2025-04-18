import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from datetime import datetime

# def get_options_data(ticker):
#     data = yf.Ticker(ticker)
#     options_dates = data.options
#     options_data = ticker.option_chain(options_dates[0])

#     return options_data.calls, options_data.puts

# jpm_calls, jpm_puts = get_options_data('JPM')

# plt.figure(figsize=(10, 5))
# plt.plot(jpm_stock_data['Close'])
# plt.title('JPM Stock Price')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.grid(True)

start_date = "2024-01-01"
end_date = "2025-01-01"

data = yf.download("NVDA", start=start_date, end=end_date)
stock_data = data.to_csv("./stock_data.csv")
stock_data = data

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S # underlying asset price
        self.K = K # strike price
        self.T = T # time to expiration (years)  --- BSM differentiates from other models since it only calculates value of european options (american options must factor in that the option can be exercied before expiration)
        self.r = r # risk-free interest rate
        self.sigma = sigma # volatility of udnerlying asset

    # determines probability adjusted moneyness of the option
    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    # probs of option expiring ITM (normal distribution)
    # d2 is d1 minus volatility over time (what is the chance the option will be excerised)
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    # CDF is Culmative Distribution Function of the std normal distrbiution
    def call_option_price(self):
        return (self.S * si.norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0))
    
    def put_option_price(self):
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.S * si.norm.cdf(-self.d1(), 0.0, 1.0))
    
bsm = BlackScholesModel(S=100, K=100, T=1, r=0.05, sigma=0.2) ## sigma will be replaced here with the volatilty function below
print(f"Call option price: {bsm.call_option_price()}") # 10.450583572185565
print(f"Put option price: {bsm.put_option_price()}") # 5.573526022256971

# to get sigma need to calc standard deviation for historical asset performance

def calculate_volatility(stock_data, window=252): # this is used for estimating sigma from above
    log_rets = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    vol = np.sqrt(window) * log_rets.std()
    return vol

nvda_vol = calculate_volatility(stock_data)
print(f"NVDA volaitlity: {nvda_vol}")

# Greeks measure sensitivity of option price to various factors
# common ones include: Delta, gamma, theta, vega, rho

# delta: measures rate of change of option price with repcet to the underlying price. call options 0-1, put options -1 to 0
# gamma: measures rate of change of delta (with respect ot underlying price).
# theta: measures rate of decline of the options value with respsect to passing of time (time decay). theta is normally negaitve which indicates loss of value as time decays
# vega: sensitivity of the option price to volatility in the underlying price. vega indicates how much the price will change given a 1% change in volatility
# rho: senstiivty of option price to changes in risk free rate. for call options, higher rate normally increadss the options value but put options decreases the value

class Greeks(BlackScholesModel):
    def delta_call(self):
        return si.norm.cdf(self.d1(), 0.0, 1.0)
    
    def delta_put(self):
        return -si.norm.cdf(self.d1(), 0.0, 1.0)
    
    # pdf is probability density function
    def gamma(self):
        return si.norm.pdf(self.d1(), 0.0, 1.0) / self.S * self.sigma * np.sqrt(self.T)
    
    def theta_call(self):
        return (-self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.s2(), 0.0, 1.0))

    def theta_put(self):
        return (-self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.s2(), 0.0, 1.0))
    
    def vega(self):
        return self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * np.sqrt(self.T)
    
    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0)
    
    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0)

greeks = Greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
print(f"delta (call): {greeks.delta_call()}") # 0-1
print(f"delta (put): {greeks.delta_put()}") # -1 - 0

prices = np.linspace(80, 120, 100)
delta_call = [Greeks(S=price, K=100, T=1, r=0.05, sigma=0.2).delta_call() for price in prices]
delta_put = [Greeks(S=price, K=100, T=1, r=0.05, sigma=0.2).delta_put() for price in prices]

def plot_call_delta(delta_call):
    plt.figure(figsize=(10, 5))
    plt.plot(prices, delta_call)
    plt.title("Delta of Call Option as underlying price changes")
    plt.xlabel('Stock Price')
    plt.ylabel('Delta')
    plt.grid(True)
    plt.show()

def plot_call_delta(delta_put):
    plt.figure(figsize=(10, 5))
    plt.plot(prices, delta_put)
    plt.title("Delta of Put Option as underlying price changes")
    plt.xlabel('Stock Price')
    plt.ylabel('Delta')
    plt.grid(True)
    plt.show()

# gamma: measures rate of change of delta (with respect ot underlying price).
# def plot_gamma():
#     gamma = Greeks()

def plot_option_sensitivity(bsm, parameter, values, option_type='call'):
    prices = []
    for value in values:
        setattr(bsm, parameter, value)

        if option_type == 'call':
            prices.append(bsm.call_option_price())
        else:
            prices.append(bsm.put_option_price())
    
    plt.figure(figsize=(10, 5))
    plt.plot(values, prices)
    plt.title(f'Sensitivity to {parameter.capitalize()}')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Option price')
    plt.grid(True)
    plt.show()

volatilities = np.linspace(0.1, 0.3, 100)
plot_option_sensitivity(bsm, 'sigma', volatilities, 'call')