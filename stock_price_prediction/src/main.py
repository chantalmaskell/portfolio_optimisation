import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

tickers = ["SPY"]

start = "2020-01-01"
end = "2025-04-05"

data = yf.download(tickers, start, end)
data.to_csv('./SPY.csv', index=False)

x = data[['High', 'Low', 'Open', 'Volume']].values #input
y = data[['Close']].values #goal to predict

#x_train 80%
#x_test 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) # state 0 ensures split can be reproduced

regModel = LinearRegression()
regModel.fit(x_train, y_train)

regModel.coef_