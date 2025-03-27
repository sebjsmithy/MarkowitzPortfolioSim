
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.optimize import minimize

print("Let's begin!")
tickers = ['AVGO', 'AAPL', 'CSCO', 'INTU', 'ORCL', "QCOM"]
end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=(365*6))

#Get all our data into a pandas DataFrame so we can manipulate it appropriately
data = yf.download(tickers, start=start_date, end=end_date)["Close"]
stock_data = pd.DataFrame(data)

#Normalise the data using log. We convert into returns. Printed to show the size of the DataFrame. 1508 rows,
# one for each trading day, and 6 columns for each stock.
log_returns = np.log(stock_data/stock_data.shift(1))
print(log_returns)

#We begin to simulate our portfolios, using a seed for the sake of reproducibility. 10,000 simulations.
np.random.seed(43)
num_of_portfolios = 10000
all_weights = np.zeros((num_of_portfolios, len(stock_data.columns)))
return_arr = np.zeros(num_of_portfolios)
vol_arr = np.zeros(num_of_portfolios)
sharpe_arr = np.zeros(num_of_portfolios)

for x in range(num_of_portfolios):
    weights = np.array(np.random.random(len(stock_data.columns)))
    weights = weights/np.sum(weights)
    all_weights[x,:] = weights

    #We use 1507 since that's the number of rows/trading days, .mean() to find the average return of each stock over
    #the time period, multiply that by the random weight, and then sum it all up to get the total portfolio return
    return_arr[x] = np.sum( (log_returns.mean() * weights * 1507 ))

    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*1507, weights)))

    # Sharpe Ratio
    sharpe_arr[x] = return_arr[x]/vol_arr[x]



print("Max sharpe ratio in the array: {}".format(sharpe_arr.max()))
print("Its location in the array: {}".format(sharpe_arr.argmax()))

print(all_weights[3449, :])
max_sharpe_ratio_return = return_arr[sharpe_arr.argmax()]
max_sharpe_ratio_vol = vol_arr[sharpe_arr.argmax()]
max_sharpe_ratio_weights = all_weights[sharpe_arr.argmax()]

#Now we can start writing code to plot the red outline of the frontier
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_returns.mean() * weights) * 1507
    vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*1507, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

def neg_sharpe(weights):
# the number 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    #return 0 if sum of the weights is 1
    return np.sum(weights)-1

cons = ({'type': 'eq', 'fun': check_sum})
bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
init_guess = [0.15,0.15,0.15,0.15, 0.15, 0.15]

opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
print(opt_results)

opt_weights = opt_results.x
print(get_ret_vol_sr(opt_weights))

frontier_y = np.linspace(-0.5, 3.0, 200)
def minimise_volatility(weights):
    return get_ret_vol_sr(weights)[1]


frontier_x = []

for possible_return in frontier_y:
    cons = ({'type': 'eq', 'fun': check_sum},
            {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})

    result = minimize(minimise_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])

plt.figure(figsize=(12,8))
plt.scatter(vol_arr, return_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sharpe_ratio_vol, max_sharpe_ratio_return,c='red', s=50) # red dot
plt.plot(frontier_x,frontier_y, 'r--', linewidth=3)
plt.savefig('cover.png')
plt.show()

# testing again. adding notes!