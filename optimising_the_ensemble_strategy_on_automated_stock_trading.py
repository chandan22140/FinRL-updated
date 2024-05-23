
# # ## install finrl library
# !pip install wrds
# !pip install swig
# !pip install -q condacolab
# import condacolab
# condacolab.install()
# !apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
# !pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
# !pip install ta
# !pip install pandas_ta
# !pip install TA-Lib
# import talib

import warnings
warnings.filterwarnings("ignore")

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
from sklearn.ensemble import RandomForestRegressor

# %matplotlib inline
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import pandas_ta as ta
import pandas as pd


from pprint import pprint

import sys
sys.path.append("../FinRL-Library")

import itertools

"""<a id='1.4'></a>
## 2.4. Create Folders
"""

import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

"""<a id='2'></a>
# Part 3. Download Data
Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.
* FinRL uses a class **YahooDownloader** to fetch data from Yahoo Finance API
* Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).

-----
class YahooDownloader:
    Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API
"""

print(DOW_30_TICKER)

# TRAIN_START_DATE = '2009-04-01'
# TRAIN_END_DATE = '2021-01-01'
# TEST_START_DATE = '2021-01-01'
# TEST_END_DATE = '2022-06-01'

TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TEST_START_DATE = '2021-10-01'
TEST_END_DATE = '2023-03-01'

df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TEST_END_DATE,
                     ticker_list = DOW_30_TICKER).fetch_data()

df['date'] = pd.to_datetime(df['date'])

df = df[df['tic'] == "AAPL"]

filtered_df = df.copy()
filtered_df = filtered_df.fillna(0)
filtered_df = filtered_df.replace(np.inf,0)
filtered_df.sort_values(['date','tic']).head()
filtered_df.set_index('date', inplace=True)  # Assuming 'date' is your datetime column

if __name__=="main":
    filtered_df.ta.strategy("all")

filtered_df.shape

from ta import add_all_ta_features
filtered_df2 = df.copy()

filtered_df2.set_index('date', inplace=True)  # Assuming 'date' is your datetime column
filtered_df2 = add_all_ta_features(
    filtered_df2, open="open", high="high", low="low", close="close", volume="volume")
filtered_df2 = filtered_df2.drop(["open", "high", "low", "close", "volume", "tic", "day"], axis=1)

filtered_df2.shape

processed = pd.concat([filtered_df, filtered_df2], axis=1, join='inner')
processed

"""# Part 4: Preprocess Data
Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
* Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
* Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.
"""

# ensure string date, and indexing is natural(1, 2, 3,4, 5, )
filtered_df3 = df.copy()
filtered_df3['date'] = filtered_df3['date'].astype(str)

print(type(filtered_df3['date'][0]))
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_turbulence=True,
                     use_vix=False,
                     user_defined_feature = True)
filtered_df3.head()
filtered_df3.set_index('date', inplace=True)  # Assuming 'date' is your datetime column
filtered_df3.reset_index(inplace=True)

filtered_df3 = fe.preprocess_data(filtered_df3)


# -----------custom_indicators_start-----------------------------------




# Compute 52 Week High/Low

def sma(price, period):
    sma = price.rolling(period).mean()
    return sma

def ao(price, period1, period2):
    median = price.rolling(2).median()
    short = sma(median, period1)
    long = sma(median, period2)
    ao = short - long
    ao_df = pd.DataFrame(ao).rename(columns = {'Close':'ao'})
    return ao_df




def advance_decline(df):
    ADVN = df['close'].diff().apply(lambda x: 1 if x > 0 else 0).cumsum()
    DECN = df['close'].diff().apply(lambda x: 1 if x < 0 else 0).cumsum()
    UNCN = len(df) - ADVN - DECN

    # Calculate difference
    difference = (ADVN - DECN) / (UNCN + 1)

    # Calculate ADL
    adline = np.cumsum(np.where(difference > 0, np.sqrt(difference), -np.sqrt(-difference)))

    # Add ADL column to DataFrame
    df['ADL'] = adline
    return df


# Function to determine time step
def timeStep(tf_in_seconds):
    tf_in_ms = tf_in_seconds * 1000
    step = np.select(
        [
            tf_in_ms <= MS_IN_MIN,
            tf_in_ms <= MS_IN_MIN * 5,
            tf_in_ms <= MS_IN_HOUR,
            tf_in_ms <= MS_IN_HOUR * 4,
            tf_in_ms <= MS_IN_HOUR * 12,
            tf_in_ms <= MS_IN_DAY,
            tf_in_ms <= MS_IN_DAY * 7
        ],
        [
            MS_IN_HOUR,
            MS_IN_HOUR * 4,
            MS_IN_DAY,
            MS_IN_DAY * 3,
            MS_IN_DAY * 7,
            MS_IN_DAY * 30.4375,
            MS_IN_DAY * 90
        ],
        default=MS_IN_DAY * 365
    )
    return int(step)

# Function to calculate Rolling VWAP
def rvwap(_src, fixed_tf_input=False, mins_input=0, hours_input=0, days_input=1, min_bars_input=10):
    if np.all(np.cumsum(_src.volume) == 0):
        raise ValueError("No volume is provided by the data vendor.")

    _days_input = days_input * MS_IN_MIN
    _hours_input = hours_input * MS_IN_HOUR
    _mins_input = mins_input * MS_IN_DAY

    # RVWAP + stdev bands
    time_in_ms = _mins_input + _hours_input + _days_input if fixed_tf_input else timeStep(timeframe.period)
    sum_src_vol = pc.totalForTimeWhen(_src * _src.volume, time_in_ms, True, min_bars_input)
    sum_vol = pc.totalForTimeWhen(_src.volume, time_in_ms, True, min_bars_input)
    sum_src_src_vol = pc.totalForTimeWhen(_src.volume * np.power(_src, 2), time_in_ms, True, min_bars_input)

    rolling_vwap = sum_src_vol / sum_vol

    return rolling_vwap

# Function to calculate Correlation Moving Average
def correlation_ma(src, len, factor=1.7):
    ma = src.rolling(window=len).mean() + src.rolling(window=len).corr(src.index, pairwise=True) * src.rolling(window=len).std() * factor
    return ma

# Function to calculate Regularized Exponential Moving Average
def reg_ma(src, len, lambda_val=0.5):
    alpha = 2 / (len + 1)
    ma = np.zeros_like(src)
    for i in range(1, len(src)):
        ma[i] = (ma[i - 1] + alpha * (src[i] - ma[i - 1]) + lambda_val * (2 * ma[i - 1] - ma[i - 2])) / (lambda_val + 1)
    return ma

# Function to calculate Repulsion Moving Average
def rep_ma(src, len):
    return src.rolling(window=len * 3).mean() + src.rolling(window=len * 2).mean() - src.rolling(window=len).mean()



# Function to calculate End Point Moving Average
def epma(src, length, offset=4):
    sum_val = 0.0
    weight_sum = 0.0
    for i in range(length):
        weight = length - i - offset
        sum_val += src.iloc[i] * weight
        weight_sum += weight
    return sum_val / weight_sum



# Function to calculate 1LC-LSMA (1 line code lsma with 3 functions)
def lc_lsma(src, length):
    return src.rolling(window=length).mean() + src.rolling(window=length).corr(src.index, pairwise=True) * src.rolling(window=length).std() * 1.7

# -----------adding columns-------------------------------
MS_IN_MIN = 60 * 1000
MS_IN_HOUR = MS_IN_MIN * 60
MS_IN_DAY = MS_IN_HOUR * 24


filtered_df3['260candle High'] = filtered_df3['high'].rolling(window=52*5).max()
filtered_df3['260candle Low'] = filtered_df3['low'].rolling(window=52*5).min()
filtered_df3['ao'] = ao(filtered_df3['close'], 5, 34)
# Compute Advance/Decline
# This is a custom calculation based on the number of advancing and declining stocks in an index or market.
filtered_df3 = advance_decline(filtered_df3)

# Compute Arnaud Legoux Moving Average
filtered_df3['ALMA_'] = filtered_df3['close'].rolling(window=9).mean()




# filtered_df3['STOCH_'] = ta.stoch(filtered_df3['high'], filtered_df3['low'], filtered_df3['close'])
# filtered_df3['PPO_'] = ta.ppo(filtered_df3['close'], fast=9, slow=26, signal=21)

# filtered_df3['KST_'] = ta.kst(filtered_df3['close'], rma1=10, rma2=15, rma3=20, rma4=30, signal=9)

# Compute Volume Indicators
filtered_df3['CMF_'] = ta.cmf(filtered_df3['high'], filtered_df3['low'], filtered_df3['close'], filtered_df3['volume'], length=20)

# Compute Volatility Indicators
filtered_df3['NATR_'] = ta.natr(filtered_df3['high'], filtered_df3['low'], filtered_df3['close'], length=14)
# filtered_df3['CHAIKINVOL_'] = ta.chaikinvol(filtered_df3['high'], filtered_df3['low'], filtered_df3['close'], filtered_df3['volume'], length=20)

# Compute Cycle Indicators
# filtered_df3['HT_TRENDLINE_'] = ta.ht_trendline(filtered_df3['close'])
filtered_df3['VIDYA_'] = ta.vidya(filtered_df3['close'], length=14, scalar=9, drift=1)



# -----------custom_indicators_end-----------------------------------



























filtered_df3 = filtered_df3.drop(["open", "high", "low", "close", "volume", "tic", "day"], axis=1)
print(type(filtered_df3['date'][0]))
filtered_df3['date'] = pd.to_datetime(filtered_df3['date'])
print(type(filtered_df3['date'][0]))

filtered_df3.set_index('date', inplace=True)

processed_final = pd.concat([processed, filtered_df3], axis=1, join='inner')
processed_final.shape

processed_final = processed_final.copy()
processed_final = processed_final.fillna(0)
processed_final = processed_final.replace(np.inf,0)
(processed_final.columns)

processed_final.to_csv("processed_final.csv")
"""<a id='4'></a>
# Part 5. Design Environment
Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.

Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.

The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.
"""

stock_dimension = 1



INDICATORS = list(processed_final.columns)
INDICATORS.remove("open")
INDICATORS.remove("high")
INDICATORS.remove("low")
INDICATORS.remove("close")
INDICATORS.remove("volume")
INDICATORS.remove("tic")
INDICATORS.remove("day")

state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity":5
}

processed_final.reset_index(inplace=True)



# -----------------------------------------------------------------------------------------------------------------
"""<a id='5'></a>
# Part 6: Implement DRL Algorithms
* The implementation of the DRL algorithms are based on **OpenAI Baselines** and **Stable Baselines**. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.
* FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
design their own DRL algorithms by adapting these DRL algorithms.

* In this notebook, we are training and validating 3 agents (A2C, PPO, DDPG) using Rolling-window Ensemble Method ([reference code](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/80415db8fa7b2179df6bd7e81ce4fe8dbf913806/model/models.py#L92))
"""

rebalance_window = 63 # rebalance_window is the number of days to retrain the model
validation_window = 63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

ensemble_agent = DRLEnsembleAgent(df=processed_final,
                 train_period=(TRAIN_START_DATE,TRAIN_END_DATE),
                 val_test_period=(TEST_START_DATE,TEST_END_DATE),
                 rebalance_window=rebalance_window,
                 validation_window=validation_window,
                 **env_kwargs)

A2C_model_kwargs = {
                    'n_steps': 5,
                    'ent_coef': 0.005,
                    'learning_rate': 0.0007
                    }

PPO_model_kwargs = {
                    "ent_coef":0.01,
                    "n_steps": 2048,
                    "learning_rate": 0.00025,
                    "batch_size": 128
                    }
PPO2_PARAMS = {
                    "ent_coef":0.02,
                    "n_steps": 2048,
                    "learning_rate": 0.00015,
                    "batch_size": 256
                    }


DDPG_model_kwargs = {
                      #"action_noise":"ornstein_uhlenbeck",
                      "buffer_size": 10_000,
                      "learning_rate": 0.0005,
                      "batch_size": 64
                    }
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.0001}

# DQN_PARAMS = {
#     "batch_size": 32,
#     "buffer_size": 1000000,
#     "learning_rate": 1e-4,
#     "learning_starts": 100, #how many steps of the model to collect transitions for before learning starts
# }

timesteps_dict = {'a2c' : 10_000,
                 'ppo' : 10_000,
                 'ppo2' : 10_000,
                 'ddpg' : 10_000,
                  'td3':10_000,
                  'sac':10_000
                 }

# df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
#                                                  PPO_model_kwargs,
#                                                   DDPG_model_kwargs,
#                                                   SAC_PARAMS,
#                                                   TD3_PARAMS,
#                                                   DQN_PARAMS,
#                                                  timesteps_dict)

df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                 PPO_model_kwargs,
                                                  DDPG_model_kwargs,
                                                  SAC_PARAMS,
                                                  TD3_PARAMS,
                                                  PPO2_PARAMS,
                                                 timesteps_dict)
# df_summary = ensemble_agent.run_ensemble_strategy(None,
#                                                  None,
#                                                   None,
#                                                   None,
#                                                   None,
#                                                   PPO2_PARAMS,
#                                                  timesteps_dict)

processed = processed_final

unique_trade_date = processed[(processed.date > TEST_START_DATE)&(processed.date <= TEST_END_DATE)].date.unique()

df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

df_account_value=pd.DataFrame()
for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
    temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble',i))
    df_account_value = df_account_value._append(temp,ignore_index=True)
sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
print('Sharpe Ratio: ',sharpe)
df_account_value=df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))

df_account_value

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
df_account_value.account_value.plot()

"""<a id='6.1'></a>
## 7.1 BackTestStats
pass in df_account_value, this information is stored in env class

"""

print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)

#baseline stats
print("==============Get Baseline Stats===========")
df_dji_ = get_baseline(
        ticker="^DJI",
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

stats = backtest_stats(df_dji_, value_col_name = 'close')

df_dji = pd.DataFrame()
df_dji['date'] = df_account_value['date']
df_dji['dji'] = df_dji_['close'] / df_dji_['close'][0] * env_kwargs["initial_amount"]
print("df_dji: ", df_dji)
df_dji.to_csv("df_dji.csv")
df_dji = df_dji.set_index(df_dji.columns[0])
print("df_dji: ", df_dji)
df_dji.to_csv("df_dji+.csv")

df_account_value.to_csv('df_account_value.csv')

"""<a id='6.2'></a>
## 7.2 BackTestPlot
"""

# Commented out IPython magic to ensure Python compatibility.


# print("==============Compare to DJIA===========")
# %matplotlib inline
# # S&P 500: ^GSPC
# # Dow Jones Index: ^DJI
# # NASDAQ 100: ^NDX
# backtest_plot(df_account_value,
#               baseline_ticker = '^DJI',
#               baseline_start = df_account_value.loc[0,'date'],
#               baseline_end = df_account_value.loc[len(df_account_value)-1,'date'])
df.to_csv("df.csv")
df_result_ensemble = pd.DataFrame({'date': df_account_value['date'], 'ensemble': df_account_value['account_value']})
df_result_ensemble = df_result_ensemble.set_index('date')

print("df_result_ensemble.columns: ", df_result_ensemble.columns)

# df_result_ensemble.drop(df_result_ensemble.columns[0], axis = 1)
print("df_trade_date: ", df_trade_date)
# df_result_ensemble['date'] = df_trade_date['datadate']
# df_result_ensemble['account_value'] = df_account_value['account_value']
df_result_ensemble.to_csv("df_result_ensemble.csv")
print("df_result_ensemble: ", df_result_ensemble)
print("==============Compare to DJIA===========")
result = pd.DataFrame()
# result = pd.merge(result, df_result_ensemble, left_index=True, right_index=True)
# result = pd.merge(result, df_dji, left_index=True, right_index=True)
result = pd.merge(df_result_ensemble, df_dji, left_index=True, right_index=True)
print("result: ", result)
result.to_csv("result.csv")
result.columns = ['ensemble', 'dji']

# %matplotlib inline
plt.rcParams["figure.figsize"] = (15,5)
plt.figure();
result.plot();

# -----------------------------------------------------------------------------------------------------------------
processed_final = processed_final.copy()

y = processed_final['close']
X = processed_final.drop('close', axis=1).drop('date', axis=1).drop('tic', axis=1)
processed_final['Increase_Decrease'] = np.where(processed_final['volume'].shift(-1) > processed_final['volume'],1,0)
processed_final['Buy_Sell_on_Open'] = np.where(processed_final['open'].shift(-1) > processed_final['open'],1,0)
processed_final['Buy_Sell'] = np.where(processed_final['close'].shift(-1) > processed_final['close'],1,0)
processed_final['Returns'] = processed_final['close'].pct_change()
processed_final = processed_final.fillna(0)

# Create decision tree classifer object
clf = RandomForestRegressor(random_state=0, n_jobs=-1)

# Train model
model = clf.fit(X, y)

# Calculate feature importances
importances = model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [processed_final.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=90)

# Show plot
plt.show()

# -----------------------------------------------------------------------------------------------------------------
