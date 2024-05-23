



import requests
import pandas as pd
import urllib.parse
import requests

apiKey = '602681ca-40cc-4eae-b759-fc123f3a0b9f'
secretKey = 'mlr8pci82k'
redirectUrl = 'https://127.0.0.1:5000/'
rurl = urllib.parse.quote(redirectUrl,safe="")
url = 'https://api-v2.upstox.com/login/authorization/token'


uri = f'https://api-v2.upstox.com/login/authorization/dialog?response_type=code&client_id={apiKey}&redirect_uri={rurl}'
print(uri)
# /--------------=-----------------------------------
code  = 'VWyp53'
headers = {
    'accept': 'application/json',
    'Api-Version': '2.0',
    'Content-Type': 'application/x-www-form-urlencoded'
}

data = {
    'code': code,
    'client_id': apiKey,
    'client_secret': secretKey,
    'redirect_uri': redirectUrl,
    'grant_type': 'authorization_code'
}

response = requests.post(url, headers=headers, data=data)
json_response = response.json()
print(json_response)

def get_historical_data(instrument='NSE_INDEX|Nifty 50', interval=None, from_date=None, to_date=None):
    if not (interval and from_date and to_date):
        url = f'https://api-v2.upstox.com/historical-candle/intraday/{instrument}/1minute' 
        headers = {
            'accept': 'application/json',
            'Api-Version': '2.0',
        }
        
        response = make_request('GET', url, headers=headers)

        candle_data = response["data"]["candles"]

        # Define column names
        columns = ["timestamp", "open", "high", "low", "close", "volume", "Open Interest"]

        # Create DataFrame
        df = pd.DataFrame(candle_data, columns=columns)

        # Convert "Timestamp" column to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.iloc[::-1]

        # df.to_csv("intraday_data.csv")

        return df
    else:
        headers = {
            'accept': 'application/json',
            'Api-Version': '2.0',
        }
        url = f'https://api-v2.upstox.com/historical-candle/{instrument}/{interval}/{to_date}/{from_date}'
        # url = f'https://api-v2.upstox.com/historical-candle/intraday/:{instrument}/:{interval}/:{to_date}/:{from_date}'

        response = make_request('GET', url, headers=headers)
        print("......")
        # print(response)
        

        candle_data = response["data"]["candles"]

        # Define column names
        columns = ["timestamp", "open", "high", "low", "close", "volume", "Open Interest"]

        # Create DataFrame
        df = pd.DataFrame(candle_data, columns=columns)

        # Convert "Timestamp" column to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.iloc[::-1]
        df.to_csv("intraday_data.csv")        
        return df

def make_request(method, url, headers=None, params=None, data=None):
    response = None

    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif method == 'POST':
            response = requests.post(url, headers=headers, params=params, json=data)
        elif method == 'PUT':
            response = requests.put(url, headers=headers, params=params, json=data)
        else:
            raise ValueError('Invalid HTTP method.')

        if response.status_code == 200:
           
            return response.json()
        else:
            
            return response

    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')
        return None
  

# api_version = # str | API Version Header
capital = 1000
data = (get_historical_data(interval='day', from_date='1993-03-01', to_date='2024-05-09'))



def get_AAPL_data():
    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2021-10-01'
    TEST_START_DATE = '2021-10-01'
    TEST_END_DATE = '2023-03-01'

    df = YahooDownloader(start_date = TRAIN_START_DATE,
                        end_date = TEST_END_DATE,
                        ticker_list = ["AAPL"]).fetch_data()

    df['date'] = pd.to_datetime(df['date'])

    df = df[df['tic'] == "AAPL"]
    return df


def get_NIFTY_data():
    df = pd.read_csv("nifty_intraday_data.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['date'] = pd.to_datetime(df['timestamp'])
    df = df.drop(['timestamp', 'Open Interest'], axis=1)
    return df




def get_NEPSE_data():
    df = pd.read_csv("nepsealpha_export_price_NEPSE_2019-05-12_2024-05-12.csv")
    df.columns = map(str.lower, df.columns)
    df.drop(columns=["symbol", "percent change"], inplace=True)
    return df