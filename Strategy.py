from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import Counter
import numpy as np
import pytz
from datetime import datetime
from zipline.api import order, symbol, record, order_target, history
'''
order() takes 2 arguments : i) a security object ii) no. of shares you want to buy
record() helps you to save values of variables at each iteration
'''
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo
#from zipline import run_algorithm
import matplotlib.pyplot as plt

###------------------------------------------------------------------Packages Imported--------------------------
#assign_stock = 'SPY'
assign_stock = 'XLB'
#assign_stock = 'BHANDHOS.BO'
#assign_stock = 'AAPL'
#assign_stock = 'MSFT'
#assign_stock = 'WIPRO.BO'
start = datetime(2014,1,1, 0, 0, 0, 0, pytz.utc).date()
end = datetime(2016,1,1,0,0,0,0, pytz.utc).date()
#data = load_bars_from_yahoo(stocks=['SPY'], start=start,end=end)
data = load_bars_from_yahoo(stocks=[assign_stock], start=start,end=end)


def initialize(context):
    #context.stocks = symbols('XLY',  # XLY Consumer Discrectionary SPDR Fund   
    #                       'XLF',  # XLF Financial SPDR Fund  
    #                       'XLK',  # XLK Technology SPDR Fund  
    #                       'XLE',  # XLE Energy SPDR Fund  
    #                       'XLV',  # XLV Health Care SPRD Fund  
    #                       'XLI',  # XLI Industrial SPDR Fund  
    #                       'XLP',  # XLP Consumer Staples SPDR Fund   
    #                       'XLB',  # XLB Materials SPDR Fund  
    #                       'XLU')  # XLU Utilities SPRD Fund
    #context.stocks = symbol('SPY')
    context.stocks = symbol('XLB')
    context.historical_bars = 100
    context.feature_window = 10
    

   

def handle_data(context, data):
    prices = history(bar_count = context.historical_bars, frequency='1d', field='price')
    print prices
    ##Carrying last 100d prices
    for stock in context.stocks:
        try:
            ma1 = data[stock].mavg(50)
            ma2 = data[stock].mavg(200)
            start_bar = context.feature_window
            price_list = prices[stock].tolist()
            X = []
            y = []
            bar = start_bar
            # feature creation
            while bar < len(price_list)-1:
                try:
                    end_price = price_list[bar+1]
                    begin_price = price_list[bar]
                    pricing_list = []
                    xx = 0
                    for _ in range(context.feature_window):
                        price = price_list[bar-(context.feature_window-xx)]
                        pricing_list.append(price)
                        xx += 1
                    features = np.around(np.diff(pricing_list) / pricing_list[:-1] * 100.0, 1)
                    #print(features)
                    if end_price > begin_price:
                        label = 1
                    else:
                        label = -1
                    bar += 1
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    bar += 1
                    print(('feature creation',str(e)))
            clf = RandomForestClassifier()
            last_prices = price_list[-context.feature_window:]
            current_features = np.around(np.diff(last_prices) / last_prices[:-1] * 100.0, 1)
            X.append(current_features)
            X = preprocessing.scale(X)
            current_features = X[-1]
            X = X[:-1]
            clf.fit(X,y)
            p = clf.predict(current_features)[0]
            print(('Prediction',p))
        except Exception as e:
            print(str(e))
        



algo_obj = TradingAlgorithm(initialize=initialize,handle_data=handle_data,capital_base=80000)
#By defualt capital = 100000
perf_manual = algo_obj.run(data)    
#perf_manual[["MA1","MA2","Price"]].plot()

#####----------------------------------------------------END-------------------------
'''
bar=10
price_list = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]
pricing_list = []
xx = 0
for _ in range(10):
    price = price_list[bar-(10-xx)]
    pricing_list.append(price)
    xx += 1
features = np.around(np.diff(pricing_list) / pricing_list[:-1] * 100.0, 1)

def handle_data(context, data):
    prices = history(bar_count = context.historical_bars, frequency='1d', field='open')
    print prices




algo_obj = TradingAlgorithm(initialize=initialize,handle_data=handle_data,capital_base=80000)
#By defualt capital = 100000
perf_manual = algo_obj.run(data)    



