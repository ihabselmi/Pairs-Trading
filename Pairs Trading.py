#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
# just set the seed for the random number generator
np.random.seed(107)

import matplotlib.pyplot as plt
np.


# In[17]:


X_returns = np.random.normal(0,1,100) # Generate daily returns
# sum them and shift all the prices up into a reasonable range
X = pd.Series(np.cumsum(X_returns), name = 'X') + 50
X.plot()


# In[18]:


# Now we generate Y. Y has a deep economic sense related to X. Vary similarly
some_noise = np.random.exponential(1,100)
Y = X + 5 + some_noise
Y.name = 'Y'
pd.concat([X,Y], axis =1).plot()


# In[19]:


# Cointegration when the difference between two process is mean reverting

(Y - X).plot() # plot the spread
plt.axhline((Y - X).mean(), color = 'red', linestyle = '--') # Add the mean
plt.xlabel('Time')
plt.legend(['Price Spread', 'Mean'])


# In[20]:


# Testing for cointegration
# Compute the p-value of the cointegration test
# will inform us as to whether the spread between the T TS is stationary
# around its mean
score, pvalue, _ = coint(X,Y)
if pvalue < 0.05:
    print ("Likely cointegrated")
else:
    print ("Likely not cointegrated")


# In[21]:


def find_cointegrated_pairs(data):
    n = data.shape[1]
    #score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            #S1 = data[keys[i]]
            #S2 = data[keys[j]]
            result = coint(data[keys[i]], data[keys[j]])
            #score = result[0]
            #pvalue = result[1]
            #score_matrix[i, j] = score
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.02:
                pairs.append((keys[i], keys[j]))
    #return score_matrix, pvalue_matrix, pairs
    return pvalue_matrix, pairs


# In[30]:


# Import DataReader
from pandas_datareader.data import DataReader

# Import date
from datetime import date

# Set start and end dates
start = date(2016, 1, 1)
end = date(2016, 12, 31)

# Set the ticker
ticker = ['SPY','AAPL','ADBE','LUV','MSFT','SKYW','QCOM', 'HPQ','JNPR','AMD','IBM', 'NVDA']

# Set the data source
data_source = 'yahoo'

# Import the stock prices
stock_prices = DataReader(ticker, data_source, start, end)

# Display and inspect the result
print(stock_prices.head())
stock_prices.info()


# In[39]:


import pandas as pd 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.max_columns', 500) 
pd.set_option('display.width', 1000) 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint 
import seaborn
from pandas_datareader import data



#symbolsIds = ['SPY','AAPL','ADBE','LUV','MSFT','SKYW','QCOM', 'HPQ','JNPR','AMD','IBM']
symbolsIds2 = ['SPY','AAPL','ADBE','LUV','MSFT','SKYW','QCOM', 'HPQ','JNPR','AMD','IBM', 'NVDA']
def load_financial_data(symbols, start_date, end_date,output_file):
    #try:
     #   df = pd.read_pickle(output_file)
      #  print('File data found...reading symbols data')
    #except FileNotFoundError:
     #   print('File not found...downloading the symbols data') 
    df = data.DataReader(symbols, 'yahoo', start_date, end_date)
    df.to_pickle(output_file)
    return df
#data=load_financial_data(symbolsIds,start_date='2018-12-12', end_date = '2019-12-12',output_file='multi_data_large.pkl')
data2=load_financial_data(symbolsIds2,start_date=date(2020,2,18), end_date = date(2020,2,19),output_file='multi_data_large.pkl')


# In[40]:


data2.head(3)


# In[113]:


pvalues, pairs = find_cointegrated_pairs(data['Adj Close'])
print(pvalues)
print(pairs)


# In[112]:


seaborn.heatmap(pvalues, xticklabels=symbolsIds, yticklabels=symbolsIds, cmap='RdYlGn_r',
                      mask = (pvalues >= 0.98))
plt.show()


# In[43]:


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint 
import matplotlib.pyplot as plt


# In[44]:


# Set a seed value to make the experience reproducible
np.random.seed(123)
# Generate Symbol1 daily returns
Symbol1_returns = np.random.normal(0, 1, 100)
# Create a series for Symbol1 prices
Symbol1_prices = pd.Series(np.cumsum(Symbol1_returns), name='Symbol1') + 10
Symbol1_prices.plot(figsize=(15,7))
plt.show()



# In[45]:


# Create a series for Symbol2 prices
# We will copy the Symbol1 behavior
noise = np.random.normal(0, 1, 100) 
Symbol2_prices = Symbol1_prices + 10 + noise 
Symbol2_prices.name = 'Symbol2' 
plt.title("Symbol 1 and Symbol 2 prices") 
Symbol1_prices.plot()
Symbol2_prices.plot()
plt.show()


# In[46]:


score, pvalue, _ = coint(Symbol1_prices, Symbol2_prices)


# In[47]:


def zscore(series):
    return (series - series.mean()) / np.std(series)


# In[48]:


ratios = Symbol1_prices / Symbol2_prices
ratios.plot()


# In[53]:


train = ratios[:75]
test = ratios[75:]
plt.axhline(ratios.mean()) 
plt.legend([' Ratio']) 
plt.show()
zscore(ratios).plot() 
plt.axhline(zscore(ratios).mean(),color="black") 
plt.axhline(1.0, color="red")
plt.axhline(-1.0, color="green")
plt.show()


# In[54]:


ratios.plot()
buy = ratios.copy()
sell = ratios.copy()
buy[zscore(ratios)>-1] = 0
sell[zscore(ratios)<1] = 0
buy.plot(color="g", linestyle="None", marker="^") 
sell.plot(color="r", linestyle="None", marker="v") 
x1,x2,y1,y2 = plt.axis() 
plt.axis((x1,x2,ratios.min(),ratios.max())) 
plt.legend(["Ratio", "Buy Signal", "Sell Signal"]) 
plt.show()


# In[56]:


Symbol1_prices.plot()
symbol1_buy[zscore(ratios)>-1] = 0 
symbol1_sell[zscore(ratios)<1] = 0 
symbol1_buy.plot(color="g", linestyle="None", marker="^") 
symbol1_sell.plot(color="r", linestyle="None", marker="v")
Symbol2_prices.plot()
symbol2_buy[zscore(ratios)<1] = 0 
symbol2_sell[zscore(ratios)>-1] = 0 
symbol2_buy.plot(color="g", linestyle="None", marker="^")
symbol2_sell.plot(color="r", linestyle="None", marker="v")
x1,x2,y1,y2 = plt.axis() 
plt.axis((x1,x2,Symbol1_prices.min(),Symbol2_prices.max())) 
plt.legend(["Symbol1", "Buy Signal", "Sell Signal","Symbol2"])
plt.show()


# In[57]:


print (pairs)


# In[58]:


Symbol1_prices = data['Adj Close']['MSFT'] 
Symbol1_prices.plot(figsize=(15,7)) 
plt.show()
Symbol2_prices = data['Adj Close']['JNPR'] 
Symbol2_prices.name = 'JNPR' 
plt.title("MSFT and JNPR prices") 
Symbol1_prices.plot() 
Symbol2_prices.plot()
plt.legend()
plt.show()


# In[62]:


def zscore(series):
    return (series - series.mean()) / np.std(series)

score, pvalue, _ = coint(Symbol1_prices, Symbol2_prices)
print(pvalue)
ratios = Symbol1_prices / Symbol2_prices
plt.title("Ration between Symbol 1 and Symbol 2 price")

ratios.plot()
plt.show()


# In[63]:


zscore(ratios).plot()
plt.title("Z-score evolution")
plt.axhline(zscore(ratios).mean(),color="black")
plt.axhline(1.0, color="red")
plt.axhline(-1.0, color="green")
plt.show()


# In[64]:


ratios.plot()
buy = ratios.copy()
sell = ratios.copy()
buy[zscore(ratios)>-1] = 0
sell[zscore(ratios)<1] = 0
buy.plot(color="g", linestyle="None", marker="^")
sell.plot(color="r", linestyle="None", marker="v")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,ratios.min(),ratios.max()))
plt.legend(["Ratio", "Buy Signal", "Sell Signal"])
plt.show()


# In[65]:


symbol1_buy=Symbol1_prices.copy()
symbol1_sell=Symbol1_prices.copy()
symbol2_buy=Symbol2_prices.copy()
symbol2_sell=Symbol2_prices.copy()


# In[66]:


Symbol1_prices.plot()
symbol1_buy[zscore(ratios)>-1] = 0
symbol1_sell[zscore(ratios)<1] = 0
symbol1_buy.plot(color="g", linestyle="None", marker="^")
symbol1_sell.plot(color="r", linestyle="None", marker="v")


# In[67]:


pair_correlation_trading_strategy = pd.DataFrame(index=Symbol1_prices.index)
pair_correlation_trading_strategy['symbol1_price']=Symbol1_prices
pair_correlation_trading_strategy['symbol1_buy']=np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol1_sell']=np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol2_buy']=np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol2_sell']=np.zeros(len(Symbol1_prices))


# In[68]:


position=0
for i in range(len(Symbol1_prices)):
    s1price=Symbol1_prices[i]
    s2price=Symbol2_prices[i]
    if not position and symbol1_buy[i]!=0:
        pair_correlation_trading_strategy['symbol1_buy'][i]=s1price
        pair_correlation_trading_strategy['symbol2_sell'][i] = s2price
        position=1
    elif not position and symbol1_sell[i]!=0:
        pair_correlation_trading_strategy['symbol1_sell'][i] = s1price
        pair_correlation_trading_strategy['symbol2_buy'][i] = s2price
        position = -1
    elif position==-1 and (symbol1_sell[i]==0 or i==len(Symbol1_prices)-1):
        pair_correlation_trading_strategy['symbol1_buy'][i] = s1price
        pair_correlation_trading_strategy['symbol2_sell'][i] = s2price
        position = 0
    elif position==1 and (symbol1_buy[i] == 0 or i==len(Symbol1_prices)-1):
        pair_correlation_trading_strategy['symbol1_sell'][i] = s1price
        pair_correlation_trading_strategy['symbol2_buy'][i] = s2price
        position = 0


# In[69]:


Symbol2_prices.plot()
symbol2_buy[zscore(ratios)<1] = 0
symbol2_sell[zscore(ratios)>-1] = 0
symbol2_buy.plot(color="g", linestyle="None", marker="^")
symbol2_sell.plot(color="r", linestyle="None", marker="v")


# In[80]:


x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,Symbol1_prices.min(),Symbol2_prices.max()))
plt.legend(["Symbol1", "Buy Signal", "Sell Signal","Symbol2"])
plt.show()


# In[71]:


Symbol1_prices.plot()
pair_correlation_trading_strategy['symbol1_buy'].plot(color="g", linestyle="None", marker="^")
pair_correlation_trading_strategy['symbol1_sell'].plot(color="r", linestyle="None", marker="v")
Symbol2_prices.plot()
pair_correlation_trading_strategy['symbol2_buy'].plot(color="g", linestyle="None", marker="^")
pair_correlation_trading_strategy['symbol2_sell'].plot(color="r", linestyle="None", marker="v")
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,Symbol1_prices.min(),Symbol2_prices.max()))
plt.legend(["Symbol1", "Buy Signal", "Sell Signal","Symbol2"])
plt.show()


# In[73]:


pair_correlation_trading_strategy['symbol1_buy'].head()



# In[74]:


pair_correlation_trading_strategy['symbol1_position']=pair_correlation_trading_strategy['symbol1_buy']-pair_correlation_trading_strategy['symbol1_sell']

pair_correlation_trading_strategy['symbol2_position']=pair_correlation_trading_strategy['symbol2_buy']-pair_correlation_trading_strategy['symbol2_sell']

pair_correlation_trading_strategy['symbol1_position'].cumsum().plot()
pair_correlation_trading_strategy['symbol2_position'].cumsum().plot()

pair_correlation_trading_strategy['total_position']=pair_correlation_trading_strategy['symbol1_position']+pair_correlation_trading_strategy['symbol2_position']
pair_correlation_trading_strategy['total_position'].cumsum().plot()
plt.title("Symbol 1 and Symbol 2 positions")
plt.legend()
plt.show()


# In[75]:


pair_correlation_trading_strategy['symbol1_price']=Symbol1_prices
pair_correlation_trading_strategy['symbol1_buy']=np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol1_sell']=np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol2_buy']=np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['symbol2_sell']=np.zeros(len(Symbol1_prices))
pair_correlation_trading_strategy['delta']=np.zeros(len(Symbol1_prices))


position=0
s1_shares = 1000000
for i in range(len(Symbol1_prices)):
    s1positions= Symbol1_prices[i] * s1_shares
    s2positions= Symbol2_prices[i] * int(s1positions/Symbol2_prices[i])
    print(Symbol1_prices[i],Symbol2_prices[i])
    delta_position=s1positions-s2positions
    if not position and symbol1_buy[i]!=0:
        pair_correlation_trading_strategy['symbol1_buy'][i]=s1positions
        pair_correlation_trading_strategy['symbol2_sell'][i] = s2positions
        pair_correlation_trading_strategy['delta'][i]=delta_position
        position=1
    elif not position and symbol1_sell[i]!=0:
        pair_correlation_trading_strategy['symbol1_sell'][i] = s1positions
        pair_correlation_trading_strategy['symbol2_buy'][i] = s2positions
        pair_correlation_trading_strategy['delta'][i] = delta_position
        position = -1
    elif position==-1 and (symbol1_sell[i]==0 or i==len(Symbol1_prices)-1):
        pair_correlation_trading_strategy['symbol1_buy'][i] = s1positions
        pair_correlation_trading_strategy['symbol2_sell'][i] = s2positions
        position = 0
    elif position==1 and (symbol1_buy[i] == 0 or i==len(Symbol1_prices)-1):
        pair_correlation_trading_strategy['symbol1_sell'][i] = s1positions
        pair_correlation_trading_strategy['symbol2_buy'][i] = s2positions
        position = 0


# In[78]:


pair_correlation_trading_strategy['symbol1_position']=pair_correlation_trading_strategy['symbol1_buy']-pair_correlation_trading_strategy['symbol1_sell']

pair_correlation_trading_strategy['symbol2_position']=pair_correlation_trading_strategy['symbol2_buy']-pair_correlation_trading_strategy['symbol2_sell']

pair_correlation_trading_strategy['symbol1_position'].cumsum().plot()
pair_correlation_trading_strategy['symbol2_position'].cumsum().plot()

pair_correlation_trading_strategy['total_position']=pair_correlation_trading_strategy['symbol1_position']+pair_correlation_trading_strategy['symbol2_position']
pair_correlation_trading_strategy['total_position'].cumsum().plot()
plt.title("Symbol 1 and Symbol 2 positions")
plt.legend()
plt.show()


# In[79]:


pair_correlation_trading_strategy['delta'].plot()
plt.title("Delta Position")
plt.show()


# In[ ]:




