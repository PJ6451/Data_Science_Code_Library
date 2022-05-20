# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:44:43 2020

@author: mjqq
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
import matplotlib.pylab as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

#### READ DATA #####
# Unfortunately this data was propietary, could not take with me 
df = pd.read_csv(r'')

### GET DATE INTO CORRECT FORMAT ####
df["partition_ledger_year_month"] = df["partition_ledger_year_month"].apply(str)
df["Year"] = df["partition_ledger_year_month"].str[0:4]
df["Month"] = df["partition_ledger_year_month"].str[4:6]

df["date"] = pd.to_datetime(df.Year+df.Month,format='%Y%m') + MonthEnd(1)
df = df.drop(columns = ['Year','Month', 'partition_ledger_year_month' ])


####### DATA EXPLORATION ########

df.index = df.date
dfs = df.amount_usd
### TEST for stationarity, look at 10 accounts at a time ######
a = df.account_id.unique()
c = a[0:11]
dfs = df[df["account_id"].isin(c)]
for x in c:
    b = [x]
    dfs = df[df["account_id"].isin(b)]
    result = seasonal_decompose(dfs.amount_usd, period = 12)
    result.plot()
    pyplot.show()

### most accounts showed some sort of trend and seasonality
### Need to transform the data before continuing

### Need to figure out if I can drop any of the other variables 

### Seperate dependent and independent variables
X = df.drop(columns = 'amount_usd')
categorical_variables = ['GL_account_description']
cont_var = ["GL_account","date","account_id"]

####### NEED TO SEE IF ACCOUNT DESCRIPTION IS USEFUL, IF THERES high correlation I'll remove it
####### OHE ########
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(X[categorical_variables])
X_ohed = one_hot_encoder.transform(X[categorical_variables]).toarray()
X_ohed = pd.DataFrame(X_ohed,
                      columns=one_hot_encoder.get_feature_names(categorical_variables))


X1 = pd.concat([X, X_ohed], axis=1)

#### Testing correlation between account_id and GL_account
y1 = X1.GL_account
X11 = X1.account_id
model = sm.OLS(y1, X11).fit()

print(model.summary())

##### Testing correlation between account_id and GL_account, with one-hot-encoded descriptions 
y2 = X1.GL_account
X12 = X1.drop(columns = ["date","GL_account", "GL_account_description"])
model = sm.OLS(y2, X12).fit()

print(model.summary())

##### High correlation between account_id and GL_account in both models (R^2 > .8 for both runs), 
##### so I will ellect to discard both GL_account and account description

##### MODEL TESTING, TRYING WITH ONE ACCOUNT AT A TIME ####
c = ["ID550385"]
##### Now I'm going to Pivot table the values, with date as the index and account ID's as the columns, with the values just being the dollar ammounts
piv = pd.pivot_table(df, index = "date", columns = "account_id", values = "amount_usd")
aaa = piv.columns
#### Need to transform columns into strings
for x in aaa:
    piv = piv.rename(columns={x: "ID" + str(x)})

piv_test  = pd.DataFrame(piv[c])

#### Standardizing dollar amounts
piv_test1 = piv_test.apply(lambda x: (x-piv_test[col].mean())/piv_test[col].std(), axis = 0)
piv_test2 = piv_test1.replace(to_replace = np.nan, value = 0)
#### Need to check for seasonality
result = seasonal_decompose(piv_test2[c], period = 12)
result.plot()
    
#### Still shows some seasonality, will take a difference to remove    
piv_test2 = piv_test1.replace(to_replace = np.nan, value = 0)
pivs = piv_test2.diff()[1:]    

#### test for normality
pivs.hist()
#### still shows some skewness but looks a lot better than it did

### determining p,d,q for ARIMA model
lag_acf = acf(pivs, nlags = 50)
lag_pacf = pacf(pivs, nlags = 50, method = 'ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle = '--', color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(piv)),linestyle = '--', color = 'gray')
plt.axhline(y=1.96/np.sqrt(len(piv)),linestyle = '--', color = 'gray')
plt.title("Autocorrelation Function")
### This plot indicates a p of 0 or 1

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle = '--', color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(piv)),linestyle = '--', color = 'gray')
plt.axhline(y=1.96/np.sqrt(len(piv)),linestyle = '--', color = 'gray')
plt.title("Partial Autocorrelation Function")
plt.tight_layout()
### This plot indicates a q of 0 or 1


### AR model, p = 1, d = 1; This model works and gives results
model = ARIMA(pivs, order = (1,1,0), freq = 'M')
results_AR = model.fit()
plt.plot(pivs)
plt.plot(results_AR.fittedvalues, color = 'red')
plt.title('RMSE: %.4f'% np.sqrt(sum((results_AR.fittedvalues - pivs.ID550385[1:])**2)))

#### MA model, q = 1, d =1; This model works and gives results
model = ARIMA(pivs, order = (0,1,1))
results_MA = model.fit(disp=-1)
plt.plot(pivs)
plt.plot(results_MA.fittedvalues, color = 'red')
plt.title('RMSE: %.4f'% np.sqrt(sum((results_MA.fittedvalues - pivs.ID550385[1:])**2)))

####ARIMA, p = 1, d = 1, q = 1; This model doesn't work and gives no results
#model = ARIMA(pivs, order = (1,1,1))
#results_ARIMA = model.fit(disp=-1)
#plt.plot(pivs)
#plt.plot(results_ARIMA.fittedvalues, color = 'red')
#plt.title('RMSE: %.4f'% np.sqrt(sum((results_ARIMA.fittedvalues - pivs.ID550385[1:])**2)))

pred_AR_trans = pd.Series(results_AR.fittedvalues)
pred_cumsum = pred_AR_trans.cumsum()
predictions_arima_st = pd.Series(piv_test2.ID550385[0:2], index = piv_test.index)
for x in range(len(predictions_arima_st)-2):
    predictions_arima_st[x+2] = predictions_arima_st[x+1] + pred_AR_trans[x]

predictions_arima_st = pd.DataFrame(predictions_arima_st)
pred = predictions_arima_st.apply(lambda x: (piv_test.ID550385.std()*x)+piv_test.ID550385.mean(), axis=0)
plt.plot(piv_test)
plt.plot(pred, color = 'red')
plt.title('RMSE: %.4f'% np.sqrt(sum((pred.ID550385 - piv_test.ID550385)**2)))

### This model did not perform well, RSME was extremely high and it was extremely manual per account
### will continue with auto_arima to automatically find values

##### The following was mainly to see how getting rid of nans from the pivot affected the table
##### nans represented months where there was no data for the account, so I figured
##### having a 0 would not necessarily hurt the model as a 0 represented no change.
##### however, I still wanted to look at the numbers
    
piv_zeros = pd.DataFrame(index = piv.columns, columns = ["Zeros1","Zeros2"])
b = 0
for col in piv.columns:
    piv_zeros.Zeros1[b] = sum(piv[col]==0)
    b = b+1

piv = piv.replace(to_replace = np.nan, value = 0)

b = 0
for col in piv.columns:
    piv_zeros.Zeros2[b] = sum(piv[col]==0)
    b = b+1

################## AUTO_ARIMA MODEL ##################

##### I set up a df for my predictions so I can just copy and paste into the scoring scv when i'm done
pred_july = pd.DataFrame(index = piv.columns, columns = ["Prediction1","Prediction2","Prediction3"])
b = 0

for col in piv.columns:
    piv_test = pd.DataFrame(piv[col])
    if sum(piv_test[col] == 0) < 42:
        #### this says that I only want to model the data if theres more than 12 rows that have actual numbers
        #### giving me actual data to work with
        #### first model sumbitted limited modeling to accounts with over 16 rows of data, now trying for 12
        
        #### Need to standardize the data and take the difference to make data non-stationary
        piv_test1 = piv_test.apply(lambda x: (x-piv_test[col].mean())/piv_test[col].std(), axis = 0)
        pivs = piv_test1.diff()[1:]
        pivs = pivs.replace(to_replace = np.nan, value = 0)
        #### this model seems to perform better then the other ARIMA model I tried
        #### plus I don't have to figure out order/seasonal order per account_id 
        
        import pmdarima as pm
        from pmdarima.model_selection import train_test_split
        train_len = 52
        y_train, y_test = train_test_split(pivs, train_size=train_len)
        fit1 = pm.auto_arima(y_train, m=12, trace=True, suppress_warnings=True)
        a = fit1.predict(n_periods = 3)
        
        ##### UNROLLING ####
        #diff is the difference between rows
        #ie: if we run it on the rows of feb and march, we get the change from feb to march
        #if we have the change and the first value, we can calcute the rest of them by adding them to the previous one
        #ie: we add the first change to the first value to get the second value
        #then just unstandardize        
        
        pred_july.iloc[b,0] = ((a[-1] + piv_test1[col][-1])*piv_test[col].std())+piv_test[col].mean()
        
    elif (sum(piv_test[col] == 0) > 42 & sum(piv_test[col] == 0) < 49):
        
        #### This is what I implemented for data that had less then 16 rows of data
        pred_july.iloc[b,0] = np.mean(piv_test[col])
    
    else:
        
        #Theres quite a few columns that have no data whatsoever, its more than likely they'll be 0
        pred_july.iloc[b,0] = 0

    b = b + 1
   
    
#### Just differencing, I was unable to test this model for submissions
b = 0
for col in piv.columns:
    piv_test = pd.DataFrame(piv[col])
    if sum(piv_test[col] == 0) < 42:
        #### this says that I only want to model the data if theres more than 12 rows that have actual numbers
        #### giving me actual data to work with
        #### first model sumbitted limited modeling to accounts with over 16 rows of data, now trying for 12
        
        #### Need to standardize the data and take the difference to make data non-stationary
        #piv_test1 = piv_test.apply(lambda x: (x-piv_test[col].mean())/piv_test[col].std(), axis = 0)
        pivs = piv_test.diff()[1:]
        pivs = pivs.replace(to_replace = np.nan, value = 0)
        #### this model seems to perform better then the other ARIMA model I tried
        #### plus I don't have to figure out order/seasonal order per account_id 
        
        import pmdarima as pm
        from pmdarima.model_selection import train_test_split
        train_len = 52
        y_train, y_test = train_test_split(pivs, train_size=train_len)
        fit1 = pm.auto_arima(y_train, m=12, trace=True, suppress_warnings=True)
        a = fit1.predict(n_periods = 3)
        
        ##### UNROLLING ####       
        
        pred_july.iloc[b,1] = (a[-1] + piv_test[col][-1])
        
    elif (sum(piv_test[col] == 0) > 42 & sum(piv_test[col] == 0) < 49):
        
        #### This is what I implemented for data that had less then 16 rows of data
        pred_july.iloc[b,1] = np.mean(piv_test[col])
    
    else:
        
        #Theres quite a few columns that have no data whatsoever, its more than likely they'll be 0
        pred_july.iloc[b,1] = 0

    b = b + 1
    
###### No differencing, this model performed the best
b = 0    
for col in piv.columns:
    piv_test = pd.DataFrame(piv[col])
    if sum(piv_test[col] == 0) < 42:
        #### this says that I only want to model the data if theres more than 12 rows that have actual numbers
        #### giving me actual data to work with
        #### first model sumbitted limited modeling to accounts with over 16 rows of data, now trying for 12
        
        #### Need to standardize the data and take the difference to make data non-stationary
        pivs = piv_test.apply(lambda x: (x-piv_test[col].mean())/piv_test[col].std(), axis = 0)
        pivs = pivs.replace(to_replace = np.nan, value = 0)
        #### this model seems to perform better then the other ARIMA model I tried
        #### plus I don't have to figure out order/seasonal order per account_id 
        
        import pmdarima as pm
        from pmdarima.model_selection import train_test_split
        train_len = 52
        y_train, y_test = train_test_split(pivs, train_size=train_len)
        fit1 = pm.auto_arima(y_train, m=12, trace=True, suppress_warnings=True)
        a = fit1.predict(n_periods = 3)
        
        ##### UNROLLING ####
        
        pred_july.iloc[b,2] = (a[-1])*piv_test[col].std()+piv_test[col].mean()
        
    elif (sum(piv_test[col] == 0) > 42 & sum(piv_test[col] == 0) < 49):
        
        #### This is what I implemented for data that had less then 16 rows of data
        pred_july.iloc[b,2] = np.mean(piv_test[col])
    
    else:
        
        #Theres quite a few columns that have no data whatsoever, its more than likely they'll be 0
        pred_july.iloc[b,2] = 0

    b = b + 1
    
#### Ultimately these models didn't perform very well, however I'm pretty happy that I was 
#### able to write all this code, understand it, and have a better understanding of how the
#### data science process works
    
#### To improve this I would definitely try other models and work on a better way to handle 
#### accounts with very few or sporadic data points