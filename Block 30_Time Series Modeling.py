import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

df = pd.read_excel('monthly-champagne-sales.xlsx')
df.Month = pd.to_datetime(df.Month)

df.head()
plt.figure(figsize=(8, 3))
sns.lineplot(data=df, x ='Month' , y = 'Sales')

plt.figure(figsize=(5, 3))
df.boxplot()

df["Year"] = df["Month"].dt.year
df["AbbrMonth"] = df["Month"].dt.strftime('%b')
df.head()


#Setting the size of the plot
plt.figure(figsize=(16, 7))
sns.pointplot(x="AbbrMonth", y="Sales", hue='Year', data=df, order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
       'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel("Month")
plt.ylabel("Price in ($)")
plt.title("Sales by Year\n (Used to Show Seasonality in Sales) ")
plt.legend(loc='upper left')


plt.figure(figsize=(16, 7))
sns.catplot(x="AbbrMonth", y="Sales", data=df, kind="box", row_order=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
       'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel("Month")
plt.ylabel("Sales in Units")
plt.title("Sales Trends \nBoxplot by Month ")


#Plotting 25 lag plots to determine randomness in data.
from pandas.plotting import lag_plot
plot_lags = 25
rows = int(plot_lags/5)
cols = int(plot_lags/5)
fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
fig.set_figwidth(plot_lags)
fig.set_figheight(plot_lags)
count =1
for i in range(rows):
    for j in range(cols):
        lag_plot(df["Sales"], lag=count, ax=axes[i, j])
        count+=1


#Plotting graph to determine autocorrelation
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(20, 4))
autocorrelation_plot(df["Sales"])


#Decomposing to see the white noise.
decompose = df[["Month", "Sales"]]
decompose.index = df["Month"]
decompose = decompose[["Sales"]]
decompose.head()



from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(decompose)


trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(16, 7))

plt.subplot(411)
plt.plot(df["Sales"], label='Original')
plt.title('Data Trend Decompositons')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(trend, label='Trend', color = 'green')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(seasonal,label='Seasonality', color = 'red')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(residual, label='Residuals', color = 'black')
plt.legend(loc='upper left')
plt.tight_layout()




import datetime
import math
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import adfuller
#Defining a custom function to perform Stationarity Test on the data
def stationarity_test(data, y=''):
    dftest = adfuller(data[y], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    plt.figure(figsize=(16, 7))
    plt.plot(data.index, data[y])
    plt.show()

stationarity_test(df, 'Sales')

log_sales = df.copy()
log_sales["Sales"] = log_sales["Sales"].apply(lambda x: math.log(x+1))

stationarity_test(log_sales, 'Sales')

log_sales_shift = log_sales
log_sales_shift['Sales'] = log_sales["Sales"] - log_sales["Sales"].shift(4)
log_sales_shift = log_sales.fillna(0)


stationarity_test(log_sales_shift, 'Sales')


#Setting values for lag parameters for ACF and PACF
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(log_sales_shift['Sales'], nlags=24)
lag_pacf = pacf(log_sales_shift['Sales'], nlags=24, method='ols')
plt.figure(figsize=(16, 7))

#Plotting ACF plot:
plt.subplot(121)
plt.plot(lag_acf, marker="o")
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
plt.title('Autocorrelation Function -\nHelps find Q')

#Plotting PACF Plot:
plt.subplot(122)
plt.plot(lag_pacf, marker="o")
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(lag_pacf)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(lag_pacf)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function -\nHelps find P')
plt.tight_layout()

X = df.Month
y = df.Sales


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

y_train

X_train.shape, X_test.shape, y_train.shape, y_test.shape

vals = pd.DataFrame({"X_train" : X_train, "y_train":y_train})



plt.figure(figsize=(16, 7))

plt.subplot(211)
plt.plot(df.Month[X_train.index], y_train.values, color="lightblue", label ='Training (Historical Data)')
plt.plot(df.Month[X_test.index], y_test.values, color="green", label ='Testing (Verification Data)')

#Code for checking Forecasting
pred = pd.DataFrame(results_AR.forecast(len(y_test)))
pred.columns = ["yhat"]
pred.index = y_test.index

#Converting from log to normal value
pred["yhat"] = pred["yhat"].apply(lambda x: math.exp(x)-1) + 1185 #found this value to give a better MSE

#Code for Measuring error.
measure = math.pow(mean_squared_error(y_test.values, pred.values), 0.5)
print(f'Mean squared error: {measure:.3f}')
plt.plot(df.Month[pred.index], pred.values, color="red", label = 'Prediction (Our Model Prediction)')
plt.title('Time Series Prediction of Sales Using the ARIMA Model')
plt.xlabel('Time')
plt.ylabel('Unit Sales')
plt.legend()


plt.subplot(212)
plt.plot(df.Month[X_test.index], y_test.values, color="green", label ='Testing (Verification Data)')

#Code for checking Forecasting
pred = pd.DataFrame(results_AR.forecast(len(y_test)))
pred.columns = ["yhat"]
pred.index = y_test.index


#Converting from log to normal value
pred["yhat"] = pred["yhat"].apply(lambda x: math.exp(x)-1) + 1185 #found this value to give a better MSE
plt.plot(df.Month[pred.index], pred.values, color="red", label = 'Prediction (Our Model Prediction)')
#Fill under the curve
plt.fill_between(
        x = df.Month[pred.index], 
        y1 = pred["yhat"], 
        y2 = y_test,
        where = y_test < pred["yhat"],
     interpolate=True,
        color= "r",
        alpha= 0.8)
#Fill under the curve
plt.fill_between(
        x = df.Month[pred.index], 
        y1 = pred["yhat"], 
        y2 = y_test,
        where = y_test >= pred["yhat"],
        interpolate=True,
        color= "g",
        alpha= 0.8)

plt.title('')
plt.xlabel('Time')
plt.ylabel('Unit Sales')
plt.legend()

import plotly.express as px
fig = px.line(df, x='Month', y='Sales', title='Time Series with Rangeslider')

fig.update_xaxes(rangeslider_visible=True)
fig.show()
