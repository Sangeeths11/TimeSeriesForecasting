# %% [markdown]
# ### Import Libraries

# %%
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

# %% [markdown]
# ### Load Data

# %%
df = pd.read_excel("../data/Superstore.xls")

# %% [markdown]
# ### Data Understanding

# %%
df.head()

# %%
df.info()

# %% [markdown]
# #### Forecasting Furniture Sales

# %%
furniture = df.loc[df['Category'] == 'Furniture']
furniture

# %% [markdown]
# #### Checking Periodicity of the Time Series

# %%
furniture['Order Date'].min(), furniture['Order Date'].max()

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# #### Dropping Columns

# %%
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode',
        'Customer ID', 'Customer Name', 'Segment', 'Country',
        'City', 'State', 'Postal Code', 'Region', 'Product ID', 
        'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']

furniture.drop(cols, axis=1, inplace=True)

# %%
furniture

# %% [markdown]
# #### Sorting the data

# %%
furniture = furniture.sort_values('Order Date')
furniture

# %% [markdown]
# #### Checking for Missing Values

# %%
furniture.isnull().sum()

# %%
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
furniture

# %% [markdown]
# #### Indexing with Time Series Data

# %%
furniture = furniture.set_index('Order Date')
furniture.index

# %%
y = furniture['Sales'].resample('MS').mean()
y

# %% [markdown]
# #### Quick peek 2017 Furniture Sales Data

# %%
y['2017':]

# %% [markdown]
# ### Visualizing Furniture Sales Time Series Data

# %%
y.plot(figsize=(15, 6))
plt.show()

# %% [markdown]
# #### Observed, Trend, Seasonal, and Residual Components 

# %% [markdown]
# 
# Time Series Decomposition of Sales: This is a method to break down the sales data over time into different parts to understand it better.
# 
# Original Series: This is the actual sales data as it was recorded.
# 
# Trend: This shows the general direction in which sales are moving over a long period, like an overall increase or decrease.
# 
# Seasonality: This captures regular patterns or cycles in sales data that repeat at specific times, like higher sales in December each year.
# 
# Residuals: These are the random fluctuations in sales data that can't be explained by the trend or seasonality, often considered as "noise" or irregular variations.

# %%
observations = y
decomposition = sm.tsa.seasonal_decompose(observations, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(15, 10))
plt.suptitle('Time Series Decomposition of Sales', fontsize=22, y=1.02)

# Original Series
plt.subplot(411)
plt.plot(observations, label='Original', color="blue")
plt.legend(loc='upper left')
plt.title('Original Series')

# Trend
plt.subplot(412)
plt.plot(trend, label='Trend', color="orange")
plt.legend(loc='upper left')
plt.title('Trend')

# Seasonality
plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color="green")
plt.legend(loc='upper left')
plt.title('Seasonality')

# Residuals
plt.subplot(414)
plt.plot(residual, label='Residuals', color="red")
plt.legend(loc='upper left')
plt.title('Residuals')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# %% [markdown]
# #### Time Series forecasting with ARIMA

# %% [markdown]
# We are going to apply one of the most commonly used method for time-series forecasting, known as ARIMA, which stands for Autoregressive Integrated Moving Average.
# 
# Parameter Selection for the ARIMA Time Series Model

# %%
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))

#Jährliche Saison
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# %% [markdown]
# p (autoregressive part): This is the number of past values included in the model, indicating how many previous time points are used to predict the current value.
# 
# d (differencing part): This indicates how many times the data need to be differenced to become stationary, which helps remove trends and stabilize the data.
# 
# q (moving average part): This is the number of previous error terms (differences between actual and predicted values) included in the model.

# %% [markdown]
# #### Grid Search for the best ARIMA parameters

# %%
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# %% [markdown]
# #### Fitting the ARIMA model

# %%
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

# %% [markdown]
# #### Visualizing the ARIMA Time Series Forecasting Results

# %%
print("Length of the time series:", len(y))

# %%
residuals = results.resid
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title('Residuals over time')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

# %% [markdown]
# There are clear fluctuations in the residuals, which may indicate seasonal patterns or trends that were not fully captured by the model. These fluctuations appear to be repeated regularly over the years.
# 
# Residuals should fluctuate around the zero line with no discernible trends or seasonal patterns if the model describes the data well. However, there are some noticeable outliers and fluctuations in this plot, suggesting that the model may not have captured all systematic patterns in the data.
# 
# There are some extreme values, particularly around the turn of 2015, which indicate that the model made major errors in these periods. These outliers could indicate unusual events or errors in the model

# %%
plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# A histogram shows the distribution of the residuals. Ideally, the residuals should follow a normal distribution that forms a bell-shaped curve.
# In this histogram, we see that the residuals are not perfectly normally distributed. There are several bars representing different frequencies of the residuals in different areas.
# 
# A normally distributed distribution should be symmetrical around the mean value. This histogram shows a certain symmetry around the mean value, but also shows some deviations.
# 
# There are some extreme residual values, particularly around -800 and 600, which indicate outliers. These could indicate special events or model errors.
# 
# Most of the residuals seem to be concentrated in the range of about -200 to 200, which indicates that the model does not make extremely large errors in most cases.

# %%
import statsmodels.api as sm
plt.figure(figsize=(10, 4))
sm.qqplot(residuals, line='s')
plt.title('QQ Plot')
plt.show()

# %% [markdown]
# Most of the points are relatively close to the red line, which indicates that the residuals roughly follow a normal distribution.
# 
# There are some points at the beginning and end (especially for extreme values) that deviate significantly from the line. These outliers indicate that the residuals for the extreme values are not normally distributed.
# 
# There is a slight S-curve, which indicates that the distribution of the residuals shows slight deviations from the normal distribution, especially in the extreme ranges.

# %%
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(10, 4))
plot_acf(residuals, lags=30)
plt.title('ACF of Residuals')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()

# %% [markdown]
# The first bar at lag 0 always has an autocorrelation of 1, as it is the correlation of the residuals with themselves.
# 
# Most of the autocorrelation values for the different lags lie within the blue confidence interval, which indicates that these autocorrelations are not significant.
# 
# As most of the autocorrelation values lie within the confidence intervals, this indicates that there is no significant autocorrelation in the residuals. This is a good sign as it indicates that the model has captured the time series structure well.

# %% [markdown]
# ### Validating forecasts

# %% [markdown]
# To help us understand the accuracy of our forecasts, we compare predicted sales to real sales of the time series, and we set forecasts to start at 2017-07-01 to the end of the data.

# %%
import plotly.graph_objs as go
import plotly.io as pio

pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()

observed_trace = go.Scatter(x=y.index, y=y, mode='lines', name='Observed')

forecast_trace = go.Scatter(x=pred.predicted_mean.index, y=pred.predicted_mean, mode='lines', name='Forecast')

ci_trace = go.Scatter(x=pred_ci.index.tolist() + pred_ci.index.tolist()[::-1],
                      y=pred_ci.iloc[:, 0].tolist() + pred_ci.iloc[:, 1].tolist()[::-1],
                      fill='toself',
                      fillcolor='rgba(0,100,80,0.2)',
                      line=dict(color='rgba(255,255,255,0)'),
                      showlegend=False,
                      name='Confidence Interval')


layout = go.Layout(title='Furniture Sales Forecast',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Furniture Sales'))


fig = go.Figure(data=[observed_trace, forecast_trace, ci_trace], layout=layout)

pio.show(fig)



# %%
print(pred_ci)

# %% [markdown]
# The line plot is showing the observed values compared to the rolling forecast predictions. Overall, our forecasts align with the true values very well, showing an upward trend starts from the beginning of the year.

# %%
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# %% [markdown]
# 
# In statistics, the mean squared error (MSE) of an estimator measures the average of the squares of the errors — that is, the average squared difference between the estimated values and what is estimated. The MSE is a measure of the quality of an estimator—it is always non-negative, and the smaller the MSE, the closer we are to finding the line of best fit.

# %%
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

# %% [markdown]
# 
# Root Mean Square Error (RMSE) tells us that our model was able to forecast the average daily furniture sales in the test set within 160.34 of the real sales. Our furniture daily sales range from around 400 to over 1200. In my opinion, this is a pretty good model so far.

# %% [markdown]
# ### Producing and visualizing forecasts

# %%
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()

observed_trace = go.Scatter(x=y.index, y=y, mode='lines', name='Observed')

forecast_trace = go.Scatter(x=pred_uc.predicted_mean.index, y=pred_uc.predicted_mean, mode='lines', name='Forecast')

ci_trace = go.Scatter(x=pred_ci.index.tolist() + pred_ci.index.tolist()[::-1],
                        y=pred_ci.iloc[:, 0].tolist() + pred_ci.iloc[:, 1].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        name='Confidence Interval')

layout = go.Layout(title='Furniture Sales Forecast',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Furniture Sales'))

fig = go.Figure(data=[observed_trace, forecast_trace, ci_trace], layout=layout)

pio.show(fig)

# %% [markdown]
# The above time series analysis for furniture makes me curious about other categories, and how do they compare with each other over time. Therefore, we are going to compare time series of furniture and office supplier.

# %% [markdown]
# ### Time Series of Furniture vs. Office Supplies

# %%
furniture = df.loc[df['Category'] == 'Furniture']
office = df.loc[df['Category'] == 'Office Supplies']
furniture.shape, office.shape

# %% [markdown]
# According to our data, there were way more number of sales from Office Supplies than from Furniture over the years.

# %% [markdown]
# ### Data Exploration

# %%
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
office.drop(cols, axis=1, inplace=True)

# %%
furniture = furniture.sort_values('Order Date')
office = office.sort_values('Order Date')

# %%
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
office = office.groupby('Order Date')['Sales'].sum().reset_index()

# %%
furniture = furniture.set_index('Order Date')
office = office.set_index('Order Date')

# %%
y_furniture = furniture['Sales'].resample('MS').mean()
y_office = office['Sales'].resample('MS').mean()

# %%
furniture = pd.DataFrame({'Order Date':y_furniture.index, 'Sales':y_furniture.values})
office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})

# %%
store = furniture.merge(office, how='inner', on='Order Date')
store.rename(columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True)
store.head()

# %%
plt.figure(figsize=(20, 8))
plt.plot(store['Order Date'], store['furniture_sales'], 'b-', label = 'furniture')
plt.plot(store['Order Date'], store['office_sales'], 'r-', label = 'office supplies')
plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('Sales of Furniture and Office Supplies')
plt.legend();

# %%
first_index = np.min(np.where(store['office_sales'] > store['furniture_sales'])[0])
first_date = store.loc[first_index, 'Order Date']

print("Office supplies first time produced higher sales than furniture is {}.".format(first_date.date()))

# %% [markdown]
# ### Time Series Modeling with Prophet

# %%
from prophet import Prophet

furniture = furniture.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
furniture_model = Prophet(interval_width=0.95)
furniture_model.fit(furniture)
office = office.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
office_model = Prophet(interval_width=0.95)
office_model.fit(office)

furniture_forecast = furniture_model.make_future_dataframe(periods=36, freq='MS')
furniture_forecast = furniture_model.predict(furniture_forecast)
office_forecast = office_model.make_future_dataframe(periods=36, freq='MS')
office_forecast = office_model.predict(office_forecast)

plt.figure(figsize=(18, 6))
furniture_model.plot(furniture_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Furniture Sales');

# %%
plt.figure(figsize=(18, 6))
office_model.plot(office_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Office Supplies Sales');

# %%
office_forecast.columns

# %%
furniture_forecast.columns

# %% [markdown]
# ### Comparing Forecasts

# %%
furniture_names = []
for column in furniture_forecast.columns:
    furniture_names.append('furniture_%s' % column)

office_names = []
for column in office_forecast.columns:
    office_names.append('office_%s' % column)

print(furniture_names)
print(office_names)

# %%
merge_furniture_forecast = furniture_forecast.copy()
merge_office_forecast = office_forecast.copy()

# %%
merge_furniture_forecast.columns = furniture_names
merge_office_forecast.columns = office_names

print("Furniture Forecast Columns: ", merge_furniture_forecast.columns)
print("Office Forecast Columns: ", merge_office_forecast.columns)

# %%
forecast = pd.merge(merge_furniture_forecast, merge_office_forecast, how='inner', left_on='furniture_ds', right_on='office_ds')
forecast = forecast.rename(columns={'furniture_ds': 'Date'}).drop('office_ds', axis=1)

# %%
print(forecast.head())

# %% [markdown]
# ### Trend and Forecast Visualization

# %%
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['furniture_trend'], 'b-', label='Furniture Trend')
plt.plot(forecast['Date'], forecast['office_trend'], 'r-', label='Office Supplies Trend')

plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Furniture vs. Office Supplies Sales Trend')

plt.legend()

plt.show()


# %%
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['furniture_yhat'], 'b-', label='Furniture Estimate')
plt.plot(forecast['Date'], forecast['office_yhat'], 'r-', label='Office Supplies Estimate')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('Furniture vs. Office Supplies Estimate');

# %% [markdown]
# ### Trends and Patterns

# %%
furniture_model.plot_components(furniture_forecast);

# %%
office_model.plot_components(office_forecast);

# %% [markdown]
# Good to see that the sales for both furniture and office supplies have been linearly increasing over time and will be keep growing, although office supplies’ growth seems slightly stronger.
# 
# The worst month for furniture is April, the worst month for office supplies is December. The best month for furniture is December, and the best month for office supplies is February.
# 
# There are many time-series analysis we can explore from now on, such as forecast with uncertainty bounds, change point and anomaly detection, forecast time-series with external data source. We have only just started.


