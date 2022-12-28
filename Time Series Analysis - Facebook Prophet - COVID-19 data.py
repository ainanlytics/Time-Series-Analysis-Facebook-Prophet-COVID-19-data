#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

# Let's read data from NY times
df = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv")

# Engineer new cases
df['new_cases'] = df.cases - df.cases.shift().fillna(0)

# Create pandas time series
df.date = pd.to_datetime(df.date)
df.set_index('date',inplace=True)
df['rolling_weekly_avg'] = df.new_cases.rolling(window=7).mean().fillna(0)

# Create timeseries readable by prophet
ts = pd.DataFrame({'ds':df.index,'y':df.new_cases})
#ts['cap'] = 30000 # unused in linear growth
#ts['floor'] = 0 # unused in linear growth
ts.head()


# In[12]:


from prophet import Prophet

# Let's create the model and fit the timeseries
prophet = Prophet()
prophet.fit(ts)

# Create a future data frame 
future = prophet.make_future_dataframe(periods=25)
forecast = prophet.predict(future)

# Display the most critical output columns from the forecast
forecast[['ds','yhat','yhat_lower','yhat_upper']].head()


# In[13]:


# plot
fig = prophet.plot(forecast)


# In[14]:


# Let's create the model and fit the timeseries
prophet = Prophet(weekly_seasonality=False, changepoint_range=1,changepoint_prior_scale=0.75)
prophet.fit(ts)

# Create a future data frame 
future = prophet.make_future_dataframe(periods=25)
forecast = prophet.predict(future)

# Display the most critical output columns from the forecast
forecast[['ds','yhat','yhat_lower','yhat_upper']].head()


# In[15]:


# Plot
fig = prophet.plot(forecast)


# In[ ]:




