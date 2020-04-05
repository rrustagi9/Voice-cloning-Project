# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 03:14:32 2020

@author: rrust
"""
#kaggle bike share 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import math

#read the data 
dataset=pd.read_csv('hour.csv')

#analysis and dropping
bikes_prep=dataset.copy()
bikes_prep = dataset.drop(['index','date','casual','registered'] ,axis =1)

#check missing values
bikes_prep.isnull().sum()

#visualisation using pandas histogram
bikes_prep['demand'].hist(rwidth=0.9)
plt.tight_layout()

#visualising the data 
plt.subplot(2,2,1)
plt.title('temp vs demand')
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],)

plt.subplot(2,2,2)
plt.title('atemp vs demand')
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],)

plt.subplot(2,2,3)
plt.title('windspeed vs demand')
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'])

plt.subplot(2,2,4)
plt.title('humidity vs demand')
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'])



#average demands for categorical features
plt.subplot(3,3,1)
plt.title('average demand per season')
cat_list=bikes_prep['season'].unique()
cat_average=bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list,cat_average)

plt.subplot(3,3,2)
plt.title('average demand per year')
cat_list=bikes_prep['year'].unique()
cat_average=bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list,cat_average)

plt.subplot(3,3,3)
plt.title('average demand per month')
cat_list=bikes_prep['month'].unique()
cat_average=bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list,cat_average)

plt.subplot(3,3,4)
plt.title('average demand per hour')
cat_list=bikes_prep['hour'].unique()
cat_average=bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list,cat_average)

plt.subplot(3,3,5)
plt.title('average demand per holiday')
cat_list=bikes_prep['holiday'].unique()
cat_average=bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list,cat_average)

plt.subplot(3,3,6)
plt.title('average demand per weekday')
cat_list=bikes_prep['weekday'].unique()
cat_average=bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list,cat_average)

plt.subplot(3,3,7)
plt.title('average demand per workingday')
cat_list=bikes_prep['workingday'].unique()
cat_average=bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list,cat_average)

plt.subplot(3,3,8)
plt.title('average demand per weather')
cat_list=bikes_prep['weather'].unique()
cat_average=bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list,cat_average)

plt.tight_layout()


#features to be dropped weekdays,years,workingday

#check for outliers
bikes_prep['demand'].describe()

bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99])


#correlation
corrrelation=bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr() 

#dropping colunms
bikes_prep=bikes_prep.drop(['atemp','year','weekday','workingday','windspeed'], axis=1)

#check autocorrelation
df1=pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.acorr(df1,maxlags=12)
#log normalize the demand
df1=bikes_prep['demand']

df2=np.log(df1)
plt.figure()
df1.hist(rwidth=0.9,bins=20)

plt.figure()
df2.hist(rwidth=0.9,bins=20)

bikes_prep['demand']=np.log(bikes_prep['demand'])

#create dummy variables
bikes_prep['season']=bikes_prep['season'].astype('category')
bikes_prep['month']=bikes_prep['month'].astype('category')
bikes_prep['hour']=bikes_prep['hour'].astype('category')
bikes_prep['weather']=bikes_prep['weather'].astype('category')
bikes_prep['holiday']=bikes_prep['holiday'].astype('category')
dummy_df=pd.get_dummies(bikes_prep,drop_first=True)


y=bikes_prep[['demand']]
x=bikes_prep.drop(['demand'],axis=1)

#create training set
tr_size=0.7*len(x)

tr_size=int(tr_size)

x_train=x.values[0:tr_size]
x_test=x.values[tr_size:len(x)]

y_train=y.values[0:tr_size]
y_test=y.values[tr_size:len(y)]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

r2_train=regressor.score(x_train,y_train)
r2_test=regressor.score(x_test,y_test)

y_pred=regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(y_test,y_pred))

