#RF model for predicting weather data for Seattle, WA from 2016 using the NOAA Climate Data Online tool

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#### Loading data #####
df = pd.read_csv(r'C:\Users\16617\Desktop\Data Science Development Code\Data-Science-Development-Code\Homework_Data\temps.csv')

###STEPS#####
#1)SPLIT DATA
#2)TRAIN RF MODEL
#3)TEST RF MODEL

#### Set up labels ######
label = df["actual"]

#### Set up features ######
features = df.drop('actual',1)
categorical_variables = ["week"]
continuous_variables = []

####### OHE FOR LOADING DATA ########
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(features[categorical_variables])
X_ohed = one_hot_encoder.transform(features[categorical_variables]).toarray()
X_ohed = pd.DataFrame(X_ohed,columns=one_hot_encoder.get_feature_names_out(categorical_variables))

G = features.drop('week',1)
features = pd.concat([G, X_ohed], axis=1)

#### Split data into test/train #####
X_train, X_test, y_train, y_test = train_test_split(features, label,test_size=0.3,random_state=123)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 123)
# TRAIN MODEL
rf.fit(X_train, y_train)

# Use RF to predict actual temps
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = 100*abs(predictions - y_test)/y_test
print('Mean Absolute Error for Loading Total:', round(np.mean(errors), 2), '%.')