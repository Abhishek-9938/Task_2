# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 18:16:25 2022

@author: HP
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
import sklearn.metrics as sm

data = pd.read_csv("Test_data_file.xlsx - Prediction.csv")

x = data[['College_T1','College_T2','Role_Manager','City_Metro','previous CTC','previous job changes','Graduation marks','Exp']]
y = data["Actual CTC"]

x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=0)

model=linear_model.LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)


print("Score=",sm.r2_score(y_test, y_pred))


try_prd = model.predict([[0,1,0,1,57081,1,84,18]]) #try pred"ictions  97.5% accurate

#ploting
plt.figure()
plt.scatter(y_test,y_pred, color='green', label='True Value')

plt.xlabel("actual ctc")
plt.ylabel("previous ctc")
plt.title('Prediction Result of Test data')
plt.legend()
plt.show()nm\

