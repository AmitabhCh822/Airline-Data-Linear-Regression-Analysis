"""
@author: Amitabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

        # pd.read_fwf() reads the .dat file by dividing the data into columns
df1 = pd.read_fwf('airline_costs.dat', index_col=0, # fwf refers to fixed width file               
                  names = ["Airline name", "W", "X", "Y", "Z", "Length (miles)", # assigning the column names
                           "Speed (miles/hour)", "Time (hours)",
                           "Number of customers", "Total operating cost",
                           "Total assets", "Investments and special funds"])
df1.to_csv('airline_costs.csv')
df = pd.read_csv('airline_costs.csv')


y1 = df.iloc[:,8]
X1 = df.iloc[:,[5,7]]
X1 = sm.add_constant(X1)
lr_model1 = sm.OLS(y1, X1).fit()

print(lr_model1.summary())
print("")
"""Regression formula : 
customers = (-0.0008)*Length + (7.8338)*Time + (-0.3366)
R-squared value = 0.479 """
print("Correlation of number of customers served with flight length is",
      np.corrcoef(df.iloc[:,8],df.iloc[:,5])[0,1])
print("Correlation of number of customers served with flight time is",
      np.corrcoef(df.iloc[:,8],df.iloc[:,7])[0,1])
print("Correlation of flight time with flight length is",
      np.corrcoef(df.iloc[:,7],df.iloc[:,5])[0,1])
print("")


y2 = df.iloc[:,10]
X2 = df.iloc[:,8]
X2 = sm.add_constant(X2)
lr_model2 = sm.OLS(y2, X2).fit()

print(lr_model2.summary())
print("")
"""Regression formula : 
Total assets = (16.7904)*customers + (-35.2099)
R-squared value = 0.338 """
print("Correlation of annual net sales with number of competitors in the area is",
      np.corrcoef(df.iloc[:,10],df.iloc[:,8])[0,1])
print("")
