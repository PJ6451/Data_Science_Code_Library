# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:07:27 2020

@author: mjqq
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv(r'C:\Users\16617\Desktop\Data Science Development Code\Data-Science-Development-Code\Homework_Data\Boston.csv')

x = df["rm"]
y = df["medv"]

plt.scatter(x,y)
plt.title("Room Number vs Median Home Value in Boston")
plt.xlabel("Average Room Number",fontsize = 18)
plt.ylabel("Median Home Value",fontsize = 18)
plt.xticks(size = 16)
plt.yticks(size = 16)
plt.show()