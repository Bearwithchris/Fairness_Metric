# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 23:15:28 2021

@author: Chris
"""


# importing required packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
  
# loading dataset
data = pd.read_csv("data.csv")
  
# draw pointplot
sns.catplot(x = "Bias Percentage",
              y = "Score",
              hue="Attributes",
              col= "Metric",
              data = data,
              kind="point",
              dodge=True,
              height=4, 
              aspect=.7);
# # show the plot
plt.show()
# This code is contributed 
# by Deepanshu Rustagi.