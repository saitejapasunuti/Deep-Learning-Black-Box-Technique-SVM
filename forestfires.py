# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 00:11:50 2020

@author: saiteja pasunuti
"""

import pandas as pd 
#for data manipulation,cleaning and analysis
import numpy as np 
#deals with  numerical data
import seaborn as sns
#for graphical representation

forestfires=pd.read_csv("D:/360digiTMG/unsupervised/mod24 Deep Learning Black Box Technique-SVM/forestfires/forestfires.csv")

forestfires.head()
forestfires.describe()
forestfires.columns

sns.boxplot(x="size_category",y="DC",data=forestfires,palette = "hls")#vertical representation
sns.boxplot(x="size_category",y="DMC",data=forestfires,palette = "hls")#horizontal representation
#The color_palette function in Seaborn returns a list of colors defined in ... used to create a palette with evenly spaced colors in HLS hue space.


##Dropping the month and day columns
forestfires.drop(["month","day"],axis=1,inplace =True)

##Normalising the data as there is scale difference
predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

fires = norm_func(predictors)

from sklearn.svm import SVC
#import support vector classification
from sklearn.model_selection import train_test_split
#split the data into train and test datasets

x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# ########### kernel = linear ###########

help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test)
# Accuracy = 0.9692307692307692

# ######### Kernel = poly################

model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) 
# Accuracy = 0.9692307692307692

############kernel = rbf###################

model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) 
# Accuracy = 0.7538461538461538

############'sigmoid'##################
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test) #Accuracy = 73%
# Accuracy = 0.7538461538461538