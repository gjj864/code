# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:01:20 2018

@author: Jerry
"""

import numpy as np
from scipy.optimize import fsolve
import random 
from scipy import stats
from copy import copy
from optimize_method import GA_optimizer, Gradient_optimizer



class LinearRegression(object):
    def __init__(self):
        self.FinalPara = None
        self.FitValue = None
        self.Mean = []
        self.Std = []
    
    def scale(self, DataArray):
        variables = DataArray.shape[1]
        data = np.zeros_like(DataArray)
        for var in range(variables):
            self.Mean.append(np.mean(DataArray[:, var]))
            self.Std.append(np.std(DataArray[:, var]))
            
            data[:, var] = (DataArray[:, var] - self.Mean[var]) / self.Std[var]
        return data
    def _get_func(self, x, DataArray, Labels):
        out_array = np.zeros_like(Labels)
        variables = DataArray.shape[1]        
        for var in range(variables):
           out_array += DataArray[:, var].reshape(-1, 1) * x[var]
        return np.mean((Labels - out_array) ** 2)    
    
    def fit(self, DataArray, labels, scale=True, intercept=True):
        self.intercept = intercept
        self.to_scale = scale
        if scale:            
            scales_data, scales_labels = self.scale(DataArray), self.scale(labels)
        else:
            scales_data, scales_labels = copy(DataArray), copy(labels)        
        
        if intercept:            
            one_join_array = np.insert(scales_data, 0, values=np.ones(len(scales_data)), axis=1)
        else:
            one_join_array = scales_data
        items, variables = one_join_array.shape
        x_initial = np.random.normal(loc=0,scale=5, size=variables)
        func = lambda x:self._get_func(x, DataArray=one_join_array,Labels=scales_labels)
        final_parameters = Gradient_optimizer().Gradient_desc(func, x_initial, rate=0.05,sa=False,t=200,cool_ratio=0.99)
        
# =============================================================================
#         for i in range(variables):
#             final_parameters[i] = final_parameters[i] * self.Std[i] + self.Mean[i]
# =============================================================================
        self.FinalPara = final_parameters
        return final_parameters
    
    def predict(self, DataArray):
        if self.to_scale:
            DataArray_to_modify = self.scale(DataArray)    
        else:
            DataArray_to_modify = copy(DataArray)
            
        if self.intercept:            
            one_join_array = np.insert(DataArray_to_modify, 0, values=np.ones(len(DataArray_to_modify)), axis=1)
        else:
            one_join_array = DataArray_to_modify
            
        num_y, num_var = one_join_array.shape      
        output = np.zeros(shape=num_y)        
        for j in range(num_var):
            output += one_join_array[:, j] * self.FinalPara[j]
        if self.to_scale:           
            output = output * self.Std[num_var] + self.Mean[num_var]
        self.FitValue = output
        return output


x1 = np.array([[1,3,6,7,10,2]]).reshape(-1, 1)
x2 = np.array([[4,2.9,0.8,3,2.4,2]]).reshape(-1, 1)

noise = np.random.normal(loc=0, scale=1, size=(x1.shape[0],1))
y = x1 * 5 - x2 * 3 + noise

x=np.c_[x1, x2]


z = LinearRegression()

z.fit(x,y,scale=False)
z.predict(x)
z.Std

x[:,1].shape

class LogisticRegression(object):
    def __init__(self):
        self.FinalPara = None
        self.FitValue = None
        self.Mean = []
        self.Std = []
    
    def scaled(self, DataArray):
        variables = DataArray.shape[1]
        data = np.zeros_like(DataArray)
        for var in range(variables):
            self.Mean.append(np.mean(DataArray[:, var]))
            self.Std.append(np.std(DataArray[:, var]))
            
            data[:, var] = (DataArray[:, var] - self.Mean[var]) / self.Std[var]
        return data
    def _get_func(self, x, DataArray, Labels):
        out_array = np.zeros_like(Labels)
        variables = DataArray.shape[1]        
        for var in range(variables):
           out_array += DataArray[:, var].reshape(-1, 1) * x[var]
           
        out_array = 1 / (1 + np.exp(-out_array))
        return -np.mean(Labels * np.log(out_array) + (1 - Labels) * np.log(1 - out_array))    
    
    def fit(self, DataArray, labels, scale=True, intercept=True):
        self.intercept = intercept
        self.to_scale = scale
        if scale:            
            scales_data= self.scaled(DataArray)
            scales_labels = copy(labels)
        else:
            scales_data = copy(DataArray)      
            scales_labels = copy(labels)  
        if intercept:            
            one_join_array = np.insert(scales_data, 0, values=np.ones(len(scales_data)), axis=1)
        else:
            one_join_array = scales_data
        items, variables = one_join_array.shape
        #x_initial = np.random.normal(loc=0,scale=1, size=variables)
        func = lambda x:self._get_func(x, DataArray=one_join_array,Labels=scales_labels)
        final_parameters = GA_optimizer().GA_minimize(func, num_variables=variables, max_iters=1000)
        
        self.FinalPara = final_parameters
        return final_parameters
    
    def scaled_pre(self, DataArray):
        variables = DataArray.shape[1]
        data = np.zeros_like(DataArray)
        for var in range(variables):
            data[:, var] = (DataArray[:, var] - self.Mean[var]) / self.Std[var]
        return data
    
    def predict(self, DataArray):
        if self.to_scale:
            DataArray_to_modify = self.scaled_pre(DataArray)    
        else:
            DataArray_to_modify = copy(DataArray)
            
        if self.intercept:            
            one_join_array = np.insert(DataArray_to_modify, 0, values=np.ones(len(DataArray_to_modify)), axis=1)
        else:
            one_join_array = DataArray_to_modify
            
        num_y, num_var = one_join_array.shape      
        output = np.zeros(shape=num_y)        
        for j in range(num_var):
            output += one_join_array[:, j] * self.FinalPara[j]
            
        output = 1 / (1 + np.exp(-output))
        self.FitValue = output
        return output   

rg = LogisticRegression()
x1 = np.array([[1,3,6,7,10,2]]).reshape(-1, 1)
x2 = np.array([[4,2.9,0.8,3,2.4,2]]).reshape(-1, 1)

noise = np.random.normal(loc=0, scale=1, size=(x1.shape[0],1))
y = x1 * 5 - x2 * 3 + noise
y[y<5]=0
y[y>=5]=1

x=np.c_[x1, x2]
rg.fit(x, y)
rg.predict(x)
