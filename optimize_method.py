# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:06:11 2018

@author: Jerry
"""

import numpy as np
from scipy.optimize import fsolve
import random 
from scipy import stats
from copy import copy


  
def func(x):
    return x[0]**2 + x[1]**2 + (x[2] - 1)**2


class Gradient_optimizer(object):
    
    def __init__(self):
        self.FinalValue = None
        self.FinalParameter = None
    
    def get_first_derivative(self, func, delta, args):
        '''
        func:目标函数    
        delta:计算精度
        args:初始参数列表
        '''
            
        derivative = []
        for i in range(len(args)):
            x = copy(args)
            x[i] += delta
            derivative.append(round((func(x) - func(args)) / delta, 5)) #定义法
        return derivative
    
    def _modify_arg(self, arg, index, delta):
        new_arg = copy(arg)
        new_arg[index] += delta
        return new_arg
    
    def cal_hess(self, func, delta, args):
        hess_mat = np.zeros((len(args), len(args)), dtype=np.float64)
    
        for i in range(len(args)):
            
            for j in range(len(args)):
    
                sec_arg = self._modify_arg(arg=args, index=j, delta=delta)
                third_arg = self._modify_arg(arg=args, index=i, delta=delta)
                first_arg = self._modify_arg(arg=sec_arg, index=i, delta=delta)
                sec_d = (func(first_arg) - func(sec_arg) -func(third_arg) + func(args)) / delta**2
                sec_d = np.around(sec_d, 6)
                hess_mat[i, j] = sec_d
        return hess_mat
    
    def Gradient_desc(self, func, args, rate, sa=False, t=None, cool_ratio=None):
    
        current_value = func(args)
        aim_arg = np.array(args, dtype=np.float64)
        
        if sa:
            #模拟退火
            T = t
            cool_ratio = cool_ratio
            current_value = func(args)
            aim_arg = np.array(args, dtype=np.float64)
            
            while True and T > 0.001:
                first_gradient = np.array(self.get_first_derivative(func, delta=0.000001, args=aim_arg), 
                                          dtype=np.float64)
                
                temp = aim_arg - rate * first_gradient
        
                modified_value = func(temp)
                if modified_value < current_value:
                    current_value = func(temp)
                    aim_arg -= rate * first_gradient
        
                elif modified_value >= current_value:
                    
                    delta_y = modified_value - current_value
                    prob = np.exp(-delta_y / T)
                    rand = random.random()
                    T = T * cool_ratio
                    if rand < prob:
                        current_value = func(temp)
                        aim_arg -= rate * first_gradient
        
        
        else:
            
            while True:
                first_gradient = np.array(self.get_first_derivative(func, delta=0.00001, args=aim_arg), 
                                          dtype=np.float64)
                
                temp = aim_arg - rate * first_gradient
        
                modified_value = func(temp)
                if modified_value < current_value:
                    current_value = func(temp)
                    aim_arg -= rate * first_gradient
                
                elif modified_value >= current_value:
                    break
        
        self.FinalValue = current_value
        self.FinalParameter = aim_arg
        print("The final value is:{:.4f}".format(current_value))
        return aim_arg
    
    def Newton_method(self, func, args, rate):
        current_value = func(args)
        now_args = np.array(args, dtype=np.float64).reshape(-1, 1)
        while True:
            direction = -np.dot(np.linalg.inv(self.cal_hess(func=func, delta=0.000001, args=now_args)), 
                                now_args)
            
            temp = now_args + rate * direction
            
            
            modified_value = func(temp)
            if modified_value < current_value:
                
                now_args += rate * direction
                current_value = copy(modified_value)
                
            else:
                break
        self.FinalValue = current_value
        self.FinalParameter = now_args  
        print("The final value is: {:.4f}".format(current_value[0]))
        return now_args

# =============================================================================
# def _get_func(x, DataArray, Labels):
#     out_array = np.zeros_like(Labels)
#     variables = DataArray.shape[1]        
#     for var in range(variables):
#           out_array += DataArray[:, var].reshape(-1, 1) * x[var]
#     return np.mean((Labels - out_array) ** 2)    
# func([2,1])
# func = lambda x:_get_func(x, DataArray=DataArray,Labels=y)   
# ga=Gradient_optimizer()
# ga.Gradient_desc(func,[2,1],0.01)
# ga.Gradient_desc(func,[1,2,3],0.01,SA=True,T=300,cool_ratio=0.95)
# ga.Newton_method(func,[1,2,3],0.4)
# =============================================================================
