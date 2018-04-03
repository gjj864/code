# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:47:43 2018

@author: Jerry
"""
import numpy as np
import operator
from copy import copy

class DecisionTree(object):
    def __init__(self):
        self.NumFeature = None
        self.FeatureNames = None
        self.Tree = None
        
    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet: 
            #获得不同类别的计数字典
            currentLabel = featVec[-1]
            labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        shannonEnt = 0
        for key in labelCounts:
            #计算信息增益
            prob = labelCounts[key] / numEntries
            shannonEnt -= prob * np.log2(prob) 
        return shannonEnt
    
    def _splitDataSet(self, dataSet, axis, value):
        #划分数据集，返回对应axis和value的数据集
        index_tf = dataSet[:, axis] == value
        other_axis = [x for x in range(dataSet.shape[1]) if x != axis]
        retDataSet = copy(dataSet[index_tf, :][:, other_axis])
        return retDataSet
    
    def _chooseBestFeature(self, dataSet):
        """
        输入：数据集
        输出：最优特征的索引
        """
        numFeatures = dataSet.shape[1] - 1  #特征数量
        baseEntropy = self.calcShannonEnt(dataSet) 
        numSamples = dataSet.shape[0]
        bestInfoGain = 0
        bestFeature = -1
        for i in range(numFeatures):
            #对每一个特征计算信息增益
            unique_values = np.unique(dataSet[:, i])
            sub_ent = 0
            for value in unique_values:
                sub_data = dataSet[dataSet[:, i] == value, :] #根据不同特征值划分数据集
                prob = len(sub_data) / numSamples
                sub_ent += prob * self.calcShannonEnt(sub_data) #循环计算该特征的信息增益
            infoGain = baseEntropy - sub_ent
            if infoGain > bestInfoGain:
                #取信息增益最大的特征
                bestFeature = i
                bestInfoGain = infoGain
        return bestFeature

    def _majorityCnt(self, classList):
        """
        输入：类别列表或array
        输出：占最多数的类别
        """
        classCount={}
        for vote in classList:
            classCount[vote] = classCount.get(vote, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        #对计数字典按value排序
        return sortedClassCount[0][0]
    
    def _get_dim(self, dataSet):
        self.NumFeature = dataSet.shape[1]
        #对无特征名称的数据集设置默认特征名
        self.FeatureNames = ["Feature No." + str(name + 1) for name in range(self.NumFeature)]

    def _fit_min_max(self, dataSet, labels, min_samples_leaf, max_depth, variables_name=None):
        all_data = np.c_[dataSet, labels]
        numSamples = all_data.shape[0]
        variables_names = copy(variables_name)        
        
        if variables_names == None:
            variables_names = copy(self.FeatureNames)
            
        if len(np.unique(all_data[:, -1])) == 1:
            return all_data[:, -1][0]
        #min_samples_leaf, max_depth均设置
        if min_samples_leaf != None and max_depth != None:        
            if all_data.shape[1] == 1 or numSamples <= min_samples_leaf or (self.NumFeature - len(variables_names)) >= max_depth:
                return self._majorityCnt(all_data[:, -1])
        #min_samples_leaf
        elif min_samples_leaf != None and max_depth == None:
            if all_data.shape[1] == 1 or numSamples <= min_samples_leaf:
                return self._majorityCnt(all_data[:, -1])
        #max_depth    
        elif min_samples_leaf == None and max_depth != None:   
            if all_data.shape[1] == 1 or (self.NumFeature - len(variables_names)) >= max_depth:
                return self._majorityCnt(all_data[:, -1])
        #完全生长
        elif min_samples_leaf == None and max_depth == None:
            if all_data.shape[1] == 1 :
                return self._majorityCnt(all_data[:, -1])
        
        best_feature = self._chooseBestFeature(all_data) #选择最好的划分特征
        best_feature_name = variables_names[best_feature] #该特征的特征名
        unique_values = np.unique(all_data[:, best_feature]) #该特征下有哪些不同的取值
        Tree = {best_feature_name:{}} #树的结构为：特征名--特征值--（所取类别 or 特征名）--循环
        
        del variables_names[best_feature] #逐步删除已用特征
        
        for value in unique_values:
        #按每个不同的特征值划分出不同的子数据集            
            sub_all_data = self._splitDataSet(all_data, best_feature, value)
            #嵌套map，每有一个return，该条分支结束
            Tree[best_feature_name][value] = self._fit_min_max(sub_all_data[:, 0:-1], sub_all_data[:, -1], min_samples_leaf, max_depth, variables_names)
            
        return Tree
    

        
    def fit(self, dataSet, labels, min_samples_leaf=None, max_depth=None, variables_name=None):
        """
        dataSet:n_array形式
        labels:对应的类别标签array
        min_samples_leaf:继续划分所需最小样本数
        max_depth:树的最大深度
        variables_name:特征名
        """
        self._get_dim(dataSet)
        self.Tree = self._fit_min_max(dataSet, labels, min_samples_leaf, max_depth, variables_name)
            
        
    def _classify(self, tree, dataVec, variables_names):   
        """
        tree:字典形式保存的树结构
        datavec：单个样本的特征向量
        variables_names：特征向量名
        """
        firstStr = list(tree.keys())[0] #根节点的特征名
        secondDict = tree[firstStr] #该特征名对应的树
        featIndex = variables_names.index(firstStr) #该特征名对应的特征向量索引
        key = dataVec[featIndex]
        valueOfFeat = secondDict[key]  #该特征对应的值或者子树
        if isinstance(valueOfFeat, dict): 
            #对应的是子树则重新嵌套运行
            classLabel = self._classify(valueOfFeat, dataVec, variables_names)
        else: 
            #对应的是值则直接输出
            classLabel = valueOfFeat  
        return classLabel
 
    def predict(self, dataSet, variables_names=None):
        """
        dataSet:n_array形式
        variables_names:特征名
        """
        if variables_names == None:
            variables_names = copy(self.FeatureNames)
            
        output_array = []
        for item in dataSet:
            output_array.append(self._classify(self.Tree, item, variables_names))
            
        return np.array(output_array)    


class BinaryTree(object):
    def __init__(self):
        self.FeatureNames = None
        self.NumFeature = None
        self.Tree = None
    
    def calGINI(self, dataSet):
        #计算对应数据集的Gini系数
        numEntries = len(dataSet)
        labelCounts = {}
        #计数字典
        for featVec in dataSet: 
            currentLabel = featVec[-1]
            labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        Gini = 1
        for key in labelCounts:
            prob = labelCounts[key] / numEntries
            Gini -= prob ** 2
        return Gini
    
    def _splitDataSet(self, dataSet, axis, value):
        '''
        输入：数据集，数据集中某一特征列，该特征列中的某个取值
        功能：将数据集按特征列的某一取值换分为左右两个子数据集
        输出：左右子数据集
        '''
        index_left = dataSet[:, axis] <= value #小于该值得子数据集索引
        index_right = np.array([not x for x in index_left])
        other_axis = [x for x in range(dataSet.shape[1]) if x != axis] #子数据集不在含有已经用于划分的特征
        left_array = copy(dataSet[index_left, :][:, other_axis])
        right_array = copy(dataSet[index_right, :][:, other_axis])
        return left_array, right_array
    
    def _chooseBestFeature(self, dataSet):
        #以gini系数为判断标准，选择最好的特征
        numFeatures = dataSet.shape[1] - 1
        numSamples = dataSet.shape[0]
        bestFeature = -1
        base_gini = 1
        for i in range(numFeatures):
            unique_values = np.unique(dataSet[:, i])            
            for value in unique_values:
                left_data, right_data = self._splitDataSet(dataSet, i, value)
                left_gini = self.calGINI(left_data)
                right_gini = self.calGINI(right_data)
                sub_gini = (len(left_data) / numSamples) * left_gini + (len(right_data) / numSamples) * right_gini
                if base_gini > sub_gini:
                    best_value = copy(value)
                    base_gini = copy(sub_gini)
                    bestFeature = copy(i)
        return bestFeature, best_value
    
    def _majorityCnt(self, classList):
        #多数表决特征
        classCount={}
        for vote in classList:
            classCount[vote] = classCount.get(vote, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
    
    def _get_dim(self, dataSet):
        self.NumFeature = dataSet.shape[1]
        self.FeatureNames = ["Feature No." + str(name + 1) for name in range(self.NumFeature)]
    
    def _fit_min_max(self, dataSet, labels, min_samples_leaf=None, max_depth=None, variables_name=None):
        all_data = np.c_[dataSet, labels]
        numSamples = all_data.shape[0]
        variables_names = copy(variables_name)        
            
        if variables_names == None:
            variables_names = copy(self.FeatureNames)
                
        if len(np.unique(all_data[:, -1])) == 1:
            return all_data[:, -1][0]
            
        if min_samples_leaf != None and max_depth != None:        
            if all_data.shape[1] == 1 or numSamples <= min_samples_leaf or (self.NumFeature - len(variables_names)) >= max_depth:
                return self._majorityCnt(all_data[:, -1])
        elif min_samples_leaf != None and max_depth == None:
            if all_data.shape[1] == 1 or numSamples <= min_samples_leaf:
                return self._majorityCnt(all_data[:, -1])
        elif min_samples_leaf == None and max_depth != None:   
            if all_data.shape[1] == 1 or (self.NumFeature - len(variables_names)) >= max_depth:
                return self._majorityCnt(all_data[:, -1])
        elif min_samples_leaf == None and max_depth == None:
            if all_data.shape[1] == 1 :
                return self._majorityCnt(all_data[:, -1])
            
        best_feature, best_value = self._chooseBestFeature(all_data)
        best_feature_name = variables_names[best_feature]
    
        Tree = {best_feature_name:{}}
            
        del variables_names[best_feature]
                       
        
        sub_all_data_left, sub_all_data_right = self._splitDataSet(all_data, best_feature, best_value)
        #左分支，取值为该value
        Tree[best_feature_name][best_value] = self._fit_min_max(sub_all_data_left[:, 0:-1], sub_all_data_left[:, -1], min_samples_leaf, max_depth, variables_names)
        #右分，取值不为该value
        Tree[best_feature_name]["false"] = self._fit_min_max(sub_all_data_right[:, 0:-1], sub_all_data_right[:, -1], min_samples_leaf, max_depth, variables_names)
                
        return Tree
    
    def fit(self, dataSet, labels, min_samples_leaf=None, max_depth=None, variables_name=None):
        """
        dataSet:n_array形式
        labels:对应的类别标签array
        min_samples_leaf:继续划分所需最小样本数
        max_depth:树的最大深度
        variables_name:特征名
        """
        self._get_dim(dataSet)
        self.Tree = self._fit_min_max(dataSet, labels, min_samples_leaf, max_depth, variables_name)
        
    def _classify(self, tree, dataVec, variables_names):       
        firstStr = list(tree.keys())[0]
        secondDict = tree[firstStr]
        featIndex = variables_names.index(firstStr)
        key = copy(dataVec[featIndex])
        if not (key in secondDict):
            key = "false"
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict): 
            classLabel = self._classify(valueOfFeat, dataVec, variables_names)
        else: 
            classLabel = valueOfFeat  
        return classLabel
 
    def predict(self, dataSet, variables_names=None):
        if variables_names == None:
            variables_names = copy(self.FeatureNames)
            
        output_array = []
        for item in dataSet:
            output_array.append(self._classify(self.Tree, item, variables_names))
            
        return np.array(output_array)



    
rg=BinaryTree()
dataSet=np.array(dataSet)
label = dataSet[:, -1]
dataSet = dataSet[:, 0:-1]
rg.fit(dataSet=dataSet, labels=label, max_depth=3,min_samples_leaf=1)
rg.Tree
rg.predict(dataSet)
