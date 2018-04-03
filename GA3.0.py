# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:16:25 2018

@author: Jerry
"""

import numpy as np
from scipy.optimize import fsolve
import random 
from scipy import stats
from copy import copy



class GA_optimizer(object):
    
    def __init__(self):
        self.EncoderLength = None
        self.FinalValue = None
        self.NumVar = None
        self.loops = []
        self.record = []
        
    def getEncodedLength(self, delta=0.0001, boundarylist=[]):  
        # 每个变量的编码长度  
        lengths = []  
        for i in boundarylist:  
            lower = i[0]  
            upper = i[1]  
            # lamnda 代表匿名函数f(x)=0,50代表搜索的初始解  
            res = fsolve(lambda x: ((upper - lower) * 1 / delta) - 2 ** x - 1, 50)  
            length = int(np.floor(res[0]))  
            lengths.append(length)  
        return lengths
        
    def Initial_population(self, populations, encoder_length):
        #初始化种群，每一行为一个个体染色体，包含多个基因段
        total_encoder_length = sum(encoder_length)  #基因段总长度
        chromosomes = np.zeros((populations, total_encoder_length))  #个体数 * 基因总长度
        #随机初始化
        for i in range(populations):
            chromosomes[i, :] = np.random.randint(0, 2, size=total_encoder_length)        
        return chromosomes
    
    def decoded_Chromosome(self, one_chromosomes):    
        """
        解码单个染色体，
        one_chromosomes：单个染色体
        boundarylist：单个特征的上下界组成的list of list
        delta：计算精度，建议不要设的太小
        """
        decodedvalues = np.zeros((1, self.NumVar))  
        chromosome = one_chromosomes.tolist()  
        start = 0  
        for index, length in enumerate(self.EncoderLength):  
                # 将一个染色体进行拆分，得到染色体片段  
            power = length - 1  
                # 解码得到的10进制数字  
            demical = 0  
            for i in range(start, length + start):  
                demical += chromosome[i] * (2 ** power)  
                power -= 1  
            lower = self.boundarylist[index][0]  
            upper = self.boundarylist[index][1]  
                #解码公式
            decodedvalue = lower + demical * (upper - lower) / (2 ** length - 1)  
            decodedvalues[0, index] = decodedvalue  
                # 开始去下一段染色体的编码  
            start += length  
        return decodedvalues 
    
    def decodedChromosome(self, chromosomes, boundarylist, delta=0.0001):  
        #apply解码所有染色体         
        return np.apply_along_axis(self.decoded_Chromosome, 1, chromosomes).reshape(-1, self.NumVar)

    def cal_adaptation_value(self, func, decodedvalues):
        """
        计算适应度
        func：适应度函数
        decodedvalues：每个染色体解码完后组成的1维array
        """
        return np.apply_along_axis(func, 1, decodedvalues)
    
    def select_populations(self, adaptation_origin, populations, elitist_selection=False):
        """
        根据适应度选择用于cross的个体
        adaptation_origin：群体的适应度array
        populations：种群
        elitist_selection：是否将上一代的最优染色体直接加入下一代
        输出:与上一代size相同的用于cross的种群
        """
        selected_index = []
        adaptation_value = np.max(adaptation_origin) - adaptation_origin #最小化函数，所以需要对原始适应度变化
        prob = adaptation_value / sum(adaptation_value)
        for index, value in enumerate(adaptation_value):
            sel_by_stat = stats.multinomial.rvs(1, prob)  #轮盘选择
            selected_index.append(np.argmax(sel_by_stat))
            
        if elitist_selection:
            #精英选择，即将最后一个个体固定为前一种群的最优个体
            selected_index.pop()
            selected_index.append(np.argmax(prob))            
        return populations[selected_index,:]
    
    def cross(self, populations, cross_prob=0.7):
        """
        单点交叉，一次交叉产生两个染色体
        cross_prob：后代有多少是通过cross产生的，其余的直接从轮盘选择产生的染色体中随机选择产生
        """
        items, num = populations.shape
        updatepopulation = np.zeros_like(populations) #构造下一代种群
        
        num_to_cross = round(items * cross_prob) #cross产生的染色体数
        if num_to_cross % 2 == 1:
            num_to_cross = num_to_cross - 1
            
        crossed_index = [] #用于cross的母体索引
        for cr in range(num_to_cross):            
            # 产生随机索引  
            index = random.sample(range(items), 2)
            #保证不会选择同一个染色体
            while (populations[index[0], :] == populations[index[1], :]).all():
                index = random.sample(range(items), 2)
                
            crossed_index.append(index[0])
            crossed_index.append(index[1])
            start = 0
            for length in self.EncoderLength:                
                crossoverPoint = random.sample(range(start + 1, start + length), 1)  #为每一段基因选择交叉点
                crossoverPoint = crossoverPoint[0]  
                #交叉
                updatepopulation[cr, start:crossoverPoint] = populations[index[0], start:crossoverPoint]
                updatepopulation[cr, crossoverPoint:(start + length)] = populations[index[1], crossoverPoint:(start + length)]
                
                updatepopulation[items - 1 - cr, start:crossoverPoint] = populations[index[1], start:crossoverPoint]
                updatepopulation[items - 1 - cr, crossoverPoint:(start + length)] = populations[index[0], crossoverPoint:(start + length)]
                #开始下一段基因
                start += length
        #不杂交的数目直接从上一种群里随机选择填充
        not_cross_index = [i for i in range(items) if i not in crossed_index] #没有杂交的个体索引
        for not_cr in np.arange(num_to_cross, items - num_to_cross).tolist():            
            index = random.sample(not_cross_index, 1)
            updatepopulation[not_cr, :] = populations[index, :]            
        return updatepopulation

    def get_poisson_mu(self, loops, max_loop):
        #根据不同的循环次数选择不同的mu，使得开始时，每段基因上越靠前的腺嘌呤越容易发生变异
        mu=[]
        for var_num in self.EncoderLength:
            mu.append(self.modify_cross_ratio(loops, max_loop, var_num, 0))
        return mu
        
    def variation_index_poisson(self, populations, mu, Prob=0.01):
        update_variation = copy(populations)
        items, num = populations.shape
        num_to_var = int(np.ceil((items * Prob))) #要变异的染色体数目
        var_index = random.sample(range(items), num_to_var) #随机索引染色体
        for index_x in var_index:
            #对索引选择的每一条染色体进行变异
            start = 0
            index_y = stats.poisson.rvs(mu=mu,size=self.NumVar) #产生服从泊松分布的索引
            #对染色体上不同的基因段变异
            for variable, length in enumerate(self.EncoderLength):
                #防止索引超出范围
                if index_y[variable] >= length:
                    index_y[variable] = length - 1 
                index_y[variable] = start + index_y[variable]
                #开始下一段变异
                start += length
            #开始变异，1变0,0变1         
            for index in index_y:
                update_variation[index_x, index] = 1 - update_variation[index_x, index]
        return update_variation
                       
    def variation(self, populations, Prob=0.01):
        #随机将某个腺嘌呤换成其对立，无poisson版本
        update_variation = copy(populations)
        items, num = populations.shape
        num_to_var = int(np.floor((items * num * Prob)))         
        var_index = random.sample(range(items * num), num_to_var)
        
        for index in var_index:
            index_x = index // num
            index_y = (index % num) - 1
            if update_variation[index_x, index_y] == 0:
                update_variation[index_x, index_y] = 1
                
            else:
                update_variation[index_x, index_y] = 0
                
        return update_variation
    
    def modify_cross_ratio(self, num_loop, max_loop, init, min_value):
        #根据当前循环次数调整输出
        return min_value + (init - min_value) * num_loop / max_loop
        
    def GA_minimize(self, func, section_list=None, max_iters=500, 
                    num_of_population=None, variation_ratio=0.005, 
                    delta=0.001, elitist_selection=True, poisson_var=True,
                    early_stop=2):
        """
        func:目标函数 \n
        section_list：参数的区间list，形如[参数1的区间list,.....]，默认为自动补全 \n
        max_iters：最大迭代次数 \n
        num_of_population：初始化种群中染色体数目 \n
        variation_ratio：变异率的最低值 \n
        delta：精度，不宜过小 \n
        elitist_selection：是否进行精英选择 \n
        poisson_var：是否使用泊松分布进行变异 \n
        early_stop：如果临近的early_stop次迭代变化不大，则结束优化，尽量不要大于3 \n
        --------------------------------------------------------------------- \n
        目标函数仅接受一个自变量，需将所有需要的自变量打包成一个list \n
        如：\n
        def func(x):
            return x[0]**2 + x[1]**2 + (x[2] - 1)**2 
        用法：\n
        GA_me = GA_optimizer()
        GA_me.GA_minimize(func,max_iters=3000,early_stop=2)
        
        """
        if section_list==None: 
            section_list = [[-10, 10]] 
            while True:            
                try:
                    func([i for i in range(len(section_list))])            
                except IndexError:
                    section_list.append([-5,5])                
                else:
                    break  
        self.delta = delta
        self.NumVar = len(section_list)    
        self.boundarylist = section_list              
        encoder_length = self.getEncodedLength(delta=delta, boundarylist=section_list)
        
        self.EncoderLength = encoder_length
        if num_of_population is None:
            num_of_population = round(2 * sum(encoder_length))
        
        populations = self.Initial_population(num_of_population, encoder_length=encoder_length)
        adaptation_total_value = 1e15
        times = 0
        for i in range(max_iters):
            #杂交遗传
            cross_prob = self.modify_cross_ratio(i, max_iters, 0.9, 0.5)
            crossed = self.cross(populations, cross_prob=cross_prob)
            #变异
            if not poisson_var:                
                mod_variation_ratio=self.modify_cross_ratio(i, max_iters, 0.015, variation_ratio)
                varied_population = self.variation(populations=crossed, Prob=mod_variation_ratio)
            else:
                mod_variation_ratio=self.modify_cross_ratio(i, max_iters, 0.1, variation_ratio)
                mu = self.get_poisson_mu(i, max_iters)
                varied_population = self.variation_index_poisson(populations=crossed,                                                                 
                                                                 mu=mu, 
                                                                 Prob = mod_variation_ratio)
            #解码
            decoded_values = self.decodedChromosome(chromosomes=varied_population,
                                                   boundarylist=section_list,
                                                   delta=delta)
            #评估
            adaptation_values = self.cal_adaptation_value(func, decodedvalues=decoded_values)
            #比较
            varied_value = np.min(adaptation_values - adaptation_total_value)
            if varied_value < 0:
                adaptation_total_value = np.min(adaptation_values)
                populations = self.select_populations(adaptation_origin=adaptation_values, 
                                                      populations=varied_population,
                                                      elitist_selection=elitist_selection)
                if np.abs(varied_value) < 0.01:
                    times += 1
                else:
                    times = 0
            if times >= early_stop:
                break
                            
        optimal_index = np.argmin(adaptation_values)
        optimal_parameter = decoded_values[optimal_index]
        self.FinalValue = np.min(adaptation_values)
        self.loops.append(i + 1)
        self.record.append(self.FinalValue)
        print("{:.0f}杀,成功吃鸡！".format(i + 1))
        print("奖品为最终的参数 %s" % str(optimal_parameter))        
        return optimal_parameter

##########################  DEMO  ################################  
def func(x):
    return x[0]**2 + np.exp(x[1]**2) + (x[2] - 1)**2    
GA_me = GA_optimizer()
for v in range(5):
    GA_me.GA_minimize(func, poisson_var=False,early_stop=4)
m0 = np.mean(GA_me.record)
lo = np.mean(GA_me.loops)
sd0 = np.std(GA_me.record)
sd1 = np.std(GA_me.loops)

print(GA_me.EncoderLength)
print(GA_me.FinalValue)
##########################  DEMO  ################################  