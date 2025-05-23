import math
import time
import random
from collections import defaultdict
from scipy import optimize
import sys
import os
import csv
import numpy as np

# __file__ 是内置变量，包含了当前 Python 文件的完整路径
# abspath 将文件的部分路径或相对路径转为绝对路径
# dirname 返回路径的目录部分, 也就是最后一个斜杠之前的部分
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的路径
parent_dir = os.path.dirname(current_dir)
# 将上级目录添加到 sys.path
sys.path.append(parent_dir)

from public.config import *
from public.utils  import *
from public.split  import *

class SVDModel(object):
    def __init__(self):
        self.user_num=0 #用户总人数（实际人数<=usr_id）
        self.item_num=0 #同理
        self.rating_scale = [0, 100]  # 用户打分范围为0：100

    def loadTrainSet(self):
        """加载训练集并构建稀疏矩阵"""
        self.lil_matrix = []  
        # 全局平均评分
        self.overall_train_mean_rating = 0.0
        # 总的打分
        overall_rating_sum = 0.0
        # 训练数据总数
        self.num_trainingData = 0
        
        with open(training_set, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                raw_uid = int(row[0])
                raw_iid = int(row[1])
                user_rate = float(row[2])  
                                              
                uid = self.user_dict[raw_uid]
                iid = self.item_dict[raw_iid]   
                
                self.lil_matrix.append((uid, iid, user_rate))
                
                overall_rating_sum += user_rate
                self.num_trainingData += 1
                
        self.overall_train_mean_rating = overall_rating_sum / self.num_trainingData
