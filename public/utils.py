import os
from collections import defaultdict
import numpy as np

 
def file_read(file_path):
    """解析TXT文件"""
    # 创建一个defaultdict对象，默认值是一个空字典{}
    user_data = defaultdict(dict)
    item_data = defaultdict(dict)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            # 读取用户头信息
            user_header = f.readline().strip()
            if not user_header:
                break
                
            # 解析用户ID和评分数量
            user_id, num_ratings = user_header.split('|')
            num_ratings = int(num_ratings)
            
            # 读取该用户的所有评分
            for _ in range(num_ratings):
                rating_line = f.readline().strip()
                item_id, score = rating_line.split()
                score = float(score)
                
                # 构建双向索引
                user_data[user_id][item_id] = score
                item_data[item_id][user_id] = score
                
    return user_data, item_data