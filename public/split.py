import random
from public.utils import *
from public.config import *
import numpy as np
import math
import pandas as pd

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

def train_test_split():
    # 读取原始数据
    user_dict, item_dict = file_read('data/train.txt')
    
    # 第一阶段：为每个物品随机选择一个评分放入训练集
    train_records = []
    item_reserved = {}  # 记录每个物品保留的评分
    
    for item_id, ratings in item_dict.items():
        # 随机选择一个用户评分
        users = list(ratings.keys())
        selected_user = random.choice(users)
        
        train_records.append({
            'user': selected_user,
            'item': item_id,
            'score': ratings[selected_user]
        })
        item_reserved[item_id] = selected_user
    
    # 第二阶段：处理剩余评分
    valid_records = []
    
    for user_id, items in user_dict.items():
        for item_id, score in items.items():
            # 跳过已保留的评分
            if item_reserved.get(item_id) == user_id:
                continue
                
            # 随机划分
            if np.random.rand() < split_size:
                valid_records.append({
                    'user': user_id,
                    'item': item_id,
                    'score': score
                })
            else:
                train_records.append({
                    'user': user_id,
                    'item': item_id,
                    'score': score
                })
    
    # 转换为DataFrame
    train_df = pd.DataFrame(train_records)
    valid_df = pd.DataFrame(valid_records) if valid_records else pd.DataFrame(columns=['user','item','score'])
    train_df = train_df.astype({'user': int, 'item': int})
    train_df = train_df.sort_values(by=['user', 'item'], ascending=[True, True], ignore_index=True)
    train_df.to_csv('Dataset/train_set.csv', index=False)
    valid_df.to_csv('Dataset/valid_set.csv', index=False)
    
    print("训练集统计:")
    print(f"用户数: {train_df['user'].nunique()}")
    print(f"物品数: {train_df['item'].nunique()}")
    print(f"评分数: {len(train_df)}")
    
    print("\n验证集统计:")
    if not valid_df.empty:
        print(f"用户数: {valid_df['user'].nunique()}")
        print(f"物品数: {valid_df['item'].nunique()}")
        print(f"评分数: {len(valid_df)}")
    else:
        print("验证集为空，请检查split_size设置")

if __name__ == '__main__':
    # train_test_split()
    # txt_to_csv('data/test.txt', 'Dataset/test_set.csv')
    print(1)