import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from public.config import *

class UserBasedCF:
    def __init__(self):
        self.user_similarity = None
        self.train_data = None
        self.user_item_matrix = None
        self.mean_user_rating = None
    
    def fit(self, train_data):
        """使用训练集训练模型（仅对有评分的位置进行计算）"""
        self.train_data = train_data
        
        # 创建用户-物品评分矩阵（未评分位置保持NaN）
        self.user_item_matrix = self.train_data.pivot_table(
            index='user', columns='item', values='score'
        )
        
        # 计算每个用户的平均评分
        self.mean_user_rating = self.user_item_matrix.mean(axis=1)
        
        # 中心化评分
        ratings_diff = self.user_item_matrix.sub(self.mean_user_rating, axis=0)
        
        # 将NaN替换为0用于相似度计算
        ratings_diff_filled = ratings_diff.fillna(0)
        
        # 计算余弦相似度
        self.user_similarity = cosine_similarity(ratings_diff_filled)
        self.user_similarity = pd.DataFrame(
            self.user_similarity, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
        
        print("\n模型训练完成")
        print(f"用户数量: {len(self.user_item_matrix.index)}")
        print(f"物品数量: {len(self.user_item_matrix.columns)}")
        
        # 保存原始评分矩阵
        self.original_ratings = self.user_item_matrix.copy()
        # 保存中心化评分矩阵
        self.ratings_for_prediction = ratings_diff
    
    def predict(self, user_id, item_id, k=5):
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return self.mean_user_rating.mean()  # 返回全局平均分
        
        # 获取目标用户的平均评分
        mean_rating = self.mean_user_rating[user_id]
        
        # 获取对该物品评过分的用户
        rated_users = self.original_ratings.index[
            self.original_ratings[item_id].notna()
        ]

       
        if len(rated_users) == 0:
            return mean_rating  # 如果没有用户评过分，返回用户平均分
        
        
        # 获取目标用户与这些用户的相似度
        sim_scores = self.user_similarity.loc[user_id, rated_users]
        
        # 获取最相似的k个用户
        top_k_users = sim_scores.nlargest(k).index
        top_k_sim = sim_scores[top_k_users]
        
        # 获取这些用户对该物品的归一化评分
        top_k_ratings_diff = self.ratings_for_prediction.loc[top_k_users, item_id]
        
        weighted_sum = (top_k_ratings_diff * top_k_sim).sum()
        if top_k_sim.sum() != 0:
            predicted_rating = mean_rating + weighted_sum / top_k_sim.sum()
        else:
            predicted_rating = mean_rating
        
        # 确保评分在合理范围内
        predicted_rating = max(10, min(100, predicted_rating))
        
        return predicted_rating
    
    def evaluate(self, test_data, k=5):
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['user']
            item_id = row['item']
            actual = row['score']
            
            pred = self.predict(user_id, item_id, k)
            predictions.append(pred)
            actuals.append(actual)
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
        print(f"模型评估结果 (RMSE): {rmse:.2f}")
        
        return rmse
    
    def report(self, valid_set, output_file="evaluation_report.csv"):
        report_df = valid_set.copy()
        
        # 为每条记录生成预测值
        report_df['predicted_score'] = report_df.apply(
            lambda row: self.predict(row['user'], row['item']), 
            axis=1
        )
        
        # 计算评估指标
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(report_df['score'], report_df['predicted_score'])),
            'MAE': mean_absolute_error(report_df['score'], report_df['predicted_score']),
            'Avg_Actual': report_df['score'].mean(),
            'Avg_Predicted': report_df['predicted_score'].mean()
        }
        
        # 保存到CSV
        output_path = output_file
        report_df.to_csv(output_path, index=False)
        print(f"评估报告已保存到: {output_path}")
        
        # 打印指标
        print("\n===== 模型评估指标 =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        return metrics

if __name__ == "__main__":

    cf = UserBasedCF()

    train_data = pd.read_csv(training_set)
    valid_data = pd.read_csv(valid_set)
    
    # 构建模型并计算相似度矩阵
    cf.fit(train_data)
    
    # 计算验证集并评估模型性能
    cf.report(valid_data)