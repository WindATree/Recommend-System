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

class ItemBasedCF:
    def __init__(self):
        self.item_similarity = None
        self.train_data = None
        self.user_item_matrix = None
        self.mean_item_rating = None
        self.global_mean = 0
        self.user_bias = None
        self.item_bias = None
    
    def fit(self, train_data):
        """使用训练集训练模型（仅对有评分的位置进行计算）"""
        self.train_data = train_data

        # 创建用户-物品评分矩阵
        self.user_item_matrix = self.train_data.pivot_table(
            index='user', columns='item', values='score'
        )

        # 计算每个物品的平均评分（可用于冷启动）
        self.mean_item_rating = self.user_item_matrix.mean(axis=0)

        # # 中心化评分（按列中心化）
        ratings_diff = self.user_item_matrix.sub(self.mean_item_rating, axis=1)

        # ratings_diff 是中心化过的评分（对每个 item）-> 皮尔逊相关系数
        # ratings_diff = self.user_item_matrix.sub(self.user_item_matrix.mean(axis=1), axis=0)

        self.global_mean = self.train_data['score'].mean()

        # 用0填充NaN用于相似度计算
        ratings_diff_filled = ratings_diff.fillna(self.global_mean)

        # 转置：物品为行，用户为列
        self.item_similarity = ratings_diff_filled.T.corr(method='pearson')
        # 计算物品相似度矩阵（基于列）
        # self.item_similarity = cosine_similarity(ratings_diff_filled.T)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )



        # 全局均值
        # self.global_mean = self.train_data['score'].mean()

        # 用户偏置
        self.user_bias = self.user_item_matrix.sub(self.mean_item_rating, axis=1).mean(axis=1).fillna(0)

        # 物品偏置
        self.item_bias = self.mean_item_rating - self.global_mean


        print("\n模型训练完成")
        print(f"用户数量: {len(self.user_item_matrix.index)}")
        print(f"物品数量: {len(self.user_item_matrix.columns)}")

        self.original_ratings = self.user_item_matrix.copy()
        self.ratings_for_prediction = ratings_diff
    
    def predict(self, user_id, item_id, k=5):
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return self.mean_item_rating.mean()  # 全局平均分

        # 获取该用户已评分的物品
        user_rated_items = self.original_ratings.loc[user_id].dropna()

        if len(user_rated_items) == 0:
            return self.mean_item_rating.get(item_id, self.mean_item_rating.mean())

        # 取出相似度
        sim_scores = self.item_similarity.loc[item_id, user_rated_items.index]

        # 选取最相似的K个物品
        top_k_items = sim_scores.nlargest(k).index
        top_k_sim = sim_scores[top_k_items]
        top_k_ratings = self.ratings_for_prediction.loc[user_id, top_k_items]

        # 添加用户偏置和物品偏置
        base_score = self.global_mean
        base_score += self.user_bias.get(user_id, 0)
        base_score += self.item_bias.get(item_id, 0)

        # 不使用偏置
        # base_score = self.mean_item_rating.get(item_id, self.global_mean)

        weighted_sum = (top_k_sim * top_k_ratings).sum()
        if top_k_sim.sum() != 0:
            pred = base_score + weighted_sum / top_k_sim.sum()
        else:
            pred = base_score

        # 限制在合理范围
        pred = max(10, min(100, pred))
        return pred
    
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

        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        print(f"模型评估结果 (RMSE): {rmse:.2f}")
        return rmse

    def report(self, valid_set, output_file="itemcf_evaluation_report.csv"):
        report_df = valid_set.copy()

        report_df['predicted_score'] = report_df.apply(
            lambda row: self.predict(row['user'], row['item']), axis=1
        )

        metrics = {
            'RMSE': np.sqrt(mean_squared_error(report_df['score'], report_df['predicted_score'])),
            'MAE': mean_absolute_error(report_df['score'], report_df['predicted_score']),
            'Avg_Actual': report_df['score'].mean(),
            'Avg_Predicted': report_df['predicted_score'].mean()
        }

        report_df.to_csv(output_file, index=False)
        print(f"评估报告已保存到: {output_file}")

        print("\n===== 模型评估指标 =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return metrics


if __name__ == "__main__":
    cf = ItemBasedCF()
    train_data = pd.read_csv(training_set)
    valid_data = pd.read_csv(valid_set)

    cf.fit(train_data)
    cf.report(valid_data)
