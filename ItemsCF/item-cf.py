import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

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
    
    def fit(self, train_data, similarity_method='pearson'):
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
        if similarity_method == 'pearson':
            self.item_similarity = ratings_diff_filled.T.corr(method='pearson')
        # 计算物品相似度矩阵（基于列）
        elif similarity_method == 'cosine':
            self.item_similarity = cosine_similarity(ratings_diff_filled.T)
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
    
    def predict(self, user_id, item_id, k=5, abs = False):
        # 类型转换 - 确保item_id是整数类型
        if isinstance(item_id, (float, np.float64)):
            item_id = int(item_id)
        
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

        # 测试sim先求绝对值
        if abs:
            top_k_sim = top_k_sim.abs()

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
    
    def evaluate(self, test_data, k=5, abs= False):
        predictions = []
        actuals = []

        for _, row in test_data.iterrows():
            user_id = row['user']
            item_id = row['item']
            actual = row['score']
            pred = self.predict(user_id, item_id, k, abs)
            predictions.append(pred)
            actuals.append(actual)

        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        print(f"模型评估结果 (RMSE): {rmse:.2f}")
        return rmse

    def report(self, valid_set, output_file="itemcf_evaluation_report.csv", k = 5, abs = False):
        report_df = valid_set.copy()

        report_df['predicted_score'] = report_df.apply(
            lambda row: self.predict(row['user'], row['item'], k, abs), axis=1
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

def compare_similarity_and_k_values(train_data, valid_data):
    """比较不同相似度计算方法和不同K值对推荐质量的影响"""
    # 测试不同的相似度计算方法
    similarity_methods = ['pearson', 'cosine']

    # 测试不同的K值（最近邻数量）
    # k_values = [5, 10, 15, 20, 25, 30, 40, 50]
    k_values = list(range(10,31))

    # 存储结果
    results = {}
    
    # 对每种相似度方法进行测试
    for method in similarity_methods:

        print(f"\n===== 测试相似度方法: {method} =====")
        rmse_values = []
        
        # 创建并训练模型
        cf = ItemBasedCF()
        cf.fit(train_data, similarity_method=method)
        # cf.fit(train_data, similarity_method='pearson')
        print(f"使用相似度方法: {method}")
        
        # 测试不同的K值
        for k in k_values:
            print(f"测试 K={k} 的性能...")
            rmse = cf.evaluate(valid_data, k=k)
            rmse_values.append(rmse)
        
        # 存储结果
        results[method] = rmse_values
    
    # 绘制比较图
    plt.figure(figsize=(10, 6))
    for method, rmse_values in results.items():
        plt.plot(k_values, rmse_values, marker='o', label=f"{method}")
    
    plt.xlabel('K值 (最近邻数量)')
    plt.ylabel('RMSE (均方根误差)')
    plt.title('不同相似度方法和K值的RMSE比较')
    # plt.title('不同预测公式和K值的RMSE比较')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    # output_file = "similarity_k_comparison.png"
    output_file = "detailed_similarity_k_comparison.png"
    # output_file = "formula_k_comparison.png"

    plt.savefig(output_file)
    print(f"\n比较结果图像已保存到: {output_file}")
    
    # 创建结果表格并保存为CSV
    results_df = pd.DataFrame(results, index=k_values)
    results_df.index.name = 'K值'
    # results_csv = "similarity_k_comparison.csv"
    results_csv = "detailed_similarity_k_comparison.csv"  
    # results_csv = "formula_k_comparison.csv"
    results_df.to_csv(results_csv)
    print(f"详细结果数据已保存到: {results_csv}")
    
    # 找出最佳组合
    best_method = None
    best_k = None
    best_rmse = float('inf')
    
    for method, rmse_values in results.items():
        for i, k in enumerate(k_values):
            if rmse_values[i] < best_rmse:
                best_rmse = rmse_values[i]
                best_method = method
                best_k = k
    
    print(f"\n最佳组合: 相似度方法={best_method}, K={best_k}, RMSE={best_rmse:.4f}")
    # print(f"\n最佳组合: ABS={best_method}, K={best_k}, RMSE={best_rmse:.4f}")
    
    return results, best_method, best_k, best_rmse

def compare_abs_and_k_values(train_data, valid_data, abs = False):
    """比较不同相似度计算方法和不同K值对推荐质量的影响"""
    # 测试不同的相似度计算方法
    # similarity_methods = ['pearson', 'cosine']

    # 测试是否取绝对值
    abs = [False, True]

    # 测试不同的K值（最近邻数量）
    # k_values = [5, 10, 15, 20, 25, 30, 40, 50]
    k_values = list(range(10,31))

    # 存储结果
    results = {}
    
    # 对每种相似度方法进行测试
    # for method in similarity_methods:

    # 对是否取绝对值测试
    for ifabs in abs:
        print(f"\n===== 测试相似度方法: {ifabs} =====")
        rmse_values = []
        
        # 创建并训练模型
        cf = ItemBasedCF()
        # cf.fit(train_data, similarity_method=method)
        cf.fit(train_data, similarity_method='pearson')
        print(f"使用相似度方法: pearson")
        
        # 测试不同的K值
        for k in k_values:
            print(f"测试 K={k} 的性能...")
            rmse = cf.evaluate(valid_data, k=k, abs=ifabs)
            rmse_values.append(rmse)
        
        # 存储结果
        results[method] = rmse_values
    
    # 绘制比较图
    plt.figure(figsize=(10, 6))
    for ifabs, rmse_values in results.items():
        plt.plot(k_values, rmse_values, marker='o', label=f"{ifabs}")
    
    plt.xlabel('K值 (最近邻数量)')
    plt.ylabel('RMSE (均方根误差)')
    # plt.title('不同相似度方法和K值的RMSE比较')
    plt.title('不同预测公式和K值的RMSE比较')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    # output_file = "similarity_k_comparison.png"
    # output_file = "detailed_similarity_k_comparison.png"
    output_file = "formula_k_comparison.png"


    plt.savefig(output_file)
    print(f"\n比较结果图像已保存到: {output_file}")
    
    # 创建结果表格并保存为CSV
    results_df = pd.DataFrame(results, index=k_values)
    results_df.index.name = 'K值'
    # results_csv = "similarity_k_comparison.csv"
    # results_csv = "detailed_similarity_k_comparison.csv"  
    results_csv = "formula_k_comparison.csv"
    results_df.to_csv(results_csv)
    print(f"详细结果数据已保存到: {results_csv}")
    
    # 找出最佳组合
    best_method = None
    best_k = None
    best_rmse = float('inf')
    
    for method, rmse_values in results.items():
        for i, k in enumerate(k_values):
            if rmse_values[i] < best_rmse:
                best_rmse = rmse_values[i]
                best_method = method
                best_k = k
    
    # print(f"\n最佳组合: 相似度方法={best_method}, K={best_k}, RMSE={best_rmse:.4f}")
    print(f"\n最佳组合: ABS={best_method}, K={best_k}, RMSE={best_rmse:.4f}")
    
    return results, best_method, best_k, best_rmse

def compare_abs(train_data, valid_data):
    # 先进行基本训练和评估
    print("\n===== 基本模型评估 =====")
    cf.fit(train_data)
    base_metrics = cf.report(valid_data)
    
    # 执行不同相似度方法和K值的比较
    # print("\n===== 开始比较不同相似度方法和K值 =====")
    print("\n===== 开始比较不同公式和K值 =====")
    results, best_method, best_k, best_rmse = compare_similarity_and_k_values(train_data, valid_data)
    
    # 使用最佳参数训练最终模型
    print("\n===== 使用最佳参数训练最终模型 =====")
    final_cf = ItemBasedCF()
    final_cf.fit(train_data, similarity_method='pearson')  # 使用最佳方法
    final_metrics = final_cf.report(valid_data, k=best_k, output_file="itemcf_best_model_report.csv", abs=best_method)
    
    # 比较基础模型和最佳模型
    print("\n===== 模型性能对比 =====")
    print(f"基础模型 RMSE: {base_metrics['RMSE']:.4f}")
    print(f"最佳模型 RMSE: {final_metrics['RMSE']:.4f}")
    print(f"改进百分比: {((base_metrics['RMSE'] - final_metrics['RMSE'])/base_metrics['RMSE']*100):.2f}%")


def compare_sim(train_data, valid_data):
    # 先进行基本训练和评估
    print("\n===== 基本模型评估 =====")
    cf.fit(train_data)
    base_metrics = cf.report(valid_data)
    
    # 执行不同相似度方法和K值的比较
    print("\n===== 开始比较不同相似度方法和K值 =====")
    # print("\n===== 开始比较不同公式和K值 =====")
    results, best_method, best_k, best_rmse = compare_similarity_and_k_values(train_data, valid_data)
    
    # 使用最佳参数训练最终模型
    print("\n===== 使用最佳参数训练最终模型 =====")
    final_cf = ItemBasedCF()
    final_cf.fit(train_data, similarity_method=best_method)  # 使用最佳方法
    final_metrics = final_cf.report(valid_data, k=best_k, output_file="itemcf_best_model_report.csv")
    
    # 比较基础模型和最佳模型
    print("\n===== 模型性能对比 =====")
    print(f"基础模型 RMSE: {base_metrics['RMSE']:.4f}")
    print(f"最佳模型 RMSE: {final_metrics['RMSE']:.4f}")
    print(f"改进百分比: {((base_metrics['RMSE'] - final_metrics['RMSE'])/base_metrics['RMSE']*100):.2f}%")


if __name__ == "__main__":
    cf = ItemBasedCF()
    train_data = pd.read_csv(training_set)
    valid_data = pd.read_csv(valid_set)

    # cf.fit(train_data)
    # cf.report(valid_data)
    compare_sim(train_data, valid_data)
    compare_abs(train_data, valid_data)
    