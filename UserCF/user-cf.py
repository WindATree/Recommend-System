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
        self.predict_func = {
            'normal': self.predict,
            'v1': self.predict_variant1,
            'v2': self.predict_variant2
        }
    
    def fit_cosine(self, train_data):
        self.train_data = train_data
        
        # 创建用户-物品评分矩阵
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
        
        print("\n模型训练完成(cos相似度)")
        
        # 保存原始评分矩阵
        self.original_ratings = self.user_item_matrix.copy()
        # 保存中心化评分矩阵
        self.ratings_for_prediction = ratings_diff
    
    def fit_pearson(self, train_data):
        self.train_data = train_data
        
        # 创建用户-物品评分矩阵
        self.user_item_matrix = self.train_data.pivot_table(
            index='user', columns='item', values='score'
        )
        
        # 计算每个用户的平均评分
        self.mean_user_rating = self.user_item_matrix.mean(axis=1)
        
        # 中心化评分
        ratings_diff = self.user_item_matrix.sub(self.mean_user_rating, axis=0)

        # 计算皮尔逊相关系数
        self.user_similarity = self.user_item_matrix.T.corr(method='pearson')
        
        print("\n模型训练完成（皮尔逊）")
        
        # 保存矩阵
        self.original_ratings = self.user_item_matrix.copy()
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
    
    def predict_variant1(self, user_id, item_id, k=5):
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
        
        # 变体1: 删除所有相关系数小于0的用户
        positive_sim_mask = top_k_sim > 0
        top_k_sim = top_k_sim[positive_sim_mask]
        
        # 如果删除后没有剩余用户，返回用户平均分
        if len(top_k_sim) == 0:
            return mean_rating
        
        # 获取筛选后用户对该物品的归一化评分
        filtered_users = top_k_sim.index
        top_k_ratings_diff = self.ratings_for_prediction.loc[filtered_users, item_id]
        
        weighted_sum = (top_k_ratings_diff * top_k_sim).sum()
        if top_k_sim.sum() != 0:
            predicted_rating = mean_rating + weighted_sum / top_k_sim.sum()
        else:
            predicted_rating = mean_rating
        
        # 确保评分在合理范围内
        predicted_rating = max(10, min(100, predicted_rating))
        
        return predicted_rating
    
    def predict_variant2(self, user_id, item_id, k=5):
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
        
        # 变体2: 对相关系数小于0的用户取相反数
        negative_sim_mask = top_k_sim < 0
        top_k_sim[negative_sim_mask] = -top_k_sim[negative_sim_mask]
        
        # 获取这些用户对该物品的归一化评分，并对负相关用户取相反数
        top_k_ratings_diff = self.ratings_for_prediction.loc[top_k_users, item_id].copy()
        top_k_ratings_diff[negative_sim_mask] = -top_k_ratings_diff[negative_sim_mask]
        
        weighted_sum = (top_k_ratings_diff * top_k_sim).sum()
        if top_k_sim.sum() != 0:
            predicted_rating = mean_rating + weighted_sum / top_k_sim.sum()
        else:
            predicted_rating = mean_rating
        
        # 确保评分在合理范围内
        predicted_rating = max(10, min(100, predicted_rating))
        
        return predicted_rating
    
    def report(self, valid_set, output_file="evaluation_report.csv",save = False, k=5, predict_func = "normal"):
        report_df = valid_set.copy()
        
        # 为每条记录生成预测值
        report_df['predicted_score'] = report_df.apply(
            lambda row: self.predict_func[predict_func](row['user'], row['item'],k), 
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
        if save:
            output_path = output_file
            report_df.to_csv(output_path, index=False)
            print(f"评估报告已保存到: {output_path}")
        
        # 打印指标
        print("\n===== 模型评估指标 =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        return metrics

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    # 设置中文字体和图表样式
    plt.style.use('ggplot')  # 使用可用的样式替代 'seaborn'
    rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    cf = UserBasedCF()
    train_data = pd.read_csv(training_set)
    valid_data = pd.read_csv(valid_set)

    #fig 1
    
    """"
    # 存储结果的字典
    results = {
        '中心化余弦': {'k': [], 'RMSE': [], 'MAE': []},
        '皮尔逊相关': {'k': [], 'RMSE': [], 'MAE': []}
    }
    
    # 测试中心化余弦相似度
    cf.fit_cosine(train_data)
    print("\n=== 中心化余弦相似度测试 ===")
    for k in range(1, 21):
        print(f"k={k}")
        metrics = cf.report(valid_data, k=k)
        results['中心化余弦']['k'].append(k)
        results['中心化余弦']['RMSE'].append(metrics['RMSE'])
        results['中心化余弦']['MAE'].append(metrics['MAE'])
    
    # 测试皮尔逊相关系数
    cf.fit_pearson(train_data)  # 使用中心化皮尔逊
    print("\n=== 皮尔逊相关系数测试 ===")
    for k in range(1, 21):
        print(f"k={k}")
        metrics = cf.report(valid_data, k=k)
        results['皮尔逊相关']['k'].append(k)
        results['皮尔逊相关']['RMSE'].append(metrics['RMSE'])
        results['皮尔逊相关']['MAE'].append(metrics['MAE'])
    
    # 创建美观的折线图
    plt.figure(figsize=(14, 6), dpi=100)
    
    # RMSE对比图
    plt.subplot(1, 2, 1)
    colors = ['#1f77b4', '#ff7f0e']  # 设置颜色
    for idx, method in enumerate(results):
        plt.plot(results[method]['k'], results[method]['RMSE'], 
                color=colors[idx],
                marker='o', linewidth=2, markersize=8,
                label=method)
    plt.xlabel('近邻数量 (k)', fontsize=12)
    plt.ylabel('RMSE (均方根误差)', fontsize=12)
    plt.title('不同相似度方法的RMSE对比', fontsize=14, pad=20)
    plt.legend(fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, 21, 2))  # 设置x轴刻度
    
    # 添加数据标签
    for method in results:
        for x, y in zip(results[method]['k'], results[method]['RMSE']):
            if x % 2 == 1:  # 只标记奇数k值
                plt.text(x, y, f'{y:.3f}', 
                        ha='center', va='bottom',
                        fontsize=8, color='black')
    
    # MAE对比图
    plt.subplot(1, 2, 2)
    for idx, method in enumerate(results):
        plt.plot(results[method]['k'], results[method]['MAE'], 
                color=colors[idx],
                marker='s', linewidth=2, markersize=8,
                label=method)
    plt.xlabel('近邻数量 (k)', fontsize=12)
    plt.ylabel('MAE (平均绝对误差)', fontsize=12)
    plt.title('不同相似度方法的MAE对比', fontsize=14, pad=20)
    plt.legend(fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, 21, 2))
    
    # 添加数据标签
    for method in results:
        for x, y in zip(results[method]['k'], results[method]['MAE']):
            if x % 2 == 1:  # 只标记奇数k值
                plt.text(x, y, f'{y:.3f}', 
                        ha='center', va='bottom',
                        fontsize=8, color='black')
    
    # 调整布局并显示
    plt.tight_layout(pad=3.0)
    
    # 保存图表
    plt.savefig('协同过滤算法评估对比.png', bbox_inches='tight', dpi=300)
    print("\n图表已保存为: 协同过滤算法评估对比.png")
    
    plt.show()
    """
    #fig2
    """
    # 存储结果的字典
    results = {
        '原始方法': {'k': [], 'RMSE': [], 'MAE': []},
        '变体1(删除负相关)': {'k': [], 'RMSE': [], 'MAE': []},
        '变体2(反转负相关)': {'k': [], 'RMSE': [], 'MAE': []}
    }

    # 使用中心化余弦相似度
    cf.fit_cosine(train_data)

    # 测试原始预测函数
    print("\n=== 原始预测函数测试 ===")
    for k in range(1, 21):
        print(f"k={k}")
        metrics = cf.report(valid_data, k=k)
        results['原始方法']['k'].append(k)
        results['原始方法']['RMSE'].append(metrics['RMSE'])
        results['原始方法']['MAE'].append(metrics['MAE'])

    # 测试变体1: 删除相关系数小于0的用户
    print("\n=== 变体1(删除负相关)测试 ===")
    for k in range(1, 21):
        print(f"k={k}")
        metrics = cf.report(valid_data, k=k, predict_func = "v1") 
        results['变体1(删除负相关)']['k'].append(k)
        results['变体1(删除负相关)']['RMSE'].append(metrics['RMSE'])
        results['变体1(删除负相关)']['MAE'].append(metrics['MAE'])

    # 测试变体2: 对负相关用户取相反数
    print("\n=== 变体2(反转负相关)测试 ===")
    for k in range(1, 21):
        print(f"k={k}")
        metrics = cf.report(valid_data, k=k,predict_func = "v2") 
        results['变体2(反转负相关)']['k'].append(k)
        results['变体2(反转负相关)']['RMSE'].append(metrics['RMSE'])
        results['变体2(反转负相关)']['MAE'].append(metrics['MAE'])

    # 创建美观的折线图
    plt.figure(figsize=(14, 6), dpi=300)

    # 设置颜色方案
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # 蓝色、绿色、红色

    # RMSE对比图
    plt.subplot(1, 2, 1)
    for idx, method in enumerate(results):
        plt.plot(results[method]['k'], results[method]['RMSE'], 
                color=colors[idx],
                marker='o', linewidth=2, markersize=6,
                label=method)

    plt.xlabel('近邻数量 (k)', fontsize=12)
    plt.ylabel('RMSE (均方根误差)', fontsize=12)
    plt.title('不同预测方法的RMSE对比', fontsize=14, pad=20)
    plt.legend(fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, 21, 2))  # 设置x轴刻度

    # 添加数据标签
    for method in results:
        for x, y in zip(results[method]['k'], results[method]['RMSE']):
            if x % 2 == 1:  # 只标记奇数k值
                plt.text(x, y, f'{y:.3f}', 
                        ha='center', va='bottom',
                        fontsize=8, color='black')

    # MAE对比图
    plt.subplot(1, 2, 2)
    for idx, method in enumerate(results):
        plt.plot(results[method]['k'], results[method]['MAE'], 
                color=colors[idx],
                marker='s', linewidth=2, markersize=6,
                label=method)

    plt.xlabel('近邻数量 (k)', fontsize=12)
    plt.ylabel('MAE (平均绝对误差)', fontsize=12)
    plt.title('不同预测方法的MAE对比', fontsize=14, pad=20)
    plt.legend(fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, 21, 2))

    # 添加数据标签
    for method in results:
        for x, y in zip(results[method]['k'], results[method]['MAE']):
            if x % 2 == 1:  # 只标记奇数k值
                plt.text(x, y, f'{y:.3f}', 
                        ha='center', va='bottom',
                        fontsize=8, color='black')

    # 调整布局并显示
    plt.tight_layout(pad=3.0)

    # 保存图表
    plt.savefig('协同过滤预测方法对比.png', bbox_inches='tight', dpi=300)
    print("\n图表已保存为: 协同过滤预测方法对比.png")

    plt.show()
    """
    # 存储结果的字典
    results = {
        '原始方法': {'k': [], 'RMSE': [], 'MAE': []},
        '变体1(删除负相关)': {'k': [], 'RMSE': [], 'MAE': []},
        '变体2(反转负相关)': {'k': [], 'RMSE': [], 'MAE': []}
    }

    # 使用中心化余弦相似度
    cf.fit_cosine(train_data)

    # 测试原始预测函数
    print("\n=== 原始预测函数测试 ===")
    for k in range(5, 55, 5):
        print(f"k={k}")
        metrics = cf.report(valid_data, k=k)
        results['原始方法']['k'].append(k)
        results['原始方法']['RMSE'].append(metrics['RMSE'])
        results['原始方法']['MAE'].append(metrics['MAE'])

    # 测试变体1: 删除相关系数小于0的用户
    print("\n=== 变体1(删除负相关)测试 ===")
    for k in range(5, 55, 5):
        print(f"k={k}")
        metrics = cf.report(valid_data, k=k, predict_func = "v1") 
        results['变体1(删除负相关)']['k'].append(k)
        results['变体1(删除负相关)']['RMSE'].append(metrics['RMSE'])
        results['变体1(删除负相关)']['MAE'].append(metrics['MAE'])

    # 测试变体2: 对负相关用户取相反数
    print("\n=== 变体2(反转负相关)测试 ===")
    for k in range(5, 55, 5):
        print(f"k={k}")
        metrics = cf.report(valid_data, k=k,predict_func = "v2") 
        results['变体2(反转负相关)']['k'].append(k)
        results['变体2(反转负相关)']['RMSE'].append(metrics['RMSE'])
        results['变体2(反转负相关)']['MAE'].append(metrics['MAE'])

    # 创建美观的折线图
    plt.figure(figsize=(14, 6), dpi=300)

    # 设置颜色方案
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # 蓝色、绿色、红色

    # RMSE对比图
    plt.subplot(1, 2, 1)
    for idx, method in enumerate(results):
        plt.plot(results[method]['k'], results[method]['RMSE'], 
                color=colors[idx],
                marker='o', linewidth=2, markersize=6,
                label=method)

    plt.xlabel('近邻数量 (k)', fontsize=12)
    plt.ylabel('RMSE (均方根误差)', fontsize=12)
    plt.title('不同预测方法的RMSE对比', fontsize=14, pad=20)
    plt.legend(fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(5, 55, 5))  # 设置x轴刻度

    # 添加数据标签
    for method in results:
        for x, y in zip(results[method]['k'], results[method]['RMSE']):
            if x % 2 == 1:  # 只标记奇数k值
                plt.text(x, y, f'{y:.3f}', 
                        ha='center', va='bottom',
                        fontsize=8, color='black')

    # MAE对比图
    plt.subplot(1, 2, 2)
    for idx, method in enumerate(results):
        plt.plot(results[method]['k'], results[method]['MAE'], 
                color=colors[idx],
                marker='s', linewidth=2, markersize=6,
                label=method)

    plt.xlabel('近邻数量 (k)', fontsize=12)
    plt.ylabel('MAE (平均绝对误差)', fontsize=12)
    plt.title('不同预测方法的MAE对比', fontsize=14, pad=20)
    plt.legend(fontsize=10, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(5, 55, 5))

    # 添加数据标签
    for method in results:
        for x, y in zip(results[method]['k'], results[method]['MAE']):
            if x % 2 == 1:  # 只标记奇数k值
                plt.text(x, y, f'{y:.3f}', 
                        ha='center', va='bottom',
                        fontsize=8, color='black')

    # 调整布局并显示
    plt.tight_layout(pad=3.0)

    # 保存图表
    plt.savefig('协同过滤预测方法对比(扩展探究).png', bbox_inches='tight', dpi=300)
    print("\n图表已保存为: 协同过滤预测方法对比(扩展探究).png")

    plt.show()
    