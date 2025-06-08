import math
import time
import random
from collections import defaultdict
from scipy import optimize
import sys
import os
import csv
import numpy as np
import tracemalloc 
import json

# 路径设置


from public.config import *
from public.utils import getProcessMemory 
from public.split import *

class SVDModel:
    def __init__(self):
        # 初始化模型参数
        self.factors = FACTORS
        self.epochs = EPOCHS
        self.learning_rate = LR
        self.LambdaUB = LAMBDAUB
        self.LambdaIB = LAMBDAIB
        self.LambdaP = LAMBDAP
        self.LambdaQ = LAMBDAQ
        self.rating_scale = [0, 100]
        
        # 数据存储
        self.user_dict = defaultdict()
        self.item_dict = defaultdict()
        self.user_num = 0
        self.item_num = 0
        self.lil_matrix = []  # 训练数据 (u, i, r)
        self.validation_set = []  # 验证数据 (u, i, r)
        
        # 模型参数
        self.P = []  # 用户因子矩阵
        self.Q = []  # 物品因子矩阵
        self.user_bias = []  # 用户偏置
        self.item_bias = []  # 物品偏置
        self.global_mean = 0.0  # 全局平均评分
        
        # 加载数据字典
        self.load_dictionaries()
        
        # 初始化模型矩阵
        self.initialize_matrices()

    def load_dictionaries(self):
        """加载用户和物品的映射字典"""
        
        # 创建用户ID到索引的映射
        user_data, _ = file_read('data/train.txt')  # 修改这里
        self.user_dict = {str(user_id): idx for idx, user_id in enumerate(user_data.keys())}
        # print(f"用户字典: {self.user_dict}")
        self.user_num = len(self.user_dict)
        
        # 创建物品ID到索引的映射
        _, item_data = file_read('data/train.txt')
        self.item_dict = {str(item_id): idx for idx, item_id in enumerate(item_data.keys())}
        self.item_num = len(self.item_dict)
        
        print(f"用户数量: {self.user_num}, 物品数量: {self.item_num}")

    def initialize_matrices(self):
        """初始化模型矩阵"""
        sqrt_factors = math.sqrt(self.factors)
        
        # 初始化用户矩阵 (用户数 × 因子数)
        self.P = [
            [random.random() / sqrt_factors for _ in range(self.factors)]
            for _ in range(self.user_num)
        ]
        
        # 初始化物品矩阵 (物品数 × 因子数)
        self.Q = [
            [random.random() / sqrt_factors for _ in range(self.factors)]
            for _ in range(self.item_num)
        ]
        
        # 初始化偏置项
        self.user_bias = [0.0 for _ in range(self.user_num)]
        self.item_bias = [0.0 for _ in range(self.item_num)]

    def load_train_set(self, train_file):
        """加载训练集"""
        self.lil_matrix = []
        total_rating = 0.0
        count = 0
        
        # 使用CSV格式训练集
        with open(train_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过标题行并打印标题行
            print(f"训练数据文件标题行: {header}")
            for row in reader:
                if len(row) < 3:
                    print(f"跳过不完整的行: {row}")
                    continue
                
                user_id = str(row[0])
                item_id = str(row[1])
                rating = float(row[2])
                # print(user_id, item_id, rating)  # 打印每行数据
                # 转换为内部索引
                uid = self.user_dict.get(user_id)
                iid = self.item_dict.get(item_id)
                # print(uid, iid, rating)  # 打印转换后的索引和评分
                if uid is not None and iid is not None:
                    self.lil_matrix.append((uid, iid, rating))
                    total_rating += rating
                    count += 1
        
        if count > 0:
            self.global_mean = total_rating / count
        else:
            self.global_mean = 50.0  # 默认平均值
        
        print(f"加载 {len(self.lil_matrix)} 条训练数据，平均评分: {self.global_mean:.2f}")

    def load_validation_set(self, valid_file):
        """加载验证集"""
        self.validation_set = []
        
        # 使用验证集
        with open(valid_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) < 3:
                    continue
                
                user_id = str(row[0])
                item_id = str(row[1])
                rating = float(row[2])
                
                # 转换为内部索引
                uid = self.user_dict.get(user_id)
                iid = self.item_dict.get(item_id)
                
                if uid is not None and iid is not None:
                    self.validation_set.append((uid, iid, rating))
        
        print(f"加载 {len(self.validation_set)} 条验证数据")

    def predict(self, u, i):
        """预测用户u对物品i的评分"""
        # 基础预测 = 全局平均 + 用户偏置 + 物品偏置
        prediction = self.global_mean + self.user_bias[u] + self.item_bias[i]
        
        # 添加隐因子点积
        prediction += sum(self.P[u][k] * self.Q[i][k] for k in range(self.factors))
        
        
        # 确保评分在有效范围内
        return max(self.rating_scale[0], min(prediction, self.rating_scale[1]))

    def train(self):
        """训练模型"""
        print("===== 开始训练 =====")
        start_time = time.time()
        best_val_rmse = float('inf')
        best_epoch = 0
        no_improve_count = 0  # 连续未改进的轮数
        patience = 5  # 连续5轮无改进则停止
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            total_error = 0.0
            
            # 随机打乱训练数据
            random.shuffle(self.lil_matrix)
            
            # 训练过程
            for u, i, r in self.lil_matrix:
                # 计算预测值
                pred = self.predict(u, i)
                error = r - pred
                total_error += error ** 2
                
                # 更新偏置
                self.user_bias[u] += self.learning_rate * (error - self.LambdaUB * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (error - self.LambdaIB * self.item_bias[i])
                
                # 更新隐因子
                for k in range(self.factors):
                    p_uk = self.P[u][k]
                    q_ik = self.Q[i][k]
                    
                    # 更新用户因子
                    self.P[u][k] += self.learning_rate * (error * q_ik - self.LambdaP * p_uk)
                    # 更新物品因子
                    self.Q[i][k] += self.learning_rate * (error * p_uk - self.LambdaQ * q_ik)
            
            # 计算训练误差
            train_rmse = math.sqrt(total_error / len(self.lil_matrix))
            
            # 验证集评估
            val_rmse, val_mae = self.evaluate(self.validation_set)
            
            # 保存最佳模型
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                no_improve_count = 0
                self.save_model()
                print(f"保存最佳模型 (epoch {epoch+1}, RMSE={val_rmse:.4f})")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"早停触发! 连续{patience}轮无改进")
                    break
                
        
            self.learning_rate *= DECAY_FACTOR
            
            # 打印epoch信息
            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch+1}/{self.epochs}: "
                  f"Train RMSE={train_rmse:.4f}, "
                  f"Val RMSE={val_rmse:.4f}, Val MAE={val_mae:.4f}, "
                  f"Time={epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"训练完成! 总用时: {total_time:.2f}秒")
        print(f"最佳验证RMSE: {best_val_rmse:.4f} (在epoch {best_epoch+1})")

        print("\n===== 使用整个训练集（包括验证集）重新训练 =====")
        self.load_model()
        print("已加载最佳模型参数")
        
        # 2. 将验证集合并到训练集中
        full_train_set = self.lil_matrix + self.validation_set
        print(f"合并后训练集大小: {len(full_train_set)} (原训练集: {len(self.lil_matrix)}, 验证集: {len(self.validation_set)})")
        self.lil_matrix = full_train_set
        
        # 3. 设置较小的学习率和较少的训练轮次进行微调
        original_lr = self.learning_rate
        self.learning_rate = original_lr * 0.1  # 使用更小的学习率
        additional_epochs = 5  # 微调轮次
        
        print(f"重新训练参数: 学习率={self.learning_rate}, 轮次={additional_epochs}")
        
        # 4. 进行微调训练
        for epoch in range(additional_epochs):
            epoch_start = time.time()
            total_error = 0.0
            random.shuffle(self.lil_matrix)
            
            for u, i, r in self.lil_matrix:
                pred = self.predict(u, i)
                error = r - pred
                total_error += error ** 2
                
                # 更新参数
                self.user_bias[u] += self.learning_rate * (error - self.LambdaUB * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (error - self.LambdaIB * self.item_bias[i])
                
                for k in range(self.factors):
                    p_uk = self.P[u][k]
                    q_ik = self.Q[i][k]
                    self.P[u][k] += self.learning_rate * (error * q_ik - self.LambdaP * p_uk)
                    self.Q[i][k] += self.learning_rate * (error * p_uk - self.LambdaQ * q_ik)
            
            # 计算训练误差
            train_rmse = math.sqrt(total_error / len(self.lil_matrix))
            epoch_time = time.time() - epoch_start
            
            # 学习率衰减
            self.learning_rate *= DECAY_FACTOR
            
            print(f"微调 Epoch {epoch+1}/{additional_epochs}: "
                  f"Train RMSE={train_rmse:.4f}, "
                  f"Time={epoch_time:.2f}s, LR={self.learning_rate:.6f}")
        
        # 5. 保存最终模型
        self.save_model()
        print("重新训练完成，最终模型已保存")
    def evaluate(self, dataset):
        """评估模型在指定数据集上的表现"""
        if not dataset:
            return float('inf')
            
        total_squared_error = 0.0
        total_abs_error = 0.0
        count = 0
        
        for u, i, r in dataset:
            pred = self.predict(u, i)
            error = r - pred
            total_squared_error += error ** 2
            total_abs_error += abs(error)
            count += 1
        
        rmse = math.sqrt(total_squared_error / count) if count > 0 else float('inf')
        mae = total_abs_error / count if count > 0 else float('inf')
        return rmse, mae
   
    
    def save_model(self):
        """保存模型到文件"""
        os.makedirs(RESULT_FOLDER, exist_ok=True)
        
        # 保存参数矩阵
        np.save(os.path.join(RESULT_FOLDER, 'P_matrix.npy'), np.array(self.P))
        np.save(os.path.join(RESULT_FOLDER, 'Q_matrix.npy'), np.array(self.Q))
        np.save(os.path.join(RESULT_FOLDER, 'user_bias.npy'), np.array(self.user_bias))
        np.save(os.path.join(RESULT_FOLDER, 'item_bias.npy'), np.array(self.item_bias))
        
        # 保存元数据
        metadata = {
            'user_dict': self.user_dict,
            'item_dict': self.item_dict,
            'user_num': self.user_num,
            'item_num': self.item_num,
            'factors': self.factors,
            'global_mean': self.global_mean
        }
        with open(os.path.join(RESULT_FOLDER, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        print(f"模型已保存到 {RESULT_FOLDER}")

    def load_model(self):
        """从文件加载模型"""
        # 加载参数矩阵
        self.P = np.load(os.path.join(RESULT_FOLDER, 'P_matrix.npy')).tolist()
        self.Q = np.load(os.path.join(RESULT_FOLDER, 'Q_matrix.npy')).tolist()
        self.user_bias = np.load(os.path.join(RESULT_FOLDER, 'user_bias.npy')).tolist()
        self.item_bias = np.load(os.path.join(RESULT_FOLDER, 'item_bias.npy')).tolist()
        
        # 加载元数据
        with open(os.path.join(RESULT_FOLDER, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            self.user_dict = metadata['user_dict']
            self.item_dict = metadata['item_dict']
            self.user_num = metadata['user_num']
            self.item_num = metadata['item_num']
            self.factors = metadata['factors']
            self.global_mean = metadata['global_mean']
        
        print("模型加载完成")

    def predict_test_set(self, test_file, output_file):
        """预测测试集并保存结果"""
        print("预测测试集...")
        test_data = []
        
        # 读取测试集
        with open(test_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) < 2:
                    continue
                user_id = int(row[0])
                item_id = int(row[1])
                test_data.append((user_id, item_id))
        
        # 预测并保存结果
        with open(output_file, 'w') as f:
            f.write("user_id,item_id,predicted_rating\n")
            for user_id, item_id in test_data:
                # 转换为内部索引
                uid = self.user_dict.get(str(user_id))
                iid = self.item_dict.get(str(item_id))
                
                if uid is not None and iid is not None:
                    pred = self.predict(uid, iid)
                else:
                    pred = self.global_mean  # 未知用户/物品使用全局平均
                
                f.write(f"{user_id},{item_id},{pred:.4f}\n")
        
        print(f"预测结果已保存到 {output_file}")

def get_memory_usage():
    """获取当前进程内存使用(MB)"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

if __name__ == "__main__":
    # 初始化内存跟踪
    tracemalloc.start()
    
    # 记录初始内存
    m1 = get_memory_usage()
    
    # 初始化模型
    model = SVDModel()
    m2 = get_memory_usage()
    
    # 加载数据
    print("加载训练数据...")
    model.load_train_set()
    print("加载验证数据...")
    model.load_validation_set()
    m3 = get_memory_usage()
    
    # 训练模型
    print("开始训练...")
    train_start = time.time()
    model.train()
    train_end = time.time()
    m4 = get_memory_usage()
    
    # 加载最佳模型
    model.load_model()
    
    # 在测试集上预测
    print("生成测试集预测...")
    test_start = time.time()
    model.predict_test_set(os.path.join(RESULT_FOLDER, 'test_predictions.csv'))
    test_end = time.time()
    m5 = get_memory_usage()
    
    # 内存报告
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print("\n===== 性能报告 =====")
    print(f"训练用时: {train_end - train_start:.2f}秒")
    print(f"测试预测用时: {test_end - test_start:.2f}秒")
    print(f"内存峰值: {peak / 1024 / 1024:.2f} MB")
    print(f"各阶段内存使用(MB):")
    print(f"初始化: {m1:.2f} -> {m2:.2f}")
    print(f"加载数据: {m2:.2f} -> {m3:.2f}")
    print(f"训练后: {m3:.2f} -> {m4:.2f}")
    print(f"预测后: {m4:.2f} -> {m5:.2f}")