import os

split_size = 0.1

# 训练集
training_set="./../Dataset/train_set.csv"
# 验证集
valid_set="./../Dataset/valid_set.csv"
# 测试集
test_set = "./../Dataset/test_set.csv" 

# 结果输出路径
RESULT_FOLDER = "./../Results/"
os.makedirs(RESULT_FOLDER, exist_ok=True)

FACTORS = 20       # 隐因子数量
EPOCHS = 50        # 训练轮数
LR = 0.005         # 初始学习率
DECAY_FACTOR = 0.9 # 学习率衰减因子
LAMBDAUB = 0.02    # 用户偏置正则化系数
LAMBDAIB = 0.02    # 物品偏置正则化系数
LAMBDAP = 0.02     # P矩阵正则化系数
LAMBDAQ = 0.02     # Q矩阵正则化系数
