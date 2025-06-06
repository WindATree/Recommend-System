import os
from collections import defaultdict
import numpy as np
import csv
import psutil

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

def txt_to_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['user', 'item'])  # 写入CSV头部
        
        user_id = None
        num_items = 0
        items_read = 0
        
        for line in infile:
            line = line.strip()
            if '|' in line:
                # 这是用户行，格式为 <user id>|<numbers of rating items>
                if user_id is not None and items_read != num_items:
                    print(f"警告: 用户 {user_id} 的项目数量不匹配，预期 {num_items}，实际 {items_read}")
                
                user_id, num_items = line.split('|')
                user_id = user_id.strip()
                num_items = int(num_items.strip())
                items_read = 0
            else:
                # 这是项目行
                if user_id is not None and items_read < num_items:
                    item_id = line.strip()
                    writer.writerow([user_id, item_id])
                    items_read += 1
                    
def csv_to_txt(input_csv, output_txt):
    try:
        with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            with open(output_txt, 'w', encoding='utf-8') as txtfile:
                for row in reader:
                    # 将CSV行转换为字符串，并写入TXT文件
                    txtfile.write(','.join(row) + '\n')
        print(f"文件已成功转换并保存到 {output_txt}")
    except FileNotFoundError:
        print(f"文件未找到: {input_csv}")
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        
               
#获取当前进程使用内存信息
def getProcessMemory():
    # 获取当前进程的内存信息
    process = psutil.Process()
    memory_info = process.memory_info()
    # 获取程序使用的内存空间大小（以字节为单位）
    memory_usage = memory_info.rss
    return memory_usage/1024/1024 #返回用了多少MB内存