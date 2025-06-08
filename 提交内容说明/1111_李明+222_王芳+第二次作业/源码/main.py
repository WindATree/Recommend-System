from public.split import train_test_split,txt_to_csv,csv_to_txt
from UserCF.user_cf import UserBasedCF
from ItemsCF.item_cf import ItemBasedCF
import pandas as pd
import os
import time
import tracemalloc

if __name__ == "__main__":

    # 转换成csv
    os.makedirs('Dataset', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    train_test_split()
    txt_to_csv('data/test.txt', 'Dataset/test_set.csv')

    train_data = pd.read_csv('Dataset/train_set.csv')
    valid_data = pd.read_csv('Dataset/valid_set.csv')
    all_train_data = pd.concat([train_data, valid_data], axis=0)
    test_data = pd.read_csv('Dataset/test_set.csv')

    # usercf

    start_time = time.time()
    tracemalloc.start()
    ucf = UserBasedCF()
    ucf.fit_cosine(all_train_data)
    ucf_time = time.time() - start_time
    ucf_current, ucf_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[UserCF] 时间: {ucf_time:.4f}s, 内存峰值: {ucf_peak / 1e6:.2f}MB")
    ucf.test(test_data, k=30,output_file="Results/usercf_predictions.csv", save = True, predict_func = "v2")
    csv_to_txt('Results/usercf_predictions.csv', 'Results/usercf_result.txt')

    # itemcf
    start_time = time.time()
    tracemalloc.start()
    icf = ItemBasedCF()
    icf.fit(all_train_data, similarity_method='pearson')
    icf_time = time.time() - start_time
    icf_current, icf_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[ItemCF] 时间: {icf_time:.4f}s, 内存峰值: {icf_peak / 1e6:.2f}MB")
    icf.test(test_data, output_file="Results/itemcf_predictions.csv", save = True,  k=27, abs = True)
    csv_to_txt('Results/itemcf_predictions.csv', 'Results/itemcf_result.txt')