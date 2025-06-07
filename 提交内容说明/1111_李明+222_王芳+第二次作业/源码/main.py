from public.split import train_test_split,txt_to_csv,csv_to_txt
from UserCF.user_cf import UserBasedCF
from ItemsCF.item_cf import ItemBasedCF
import pandas as pd
import os

if __name__ == "__main__":

    # 转换成csv
    os.makedirs('Dataset', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    train_test_split()
    txt_to_csv('data/test.txt', 'Dataset/test_set.csv')

    train_data = pd.read_csv('Dataset/train_set.csv')
    valid_data = pd.read_csv('Dataset/valid_set.csv')
    test_data = pd.read_csv('Dataset/test_set.csv')

    # usercf

    ucf = UserBasedCF()
    ucf.fit_cosine(train_data)
    ucf.test(test_data, k=30,output_file="Results/usercf_predictions.csv", save = True, predict_func = "v2") 
    csv_to_txt('Results/usercf_predictions.csv', 'Results/usercf_result.txt')

    # itemcf
    icf = ItemBasedCF()
    icf.fit(train_data, similarity_method='pearson')
    icf.test(test_data, output_file="Results/itemcf_predictions.csv", save = True,  k=25, abs = True)
    csv_to_txt('Results/itemcf_predictions.csv', 'Results/itemcf_result.txt')