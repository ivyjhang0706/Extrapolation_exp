"""
Created on Wed August 21 16:20:00 2024

author: Wayne
"""

import os
import pandas as pd


def load_txt_file(file_path):
    if not os.path.exists(file_path):
        return print('No {} file'.format(file_path))

    txt_list = []
    file = open(file_path, 'r')
    for line in file.readlines():
        txt_list.append(float(line[:-1]))
    return txt_list


def load_parquet_file(file_path):
    if os.path.isfile(file_path):
        data_pq = pd.read_parquet(file_path, engine='pyarrow')
        return data_pq
    else:
        print("No parquet file '{}'".format(file_path))
        exit()

if __name__ == '__main__':
    df_data = load_parquet_file('D:/Wayne/GlucoseModel_New/Model/TwoClasses_SVMModel/1672/Features_005_2024_09_27_1142.parquet')
    print()