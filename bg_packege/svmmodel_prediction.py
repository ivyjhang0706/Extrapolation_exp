"""
Created on Wed Jun 24 15:22:00 2024

author: Wayne

Revise Record:


"""
import numpy as np
import pandas as pd
import sys
import os
import re
import datetime
import joblib
import shutil

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE
from collections import OrderedDict
##from tabulate import tabulate
import warnings

from bg_packege import data_management
from bg_packege import waves_detection
from bg_packege import features_extraction
from SWMlib.srj import dl_data
import globals

warnings.simplefilter(action="ignore")


def get_code_version(filename):
    return filename.split('_')[1]


def calculate_outlier(array):
    iqr = np.percentile(array, 75) - np.percentile(array, 25)
    filters = ((np.percentile(array, 75) - 1.5*iqr <= array)
               & (array <= np.percentile(array, 75) + 1.5 * iqr)
               )
    return filters


def delete_outlier(dataset_x, dataset_y, features, label):

    dataset = pd.concat([dataset_x, dataset_y], axis=1)
    dataset_x = dataset[features]
    filters = np.zeros(dataset_x.shape).T

    for i in range(len(features)):
        curr_data_x = dataset_x[features[i]].copy().reset_index(drop=True)
        filters[i] = calculate_outlier(curr_data_x)

    columns_to_keep = np.all(filters == 1, axis=0)
    filtered_x = dataset_x[columns_to_keep]
    filtered_y = dataset_y[columns_to_keep]
    filtered_x.loc[:, label] = filtered_y

    filtered_dataset_x = filtered_x[features]
    filtered_dataset_y = filtered_x[label]

    return filtered_dataset_x, filtered_dataset_y


def cal_performance_confusion(matrix, class_num=2):

    # formula: matrix[[tn, fp], [fn, tp]]
    if class_num == 2:
        tn = matrix[0, 0]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        tp = matrix[1, 1]

    else:
        total = sum(sum(matrix))
        tp = np.trace(matrix)
        fn = np.sum(matrix, axis=1) - np.diag(matrix)
        fp = np.sum(matrix, axis=0) - np.diag(matrix)
        tn = total - (tp+fn+fp)

    accuracy = (tn+tp) / sum(sum(matrix))
    precision = tp / (tp+fp)
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    try:
        f1 = 2 * (precision*sensitivity) / (precision+sensitivity)
    except ZeroDivisionError:
        f1 = 0.0

    return round(accuracy*100, 2), round(sensitivity*100, 2), round(specificity*100, 2), round(precision*100, 2), round(f1*100, 2)


def extract_date_from_filename(filename):
    # 假設日期格式固定為 YYYYMMDD
    match = re.search(r'\d{4}_\d{2}_\d{2}_\d{4}', filename)
    if match:
        return datetime.datetime.strptime(match.group(), '%Y_%m_%d_%H%M')
    return None


def find_best_or_latest_file(directory_path, file_type):
    """
    file_type= 'Features_':特徵擷取後的parquet檔,
               'Performance_': 績效,
               'StdScaler_': 模型,
               'Record_': 紀錄Quality check標準
    """
    best_perf_filepath = os.path.join(directory_path, 'Historic_Best_Performance.txt')
    if os.path.exists(best_perf_filepath):
        return best_perf_filepath

    else:  # 還沒有績效最好的 選擇最新訓練的模型
        latest_file = None
        latest_date = None
        threshold_date = datetime.datetime.strptime('2024-11-01 00:00:00', '%Y-%m-%d %H:%M:%S') # 取有做Quality check (最原始)
        for name in os.listdir(directory_path):

            if name.startswith(file_type):
                file_date = extract_date_from_filename(name)
                if file_date:
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = name

        return latest_file

def rename_best_performance_file(file_path, keep_days):
    """
    Rules
    1. 保留績效最好的特徵擷取parquet檔, 績效txt檔, 正規化模型, 特徵選取txt檔
    2. 刪除6個月前的檔案
    Returns:
    """
    # file_path = 'D:/SWM_Wayne/algorithm_blood_glucose/Model/Best_ThreeClasses_SVMModel/1727'
    dict_files = {}

    for fn in os.listdir(file_path):
        if fn.startswith('Performance'):
            curr_performance_filename = fn
            curr_acc = None
            curr_model_filename = None
            curr_std_filename = None
            curr_features_filename = None

            file_date = extract_date_from_filename(fn)
            file_content = open(os.path.join(file_path, fn))

            for i in file_content.readlines():
                if i.startswith('Acc'):
                    curr_acc = float(i.split(':')[1][:-1])
                if i.startswith('Model_Name'):
                    curr_model_filename = i.split(':')[1][:-1]
                if i.startswith('Standard_Scaler_Name'):
                    curr_std_filename = i.split(':')[1][:-1]
                if i.startswith('Feature_Selection_Name'):
                    curr_features_filename = i.split(':')[1][:-1]

            dict_files[file_date] = {'Acc': curr_acc,
                                     'Performance_FileName': curr_performance_filename,
                                     'Model_FileName': curr_model_filename,
                                     'Std_FileName': curr_std_filename,
                                     'Features_FileName': curr_features_filename
                                     }

    now_time = datetime.datetime.now()
    df_files = pd.DataFrame(dict_files).T
    df_files['Acc'] = pd.to_numeric(df_files['Acc'], errors='coerce')
    max_index_time = df_files['Acc'].idxmax()
    max_acc_row = df_files.loc[max_index_time]
    max_acc_perf_filename = max_acc_row['Performance_FileName']
    deadline_time = now_time - datetime.timedelta(days=keep_days)

    # del_df = df_files[:deadline_time]
    # keep_df = df_files[deadline_time:]
    del_df = df_files[df_files.index < deadline_time]
    keep_df = df_files[df_files.index >= deadline_time]
    best_perf_filename = 'Historic_Best_Performance.txt'
    def del_over_deadline_files(column_name):
        keep_features = keep_df[column_name]
        for filename in del_df[column_name]:
            if filename not in keep_features:
                os.remove(os.path.join(file_path, filename))
                print('Deleted file: {}'.format(filename))

    if max_index_time >= deadline_time:  # 最好的績效在keep_days(180天)內

        del_over_deadline_files('Performance_FileName')
        del_over_deadline_files('Features_FileName')
        del_over_deadline_files('Model_FileName')
        del_over_deadline_files('Std_FileName')

        if os.path.exists(os.path.join(file_path, best_perf_filename)):
            os.remove(os.path.join(file_path, best_perf_filename))
        shutil.copy(os.path.join(file_path, max_acc_perf_filename), os.path.join(file_path, best_perf_filename))

    else:  # 最好的績效在keep_days之前
        if os.path.exists(os.path.join(file_path, best_perf_filename)):
            os.remove(os.path.join(file_path, best_perf_filename))

        ## 下面這段可以再改進 (改成不用複製新的就能保留檔案)
        shutil.copy(os.path.join(file_path, max_acc_perf_filename), os.path.join(file_path, best_perf_filename))
        shutil.copy(os.path.join(file_path, max_acc_row['Model_FileName']),
                    os.path.join(file_path, 'Best_'+max_acc_row['Model_FileName']))
        shutil.copy(os.path.join(file_path, max_acc_row['Features_FileName']),
                    os.path.join(file_path, 'Best_'+max_acc_row['Features_FileName']))
        shutil.copy(os.path.join(file_path, max_acc_row['Std_FileName']), os.path.join(file_path, 'Best_'+max_acc_row['Std_FileName']))

        del_over_deadline_files('Performance_FileName')
        del_over_deadline_files('Features_FileName')
        del_over_deadline_files('Model_FileName')
        del_over_deadline_files('Std_FileName')

        shutil.copy(os.path.join(file_path, 'Best_' + max_acc_row['Model_FileName']),
                    os.path.join(file_path, max_acc_row['Model_FileName']))
        shutil.copy(os.path.join(file_path, 'Best_' + max_acc_row['Features_FileName']),
                    os.path.join(file_path, max_acc_row['Features_FileName']))
        shutil.copy(os.path.join(file_path, 'Best_' + max_acc_row['Std_FileName']),
                    os.path.join(file_path, max_acc_row['Std_FileName']))

    print(max_acc_row)
    print()

def decide_classify_level(df_features_parqfile):  # 決定要建立的分類模型
    """

    Parameters
    ----------
    df_features_parqfile

    Returns
    -------
    classify_level [str]:
    'High_Normal_Low'
    'High_Normal'
    'Low_Normal'
    'High_Low'
    """

    classify_level = "None"
    df_features_parqfile = df_features_parqfile.dropna()
    df_train = df_features_parqfile[df_features_parqfile["Dataset"] == "Train"]

    # 會看train跟test
    df_test = df_features_parqfile[df_features_parqfile["Dataset"] == "Test"]
    normal_bg_count = min(len(df_train[df_train['BG_Level'] == 'Normal']), len(df_test[df_test['BG_Level'] == 'Normal']))
    low_bg_count = min(len(df_train[df_train['BG_Level'] == 'Low']), len(df_test[df_test['BG_Level'] == 'Low']))
    high_bg_count = min(len(df_train[df_train['BG_Level'] == 'High']), len(df_test[df_test['BG_Level'] == 'High']))

    # 只看train
    # normal_bg_count = len(df_train[df_train['BG_Level'] == 'Normal'])
    # low_bg_count = len(df_train[df_train['BG_Level'] == 'Low'])
    # high_bg_count = len(df_train[df_train['BG_Level'] == 'High'])

    # low_bg_count = len(df_train[df_train['BG_Level'] == 'Low'])
    # high_bg_count = len(df_train[df_train['BG_Level'] == 'High'])

    if normal_bg_count > 0:

        if high_bg_count > 0 and low_bg_count > 0:  # 低中高血糖都有資料 分三類
            classify_level = 'High_Normal_Low'
        elif high_bg_count > 0:  # 只有中高血糖有資料
            classify_level = 'High_Normal'
        elif low_bg_count > 0:  # 只有中低血糖有資料
            classify_level = 'Low_Normal'
        else:  # 只有中血糖資料
            globals.errorcode = -505
            globals.message = ("An error occurred in the decide_classify_level function "
                               "of the SVM_Model_Builder_Predictor.py "
                               "from svmmodel_prediction.py: "
                               "Only or no normal data."
                               )
            return classify_level

    else:  # 沒有中血糖資料
        if high_bg_count > 0 and low_bg_count > 0:  # 低高血糖都有資料
            classify_level = 'High_Low'
        else:  # 高中低都沒有資料
            globals.errorcode = -505
            globals.message = ("An error occurred in the decide_classify_level function "
                               "of the SVM_Model_Builder_Predictor.py "
                               "from svmmodel_prediction.py: "
                               "Only or no normal data."
                               )
            return classify_level

    return classify_level

# 從uuid找已建立的模型類型
def know_modeled_type(uuid, model_path):

    """

    Parameters
    ----------
    uuid
    model_path

    Returns
    -------
    modeled_type: 'ThreeClasses_SVMModel', 'TwoClasses_SVMModel'
    modeled_level:
    'High_Normal_Low'
    'High_Normal'
    'Low_Normal'
    'High_Low'
    """

    two_classes_path = os.path.join(model_path, 'Best_TwoClasses_SVMModel', uuid)
    three_classes_path = os.path.join(model_path, 'Best_ThreeClasses_SVMModel', uuid)

    modeled_type = None
    modeled_level = None

    if os.path.exists(three_classes_path):
        modeled_type = 'ThreeClasses'
        modeled_level = 'High_Normal_Low'

    elif os.path.exists(two_classes_path):
        pref_txt_filename = find_best_or_latest_file(two_classes_path, "Performance_")
        lines = open(os.path.join(two_classes_path, pref_txt_filename)).readlines()
        for line in lines:
            if 'Two Classes Model' in line:
                modeled_type = 'TwoClasses'
                modeled_level = line.split(':')[1]
    else:
        print('No found built model to predict bg')

    return modeled_type, modeled_level

class SvmModelTwoClasses:
    def __init__(self, classify_group):
        self.label = 'BG_Category'
        self.classify_group = classify_group

    def clean_data(self, df):  # parquet file:  0: Normal, 1:High, 2:Low
        df = df.dropna()
        if self.classify_group == 'Low_Normal':  # 改為 0:Normal, 1:Low
            df = df[df[self.label] != 1]  # 只保留低血糖與正常血糖
            df[self.label] = pd.Series(df[self.label]).replace(2, 1)
            return df

        elif self.classify_group == 'High_Normal':  # 0:Normal, 1:High
            df = df[df[self.label] != 2]  # 只保留高血糖與正常血糖
            return df

        elif self.classify_group == 'High_Low':  # 0:Low, 1:High
            df = df[df[self.label] != 0]
            df[self.label] = pd.Series(df[self.label]).replace(1, 0)
            df[self.label] = pd.Series(df[self.label]).replace(2, 1)
            return df

    def build_model(self,  train_x, train_y, val_x):
        clf = svm.SVC(
            kernel='linear', C=1, gamma='auto',  # kernels_list = ['rbf', 'linear', 'poly']
            cache_size=500, random_state=42, probability=True
        )
        clf.fit(train_x, train_y)
        train_y_scores = clf.predict_proba(train_x)[:, 1]
        precision, sensitivity, thresholds = precision_recall_curve(train_y, train_y_scores)
        f1_scores = 2 * precision * sensitivity / (precision + sensitivity)
        best_threshold = thresholds[np.argmax(f1_scores[~np.isnan(f1_scores)])]
        train_pred = (train_y_scores >= best_threshold).astype(int)

        val_y_scores = clf.predict_proba(val_x)[:, 1]
        val_pred = (val_y_scores >= best_threshold).astype(int)

        return clf, train_pred, val_pred, best_threshold

    def predict_model(self, clf, data_x, best_threshold):

        data_prob = clf.predict_proba(data_x)
        data_y_scores = data_prob[:, 1]
        data_pred = (data_y_scores >= best_threshold).astype(int)

        return data_pred, data_prob

    def cal_performance(self, y_true, y_pred):

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        matrix = confusion_matrix(y_true, y_pred)
        accuracy, sensitivity, specificity, precision, f1score = cal_performance_confusion(matrix)
        tn, fp, fn, tp = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        auc = round(metrics.auc(fpr, tpr), 2)
        dict_performance = {
            'Accuracy': accuracy, 'Sensitivity': sensitivity,
            'Specificity': specificity, 'Precision': precision,
            'F1score': f1score, 'AUC': auc, 'TN': tn,
            'FP': fp, 'FN': fn, 'TP': tp
        }

        return dict_performance

    def save_performance(self, dict_performance, model_threshold, save_filename, model_filename, stdscaler_filename, feature_select_filename):

        try:
            acc_dataset = dict_performance['Accuracy']
            sen_dataset = dict_performance['Sensitivity']
            spe_dataset = dict_performance['Specificity']
            pre_dataset = dict_performance['Precision']
            f1_dataset = dict_performance['F1score']
            auc_dataset = dict_performance['AUC']
            tp = dict_performance['TP']
            fp = dict_performance['FP']
            tn = dict_performance['TN']
            fn = dict_performance['FN']

            f = open(save_filename, 'w')
            f.write('Two Classes Model:{}\n'.format(self.classify_group))
            f.write('Sensitivity:{}\n'.format(sen_dataset))
            f.write('Precision:{}\n'.format(pre_dataset))
            f.write('Specificity:{}\n'.format(spe_dataset))
            f.write('-------------\n')
            ## 要再改 兩類分類不一定是high 有可能是low
            f.write('High_Sensitivity:{}\n'.format(sen_dataset))
            f.write('Normal_Sensitivity:{}\n'.format(spe_dataset))
            f.write('-------------\n')
            f.write('F1:{}\n'.format(f1_dataset))
            f.write('Acc:{}\n'.format(acc_dataset))
            f.write('AUC:{}\n'.format(auc_dataset))
            f.write('TP:{}, FP:{}, FN:{}, TN:{}\n'.format(tp, fp, fn, tn))
            f.write('Model_Name:{}\n'.format(model_filename))
            f.write('Model_Threshold:{}\n'.format(model_threshold))
            f.write('Standard_Scaler_Name:{}\n'.format(stdscaler_filename))
            f.write('Feature_Selection_Name:{}\n'.format(feature_select_filename))
            f.close()

        except KeyError:
            print('Error: Not find key value')

    def save_performance_to_excel(self, uuid, excel_filename, dict_performance):
        if os.path.exists(excel_filename):
            df_perf = pd.read_excel(excel_filename)
        else:
            df_perf = pd.DataFrame({
                "UUID": [], "Model": [], "Sensitivity": [],
                "Precision": [], "Specificity": [], "F1": [], "AUC": [],
                "ACC": [], "TP": [], "FP": [],
                "FN": [], "TN": []
            })

        model = 'Two Classes'
        accuracy = dict_performance['Accuracy']
        sensitivity = dict_performance['Sensitivity']
        specificity = dict_performance['Specificity']
        precision = dict_performance['Precision']
        f1score = dict_performance['F1score']
        tp = dict_performance['TP']
        fp = dict_performance['FP']
        tn = dict_performance['TN']
        fn = dict_performance['FN']
        auc = dict_performance['AUC']

        new_row = pd.DataFrame({
            "UUID": uuid, "Model": model, "Sensitivity": sensitivity,
            "Precision": precision, "Specificity": specificity, "F1": f1score,
            "AUC": auc, "ACC": accuracy, "TP": tp, "FP": fp,
            "FN": fn, "TN": tn
        }, index=[0])

        df_perf = pd.concat([df_perf, new_row], ignore_index=True)
        df_perf.to_excel(excel_filename, index=False)

    def load_modeled_info(self, model_info_filepath, pref_txt_filename):

        features_selection = []

        f = open(os.path.join(model_info_filepath, pref_txt_filename))
        for line in f.readlines():
            if "Model_Name" in line:
                model_filename = line.split(":")[1][:-1]  # [:-1]刪除換行符號
            #### 修改!
            if "Model_Threshold" in line:
                threshold_str = line.split(":")[1][:-1]
                if threshold_str == 'None':
                    model_threshold = 0.5
                else:
                    model_threshold = round(float(threshold_str), 3)
            if "Standard_Scaler" in line:
                stdscaler_filename = line.split(":")[1][:-1]
            if "Feature_Selection" in line:
                feature_selection_filename = line.split(":")[1][:-1]
        f.close()

        # Load model
        model_fullpath = os.path.join(model_info_filepath, model_filename)
        stdscaler_fullpath = os.path.join(model_info_filepath, stdscaler_filename)
        loaded_clf = joblib.load(model_fullpath)
        loaded_stdscaler = joblib.load(stdscaler_fullpath)
        feature_selection_fullpath = os.path.join(model_info_filepath, feature_selection_filename)
        if os.path.exists(feature_selection_fullpath):
            f = open(feature_selection_fullpath, 'r')
            for col in f.readlines():
                features_selection.append(str(col[:-1]))
            # print(features_selection)

        return loaded_clf, loaded_stdscaler, model_threshold, features_selection

class SvmModelThreeClasses:
    def __init__(self, classify_group):
        self.label = 'BG_Category'
        self.classify_group = classify_group

    def clean_data(self, df):  # 0: Normal, 1:High, 2:Low
        df = df.dropna()
        normal_bg_count = len(df[df[self.label] == 0])
        high_bg_count = len(df[df[self.label] == 1])
        low_bg_count = len(df[df[self.label] == 2])
        # print('Count of normal BG: {}'.format(normal_bg_count))
        # print('Count of low BG: {}'.format(low_bg_count))
        # print('Count of high BG: {}'.format(high_bg_count))

        return df

    def build_model(self, train_x, train_y, val_x):
        clf = svm.SVC(
            kernel='linear', C=1, gamma='auto',  # kernels_list = ['rbf', 'linear', 'poly']
            cache_size=500, random_state=42, probability=True, decision_function_shape='ovo'
        )
        clf.fit(train_x, train_y)

        best_threshold = None  # 還沒加入動態調整Threshold
        # train_pred = clf.predict(train_x)
        train_prob = clf.predict_proba(train_x)
        train_pred = np.argmax(train_prob, axis=1)
        # val_pred = clf.predict(val_x)
        val_prob = clf.predict_proba(val_x)
        val_pred = np.argmax(val_prob, axis=1)

        return clf, train_pred, val_pred, best_threshold

    def predict_model(self, clf, data_x, best_threshold=None):

        if isinstance(data_x, np.ndarray):
            data_x = pd.DataFrame(data_x)

        data_x = data_x.dropna()

        if len(data_x) == 1:
            data_prob = clf.predict_proba(data_x)
            data_pred = int(np.argmax(data_prob))

        elif len(data_x) >1:
            data_prob = clf.predict_proba(data_x)
            data_pred = np.argmax(data_prob, axis=1)

        else: # data_x=0
            return None, None

        # if len(df_data) == 1:  # input sample=1
        #     if np.isnan(data_x).any():  # 有特徵擷取為空
        #         return None, None
        #     else:
        #         data_prob = clf.predict_proba(data_x)
        #         data_pred = int(np.argmax(data_prob))
        #
        # else:  # input sample>1
        #     data_prob = clf.predict_proba(data_x)
        #     data_pred = np.argmax(data_prob, axis=1)

        return data_pred, data_prob

    def cal_theclass_perf(self, tc, confusion_matrix):

        tp = confusion_matrix[tc][tc]
        fn = int(np.sum(confusion_matrix[tc]) - tp)
        fp = int(np.sum(np.array(confusion_matrix)[:, tc]) - tp)
        tn = int(np.sum(confusion_matrix) - (tp + fn + fp))

        try:
            accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        except ZeroDivisionError:
            accuracy = 0.0
        try:
            precision = tp / (tp+fp)
        except ZeroDivisionError:
            precision = 0.0
        try:
            specificity = tn / (tn+fp)
        except ZeroDivisionError:
            specificity = 0.0
        try:
            sensitivity = tp / (tp+fn)
        except ZeroDivisionError:
            sensitivity = 0.0
        try:
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        except ZeroDivisionError:
            f1_score = 0.0

        dict_performance = {
            'Accuracy': round(accuracy*100, 2), 'Sensitivity': round(sensitivity*100, 2),
            'Specificity': round(specificity*100, 2), 'Precision': round(precision*100, 2),
            'F1score': round(f1_score*100, 2), 'TN': tn,
            'FP': fp, 'FN': fn, 'TP': tp
        }

        return dict_performance

    def cal_micro_avg(self, classA, classB, classC):

        # Macro-average: (Sensitivity_A+Sensitivity_B+Sensitivity_C) / N
        # Micro-average: (TP_A+TP_B+TP_C) / (TP_A+FN_A+TP_B+FN_B+TP_C+FN_C)

        sensitivity_micro = ((classA['TP'] + classB['TP'] + classC['TP'])
                             / (classA['TP'] + classB['TP'] + classC['TP']
                                + classA['FN'] + classB['FN'] + classC['FN']))

        precision_micro = ((classA['TP'] + classB['TP'] + classC['TP'])
                           / (classA['TP'] + classB['TP'] + classC['TP']
                              + classA['FP'] + classB['FP'] + classC['FP']))

        specificity_micro = ((classA['TN'] + classB['TN'] + classC['TN'])
                             / (classA['TN'] + classB['TN'] + classC['TN']
                                + classA['FP'] + classB['FP'] + classC['FP']))

        classA_subject = classA['TP']+classA['FN']
        classB_subject = classB['TP']+classB['FN']
        classC_subject = classC['TP']+classC['FN']

        try:
            sensitivity_macro = (((classA['TP']/(classA['TP']+classA['FN'])*classA_subject)
                                 +(classB['TP']/(classB['TP']+classB['FN'])*classB_subject)
                                 +(classC['TP']/(classC['TP']+classC['FN'])*classC_subject))
                                    / (classA_subject+classB_subject+classC_subject))

            precision_macro = (((classA['TP']/(classA['TP']+classA['FP'])*classA_subject)
                                 +(classB['TP']/(classB['TP']+classB['FP'])*classB_subject)
                                 +(classC['TP']/(classC['TP']+classC['FP'])*classC_subject))
                                    / (classA_subject+classB_subject+classC_subject))

            specificity_macro = (((classA['TN']/(classA['TN']+classA['FP'])*classA_subject)
                                 +(classB['TN']/(classB['TN']+classB['FP'])*classB_subject)
                                 +(classC['TN']/(classC['TN']+classC['FP'])*classC_subject))
                                    / (classA_subject+classB_subject+classC_subject))
        except ZeroDivisionError:
            sensitivity_macro = 0.0
            precision_macro = 0.0
            specificity_macro = 0.0

        try:
            f1_micro = 2 * (precision_micro*sensitivity_micro) / (precision_micro+sensitivity_micro)
            f1_macro = 2 * (precision_macro*sensitivity_macro) / (precision_macro+sensitivity_macro)
        except ZeroDivisionError:
            f1_micro = 0.0
            f1_macro = 0.0

        dict_perf = {'Micro_avg': {'Sensitivity': round(sensitivity_micro*100, 2),
                     'Precision': round(precision_micro*100, 2),
                     'Specificity': round(specificity_micro*100, 2),
                     'F1score': round(f1_micro*100, 2)
                     },'Macro_avg': {
                     'Sensitivity': round(sensitivity_macro * 100, 2),
                     'Precision': round(precision_macro * 100, 2),
                     'Specificity': round(specificity_macro * 100, 2),
                     'F1score': round(f1_macro * 100, 2)
                     }}

        return dict_perf

    # 為跟Dennis的NN多類模型相同 將三類轉成二類計算績效
    def cal_performance(self, y_true, y_pred):
        """
        Returns:
            dict_performance
            format:
            {'Normal':
                {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1score', 'TN', 'FP', 'FN', 'TP'},
            'Low':
                {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1score', 'TN', 'FP', 'FN', 'TP'},
            'High':
                {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1score', 'TN', 'FP', 'FN', 'TP'},
            'TwoLevel':
                {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1score', 'TN', 'FP', 'FN', 'TP'},
            'Micro_avg':
                {'Sensitivity', 'Precision', 'Specificity', 'F1score'},
            'Macro_avg':
                {'Sensitivity', 'Precision', 'Specificity', 'F1score'},
            'Confusion_Matrix':
            }
        """

        cf_mx = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        dict_normal_perf = self.cal_theclass_perf(0, cf_mx)
        dict_high_perf = self.cal_theclass_perf(1, cf_mx)
        dict_low_perf = self.cal_theclass_perf(2, cf_mx)

        dict_performance = {'Normal': dict_normal_perf, 'Low': dict_low_perf, 'High': dict_high_perf}
        dict_performance.update(self.cal_micro_avg(dict_normal_perf, dict_low_perf, dict_high_perf))
        dict_performance['Confusion_Matrix'] = cf_mx

        # 3類改2類
        y_true[y_true == 2] = 1  # 將所有2轉為1 -> 中血糖0 低高血糖為1
        y_pred[y_pred == 2] = 1

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        matrix = confusion_matrix(y_true, y_pred)
        accuracy, sensitivity, specificity, precision, f1score = cal_performance_confusion(matrix)
        tn, fp, fn, tp = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        auc = round(metrics.auc(fpr, tpr), 2)
        dict_twolevel = {'TwoLevel':
                             {'Accuracy': accuracy, 'Sensitivity': sensitivity,
                              'Specificity': specificity, 'Precision': precision,
                              'F1score': f1score, 'AUC': auc, 'TN': tn,
                              'FP': fp, 'FN': fn, 'TP': tp}
                         }
        dict_performance.update(dict_twolevel)
        return dict_performance


    def save_performance(self, dict_performance, model_threshold, save_filename, model_filename, stdscaler_filename, feature_select_filename):

        try:
            acc_macro = dict_performance['Normal']['Accuracy']
            sen_macro = dict_performance['Macro_avg']['Sensitivity']
            spe_macro = dict_performance['Macro_avg']['Specificity']
            pre_macro = dict_performance['Macro_avg']['Precision']
            f1_macro = dict_performance['Macro_avg']['F1score']

            sensitivity_high = dict_performance['High']['Sensitivity']
            sensitivity_low = dict_performance['Low']['Sensitivity']
            sensitivity_normal = dict_performance['Normal']['Sensitivity']

            tp_twolevel = dict_performance['TwoLevel']['TP']
            fp_twolevel = dict_performance['TwoLevel']['FP']
            tn_twolevel = dict_performance['TwoLevel']['TN']
            fn_twolevel = dict_performance['TwoLevel']['FN']

            cm = dict_performance['Confusion_Matrix']

            f = open(save_filename, 'w')
            f.write('Three Classes Model:{}\n'.format(self.classify_group))
            # 為跟Dennis的NN多類模型相同 將三類轉成二類儲存Sensitivity, Specificity, F1, Acc, TP, FP, FN, TN
            f.write('Sensitivity:{}\n'.format(sen_macro))
            f.write('Specificity:{}\n'.format(spe_macro))
            f.write('-----------------------\n')
            f.write('High_Sensitivity:{}\n'.format(sensitivity_high))
            f.write('Low_Sensitivity:{}\n'.format(sensitivity_low))
            f.write('Normal_Sensitivity:{}\n'.format(sensitivity_normal))
            f.write('-----------------------\n')
            f.write('F1:{}\n'.format(f1_macro))
            f.write('Acc:{}\n'.format(acc_macro))
            f.write('TP:{}, FP:{}, FN:{}, TN:{}\n'.format(tp_twolevel, fp_twolevel, fn_twolevel, tn_twolevel))
            f.write('Confusion Matrix: \n{}\n'.format(cm))
            f.write('\n')
            f.write('Model_Name:{}\n'.format(model_filename))
            f.write('Model_Threshold:{}\n'.format(model_threshold))
            f.write('Standard_Scaler_Name:{}\n'.format(stdscaler_filename))
            f.write('Feature_Selection_Name:{}\n'.format(feature_select_filename))
            f.close()

        except KeyError:
            print('Error: Not find key value')

    def save_performance_to_excel(self, uuid, excel_filename, dict_performance):
        if os.path.exists(excel_filename):
            df_perf = pd.read_excel(excel_filename)
        else:
            df_perf = pd.DataFrame({
                "UUID": [], "Model": [], "Sensitivity": [],
                "Precision": [], "Specificity": [], "F1": [], "AUC": [],
                "ACC": [], "TP": [], "FP": [],
                "FN": [], "TN": []
            })
        model = 'Three Classes'
        try:
            accuracy = dict_performance['Normal']['Accuracy']
            sensitivity = dict_performance['Micro_avg']['Sensitivity']
            specificity = dict_performance['Micro_avg']['Specificity']
            precision = dict_performance['Micro_avg']['Precision']
            f1score = dict_performance['Micro_avg']['F1score']
        except KeyError:
            print('Key Error')

        tp = None
        fp = None
        tn = None
        fn = None
        auc = None

        new_row = pd.DataFrame({
            "UUID": uuid, "Model": model, "Sensitivity": sensitivity,
            "Precision": precision, "Specificity": specificity, "F1": f1score,
            "AUC": auc, "ACC": accuracy, "TP": tp, "FP": fp,
            "FN": fn, "TN": tn
        }, index=[0])

        df_perf = pd.concat([df_perf, new_row], ignore_index=True)
        df_perf.to_excel(excel_filename, index=False)

    def load_modeled_info(self, model_info_filepath, pref_txt_filename):

        features_selection = []
        model_threshold = 0.5  # initial

        f = open(os.path.join(model_info_filepath, pref_txt_filename))
        for line in f.readlines():
            if "Model_Name" in line:
                model_filename = line.split(":")[1][:-1]  # [:-1]刪除換行符號
            if "Standard_Scaler" in line:
                stdscaler_filename = line.split(":")[1][:-1]
            if "Feature_Selection" in line:
                feature_selection_filename = line.split(":")[1][:-1]
        f.close()

        # Load model
        model_fullpath = os.path.join(model_info_filepath, model_filename)
        stdscaler_fullpath = os.path.join(model_info_filepath, stdscaler_filename)
        loaded_clf = joblib.load(model_fullpath)
        loaded_stdscaler = joblib.load(stdscaler_fullpath)
        feature_selection_fullpath = os.path.join(model_info_filepath, feature_selection_filename)
        if os.path.exists(feature_selection_fullpath):
            f = open(feature_selection_fullpath, 'r')
            for col in f.readlines():
                features_selection.append(str(col[:-1]))

        return loaded_clf, loaded_stdscaler, model_threshold, features_selection

class SvmModelHelper:
    def __init__(self, uuid, results_filepath):
        self.uuid = uuid
        self.results_filepath = results_filepath

    def train_svm_model(self, features_parqpath):

        """

        Parameters
        ----------
        results_filepath
        classify_level[str]: 'High', 'Low'

        Returns
        -------

        """

        is_down_sampling = False
        is_delete_outlier = True
        is_over_sampling = True

        label = 'BG_Category'
        # Selected features
        features = [
            'rr_interval', 'hr', 'pr_duration',
            'pr_corrections3', 'qr_duration', 'qr_corrections3',
            'rs_duration', 'rs_corrections3', 'rt_duration',
            'rt_corrections3', 'pq_duration',  'pq_corrections3',
            'ps_duration', 'ps_corrections3', 'pt_duration',
            'pt_corrections3', 'qs_duration', 'qs_corrections3',
            'qt_duration', 'qt_corrections3', 'st_duration',
            'st_corrections3', 'p_left_sharp', 'p_right_sharp',
            'r_left_sharp', 'r_right_sharp', 't_left_sharp',
            't_right_sharp'
        ]
        # All features
        # features = [
        #     "rr_interval", "hr",
        #     "p_value", "q_value", "r_value", "s_value", "t_value",
        #     "pr_duration", "pr_amplitude", "pr_distances", "pr_directions",
        #     "pr_slope", "pr_corrections3",
        #     "qr_duration", "qr_amplitude", "qr_distances", "qr_directions",
        #     "qr_slope", "qr_corrections3",
        #     "rs_duration", "rs_amplitude", "rs_distances", "rs_directions",
        #     "rs_slope", "rs_corrections3",
        #     "rt_duration", "rt_amplitude", "rt_distances", "rt_directions",
        #     "rt_slope", "rt_corrections3",
        #     "pq_duration", "pq_amplitude", "pq_distances", "pq_directions",
        #     "pq_slope", "pq_corrections3",
        #     "ps_duration", "ps_amplitude", "ps_distances", "ps_directions",
        #     "ps_slope", "ps_corrections3",
        #     "pt_duration", "pt_amplitude", "pt_distances", "pt_directions",
        #     "pt_slope", "pt_corrections3",
        #     "qs_duration", "qs_amplitude", "qs_distances", "qs_directions",
        #     "qs_slope", "qs_corrections3",
        #     "qt_duration", "qt_amplitude", "qt_distances", "qt_directions",
        #     "qt_slope", "qt_corrections3",
        #     "st_duration", "st_amplitude", "st_distances", "st_directions",
        #     "st_slope", "st_corrections3",
        #     "p_left_slope", "p_right_slope", "p_left_sharp", "p_right_sharp", "p_tilt",
        #     "r_left_slope", "r_right_slope", "r_left_sharp", "r_right_sharp", "r_tilt",
        #     "t_left_slope", "t_right_slope", "t_left_sharp", "t_right_sharp", "t_tilt"
        # ]


        # Load dataset
        features_filename = find_best_or_latest_file(features_parqpath, "Features")
        if features_filename == None:
            globals.errorcode = -506
            globals.message = (
                "An error occurred in the build_model function of the SVM_Model_Builder_Predictor.py "
                "from bg_package.swmmodel_prediction: "
                "fail to read the result of features parquet file"
            )
            return

        features_parquet_fullpath = os.path.join(features_parqpath, features_filename)
        code_version = get_code_version(features_filename)
        load_filetime = extract_date_from_filename(features_filename)
        load_filetime_str = load_filetime.strftime("%Y_%m_%d_%H%M")
        current_time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")

        # Set file name
        perf_txt_filename = "Performance_{}_{}.txt".format(code_version, current_time_str)
        train_perf_txt_filename = "Train_Performance_{}_{}.txt".format(code_version, current_time_str)
        model_name = "SvmModel_{}_{}.pkl".format(code_version, load_filetime_str)
        stdscaler_name = "StdScaler_Model_{}_{}.bin".format(code_version, load_filetime_str)
        feature_select_name = 'Feature_Selection_{}_{}.txt'.format(code_version, current_time_str)

        df_personaldata = data_management.load_parquet_file(features_parquet_fullpath)
        # 把Parquet檔裡的字串標籤轉為數字
        df_personaldata[label] = df_personaldata['BG_Level'].apply(
            lambda x: 0 if x == 'Normal' else 1 if x == 'High' else 2 if x == 'Low' else None
        )

        # Decide model type
        classify_level = decide_classify_level(df_personaldata)

        # 決定svm model的分類方式
        if classify_level == 'High_Normal_Low':  # 0: Normal, 1:High, 2:Low
            print('Three level')
            svm_model_obj = SvmModelThreeClasses(classify_level)
            model_type = 'Best_ThreeClasses_SVMModel'
        else:
            svm_model_obj = SvmModelTwoClasses(classify_level)
            model_type = 'Best_TwoClasses_SVMModel'

        # Clean dataset
        df_personaldata = svm_model_obj.clean_data(df_personaldata)

        the_uuid_perf_path = os.path.join(self.results_filepath, model_type, self.uuid)
        if not os.path.exists(the_uuid_perf_path):
            os.makedirs(the_uuid_perf_path)

        save_model_fullpath = os.path.join(the_uuid_perf_path, model_name)
        save_perftxt_fullpath = os.path.join(the_uuid_perf_path, perf_txt_filename)
        save_stdscaler_fullpath = os.path.join(the_uuid_perf_path, stdscaler_name)
        save_featureselection_fullpath = os.path.join(the_uuid_perf_path, feature_select_name)

        # Split dataset
        df_train = df_personaldata[df_personaldata["Dataset"] == "Train"]
        df_val = df_personaldata[df_personaldata["Dataset"] == "Test"]

        # Print dataset distribution
        # print('Count of training dataset: {}'.format(len(df_train)))
        # print('Count of testing dataset: {}'.format(len(df_val)))

        # Feature selection
        train_x, train_y = df_train[features], df_train[label]
        val_x, val_y = df_val[features], df_val[label]

        # ------  Preprocessing ---------
        # Data balance
        no_data_set = ''
        if len(df_train) == 0:
            no_data_set += 'Train'
        if len(df_val) == 0:
            no_data_set += 'Test'

        if no_data_set:
            globals.errorcode = -507
            globals.message = (
                "An error occurred in the build_svm_model function of the SVM_Model_Builder_Predictor.py"
                "from bg_package.swmmodel_prediction: "
                "No features extraction in {} dataset".format(no_data_set)
            )
            return

        # Down sampling
        if is_down_sampling:
            rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')
            train_x, train_y = rus.fit_resample(train_x, train_y)
            # val_x, val_y = rus.fit_resample(val_x, val_y)

        # Over sampling (SMOTE)
        if is_over_sampling:
            sm = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
            train_x, train_y = TomekLinks().fit_resample(train_x, train_y)
            train_x, train_y = sm.fit_resample(train_x, train_y)

        # Delete outlier
        if is_delete_outlier:
            train_x, train_y = delete_outlier(train_x, train_y, features, label)

        # Normalize features
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)
        train_x = pd.DataFrame(train_x, columns=features)
        val_x = pd.DataFrame(val_x, columns=features)

        # ------  Model ---------
        # 1. Build the model
        clf, train_pred, val_pred, best_threshold = svm_model_obj.build_model(train_x, train_y, val_x)

        # 2. Select feature by using recursive feature elimination
        # (1) RFE
        rfe_selector = RFE(estimator=clf, n_features_to_select=10, step=2)
        # n_features_to_select是指定選取特徵的個數 每個模型特徵數相同
        clf_pipeline = make_pipeline(rfe_selector, clf)
        clf_pipeline.fit(train_x, train_y)

        # (2) RFECV (RFE+Cross validation)
        # rfecv_selector = RFECV(clf, min_features_to_select=4, step=1, cv=10, scoring='f1_weighted', n_jobs=2)
        # # min_features_to_select是至少要選取特徵的個數 每個模型特徵數不同
        # rfecv_selector.fit(train_x, train_y)

        #  get selected features from recursive feature elimination
        selected_features_indices = rfe_selector.get_support(indices=True)
        selected_features_names = list(train_x.columns[selected_features_indices])

        # plt.figure()
        # cv_results = pd.DataFrame(rfecv_selector.cv_results_)
        # plt.xlabel("Number of features selected")
        # plt.ylabel("Mean test accuracy")
        # plt.errorbar(
        #     x=pd.Series(cv_results.index),
        #     y=cv_results["mean_test_score"],
        #     yerr=cv_results["std_test_score"],
        # )
        # plt.title(
        #     "UUID{}\nRecursive Feature Elimination \n"
        #     "with correlated features".format(uuid)
        # )
        # plt.savefig('{}_RFECV_.png'.format(uuid))

        # if "hr" not in selected_features_names:
        #     selected_features_names.append("hr")
        # if "rr_interval" not in selected_features_names:
        #     selected_features_names.append("rr_interval")

        train_x = train_x[selected_features_names]
        val_x = val_x[selected_features_names]

        # 3. Fit the model
        clf.fit(train_x, train_y)

        # Plot Feature's importance
        # plt.figure(figsize=(6,3))
        # imp, names = zip(*sorted(zip(clf.coef_[0], np.array(selected_features_colnames))))
        # plt.barh(range(len(names)), imp, align='center')
        # plt.yticks(range(len(names)), names)
        # plt.show()

        # Save the model
        joblib.dump(clf, save_model_fullpath)
        joblib.dump(scaler, save_stdscaler_fullpath, compress=True)  # Save the normalize model
        f = open(save_featureselection_fullpath, 'w')
        for name in selected_features_names:
            f.write(str(name)+'\n')
        f.close()

        # ------  Performance ---------
        train_pred, train_prob = svm_model_obj.predict_model(clf, train_x, best_threshold)
        val_pred, val_prob = svm_model_obj.predict_model(clf, val_x, best_threshold)
        dict_performance_train = svm_model_obj.cal_performance(train_y, train_pred)
        dict_performance_val = svm_model_obj.cal_performance(val_y, val_pred)
        svm_model_obj.save_performance(dict_performance_val,
                                       best_threshold,
                                       save_perftxt_fullpath,
                                       model_name,
                                       stdscaler_name,
                                       feature_select_name
                                       )

        # Save performance to excel (develop in test)
        # excel_filename_train = "SVM_Performance_Train.xlsx"
        # excel_filename_val = "SVM_Performance_Test.xlsx"
        # svm_model_obj.save_performance_to_excel(self.uuid, excel_filename_train, dict_performance_train)
        # svm_model_obj.save_performance_to_excel(self.uuid, excel_filename_val, dict_performance_val)

        # ----- Clean previous performance files -------
        # clean_invalid_files(keep_days=180)
        # 比較最好performance的txt 存成hHistoric_Best_Performance.txt 並把keep_days以前的files刪除
        # keep_days未來要寫到最外面給server取
        rename_best_performance_file(the_uuid_perf_path, keep_days=180)

        return

    # 10s ECG
    def predict_bg(self, ecg_10s):

        ##good_quality_wave_count = 0
        wave_info = waves_detection.WaveDetector()

        ##  要再改 目前為了對照標準化
        features = [
            'rr_interval', 'hr', 'pr_duration',
            'pr_corrections3', 'qr_duration', 'qr_corrections3',
            'rs_duration', 'rs_corrections3', 'rt_duration',
            'rt_corrections3', 'pq_duration', 'pq_corrections3',
            'ps_duration', 'ps_corrections3', 'pt_duration',
            'pt_corrections3', 'qs_duration', 'qs_corrections3',
            'qt_duration', 'qt_corrections3', 'st_duration',
            'st_corrections3', 'p_left_sharp', 'p_right_sharp',
            'r_left_sharp', 'r_right_sharp', 't_left_sharp',
            't_right_sharp'
        ]

        modeled_type, modeled_level = know_modeled_type(self.uuid, self.results_filepath)
        model_info_filepath = os.path.join(self.results_filepath, "Best_"+modeled_type+"_SVMModel", self.uuid)

        if modeled_type == 'ThreeClasses':
            svm_model_obj = SvmModelThreeClasses(modeled_level)
            # print('Use three class svm model')
        else:  # modeled_type == 'TwoClasses_SVMModel'
            svm_model_obj = SvmModelTwoClasses(modeled_level)
            # print('Use two class svm model')

        # Find the last model file
        pref_txt_filename = find_best_or_latest_file(model_info_filepath, "Performance_")
        if pref_txt_filename is None:
            globals.errorcode = -603
            globals.message = ("An error occurred in the predict_bg function of the SVM_Model_Builder_Predictor.py"
                               "from bg_package.svmmodel_prediction: "
                               "No built model for uuid {} exists".format(self.uuid)
                               )
            return

        loaded_clf, loaded_stdscaler, model_threshold, features_selection = svm_model_obj.load_modeled_info(model_info_filepath, pref_txt_filename)

        dict_waves_detection = wave_info.generate_waves_info(ecg_10s, self.uuid, measure_type=0)
        ##dict_features_10s, good_quality_wave_count = features_extraction.extract_features(dict_waves_detection, good_quality_wave_count)
        dict_features_10s, flag = features_extraction.extract_features(dict_waves_detection) ##, good_quality_wave_count)

        if dict_features_10s['type']:  # 有擷取到features
            df_features_resample = features_extraction.get_mean_in_window(dict_features_10s)  # 取平均
            test_x_temp = df_features_resample[features]  # Feature selection
            test_x_temp = loaded_stdscaler.transform(test_x_temp)  # Normalization
            df_test_x = pd.DataFrame(test_x_temp, columns=features)
            test_x = df_test_x[features_selection].to_numpy()

            try:
                test_pred, test_prob = svm_model_obj.predict_model(loaded_clf, test_x, model_threshold)
                return test_pred, test_prob
            except:
                return None, None

        else:
            return None, None
