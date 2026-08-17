import os
import shutil
import sys
import argparse
import numpy as np
import time
from scipy.fftpack import fft
import pywt
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader       
##from torch.utils.data import TensorDataset
from natsort import natsorted
from scipy import signal
import neurokit2 as nk
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

import torch.optim as optim
import torch.nn as nn
import pandas as pd
import ujson as json
from datetime import datetime, timedelta
import math
import glob
import csv
from time import strftime
import matplotlib.pyplot as plt
###from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
import globals
import joblib
from sklearn.model_selection import KFold
from zipfile import ZipFile
from multiprocessing import Pool


import sympy
#from pysr import PySRRegressor
import xgboost as xgb
from xgboost import plot_importance
# import shap
from sklearn.model_selection import train_test_split,GridSearchCV
from pathlib import Path

if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))



from SWMlib.motion import *  ##ahrs
from SWMlib.ecg.baseline import baseline_remove 
from SWMlib.ecg.rpeak import rpeak_detection 
from SWMlib.ecg.quality_check import ecg_quality_check_v3
from SWMlib.common.calculation import normalization 
from SWMlib.common import data_load_concate  
from bg_packege import features_extraction
from SWMlib.ecg.noise_remove import remove_spike,emg_detector_remover 

from collections import Counter, defaultdict
from statistics import median

performance_table=[]

def get_version(): ###取得版本號

    return '007'

    
def check_infinity_or_too_large(X):
    # Check for infinity
    if np.any(np.isinf(X)):
        return True, "Contains infinity"
    
    # Check for values too large for dtype('float32')
    max_float32 = np.finfo(np.float32).max
    if np.any(X > max_float32):
        return True, "Contains value too large for float32"
    
    return False, "No issues found"
'''
used_feature_dic=[(0,'uuid'),(0,'type'),(1,'rr_interval'),(0,'hr'),(0,'a_score'),(0,'p_score'),(0,'r_score'),(0,'p_stability'),(0,'q_stability'),
                    (0,'s_stability'),(0,'t_stability'),(0,'p_value'),(0,'q_value'),(0,'r_value'),(0,'s_value'),(0,'t_value'),(0,'pr_duration'),
                    (0,'pr_amplitude'),(0,'pr_distances'),(0,'pr_directions'),(0,'pr_slope'),(0,'pr_corrections3'),(0,'qr_duration'),(0,'qr_amplitude'),
                    (0,'qr_distances'),(0,'qr_directions'),(0,'qr_slope'),(0,'qr_corrections3'),(0,'rs_duration'),(0,'rs_amplitude'),(0,'rs_distances'),
                    (0,'rs_directions'),(0,'rs_slope'),(0,'rs_corrections3'),(1,'rt_duration'),(0,'rt_amplitude'),(0,'rt_distances'),(0,'rt_directions'),
                    (0,'rt_slope'),(0,'rt_corrections3'),(0,'pq_duration'),(0,'pq_amplitude'),(0,'pq_distances'),(0,'pq_directions'),(0,'pq_slope'),
                    (0,'pq_corrections3'),(0,'ps_duration'),(0,'ps_amplitude'),(0,'ps_distances'),(0,'ps_directions'),(0,'ps_slope'),(0,'ps_corrections3'),
                    (1,'pt_duration'),(0,'pt_amplitude'),(0,'pt_distances'),(0,'pt_directions'),(0,'pt_slope'),(0,'pt_corrections3'),(0,'qs_duration'),
                    (0,'qs_amplitude'),(0,'qs_distances'),(0,'qs_directions'),(0,'qs_slope'),(0,'qs_corrections3'),(1,'qt_duration'),(0,'qt_amplitude'),
                    (1,'qt_distances'),(0,'qt_directions'),(0,'qt_slope'),(0,'qt_corrections3'),(0,'st_duration'),(0,'st_amplitude'),(0,'st_distances'),
                    (0,'st_directions'),(0,'st_slope'),(1,'st_corrections3'),(0,'p_left_slope'),(0,'p_right_slope'),(0,'p_left_sharp'),(0,'p_right_sharp'),
                    (0,'p_tilt'),(0,'r_left_slope'),(0,'r_right_slope'),(0,'r_left_sharp'),(0,'r_right_sharp'),(0,'r_tilt'),(1,'t_left_slope'),(1,'t_right_slope'),
                    (1,'t_left_sharp'),(1,'t_right_sharp'),(0,'t_tilt'),(0,'Dataset'),(0,'BG_Level'),(1,'ratio of dif_qs_amp/dif_qr_amp'),(1,'ratio of dif_tr_amp/dif_st_amp'),
                    (1,'ratio of tr_amp'),(1,'ratio of st_amp')]
'''

used_feature_dic=[(0,'uuid'),(0,'type'),(1,'rr_interval'),(0,'hr'),(0,'a_score'),(0,'p_score'),(0,'r_score'),(0,'p_stability'),(0,'q_stability'),
                    (0,'s_stability'),(0,'t_stability'),(0,'p_value'),(0,'q_value'),(0,'r_value'),(0,'s_value'),(0,'t_value'),(0,'pr_duration'),
                    (0,'pr_amplitude'),(0,'pr_distances'),(0,'pr_directions'),(0,'pr_slope'),(0,'pr_corrections3'),(0,'qr_duration'),(0,'qr_amplitude'),
                    (0,'qr_distances'),(0,'qr_directions'),(0,'qr_slope'),(0,'qr_corrections3'),(0,'rs_duration'),(0,'rs_amplitude'),(0,'rs_distances'),
                    (0,'rs_directions'),(0,'rs_slope'),(0,'rs_corrections3'),(1,'rt_duration'),(0,'rt_amplitude'),(0,'rt_distances'),(0,'rt_directions'),
                    (0,'rt_slope'),(0,'rt_corrections3'),(0,'pq_duration'),(0,'pq_amplitude'),(0,'pq_distances'),(0,'pq_directions'),(0,'pq_slope'),
                    (0,'pq_corrections3'),(0,'ps_duration'),(0,'ps_amplitude'),(0,'ps_distances'),(0,'ps_directions'),(0,'ps_slope'),(0,'ps_corrections3'),
                    (1,'pt_duration'),(0,'pt_amplitude'),(0,'pt_distances'),(0,'pt_directions'),(0,'pt_slope'),(0,'pt_corrections3'),(0,'qs_duration'),
                    (0,'qs_amplitude'),(0,'qs_distances'),(0,'qs_directions'),(0,'qs_slope'),(0,'qs_corrections3'),(1,'qt_duration'),(0,'qt_amplitude'),
                    (1,'qt_distances'),(0,'qt_directions'),(0,'qt_slope'),(0,'qt_corrections3'),(0,'st_duration'),(0,'st_amplitude'),(0,'st_distances'),
                    (0,'st_directions'),(0,'st_slope'),(1,'st_corrections3'),(0,'p_left_slope'),(0,'p_right_slope'),(0,'p_left_sharp'),(0,'p_right_sharp'),
                    (0,'p_tilt'),(0,'r_left_slope'),(0,'r_right_slope'),(0,'r_left_sharp'),(0,'r_right_sharp'),(0,'r_tilt'),(1,'t_left_slope'),(1,'t_right_slope'),
                    (1,'t_left_sharp'),(1,'t_right_sharp'),(0,'t_tilt'), (1,'qrs_area'),(1,'st_area'),(1,'twave_cog') ,(0,'Dataset'),(0,'BG_Level'),(1,'ratio of dif_qs_amp/dif_qr_amp'),(1,'ratio of dif_tr_amp/dif_st_amp'),
                    (1,'ratio of tr_amp'),(1,'ratio of st_amp')]




class DataArrangement:

    def __init__(self):
        self.basepath = os.path.dirname(__file__)      

    '''
    def unzip_file(self,uuid,start_time,end_time,db_path,export_path):  ####將受測者zip檔解壓縮(只適用於health server)

        uuid_export_path = os.path.join(export_path, uuid)        
        if not os.path.exists(uuid_export_path): 
            os.makedirs(uuid_export_path)      
       
        UnzipFileNameList = data_load_concate.search_unzip_file(db_path=db_path, target_uuid=uuid, start_time=start_time, end_time=end_time, export_path=uuid_export_path)   
    '''

    def unzip_file(self,uuid,start_time,end_time,db_path,export_path):  ####將受測者zip檔解壓縮
        try:
            uuid_export_path = os.path.join(export_path)  ##, uuid)        
            if not os.path.exists(uuid_export_path): 
                os.makedirs(uuid_export_path)      
        
            ##UnzipFileNameList = data_load_concate.search_unzip_file(db_path=db_path, target_uuid=uuid, start_time=start_time, end_time=end_time, export_path=uuid_export_path)   
            ##self.serach_zip_file_and_unzip(uuid,server_db_path,start_time,end_time,uuid_export_path)
            zipfile_list=os.listdir(server_db_path)
        
            for zipfilename in zipfile_list:
                file_name_split_array=zipfilename.split('_')           
                if (file_name_split_array[0]==uuid):              
                    with ZipFile(os.path.join(server_db_path,zipfilename),"r") as zip:
                        zip.extractall(export_path)  ##export srj files in the path
        except:
            errorcode="-900"
            message="An error occurs in the unzip_file function of Regression_Model_Predictor.py: fail to unzip files!"
            return errorcode, message

    
    ##def data_processing(self,uuid,start_time,end_time,srj_db_path,glucosedata_path,basepath,server_db_path): ##測試用
    def data_processing(
        self,
        uuid,
        srj_db_path,
        glucosedata_path,
        basepath,
        processnum,
        splitting_ratio="",
        run_rel=None,
        features_full_rel=None,
        downsample_ratio=0.3,
        downsample_cap_ref="median",
        skip_feature_extract=False,
        skip_normalize=False,
        do_downsample=True,
        seed=42,
    ):
        """
        方案 A：
        - GlucoseData / normalize：固定在 splitting_ratio（例如 70_30）
        - 完整特徵：寫到 features_full_rel（例如 70_30/full）
        - 實驗產出：run_rel（例如 70_30/ds_median_r0.30）
          從 full copy 過去後再 downsample，不重抽 ECG

        skip_feature_extract=True：略過抽特徵，要求 full 已存在，只做 copy(+downsample)
        """
        errorcode = "0"
        message = ""

        features_full_rel = features_full_rel or os.path.join(splitting_ratio, "full")
        run_rel = run_rel or features_full_rel

        ##測試用
        '''
        if(server_db_path!=""):  ###測試用,需要抓雲端上的zip檔案 
            print('Start unzippig file!')      
            self.unzip_file(uuid=uuid,start_time=start_time,end_time=end_time,db_path=server_db_path,export_path=srj_db_path) ##自雲端資料夾中將ECG壓縮檔解壓縮成srj檔放置到srj_db_path路徑下                    
        
            if int(errorcode)<0:
                return errorcode, message 
        '''

        if not skip_normalize:
            print("Start data copying and normalizing...")
            errorcode, message = self.data_copying_normalizing(
                uuid=uuid, basepath=basepath, splitting_ratio=splitting_ratio
            )
            if int(errorcode) < 0:
                return errorcode, message
        else:
            print("Skip normalizing (--skip-normalize)")

        full_features_uuid = os.path.join(
            basepath, features_full_rel, "Regression_Features", uuid
        )

        if not skip_feature_extract:
            print(f"Start feature extracting → {features_full_rel}/Regression_Features/{uuid}")
            errorcode, message = self.regression_feature_extraction(
                uuid=uuid,
                export_txtfile_path=basepath,
                splitting_ratio=features_full_rel,
                glucose_splitting_ratio=splitting_ratio,
            )
            if int(errorcode) < 0:
                return errorcode, message
        else:
            print(f"Skip feature extract；使用既有特徵：{full_features_uuid}")
            if not os.path.isdir(full_features_uuid):
                errorcode = "-1"
                message = (
                    f"--skip-feature-extract 但特徵目錄不存在：{full_features_uuid}。"
                    f"方案A請先建立 70_30/full；原始 data_balanced 請用 --legacy-balanced "
                    f"並確認 70_30/Regression_Features/{{uuid}} 存在。"
                )
                return errorcode, message

        # 實驗目錄：從 full copy 一份，再只在這份上 downsample（保護 full 不被砍）
        if os.path.normpath(run_rel) != os.path.normpath(features_full_rel):
            print(f"Copy features: {features_full_rel} → {run_rel} (uuid={uuid})")
            ok, copy_msg = copy_regression_features_uuid(
                basepath, features_full_rel, run_rel, uuid
            )
            if not ok:
                return "-1", copy_msg
        elif do_downsample:
            errorcode = "-1"
            message = (
                "拒絕對來源特徵目錄做 in-place downsample（會弄壞可重用的完整集）。"
                "請使用實驗目錄 run_rel（例如 70_30/ds_median_r0.30）。"
            )
            return errorcode, message
        else:
            print(f"使用既有特徵目錄訓練（不做 downsample / 不 re-add Remove_Data）：{run_rel}")

        if do_downsample:
            print(
                f"Start data downsample on {run_rel} "
                f"(cap_ref={downsample_cap_ref}, ratio={downsample_ratio})..."
            )
            errorcode, message = self.data_downsample_by_ratio(
                uuid=uuid,
                txtfile_path=basepath,
                splitting_ratio=run_rel,
                ratio=downsample_ratio,
                cap_ref=downsample_cap_ref,
                seed=seed,
            )
            if int(errorcode) < 0:
                return errorcode, message
        else:
            print("Skip downsample (--no-downsample)")

        message = "data processing is done!"
        return errorcode, message
    

    def data_parsing(self,uuid,srj_db_path,glucosedata_path,export_txtfile_path, processnum=8): 
        
        '''
        Parsing data with time before and after 3-min of CGM recording time and is in static state
        Each output text file contains 10-second ecg segment
        '''
        
        errorcode="0"
        message=""

        n_10sec=1 ##設定片段長度為2500的n倍數
        clip_length=n_10sec*2500
        export_rawdata_path=os.path.join(export_txtfile_path,str(clip_length)+"_motion")        
        if not os.path.exists(export_rawdata_path): 
            os.makedirs(export_rawdata_path) ###創建ECG擷取後存放資料夾  

        uuid_temp_path = os.path.join(export_rawdata_path, uuid)
        if not os.path.exists(uuid_temp_path): 
            os.makedirs(uuid_temp_path) ### 創建uuid的資料夾 
            
        uuid_static_path = os.path.join(uuid_temp_path, "static")
        if not os.path.exists(uuid_static_path): 
            os.makedirs(uuid_static_path) ### 創建static的資料夾 
        
        uuid_dynamic_path = os.path.join(uuid_temp_path, "dynamic")
        if not os.path.exists(uuid_dynamic_path): 
            os.makedirs(uuid_dynamic_path) ### 創建dynamic的資料夾 
        
        glucosedata_path=os.path.join(glucosedata_path,uuid+'.csv')
        glucose_csv_file_list=glob.glob(glucosedata_path)      
        print('glucosedata_path:',glucosedata_path)
        ecgdata_path=os.path.join(srj_db_path,'*.srj')
        srj_file_list=glob.glob(ecgdata_path) 
        
        '''
        mode='motion'
        with open(glucose_csv_file_list[0], newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f, delimiter=','))         
            for row in rows: ###檢察是否有motion資料
                recored_time = row[0]
                glucose_value = row[1]                        
                if glucose_value != "":
                    datetime_object = datetime.strptime(recored_time[2:], '%y/%m/%d %H:%M')
                    time_change = timedelta(minutes=3)
                    new_time_before = datetime_object - time_change
                    new_time_after = datetime_object + time_change
                    start_time_str = new_time_before.strftime("20%y%m%d %H%M%S")
                    end_time_str = new_time_after.strftime("20%y%m%d %H%M%S")
                    
                    ecg_data_array, motion_data_array, _, _ = self._dataconcate(srj_db_path, srj_file_list, start_time_str, end_time_str, n_10sec)
                    if (ecg_data_array): ##有取得ECG資料
                        if(len(motion_data_array[0])==0): ##檢查是否有motion資料
                            mode='rawdata'     
                        else:
                            mode='motion'                           
                       
                        break                           

        if(mode!='rawdata'): ##有motion data
            
            if(len(glucose_csv_file_list)==0):           
                errorcode="-902"
                message="An error occurs in the data-parsing function of Regression_Model_Predictor.py: no glucose csv files exist!"          
                return errorcode, message
                    
            try:
                with open(glucose_csv_file_list[0], newline='', encoding='utf-8') as f:
                                
                    rows = list(csv.reader(f, delimiter=',')) 
                    chunks = self._split_chunks(rows, processnum)               
                    args = [(chunk, srj_db_path, srj_file_list, uuid, n_10sec, uuid_dynamic_path, uuid_static_path) for chunk in chunks]

                    with Pool(processes=processnum) as pool:
                        pool.starmap(self._process_chunk, args)
                            
            except:
                errorcode="-903"
                message="An error occurs in the data-parsing function of Regression_Model_Predictor.py: fail to do multiprocessing analaysis!"
                return errorcode, message
            
        else: ##直接取RawData的資料到2500_motion/static
        '''
        
        print('Directly getting data form RawData folder...')
        source_folder=os.path.join(export_txtfile_path,'RawData',uuid)
        for filename in os.listdir(source_folder):
            ##print('filename:',filename)
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(uuid_static_path, filename)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_file) 

        
        return errorcode, message

    def _split_chunks(self, data, num_chunks):
        
        chunk_size = math.ceil(len(data) / num_chunks)
        
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

      
    def _process_chunk(self,chunk, srj_db_path, srj_file_list, uuid, n_10sec, uuid_dynamic_path, uuid_static_path):
       
        for row in chunk:
            recored_time = row[0]
            glucose_value = row[1]
                        
            if glucose_value != "":
                datetime_object = datetime.strptime(recored_time[2:], '%y/%m/%d %H:%M')
                time_change = timedelta(minutes=3)
                new_time_before = datetime_object - time_change
                new_time_after = datetime_object + time_change
                current_time_str = datetime_object.strftime("20%y%m%d %H%M%S")
                start_time_str = new_time_before.strftime("20%y%m%d %H%M%S")
                end_time_str = new_time_after.strftime("20%y%m%d %H%M%S")
                
                ecg_data_array, motion_data_array, _, _ = self._dataconcate(srj_db_path, srj_file_list, start_time_str, end_time_str, n_10sec)
               
                for i in range(len(ecg_data_array)):
                    current_ecg_data = ecg_data_array[i]
                    current_ecg_data=baseline_remove(current_ecg_data)  ##新增基線拉直
                    current_motion_data = motion_data_array[i]

                    if len(current_motion_data) > 0:
                        flag = self.fast_motion_analysis(current_motion_data)
                        if flag:
                            self._write_ecg_data(uuid, current_ecg_data, current_time_str, 1, i, glucose_value, uuid_dynamic_path)
                        else:
                            self._write_ecg_data(uuid, current_ecg_data, current_time_str, 0, i, glucose_value, uuid_static_path)

    def read_text_normalization(self,textfile_path,level):
        
        ecg_array=[]
        with open(textfile_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Append the cleaned line to the list
                ecg_array.append(int(line.strip()))

        
        ecg_norm=normalization(ecg_array)

        ecg_norm=ecg_norm*level

        return ecg_norm
    
    def save_array_to_txt(self, data_array, destination_txt_path):
      
        with open(destination_txt_path, 'w', encoding='utf-8') as f:
            for item in data_array:
                f.write(f"{item}\n")

    

    def data_copying_normalizing(self,uuid,basepath,splitting_ratio): 
        
        '''
        normalizaing ecg data in the text filses from rawdata folder and save it to the NormalizedData folder
        '''
        
        errorcode="0"
        message=""      
        
        '''
        regression_glucose_uuid_path=os.path.join(basepath,splitting_ratio,'GlucoseRegressionData',uuid)  ###根據splitting_ration分為train and test資料夾
        if not os.path.exists(regression_glucose_uuid_path):
            os.makedirs(regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid存放normal,low,high資料


        train_regression_glucose_uuid_path=os.path.join(regression_glucose_uuid_path,'Train')
        if not os.path.exists(train_regression_glucose_uuid_path):
            os.makedirs(train_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Train存放資料夾
            
        train_normal_regression_glucose_uuid_path=os.path.join(train_regression_glucose_uuid_path,'Normal')
        if not os.path.exists(train_normal_regression_glucose_uuid_path):
            os.makedirs(train_normal_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Train/Normal存放資料夾

        train_low_regression_glucose_uuid_path=os.path.join(train_regression_glucose_uuid_path,'Low')
        if not os.path.exists(train_low_regression_glucose_uuid_path):
            os.makedirs(train_low_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Train/Low存放資料夾

        train_high_regression_glucose_uuid_path=os.path.join(train_regression_glucose_uuid_path,'High')
        if not os.path.exists(train_high_regression_glucose_uuid_path):
            os.makedirs(train_high_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Train/High存放資料夾    

            
        test_regression_glucose_uuid_path=os.path.join(regression_glucose_uuid_path,'Test')
        if not os.path.exists(test_regression_glucose_uuid_path):
            os.makedirs(test_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Test存放資料夾

        test_normal_regression_glucose_uuid_path=os.path.join(test_regression_glucose_uuid_path,'Normal')
        if not os.path.exists(test_normal_regression_glucose_uuid_path):
            os.makedirs(test_normal_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Test/Normal存放資料夾

        test_low_regression_glucose_uuid_path=os.path.join(test_regression_glucose_uuid_path,'Low')
        if not os.path.exists(test_low_regression_glucose_uuid_path):
            os.makedirs(test_low_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Test/Low存放資料夾

        test_high_regression_glucose_uuid_path=os.path.join(test_regression_glucose_uuid_path,'High')
        if not os.path.exists(test_high_regression_glucose_uuid_path):
            os.makedirs(test_high_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Test/High存放資料夾
        '''               
        
        normalized_data_uuid_path=os.path.join(basepath,"NormalizedData",uuid)
        if not os.path.exists(normalized_data_uuid_path):
            os.makedirs(normalized_data_uuid_path) ###創建normalized_data_path存放正規化資料夾

        dataset_array=['Train','Test']
        category_name_array=['Normal','High','Low']

        for dataset_name in dataset_array:
            for category_name in category_name_array:
                source_name_folder=os.path.join(basepath,splitting_ratio,'GlucoseData',uuid,dataset_name,category_name)  ##檔案名稱來自GlucoseData資料夾
                ##destination_folder=os.path.join(regression_glucose_uuid_path,dataset_name,category_name)  ##RawData下相同檔名的檔案移動至GlucoseRegressionData資料夾
            
                filenamelist=os.listdir(source_name_folder) 
                for filename in filenamelist:
                    ##destination_txt_path=os.path.join(destination_folder,filename)
                    destination_txt_path=os.path.join(normalized_data_uuid_path,filename)
                    source_txt_path = os.path.join(basepath,"RawData",uuid,filename) ##RawData資料夾下檔案

                    if(not os.path.isfile(destination_txt_path)):  
                        if (os.path.isfile(source_txt_path)): ##RawData資料夾下有此檔案名稱
                            normalized_ecg=self.read_text_normalization(source_txt_path,level=300) ###正規化為0~300
                            self.save_array_to_txt(normalized_ecg, destination_txt_path)
                            ##shutil.copy(source_txt_path, destination_txt_path)    
                     
          
        ##except:
        ##    errorcode="-904"
        ##    message="An error occurs in the data_copying_normalizing function of Regression_Model_Predictor.py: fail to copy files from Rawdata folder to GlucoseRegressionData folder"
        ##    return errorcode, message 
        
       
  
        return errorcode, message        
             

    
    def data_splitting(self,uuid,basepath,splitting_ratio): 
        
        '''
        1. Split ECG text files in raw data folder(2500_moiton) into normal, low, and high category
        2. split normal, low, high data into train and test folder
        '''
        
        errorcode="0"
        message=""
       
        low_thres=80
        high_thres=175
        
      
        dataset_path=os.path.join(basepath,'Dataset_Regression') ##將ECG raw text file分成normal, low and high, 還沒有根據spltting_ratio分train and test
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path) ###創建Dataset_Regression存放資料夾

        dataset_uuid_path=os.path.join(dataset_path,uuid)
        if not os.path.exists(dataset_uuid_path):
            os.makedirs(dataset_uuid_path) ###創建Dataset_Regression存放資料夾

        normal_dataset_uuid_path=os.path.join(dataset_uuid_path,'Normal')
        if not os.path.exists(normal_dataset_uuid_path):
            os.makedirs(normal_dataset_uuid_path) ###創建Dataset_Regression/Normal存放資料夾

        low_dataset_uuid_path=os.path.join(dataset_uuid_path,'Low')
        if not os.path.exists(low_dataset_uuid_path):
            os.makedirs(low_dataset_uuid_path) ###創建Dataset_Regression/Low存放資料夾

        high_dataset_uuid_path=os.path.join(dataset_uuid_path,'High')
        if not os.path.exists(high_dataset_uuid_path):
            os.makedirs(high_dataset_uuid_path) ###創建Dataset_Regression/Normal存放資料夾        

        
        rawdata_uuid_path = os.path.join(basepath, "2500_motion", uuid, "static") ##來源資料夾
                     

        file_list=os.listdir(rawdata_uuid_path)
        ##if(True):
        try:
            for filename in file_list:
                if filename.endswith('.txt'):
                    string_array=filename.split('_')
                    value_string=(string_array[-1]).split('.')
                    glucose_value=int(value_string[0])
                
                    source_txt_path=os.path.join(rawdata_uuid_path,filename)
                    if(glucose_value<=low_thres):
                        destination_txt_path=os.path.join(low_dataset_uuid_path,filename)                    
                    elif(glucose_value>=high_thres):
                        destination_txt_path=os.path.join(high_dataset_uuid_path,filename)
                    else:
                        destination_txt_path=os.path.join(normal_dataset_uuid_path,filename)  
                
                    if(not os.path.isfile(destination_txt_path)):  
                        if (os.path.isfile(source_txt_path)):
                            shutil.copy(source_txt_path, destination_txt_path)                    

        
            regression_glucose_uuid_path=os.path.join(basepath,splitting_ratio,'GlucoseRegressionData',uuid)  ###根據splitting_ration分為train and test資料夾
            if not os.path.exists(regression_glucose_uuid_path):
                os.makedirs(regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid存放normal,low,high資料


            train_regression_glucose_uuid_path=os.path.join(regression_glucose_uuid_path,'Train')
            if not os.path.exists(train_regression_glucose_uuid_path):
                os.makedirs(train_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Train存放資料夾
            
            train_normal_regression_glucose_uuid_path=os.path.join(train_regression_glucose_uuid_path,'Normal')
            if not os.path.exists(train_normal_regression_glucose_uuid_path):
                os.makedirs(train_normal_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Train/Normal存放資料夾

            train_low_regression_glucose_uuid_path=os.path.join(train_regression_glucose_uuid_path,'Low')
            if not os.path.exists(train_low_regression_glucose_uuid_path):
                os.makedirs(train_low_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Train/Low存放資料夾

            train_high_regression_glucose_uuid_path=os.path.join(train_regression_glucose_uuid_path,'High')
            if not os.path.exists(train_high_regression_glucose_uuid_path):
                os.makedirs(train_high_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Train/High存放資料夾    

            
            test_regression_glucose_uuid_path=os.path.join(regression_glucose_uuid_path,'Test')
            if not os.path.exists(test_regression_glucose_uuid_path):
                os.makedirs(test_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Test存放資料夾

            test_normal_regression_glucose_uuid_path=os.path.join(test_regression_glucose_uuid_path,'Normal')
            if not os.path.exists(test_normal_regression_glucose_uuid_path):
                os.makedirs(test_normal_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Test/Normal存放資料夾

            test_low_regression_glucose_uuid_path=os.path.join(test_regression_glucose_uuid_path,'Low')
            if not os.path.exists(test_low_regression_glucose_uuid_path):
                os.makedirs(test_low_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Test/Low存放資料夾

            test_high_regression_glucose_uuid_path=os.path.join(test_regression_glucose_uuid_path,'High')
            if not os.path.exists(test_high_regression_glucose_uuid_path):
                os.makedirs(test_high_regression_glucose_uuid_path) ###創建GlucoseRegressionData_uuid/Test/High存放資料夾
          
        except:
            errorcode="-904"
            message="An error occurs in the data_splitting function of Regression_Model_Predictor.py: fail to read text files in the 2500_motion folder or fail to copy files to the Dataset_Regression folder"
            return errorcode, message 
        
        errorcode, message=self._train_test_splitting(normal_dataset_uuid_path, regression_glucose_uuid_path, 'Normal', splitting_ratio) ##根據splitting_ration把dataset_regression資料夾中的Normal資料分為train and test資料夾
        if(int(errorcode)<0):
            return errorcode, message
    
        errorcode, message=self._train_test_splitting(high_dataset_uuid_path, regression_glucose_uuid_path, 'High', splitting_ratio)    
        if(int(errorcode)<0):
            return errorcode, message
    
        errorcode, message=self._train_test_splitting(low_dataset_uuid_path, regression_glucose_uuid_path, 'Low', splitting_ratio)                 
        if(int(errorcode)<0):
            return errorcode, message
  
        return errorcode, message        
             
    
    def _train_test_splitting(self, data_path, newpath, type, splitting_ratio):

        errorcode = "0"
        message = ""
        
        splitting_ratio_value=float((splitting_ratio.split("_"))[0])/100.0
        file_list = np.array(natsorted(os.listdir(data_path)))
        
        try: 
            if len(file_list)==0:
                print('Running _train_test_splitting function: no file in', data_path)
            else:               
                fileInfoList = np.array([np.array(x) for x in np.char.split(np.array([f[:-4] for f in file_list]), '_')])
            
                start_time = fileInfoList[0][1][:8]
                end_time = fileInfoList[-1][1][:8]
                BGs=fileInfoList[:, -1].astype(np.int16)
            
                min_glucose_value = min(BGs)
                max_glucose_value = max(BGs)
               
                dates_special = [] 
                file_list = os.listdir(data_path)            
                for filename in file_list:                
                    try:
                        parts = filename.split("_")
                        date_str = parts[1][0:8]  # YYYYMMDD
                        file_date = datetime.strptime(date_str, "%Y%m%d")
                        val_str = parts[-1].split(".")[0] 
                        val = int(val_str)                    
                        dates_special.append(file_date)  
                        
                    except Exception as e:
                        print("File parsing error in the _movedata function, but you can ignor it!:", filename, " reason:", e)  
             
            
                if not dates_special:
                    print('no any low- or high-glucose data!')                
                else:
                    dates_unique = sorted(set(dates_special))                         
                    new_start_date = min(dates_unique)     ###找到第一個高血糖日期作為新的起始點           
                    end_date = dates_unique[-1]        
                    total_days = (end_date - new_start_date).days + 1
                    days_in_r_percent = int(total_days * splitting_ratio_value) 
                    split_datetime = new_start_date + timedelta(days=days_in_r_percent - 1)
                    print('special start_date:', new_start_date, 'special end_date:', end_date, 'special split_datetime:', split_datetime)

                                   
                    for i, filename in enumerate(file_list):
                        current_item = file_list[i]
                        current_glucose_value = BGs[i]
                        if current_glucose_value == min_glucose_value or current_glucose_value == max_glucose_value:  ###把最大最小值放在Training data
                            current_path = os.path.join(data_path, current_item)
                            copyto_path = os.path.join(newpath, 'Train', type, filename)
                            if (not os.path.isfile(copyto_path)):
                                if (os.path.isfile(current_path)):
                                    shutil.move(current_path, copyto_path)

                        else:
                            current_datetime=datetime.strptime(fileInfoList[i][1][:8], "%Y%m%d")
                            if (current_datetime > split_datetime):
                                current_path = os.path.join(data_path, filename)
                                copyto_path = os.path.join(newpath, 'Test', type, filename)
                                if (not os.path.isfile(copyto_path)):
                                    if (os.path.isfile(current_path)):
                                        shutil.move(current_path, copyto_path)

                            else:  ## 在trainingset的time string陣列中，分到training set
                                current_path = os.path.join(data_path, filename)
                                copyto_path = os.path.join(newpath, 'Train', type, filename)
                                if (not os.path.isfile(copyto_path)):
                                    if (os.path.isfile(current_path)):
                                        shutil.move(current_path, copyto_path)
        
        except:
            errorcode="-905"
            message="An error occurs in the _train_test_splitting function of Regression_Model_Predictor: fail to splitting data into train and test folder"

        return errorcode, message
    
                
    def fast_motion_analysis(self,motion_data,sample_rate=2):

        '''
        Determine static or dynamic of motion data.
        
        Calculate the stantard deviation of the linear acceleration in global coordinate within 2 seconds.
        Set a threshold of stantard deviation to determine static or dynamic.
        
        input ---
            motion_data: 2D list of raw motion data
            sample_rate: int or float(Sample rate of motion data)

        output ---
                1-D list of flags 
                0: dynamic state
                1: static state
        '''
      

        # 將Motion raw data 轉成 numpy array
        sample_num = len(motion_data)
        motion_data = np.array(motion_data)

        # 將Motion raw data 轉成對應的數值(dps, g, ...)
        F = np.concatenate((np.ones(3)/114.28, np.ones(3)*0.000061, np.ones(3)*1.5, 1), axis = None)
        Values = motion_data * F

        # 擷取ACC and GYR 
        ACC = Values[:,3:6]
        GYR = Values[:,0:3]
        
        # 計算 global & body 座標系的轉換矩陣: C
        orientation = ahrs.filters.Madgwick(acc=ACC*9.80665, gyr=GYR*np.pi/180, frequency=sample_rate)
        C = ahrs.common.orientation.q2R(orientation.Q)
        
        # 將 ACC 轉至 global 坐標系, 扣除 gravity 後得到空間中實際的加速度
        linACC = np.zeros(ACC.shape)
        for i in range(0, sample_num):
            ACCglob = np.dot(ACC[i], C[i].T)
            linACC[i] = ACCglob - [0,0,1] # notice: C in matlab is the transpose of C in here
        linACC = linACC * 9.81 # g to m/s^2
        
        # 計算實際加速度 2 秒內的標準差
        w_size = 2 * sample_rate
        # 設定標準差的閥值，以定義動態或靜態
        movingTH = 0.5
        for i in range(0, sample_num):
            if i < w_size:
                window = linACC[:i+1]
            else:
                window = linACC[i-w_size+1:i+1]

            if window.shape[0] > 1:
                std=np.std(window, axis=0)
                # 三軸中任一軸的標準差大於閥值就視為動態
                if np.any(std>movingTH):
                    return 1
        return 0  
    
    
    def regression_feature_extraction(self,uuid,export_txtfile_path,splitting_ratio="",glucose_splitting_ratio=None):
        
        '''
        extract features from ecg signal in text files of raw data folder(2500_motion), 
        and then save results in Regression_Features folder

        splitting_ratio: 特徵輸出路徑前綴（例如 70_30/full）
        glucose_splitting_ratio: GlucoseData 索引路徑前綴（例如 70_30）；預設與 splitting_ratio 相同
        '''

        errorcode="0"
        message=""
        if glucose_splitting_ratio is None:
            glucose_splitting_ratio = splitting_ratio
 
        ##----將特徵按照血糖值高低放到分成Dataset中的高中低資料夾----
        features_filepath=os.path.join(export_txtfile_path,splitting_ratio,'Regression_Features') ##創建Regresion_Features資料夾
        if not os.path.exists(features_filepath): os.makedirs(features_filepath)
            
        features_uuid_path=os.path.join(features_filepath,uuid) ##創建Regresion_Features/uuid 資料夾
        if not os.path.exists(features_uuid_path): os.makedirs(features_uuid_path)

        path_train=os.path.join(features_uuid_path,'Train') ##創建Regresion_Features/uuid/train 資料夾
        if not os.path.exists(path_train): os.makedirs(path_train)
              
        path_train_high=os.path.join(path_train,'High') ##創建Regresion_Features/uuid/train/high 資料夾
        if not os.path.exists(path_train_high): os.makedirs(path_train_high)

        path_train_low=os.path.join(path_train,'Low') ##創建Regresion_Features/uuid/train/high 資料夾
        if not os.path.exists(path_train_low): os.makedirs(path_train_low)

        path_train_normal=os.path.join(path_train,'Normal') ##創建Regresion_Features/uuid/train/high 資料夾
        if not os.path.exists(path_train_normal): os.makedirs(path_train_normal)       
                
        
        path_test=os.path.join(features_uuid_path,'Test') ##創建Regresion_Features/uuid/test 資料夾
        if not os.path.exists(path_test): os.makedirs(path_test)

        path_test_high=os.path.join(path_test,'High') ##創建Regresion_Features/uuid/test/high 資料夾
        if not os.path.exists(path_test_high): os.makedirs(path_test_high)

        path_test_low=os.path.join(path_test,'Low') ##創建Regresion_Features/uuid/test/low 資料夾
        if not os.path.exists(path_test_low): os.makedirs(path_test_low)
        
        path_test_normal=os.path.join(path_test,'Normal') ##創建Regresion_Features/uuid/test/normal 資料夾
        if not os.path.exists(path_test_normal): os.makedirs(path_test_normal)  

               
        ## Set files path
       
        ##uuid_trainindex_path = os.path.join(export_txtfile_path, splitting_ratio, "GlucoseRegressionData", uuid, "Train")  ##舊方法
        ##uuid_testindex_path = os.path.join(export_txtfile_path, splitting_ratio, "GlucoseRegressionData", uuid, "Test")     ##舊方法   
        ##uuid_rawdata_path = os.path.join(export_txtfile_path, "2500_motion", uuid, "static") ##舊方法
        
        uuid_trainindex_path = os.path.join(export_txtfile_path, glucose_splitting_ratio, "GlucoseData", uuid, "Train")  
        uuid_testindex_path = os.path.join(export_txtfile_path, glucose_splitting_ratio, "GlucoseData", uuid, "Test")     
        uuid_rawdata_path = os.path.join(export_txtfile_path, "NormalizedData", uuid)
        ##uuid_rawdata_path = os.path.join(export_txtfile_path, "RawData", uuid)

                
        globals.initialize()
        code_version = get_version()
        current_time_str = strftime("%Y_%m_%d_%H%M", time.localtime())

        ## Feature extraction
        print('Start extracting features for various features...')        
        features_parquet_filename = "Features_{}_{}.parquet".format(code_version, current_time_str)
        df_features = features_extraction.load_rawdata_extract_features_multiprocess(uuid_rawdata_path, uuid_trainindex_path, uuid_testindex_path, features_uuid_path, features_parquet_filename,True)  ##講特徵寫到Dataset資料夾
        if(globals.errorcode<0):
            errorcode=str(globals.errorcode)
            message=globals.message

        return errorcode,message
    

    def _dataconcate(self,srj_db_path,srj_file_list,start_time,end_time,n_10sec=1): ##資料串接

        '''
        input ---
        db_path: srj檔案所在路徑
        srj_file_list: srj檔案列表
        start_time: 設定串接資料的開始日期(可能包含時分秒，有時分秒之格式格式 20240304 234651)  
        end_time: 設定串接資料的結束日期(可能包含時分秒，有時分秒之格式格式 20240304 234651)  
        n_10sec: the number of 10-second ecg segment 

        output ---
        ecg_data_array: 所有srj檔中每個tt片段ECG資料串接後的序列
        motion_data_array:  所有srj檔中每個tt片段motion資料串接後的序列
        breath_data_array:  所有srj檔中每個tt片段breath資料串接後的序列
        temp_data_array: 所有srj檔中每個tt片段temp資料串接後的序列
        '''
     
        ecg_data_array=[]
        motion_data_array=[]
        breath_data_array=[]
        temp_data_array=[]

        if(len(start_time)>8): ##有時分秒之格式
            start_time = datetime.strptime(start_time,"%Y%m%d %H%M%S")
            end_time = datetime.strptime(end_time,"%Y%m%d %H%M%S")
        else:   ##只有年月日之格式
            start_time = datetime.strptime(start_time,"%Y%m%d")
            end_time=datetime.strptime(end_time,"%Y%m%d") 
      
        for index in range(len(srj_file_list)):
            ecg_data_clip=[]
            motion_data_clip=[]
            breath_data_clip=[]
            temp_data_clip=[]
            count=0             
            current_file_path=os.path.join(srj_db_path,srj_file_list[index])   
            
            with open(current_file_path,"r") as srj:
                line = srj.readline()
                while line:                    
                    data = json.loads(line)
                    motions = data["rows"]["motions"]
                    ecgs=data["rows"]["ecgs"]
                    breaths=data["rows"]["breaths"]
                    temps=data["rows"]["temps"]
                    tt=data["tt"]
                    tt=int(tt/1000)
                    nowtime=datetime.fromtimestamp(tt)
                    if n_10sec==1:
                        if(nowtime>=start_time and nowtime<=end_time): ###針對當天的時段Concate Data                                                              
                            motion_data_array.append(motions)                    
                            ecg_data_array.append(ecgs) 
                            breath_data_array.append(breaths)                         
                            temp_data_array.append(temps)         
                    else:
                        if(nowtime>=start_time and nowtime<=end_time): ###針對當天的時段Concate Data   
                            motion_data_clip.extend(motions)                    
                            ecg_data_clip.extend(ecgs) 
                            breath_data_clip.extend(breaths)                         
                            temp_data_clip.extend(temps)        
                            count+=1  

                        if count==n_10sec:
                            motion_data_array.append(motion_data_clip)                    
                            ecg_data_array.append(ecg_data_clip) 
                            breath_data_array.append(breath_data_clip)                         
                            temp_data_array.append(temp_data_clip)       
                            ecg_data_clip=[]
                            motion_data_clip=[]
                            breath_data_clip=[]
                            temp_data_clip=[]
                            count=0

                    line = srj.readline()          

        return ecg_data_array,motion_data_array,breath_data_array,temp_data_array


    def _write_ecg_data(self,uuid,ecg_data_array,time_str,flag,index,glucose_value,uuid_temp_path):         
        
        time_str=time_str.replace(" ","")
        path = os.path.join(uuid_temp_path,uuid+'_'+time_str+'_'+str(flag)+'_'+str(index)+'_'+str(glucose_value)+'.txt')
        f = open(path, 'w')
        for i,ecgvalue in enumerate(ecg_data_array):
            f.write(str(ecgvalue))
            f.write("\n")
        
        f.close()
    
    
    # 只有中血糖
    # def data_balanced(self,uuid,txtfile_path,splitting_ratio=''):
    #     '''

    #     '''

    #     errorcode="0"
    #     message=""  

    #     data_balanced_filepath=os.path.join(txtfile_path,splitting_ratio,'Regression_Features',uuid, 'Remove_Data') ##創建Remove_Data資料夾
    #     if not os.path.exists(data_balanced_filepath): os.makedirs(data_balanced_filepath)

    #     path_train=os.path.join(data_balanced_filepath,'Train') ##創建Remove_Data/train 資料夾
    #     if not os.path.exists(path_train): os.makedirs(path_train)

    #     path_train_normal=os.path.join(path_train,'Normal') ##創建Remove_Data/train/normal 資料夾
    #     if not os.path.exists(path_train_normal): os.makedirs(path_train_normal)  


    #     uuid_trainindex_path = os.path.join(txtfile_path, splitting_ratio, "Regression_Features", uuid, "Train", "Normal")  
        
    #     # 1. 檢查原始資料夾是否存在
    #     if not os.path.exists(uuid_trainindex_path):
    #         print(f"資料夾不存在：{uuid_trainindex_path}")
    #         return errorcode, message

    #     # 2. 讀取txt，存取血糖值
    #     files =[
    #         f for f in os.listdir(uuid_trainindex_path)
    #         if f.endswith(".txt")
    #     ]

    #     file_info = []

    #     for filename in files:

    #         level_str = filename.split("_")[-1].split(".")[0]
    #         level = int(level_str)

    #         file_info.append({
    #                     "filename": filename,
    #                     "level": level
    #                 })

    #     print("files總筆數", len(files))
    #     print("可解析檔案數:", len(file_info))


    #     # 如果沒有可處理資料，直接結束
    #     if len(file_info) == 0:
    #         print("沒有找到txt檔案")
    #         return errorcode, message


    #     # 3. 統計每個血糖值的筆數
    #     level_counts = Counter(item["level"] for item in file_info)

    #     # 依照筆數降冪排序
    #     sorted_level_counts = sorted(
    #         level_counts.items(),
    #         key=lambda x: x[1],
    #         reverse=True
    #     )

    #     print("血糖值筆數降冪排序:")
    #     for level, count in sorted_level_counts:
    #         print(f"value={level}, count={count}")
        

    #     # 4. 使用 true count 計算 Q1、Q3、IQR、outlier threshold
    #     # 先取出排序後的 count
    #     count_list = [count for level, count in sorted_level_counts]

    #     q1 = np.percentile(count_list, 25)
    #     q3 = np.percentile(count_list, 75)
    #     iqr = q3 - q1
    #     outlier_bound = q3 + 1.5 * iqr

    #     # 搬移後目標保留數量：Q3
    #     baseline_count = int(np.ceil(q3))

    #     print(f"Q1: {q1}")
    #     print(f"Q3: {q3}")
    #     print(f"IQR: {iqr}")
    #     print(f"outlier outlier_bound = Q3 + 1.5*IQR: {outlier_bound}")
    #     print(f"outlier 搬移後保留 baseline_count(Q3): {baseline_count}")

    #     # 5. 輸出 CSV：filename、value、count
    #     csv_path = os.path.join(
    #         path_train,
    #         f"{uuid}_filename_value_count.csv"
    #     )

    #     csv_rows = []

    #     for item in file_info:
    #         filename = item["filename"]
    #         level = item["level"]
    #         count = level_counts[level]

    #         csv_rows.append({
    #             "filename": filename,
    #             "value": level,
    #             "count": count
    #         })

    #     # 依照 count 降冪排序
    #     csv_rows = sorted(
    #         csv_rows,
    #         key=lambda x: (-x["count"], x["value"], x["filename"])
    #     )

    #     with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["filename", "value", "count"])

    #         for row in csv_rows:
    #             writer.writerow([
    #                 row["filename"],
    #                 row["value"],
    #                 row["count"]
    #             ])

    #     print(f"已輸出 CSV: {csv_path}")


    #     # 6. 依照血糖值分組檔案
    #     files_by_level = defaultdict(list)

    #     for item in file_info:
    #         files_by_level[item["level"]].append(item["filename"])

    #     # 每個 value 裡面的 filename 排序，讓每次執行結果固定
    #     for level in files_by_level:
    #         files_by_level[level] = sorted(files_by_level[level])


    #     # 7. 找出 count > outlier_bound的血糖值，也就是異常高筆數
    #     values_to_balance = [
    #         level for level, count in level_counts.items()
    #         if count > outlier_bound
    #     ]

    #     values_to_balance = sorted(
    #         values_to_balance,
    #         key=lambda level: level_counts[level],
    #         reverse=True
    #     )

    #     print("需要平衡的 outlier 血糖值:")
    #     for level in values_to_balance:
    #         print(
    #             f"value={level}, count={level_counts[level]}, "
    #             f"outlier_bound={outlier_bound}, 保留到={baseline_count}"
    #         )

    #     if len(values_to_balance) == 0:
    #         message = (
    #             f"沒有血糖值的筆數超過 Q3 + 1.5*IQR，"
    #             f"Q3={q3}，outlier_bound={outlier_bound}，不需要搬移資料"
    #         )
    #         print(message)
    #         return errorcode, message
        

    #     # 8. 搬移資料
    #     moved_count = 0
    #     kept_count_total = 0

    #     for level in values_to_balance:

    #         filenames = files_by_level[level]
    #         current_count = len(filenames)
    #         keep_count = baseline_count

    #         print(
    #             f"value={level}: 原本 {current_count} 筆，"
    #             f"目標保留 {keep_count} 筆，"
    #             f"預計搬移 {current_count - keep_count} 筆"
    #         )

    #         # 浮點累積法：從 current_count 筆中，平均分散挑出 keep_count 筆保留，其餘搬移
    #         keep_indices = set()

    #         for k in range(keep_count):
    #             keep_idx = int(k * current_count / keep_count)
    #             keep_indices.add(keep_idx)

    #         kept_count_this_value = 0
    #         moved_count_this_value = 0

    #         for idx, filename in enumerate(filenames):

    #             source_file = os.path.join(uuid_trainindex_path, filename)
    #             target_file = os.path.join(path_train_normal, filename)

    #             # 在 keep_indices 裡面的檔案保留，不搬移
    #             if idx in keep_indices:
    #                 kept_count_this_value += 1
    #                 kept_count_total += 1
    #                 continue

    #             # 不在 keep_indices 裡面的檔案搬移
    #             if os.path.exists(source_file):
    #                 shutil.move(source_file, target_file)
    #                 print(f"move: {filename}")
    #                 moved_count += 1
    #                 moved_count_this_value += 1
    #             else:
    #                 print(f"來源檔案不存在，略過：{source_file}")

    #         print(
    #             f"value={level} 完成："
    #             f"保留 {kept_count_this_value} 筆，"
    #             f"搬移 {moved_count_this_value} 筆"
    #         )

    #     print("資料平衡完成")
    #     print(f"baseline_count: {baseline_count}")
    #     print(f"總搬移數量：{moved_count}")
    #     print(f"被平衡 value 的總保留數量：{kept_count_total}")

    #     message = (
    #         f"資料平衡完成，Q3={q3}outlier_bound={outlier_bound}，"
    #         f"baseline_count={baseline_count}，總搬移數量={moved_count}"
    #     )

    #     return errorcode, message

    # 新版
    def data_balanced(self,uuid,txtfile_path,splitting_ratio=''):
        '''

        '''

        errorcode="0"
        message=""  

        data_balanced_filepath=os.path.join(txtfile_path,splitting_ratio,'Regression_Features',uuid, 'Remove_Data') ##創建Remove_Data資料夾
        if not os.path.exists(data_balanced_filepath): os.makedirs(data_balanced_filepath)

        path_train=os.path.join(data_balanced_filepath,'Train') ##創建Remove_Data/train 資料夾
        if not os.path.exists(path_train): os.makedirs(path_train)

        data_types = ["Normal", "Low", "High"]
        summary_messages = []

        for data_type in data_types:
            print(f"========== 開始處理 {data_type} ==========")

            path_train_type = os.path.join(path_train, data_type)
            if not os.path.exists(path_train_type):
                os.makedirs(path_train_type)

            uuid_trainindex_path = os.path.join(
                txtfile_path,
                splitting_ratio,
                "Regression_Features",
                uuid,
                "Train",
                data_type
            )

        
            # 1. 檢查原始資料夾是否存在
            if not os.path.exists(uuid_trainindex_path):
                msg = f"{data_type} 資料夾不存在：{uuid_trainindex_path}"
                print(msg)
                summary_messages.append(msg)
                continue


            # 2. 讀取txt，存取血糖值
            files =[
                f for f in os.listdir(uuid_trainindex_path)
                if f.endswith(".txt")
            ]

            file_info = []

            for filename in files:

                level_str = filename.split("_")[-1].split(".")[0]
                level = int(level_str)

                file_info.append({
                            "filename": filename,
                            "level": level
                        })

            #print(f"{data_type} files總筆數", len(files))
            #print(f"{data_type} 可解析檔案數:", len(file_info))


            # 如果沒有可處理資料，直接處理下一個類別
            if len(file_info) == 0:
                msg = f"{data_type} 沒有找到txt檔案"
                print(msg)
                summary_messages.append(msg)
                continue


            # 3. 統計每個血糖值的筆數
            level_counts = Counter(item["level"] for item in file_info)

            # 依照筆數降冪排序
            sorted_level_counts = sorted(
                level_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            #print(f"{data_type} 血糖值筆數降冪排序:")
            for level, count in sorted_level_counts:
                print(f"value={level}, count={count}")
            

            # 4. 使用 true count 計算 Q1、Q3、IQR、outlier threshold
            # 先取出排序後的 count
            count_list = [count for level, count in sorted_level_counts]

            q1 = np.percentile(count_list, 25)
            q3 = np.percentile(count_list, 75)
            iqr = q3 - q1
            outlier_bound = q3 + 1.5 * iqr

            # 搬移後目標保留數量：Q3
            baseline_count = int(np.ceil(q3))

            # print(f"Q1: {q1}")
            # print(f"Q3: {q3}")
            # print(f"IQR: {iqr}")
            # print(f"outlier outlier_bound = Q3 + 1.5*IQR: {outlier_bound}")
            # print(f"outlier 搬移後保留 baseline_count(Q3): {baseline_count}")

            # 5. 輸出 CSV：filename、value、count
            csv_path = os.path.join(
                path_train,
                f"{uuid}_{data_type}_filename_value_count.csv"
            )

            csv_rows = []

            for item in file_info:
                filename = item["filename"]
                level = item["level"]
                count = level_counts[level]

                csv_rows.append({
                    "filename": filename,
                    "value": level,
                    "count": count
                })

            # 依照 count 降冪排序
            csv_rows = sorted(
                csv_rows,
                key=lambda x: (-x["count"], x["value"], x["filename"])
            )

            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "value", "count"])

                for row in csv_rows:
                    writer.writerow([
                        row["filename"],
                        row["value"],
                        row["count"]
                    ])

            #print(f"已輸出 CSV: {csv_path}")


            # 6. 依照血糖值分組檔案
            files_by_level = defaultdict(list)

            for item in file_info:
                files_by_level[item["level"]].append(item["filename"])

            # 每個 value 裡面的 filename 排序，讓每次執行結果固定
            for level in files_by_level:
                files_by_level[level] = sorted(files_by_level[level])


            # 7. 找出 count > outlier_bound的血糖值，也就是異常高筆數
            values_to_balance = [
                level for level, count in level_counts.items()
                if count > outlier_bound
            ]

            values_to_balance = sorted(
                values_to_balance,
                key=lambda level: level_counts[level],
                reverse=True
            )

            #print(f"{data_type} 需要平衡的 outlier 血糖值:")
            # for level in values_to_balance:
            #     print(
            #         f"value={level}, count={level_counts[level]}, "
            #         f"outlier_bound={outlier_bound}, 保留到={baseline_count}"
            #     )

            if len(values_to_balance) == 0:
                msg = (
                    f"{data_type} 沒有血糖值的筆數超過 Q3 + 1.5*IQR，"
                    f"Q3={q3}，outlier_bound={outlier_bound}，不需要搬移資料"
                )
                print(msg)
                summary_messages.append(msg)
                continue
            

            # 8. 搬移資料
            moved_count = 0
            kept_count_total = 0

            for level in values_to_balance:

                filenames = files_by_level[level]
                current_count = len(filenames)
                keep_count = baseline_count

                # print(
                #     f"value={level}: 原本 {current_count} 筆，"
                #     f"目標保留 {keep_count} 筆，"
                #     f"預計搬移 {current_count - keep_count} 筆"
                # )

                # 浮點累積法：從 current_count 筆中，平均分散挑出 keep_count 筆保留，其餘搬移
                keep_indices = set()

                for k in range(keep_count):
                    keep_idx = int(k * current_count / keep_count)
                    keep_indices.add(keep_idx)

                kept_count_this_value = 0
                moved_count_this_value = 0

                for idx, filename in enumerate(filenames):

                    source_file = os.path.join(uuid_trainindex_path, filename)
                    target_file = os.path.join(path_train_type, filename)

                    # 在 keep_indices 裡面的檔案保留，不搬移
                    if idx in keep_indices:
                        kept_count_this_value += 1
                        kept_count_total += 1
                        continue

                    # 不在 keep_indices 裡面的檔案搬移
                    if os.path.exists(source_file):
                        shutil.move(source_file, target_file)
                        #print(f"move: {filename}")
                        moved_count += 1
                        moved_count_this_value += 1
                    else:
                        print(f"來源檔案不存在，略過：{source_file}")

                # print(
                #     f"value={level} 完成："
                #     f"保留 {kept_count_this_value} 筆，"
                #     f"搬移 {moved_count_this_value} 筆"
                # )

            msg = (
                f"{data_type} 資料平衡完成，Q3={q3}，outlier_bound={outlier_bound}，"
                f"baseline_count={baseline_count}，總搬移數量={moved_count}"
            )
            print(msg)
            print(f"被平衡 value 的總保留數量：{kept_count_total}")

            summary_messages.append(msg)

        message = "；".join(summary_messages)
        print("Normal / Low / High 資料平衡處理完成")

        return errorcode, message


    def data_downsample_by_ratio(
        self,
        uuid,
        txtfile_path,
        splitting_ratio="",
        ratio=0.3,
        cap_ref="median",
        seed=42,
    ):
        """
        依「每血糖值上限」做 down-sampling（取代 IQR outlier remove）。

        keep(g) = min(count(g), cap)
        - cap_ref="median": cap = ceil(median_count * ratio)
        - cap_ref="max":    cap = floor(max_count * ratio)

        建議試 ratio ∈ {0.2, 0.3, 0.5, 0.7, 1.0}
        - median + r=1.0：砍到中位數
        - max + r=1.0：幾乎不砍
        - median + r=0.3：多數值最多保留「中位數筆數的 30%」

        超過 cap 的檔案 move 到 Regression_Features/{uuid}/Remove_Data/Train/{Normal,Low,High}
        """
        errorcode = "0"
        message = ""

        if ratio <= 0:
            return "-1", f"ratio 必須 > 0，收到 {ratio}"
        if cap_ref not in ("median", "max"):
            return "-1", f"cap_ref 只能是 'median' 或 'max'，收到 {cap_ref}"

        data_balanced_filepath = os.path.join(
            txtfile_path, splitting_ratio, "Regression_Features", uuid, "Remove_Data"
        )
        os.makedirs(data_balanced_filepath, exist_ok=True)
        path_train = os.path.join(data_balanced_filepath, "Train")
        os.makedirs(path_train, exist_ok=True)

        rng = np.random.default_rng(seed)
        data_types = ["Normal", "Low", "High"]
        summary_messages = []

        for data_type in data_types:
            print(f"========== downsample {data_type} (cap_ref={cap_ref}, ratio={ratio}) ==========")

            path_train_type = os.path.join(path_train, data_type)
            os.makedirs(path_train_type, exist_ok=True)

            uuid_trainindex_path = os.path.join(
                txtfile_path,
                splitting_ratio,
                "Regression_Features",
                uuid,
                "Train",
                data_type,
            )

            if not os.path.exists(uuid_trainindex_path):
                msg = f"{data_type} 資料夾不存在：{uuid_trainindex_path}"
                print(msg)
                summary_messages.append(msg)
                continue

            files = [f for f in os.listdir(uuid_trainindex_path) if f.endswith(".txt")]
            file_info = []
            for filename in files:
                try:
                    level = int(filename.split("_")[-1].split(".")[0])
                except ValueError:
                    print(f"[downsample] 檔名格式異常，跳過: {filename}")
                    continue
                file_info.append({"filename": filename, "level": level})

            if len(file_info) == 0:
                msg = f"{data_type} 沒有找到 txt 檔案"
                print(msg)
                summary_messages.append(msg)
                continue

            level_counts = Counter(item["level"] for item in file_info)
            count_list = list(level_counts.values())
            max_count = int(np.max(count_list))
            median_count = float(np.median(count_list))

            if cap_ref == "median":
                cap = int(np.ceil(median_count * ratio))
            else:
                cap = int(np.floor(max_count * ratio))
            cap = max(cap, 1)  # 至少留 1，避免 cap=0 清空該血糖值

            print(
                f"{data_type}: n_files={len(file_info)}, n_levels={len(level_counts)}, "
                f"max_count={max_count}, median_count={median_count:.1f}, cap={cap}"
            )

            files_by_level = defaultdict(list)
            for item in file_info:
                files_by_level[item["level"]].append(item["filename"])
            for level in files_by_level:
                files_by_level[level] = sorted(files_by_level[level])

            # 紀錄每個血糖值的處理結果
            summary_csv_path = os.path.join(
                path_train,
                f"{uuid}_{data_type}_downsample_{cap_ref}_r{ratio:.2f}.csv",
            )
            summary_rows = []
            moved_count = 0
            kept_count_total = 0
            levels_trimmed = 0

            for level in sorted(files_by_level.keys()):
                filenames = files_by_level[level]
                current_count = len(filenames)
                keep_count = min(current_count, cap)

                if keep_count >= current_count:
                    # 本來就低於上限：全留
                    kept_count_total += current_count
                    summary_rows.append(
                        [level, current_count, keep_count, 0, "keep_all"]
                    )
                    continue

                levels_trimmed += 1
                # 固定 seed：先打亂再取前 keep_count（可重現）
                order = np.arange(current_count)
                rng.shuffle(order)
                keep_indices = set(order[:keep_count].tolist())

                moved_this = 0
                for idx, filename in enumerate(filenames):
                    source_file = os.path.join(uuid_trainindex_path, filename)
                    if idx in keep_indices:
                        kept_count_total += 1
                        continue
                    target_file = os.path.join(path_train_type, filename)
                    if os.path.exists(source_file):
                        shutil.move(source_file, target_file)
                        moved_count += 1
                        moved_this += 1
                    else:
                        print(f"來源檔案不存在，略過：{source_file}")

                summary_rows.append(
                    [level, current_count, keep_count, moved_this, "trimmed"]
                )

            with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["bg_value", "original_count", "keep_count", "moved_count", "action"]
                )
                writer.writerows(summary_rows)

            msg = (
                f"{data_type} downsample 完成：cap_ref={cap_ref}, ratio={ratio}, cap={cap}, "
                f"trimmed_levels={levels_trimmed}, kept={kept_count_total}, moved={moved_count}"
            )
            print(msg)
            print(f"summary csv: {summary_csv_path}")
            summary_messages.append(msg)

        message = "；".join(summary_messages)
        print("Normal / Low / High downsample 處理完成")
        return errorcode, message


##-------------For regression model--------------
class Regression_ECGDataset(Dataset):
    def __init__(self, dir_path, used_feature_array, type='Normal', method='raw'):
        self.dir_path = os.path.abspath(dir_path)
        self.method = method
        ##self.data_len = data_len 
                
        if(type=='Normal'):        
            normalLabels, normalSigs = self.read_files('Normal',used_feature_array)           
        elif(type=='High'):
            normalLabels, normalSigs = self.read_files('High',used_feature_array)
        else:
            normalLabels, normalSigs = self.read_files('Low',used_feature_array)

        self.Signals = normalSigs      
        self.Labels = normalLabels        
       

    def __getitem__(self, index):
        signal = self.Signals[index]
        label = self.Labels[index]
        
        return signal, label

    def __len__(self):
        return len(self.Labels)

    ## Parsing files in folder
    def read_files(self, foldername,used_feature_array):
        
        Sig = []
        Label=[]
        self.file_list = []
        self.channel = 0

        for filename in os.listdir(os.path.join(self.dir_path, foldername)):
            file_path = os.path.join(self.dir_path, foldername, filename)
                        
            Sig.append(self.load_data(file_path, used_feature_array))

            self.file_list.append(filename)

            string_array=filename.split('_')
            value_string=(string_array[-1]).split('.')
            value=int(value_string[0])
            Label.append(value) 
                       

        Sig = torch.FloatTensor(Sig)
        Label = torch.FloatTensor(Label)
        
        if self.channel == 1:
            Sig = Sig.unsqueeze(1)
       
              
        return Label, Sig

    ## Read data in files & Preprocessing
    def load_data(self, filepath, used_feature_array):
       
        values = np.genfromtxt(filepath, delimiter = '')       
        values=values[:, 1] ##取得第2個column的數值              
        feature_indices = np.where(used_feature_array == 1)[0]       
        sig = values[feature_indices]
        sig.reshape(len(feature_indices), 1)
                
        self.channel = 1   
       
               
        if self.method == 'raw' or self.method == 'meta':
            # 'meta' 是訓練管線名稱，讀特徵與 raw 相同
            return sig.tolist()
        
        if self.method == 'time':
            sig = self.normalize1(sig)
            return sig.tolist()
        
               
        if self.method == 'freq':
            sig = self.normalize1(sig)
            #sig = self.FFT(sig)
            sig = self.DWT(sig)
            self.channel = len(sig)          
            return sig.tolist()
        
        if self.method == 'combine':
            sig = self.normalize1(sig)
            x1 = sig.tolist()
            #x2 = self.FFT(sig).tolist()
            x3 = self.DWT(sig).tolist()
            #Comb = [x1, x2, x3]
            x3.append(x1)
            self.channel = len(x3)
            return x3

        raise ValueError(f"Unknown feature load method: {self.method!r}")
    def remove_collinearity(self, x, y, corr_threshold=0.95, vif_threshold=15.0, min_features=5):
        """
        x: shape (n_samples, n_features)
        y: shape (n_samples,)  目標變數(Glucose)
        回傳: 篩選後的 x，以及保留下來的「原始特徵索引」keep_idx

        Step1 Pearson: |r_ij| > corr_threshold 時，保留與 Glucose 絕對相關較高的那個特徵
        Step2 VIF: 逐一刪除 VIF 最大者，但至少保留 min_features 個特徵
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_features = x.shape[1]
        keep_idx = list(range(n_features))

        # ---- Step 1: Pearson 相關係數，兩兩比對；衝突時保留與 Glucose 相關較高者 ----
        corr_matrix = np.corrcoef(x, rowvar=False)  # (n_features, n_features)
        target_corr = np.zeros(n_features)
        for i in range(n_features):
            if np.std(x[:, i]) == 0 or np.std(y) == 0:
                target_corr[i] = 0.0
            else:
                target_corr[i] = abs(np.corrcoef(x[:, i], y)[0, 1])
        target_corr = np.nan_to_num(target_corr, nan=0.0)

        to_drop = set()
        for i in range(n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue
                if abs(corr_matrix[i, j]) > corr_threshold:
                    # 保留與 Glucose 絕對相關較高者；相等時保留較小索引 i
                    if target_corr[i] >= target_corr[j]:
                        to_drop.add(j)
                    else:
                        to_drop.add(i)
                        break

        keep_idx = [idx for idx in keep_idx if idx not in to_drop]
        x = x[:, keep_idx]

        # ---- Step 2: VIF，逐一刪除最嚴重的，直到都低於門檻；至少保留 min_features 個 ----
        while x.shape[1] > min_features:
            corr = np.corrcoef(x, rowvar=False)
            try:
                vif = np.diag(np.linalg.inv(corr))
            except np.linalg.LinAlgError:
                break  # 矩陣奇異，無法再算，直接停止
            worst = np.argmax(vif)
            if vif[worst] < vif_threshold:
                break
            x = np.delete(x, worst, axis=1)
            del keep_idx[worst]

        return x, keep_idx
    def normalize1(self, x):
        x_max = np.max(x)
        x_min = np.min(x)
        x_norm = (x - x_min) / (x_max - x_min+1)  
        return x_norm
    
    def FFT(self, x):
        y = fft(x)
        P2 = abs(y) / len(x)
        P1 = P2[range(len(x)//2)]
        return P2
    
    def DWT(self, x):
        ##sig = np.zeros(2504)
        sig = np.zeros(150)
        sig[:len(x)] = x
        coeffs = pywt.swt(sig, 'sym4', level=1, trim_approx=True)       
        coeffs = np.array(coeffs)
                  
        return coeffs

def save_predictions_csv(file_names, y_pred, target_data, output_folder, current_time_str, model_type):
    print(f"\n{'Index':<8} {'檔案名稱':<30} {'預測值':>10} {'真實值':>10} {'誤差':>10}")
    print("-" * 72)
    for idx, (fname, pred, target) in enumerate(zip(file_names, y_pred, target_data)):
        print(f"{idx:<8} {fname:<30} {pred:>10.4f} {float(target):>10.4f} {abs(pred - float(target)):>10.4f}")

    csv_path = os.path.join(output_folder, f"Predictions_{current_time_str}_{model_type}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["檔案名稱", "預測值", "真實值"])
        for fname, pred, target in zip(file_names, y_pred, target_data):
            writer.writerow([fname, f"{pred:.4f}", f"{float(target):.4f}"])
    print(f"[已儲存 CSV] {csv_path}")


def clip_train_features_by_label_group(
    X_train,
    y_train,

    min_group_samples=30
    ):
    """
    Only apply to train data.

    Labels are grouped by value range.
    Example:
        min label = 69
        label_group_range = 5
        group_width = 6

        group 0: 69–74
        group 1: 75–80
        group 2: 81–86

    For each group and each feature:
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        values are clipped to [lower, upper]
    """
    clip_summary = []
    X_df = pd.DataFrame(X_train)
    y_series = pd.Series(y_train, name="label")

    df = X_df.copy()
    df["label"] = y_series.values

    feature_cols = X_df.columns.tolist()

    for label_value, group_idx in df.groupby("label").groups.items():

        group_data = df.loc[group_idx, feature_cols]

        if len(group_data) < min_group_samples:
            continue

        for feature in feature_cols:
            values = group_data[feature]

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            lower_count = (values < lower).sum()
            upper_count = (values > upper).sum()
            clip_count = lower_count + upper_count

            df.loc[group_idx, feature] = values.clip(lower=lower, upper=upper)

            
            clip_summary.append({
                "label": label_value,
                "feature": feature,
                "sample_count": len(values),
                "lower_count": lower_count,
                "upper_count": upper_count,
                "clip_count": clip_count,
                "clip_ratio": clip_count / len(values)
            })

    X_train_clipped = df[feature_cols].values

    clip_summary_df = pd.DataFrame(clip_summary)

    clip_summary_df.to_csv(
        "clip_summary.csv",
        index=False
    )
    total_values = X_train_clipped.size

    if len(clip_summary_df) > 0:
        total_clipped = clip_summary_df["clip_count"].sum()
    else:
        total_clipped = 0

    overall_ratio = total_clipped / total_values

    print(f"Overall clipped ratio: {overall_ratio:.4%}")
    return X_train_clipped


def BuildRegressionModel(uuid,basepath,splitting_ratio,model_type_array):
    
    '''
    Build normal, low or high regression models according to the glucose category in model_type_array
    '''

    errorcode="0"
    message=""
    status=1

    classes_num=len(model_type_array)
    current_time_str=strftime("%Y_%m_%d_%H%M", time.localtime()) ###取得模型訓練時當前時間
    code_version=get_version()   ##取得模型訓練時當前程式版本

    model_current_best_performance_txtfile=[]
    model_historic_best_performance_txtfile=[]   
    if(classes_num==1): ##只有normal, 但依然放在兩類的資料夾內
        if(model_type_array[0]=="Normal"):
            print('Start building the regression models of normal class!')        
            regression_model_output_folder = os.path.join(basepath,splitting_ratio,"Best_TwoClasses_Regression_Model", uuid)  
            if not os.path.exists(regression_model_output_folder): 
                os.makedirs(regression_model_output_folder)   
            model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_Normal.txt"))    
            model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_Normal.txt")) 
        

        if(model_type_array[0]=="High"):
            print('Start building the regression models of high class!')        
            regression_model_output_folder = os.path.join(basepath,splitting_ratio,"Best_TwoClasses_Regression_Model", uuid)  
            if not os.path.exists(regression_model_output_folder): 
                os.makedirs(regression_model_output_folder)   
            model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_High.txt"))    
            model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_High.txt")) 
        
       
    elif(classes_num==2):
        print('Start building the regression models of two classes!')        
        regression_model_output_folder = os.path.join(basepath,splitting_ratio,"Best_TwoClasses_Regression_Model", uuid)  
        if not os.path.exists(regression_model_output_folder): 
            os.makedirs(regression_model_output_folder)   

        if(model_type_array[0]=="Normal" and model_type_array[1]=="High"):
            model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_Normal.txt"))
            model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_High.txt"))        
            model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_Normal.txt"))   
            model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_High.txt"))
        else:  ##normal and low
            model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_Normal.txt"))
            model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_Low.txt"))        
            model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_Normal.txt"))   
            model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_Low.txt"))     

    elif(classes_num==3):
        print('Start building the regression models of three classes!')      
        regression_model_output_folder = os.path.join(basepath,splitting_ratio,"Best_ThreeClasses_Regression_Model", uuid) 
        if not os.path.exists(regression_model_output_folder): 
            os.makedirs(regression_model_output_folder) 

        model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_Normal.txt"))
        model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_High.txt"))
        model_current_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Performance_"+code_version+"_"+current_time_str+"_Low.txt"))      
        model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_Normal.txt"))   
        model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_High.txt")) 
        model_historic_best_performance_txtfile.append(os.path.join(regression_model_output_folder,"Historic_Best_Performance_Low.txt"))   
      
      
    # Dataset 特徵載入用 raw（load_data 無 'meta' 分支；'meta' 是訓練管線不是讀檔 mode）
    feature_method = 'raw'
    # 依各 uuid×class 的 Train 筆數自動選 mode：n<1200 → manual(SVR-RBF)；n≥1200 → meta(RFECV-linear+XGB)
    MODE_N_THRESHOLD = 1200
    num_cv=3  ##srv_auto中的cross validation切割個數（僅 auto/meta 使用）

    used_feature_array = np.array([row[0] for row in used_feature_dic]) 
    false_count=0
    for i, model_type in enumerate(model_type_array): ##個別一個一個訓練

        print('It is dealing with '+model_type+' model...')
        
        current_path=os.path.join(basepath,splitting_ratio,'Regression_Features',uuid,'Train')    
        if(model_type=='Normal'):
            filelist_model_type=os.listdir(os.path.join(current_path,'Normal'))    
        elif(model_type=='High'):
            filelist_model_type=os.listdir(os.path.join(current_path,'High'))
        else:
            filelist_model_type=os.listdir(os.path.join(current_path,'Low'))    

        if(len(filelist_model_type)<num_cv):  ##如果樣本個數少於num_cv, 此類別的回歸模型不訓練
            continue 
        
        current_test_path=os.path.join(basepath,splitting_ratio,'Regression_Features',uuid,'Test')  
        if(model_type=='Normal'):
            filelist_test_model_type=os.listdir(os.path.join(current_test_path,'Normal'))     
        elif(model_type=='High'):
            filelist_test_model_type=os.listdir(os.path.join(current_test_path,'High'))
        else:
            filelist_test_model_type=os.listdir(os.path.join(current_test_path,'Low'))    

       
        if(len(filelist_model_type)==0 or len(filelist_test_model_type)==0):        
            print('Runing the BuildRegressionModel function of Regression_Model_Predictor.py: No training or testing data!')
            false_count=false_count+1
            continue

        n_train = len(filelist_model_type)
        if n_train < MODE_N_THRESHOLD:
            training_mode = 'manual'
        else:
            training_mode = 'meta'
        print(f'[mode] uuid={uuid} {model_type} n_train={n_train} → {training_mode} (threshold={MODE_N_THRESHOLD})')
           

        if not os.path.exists(model_current_best_performance_txtfile[i]): ###沒有model_current_best_performance_txtfile檔案，先創造檔案
            with open(model_current_best_performance_txtfile[i], "w") as file:
                file.write(f"largest_value:{0:.1f}\n")
                file.write(f"lowest_value:{0:.1f}\n")
                file.write("features_array: []\n")
                file.write("MSE:0\n")
                file.write("MARD:0\n") 

        if not os.path.exists(model_historic_best_performance_txtfile[i]): ###沒有model_historic_best_performance_txtfile檔案，先創造檔案
            with open(model_historic_best_performance_txtfile[i], "w") as file:
                file.write(f"largest_value:{0:.1f}\n")
                file.write(f"lowest_value:{0:.1f}\n")
                file.write("features_array: []\n")
                file.write("MSE:0\n")
                file.write("MARD:0\n")        
                file.write("Model_Name: ")

        dataset = Regression_ECGDataset(dir_path=current_path, used_feature_array=used_feature_array, type=model_type, method=feature_method)   
        dataset_size = len(dataset)
        indices = list(range(dataset_size))    
        #train_sampler = SubsetRandomSampler(indices)
        #train_loader = DataLoader(dataset, sampler=train_sampler) 
        train_loader = DataLoader(dataset, shuffle=False)  #jane新增
        
        testdata = Regression_ECGDataset(dir_path=current_test_path, used_feature_array=used_feature_array, type=model_type, method=feature_method)   
        test_loader = DataLoader(testdata)  

        ##---------Building SVM model----------   
        X_all = []
        y_all = []    
        MSE = 0,0
        data_all = []
        target_all = []

        for batch_x, batch_y in train_loader:       
            X_all.append(batch_x.squeeze().numpy())  # Squeeze to remove extra dimensions and convert to numpy
            a=batch_x.squeeze().numpy()
            Flag, Message = check_infinity_or_too_large(a)
            if Flag:
                print(a)

            y_all.append(batch_y.numpy())  # Convert targets to numpy   
        
        ## Stack all batches together (this combines the data from all batches)
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)
            
        ## Handle NaN values by replacing with column means
        col_means = np.nanmean(X_all, axis=0)  # Calculate column means ignoring NaNs
        nan_indices = np.where(np.isnan(X_all))  # Find the NaN indices
        X_all[nan_indices] = np.take(col_means, nan_indices[1])  # Replace NaNs with column means
        print('col_means:',col_means)

        file_names_all = []  # ← 新增
        for idx, (data, target) in enumerate(test_loader):  
            data_all.append(data.squeeze().numpy())
            target_all.append(target.numpy())
            # 取得檔案名稱（Regression_ECGDataset 需要有 .file_list 或類似屬性）
            file_names_all.append(testdata.file_list[idx])  # ← 新增，請確認屬性名稱

        data_all = np.vstack(data_all)
        target_all = np.concatenate(target_all)
        
        nan_idx_test = np.where(np.isnan(data_all))
        data_all[nan_idx_test] = np.take(col_means, nan_idx_test[1])  ##為了讓分布符合母群體分布，所以取train的mean來使用

        ## 共線性篩選（只在 train fit；keep_idx 套用到 test）
        n_feat_before = X_all.shape[1]
        X_all, keep_idx = dataset.remove_collinearity(X_all, y_all)
        data_all = data_all[:, keep_idx]
        col_means = np.asarray(col_means)[keep_idx]
        print(f'[collinearity] {model_type}: kept {len(keep_idx)}/{n_feat_before} features, keep_idx={keep_idx}')

        ## Scale the features using StandardScaler
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        data_all_scaled = scaler.transform(data_all)  ##這邊不使用fit_transform

        Bias = 0.0
        MAE = 0.0
        RMSE = 0.0
        selected_feature_indices = None
        best_y_pred = None

        if(training_mode=='xgboost'):
            
            MSE,MARD,Bias,MAE,RMSE,max_glucose_value,min_glucose_value,model, y_pred=_xgboost(X_all_scaled,y_all,data_all_scaled,target_all)
            current_rfecv_name='None'
            current_svr_model_name="xGB_Model_"+current_time_str+"_"+model_type+".json"
            save_path=os.path.join(regression_model_output_folder,current_svr_model_name)
            best_y_pred = y_pred
            model.save_model(save_path)  
        
        elif(training_mode=='manual'):
            print('manual mode (SVR-RBF, no RFECV / no XGB)')
            MSE,MARD,max_glucose_value,min_glucose_value,model,y_pred_train,y_pred_test = _svr_manual(
                X_all_scaled, y_all, data_all_scaled, target_all
            )
            errors = np.abs(target_all - y_pred_test)
            Bias = float(np.mean(y_pred_test - target_all))
            MAE = float(np.mean(errors))
            RMSE = float(np.sqrt(MSE))
            current_rfecv_name = 'None'
            current_svr_model_name = 'SVR_Model_' + current_time_str + '_' + model_type + '.pkl'
            joblib.dump(model, os.path.join(regression_model_output_folder, current_svr_model_name))
            best_y_pred = y_pred_test

        else:
            # meta：RFECV-linear SVR → SVR pred 當 XGB 特徵 → XGB（不再與 manual RBF 比 MARD）
            print('meta mode (RFECV-linear + XGB, no manual compare)')
            MSE_auto,MARD_auto,max_glucose_value_auto,min_glucose_value_auto,rfecv,svr,selected_feature_indices,y_pred_train_auto,y_pred_test_auto = _svr_auto(X_all_scaled,y_all,data_all_scaled,target_all,num_cv)
            current_rfecv_name='Rfecv_Model_'+current_time_str+"_"+model_type+'.pkl'
            current_svr_model_name='SVR_Model_'+current_time_str+"_"+model_type+'.pkl'
            joblib.dump(rfecv, os.path.join(regression_model_output_folder,current_rfecv_name))
            joblib.dump(svr, os.path.join(regression_model_output_folder,current_svr_model_name))

            mode = 'auto'
            best_svr_model = svr
            best_rfecv = rfecv
            best_selected_feature_indices = selected_feature_indices
            best_y_pred_train = y_pred_train_auto
            best_y_pred_test = y_pred_test_auto

            # =========================
            # 加入 SVR prediction 當作 XGBoost feature
            # =========================
            X_train_residual = np.column_stack((X_all_scaled, best_y_pred_train))
            X_test_residual = np.column_stack((data_all_scaled, best_y_pred_test))

            # =========================
            # 進入 XGBoost
            # =========================
            MSE, MARD, Bias, MAE, RMSE,max_glucose_value,min_glucose_value,xgb_model, y_pred=_xgboost(X_train_residual,y_all,X_test_residual,target_all)
            current_rfecv_name='None'
            current_svr_model_name="xGB_Model_"+current_time_str+"_"+model_type+".json"
            save_path=os.path.join(regression_model_output_folder,current_svr_model_name)
            best_y_pred = y_pred
            xgb_model.save_model(save_path)

            # 把meta model存出
            current_meta_model_name = 'Meta_Model_' + current_time_str + '_' + model_type + '.pkl'
            meta_model_path = os.path.join(regression_model_output_folder, current_meta_model_name)

            meta_model_package = {
                'mode': mode,
                'feature_scaler': scaler,
                'collinearity_keep_idx': keep_idx,
                'svr_model': best_svr_model,
                'rfecv': best_rfecv,
                'selected_feature_indices': best_selected_feature_indices,
                'xgb_model': xgb_model,
                'model_type': model_type
            }

            joblib.dump(meta_model_package, meta_model_path)

            print('Meta model saved:', meta_model_path)



        save_predictions_csv(
            file_names_all, best_y_pred, target_all,
            regression_model_output_folder, current_time_str, model_type
        )      
              
        print(f"MSE:{MSE:.4f}, MARD:{MARD:.4f}, Bias:{Bias:.4f}, MAE:{MAE:.4f}, RMSE:{RMSE:.4f}, largest_value:{max_glucose_value:.1f}, lowest_value:{min_glucose_value:.1f}")        
        
        with open(model_current_best_performance_txtfile[i], 'w') as f:
            f.write(f"largest_value:{max_glucose_value:.1f}\n")
            f.write(f"lowest_value:{min_glucose_value:.1f}\n")
            if(training_mode=='xgboost' or training_mode=='manual' or training_mode=='symbolic'):
                f.write("used feature array: all"+"\n")
            else:
                f.write("used feature array: "+str(selected_feature_indices)+"\n")

            f.write(f"MSE:{MSE:.4f}\n")
            f.write(f"MARD:{MARD:.4f}\n")
            f.write(f"Bias:{Bias:.4f}\n")
            f.write(f"MAE:{MAE:.4f}\n")
            f.write(f"RMSE:{RMSE:.4f}\n")

        
        col_means_txtfile_path=os.path.join(regression_model_output_folder,"cols_mean.txt")
        with open(col_means_txtfile_path, 'w') as f:
            for value in col_means:
                f.write(str(value)+"\n")

        keep_idx_txtfile_path=os.path.join(regression_model_output_folder,"collinearity_keep_idx.txt")
        with open(keep_idx_txtfile_path, 'w') as f:
            for idx in keep_idx:
                f.write(str(idx)+"\n")
        

        historic_best_largest_value=0
        historic_best_lowest_value=0
        historic_best_MARD=0                
        performance_txtfile = open(model_historic_best_performance_txtfile[i])  ###讀取Normal,High or Low的historic performance 績效     
        for line in performance_txtfile.readlines():
            line_string=line.split(":")
            if(line_string[0]=="largest_value"):
                historic_best_largest_value=float(line_string[1])

            if(line_string[0]=="lowest_value"):
                historic_best_lowest_value=float(line_string[1])

            if(line_string[0]=="MARD"):
                historic_best_MARD=float(line_string[1])  
        
        performance_txtfile.close()

        ##檢查是否更新需要績效
        if((historic_best_largest_value-historic_best_lowest_value)==0): ###第一次執行，直接將結果也寫入historic best performance file
            print("Model is trained for the first time...")
            status=1
            with open(model_historic_best_performance_txtfile[i], 'w') as f:  ###寫入historic_best_performance
                f.write(f"largest_value:{max_glucose_value:.1f}\n")
                f.write(f"lowest_value:{min_glucose_value:.1f}\n")
                if(training_mode=='xgboost' or training_mode=='manual' or training_mode=='symbolic'):
                    f.write("used feature array: all"+"\n")
                else:
                    f.write("used feature array: "+str(selected_feature_indices)+"\n")
                
                f.write(f"MSE:{MSE:.4f}\n")
                f.write(f"MARD:{MARD:.4f}\n")
                f.write(f"Bias:{Bias:.4f}\n")
                f.write(f"MAE:{MAE:.4f}\n")
                f.write(f"RMSE:{RMSE:.4f}\n")
                f.write(f"Rfecv_Name:"+current_rfecv_name+"\n")
                f.write(f"Model_Name:"+current_svr_model_name+"\n")               
        else:
            print("Model have been training several times...")
            status=0
            if(True):
            ##if((max_glucose_value-min_glucose_value)-3>=(historic_best_largest_value-historic_best_lowest_value)): ##新的測試資料的最大最小值範圍更廣
                if(MARD<historic_best_MARD):
                    status=1 ##比之前績效好
                    with open(model_historic_best_performance_txtfile[i], 'w') as f:  ###寫入historic_best_performance
                        f.write(f"largest_value:{max_glucose_value:.1f}\n")
                        f.write(f"lowest_value:{min_glucose_value:.1f}\n")
                        if(training_mode=='xgboost' or training_mode=='manual' or training_mode=='symbolic'):
                            f.write("used feature array: all"+"\n")
                        else:
                            f.write("used feature array: "+str(selected_feature_indices)+"\n")

                        f.write(f"MSE:{MSE:.4f}\n")
                        f.write(f"MARD:{MARD:.4f}\n")
                        f.write(f"Bias:{Bias:.4f}\n")
                        f.write(f"MAE:{MAE:.4f}\n")
                        f.write(f"RMSE:{RMSE:.4f}\n")
                        f.write(f"Rfecv_Name:"+current_rfecv_name+"\n")
                        f.write(f"Model_Name:"+current_svr_model_name+"\n")   

        performance_table.append([str(uuid), str(f"MSE:{MSE:.4f}"), str(f"MARD:{MARD:.4f}"), str(f"Bias:{Bias:.4f}"), str(f"MAE:{MAE:.4f}"), str(f"RMSE:{RMSE:.4f}"), str(f"largest_value:{max_glucose_value:.1f}"), str(f"lowest_value:{min_glucose_value:.1f}")])
    
    if(false_count==len(model_type_array)):  ##各類別都沒有訓練成功
        status=-1
        errorcode="-907"
        message="An error occurs in the BuildRegressionModel function of Regression_Model_Predictor.py: No any regression model have been built!"

    return status, errorcode, message



def _xgboost(X,y,test_data,target_data):

  
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    # -------------------------------
    # 3. Define XGBoost regressor
    # -------------------------------
    
    
    model = xgb.XGBRegressor(
        n_estimators=1000,       # number of boosting rounds
        learning_rate=0.10,     # step size shrinkage
        max_depth=8,            # tree depth
        subsample=0.7,          # row sampling
        colsample_bytree=0.8,   # feature sampling        
        tree_method="hist",     # efficient for larger data
        random_state=42,
        eval_metric="mape", ##"rmse",
        early_stopping_rounds=20,
    )
    

    # -------------------------------
    # 4. Train model
    # -------------------------------
    initial_model = xgb.XGBRegressor(random_state=42)
    
    '''
    param_grid = {
    'n_estimators': [800, 1000, 1200],
    'learning_rate': [0.01, 0.05, 0.1,0.15],
    'max_depth': [5, 6, 7, 8],
    ##'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
    }

    
    grid_search = GridSearchCV(
    estimator=initial_model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',  
    cv=5,
    verbose=1,
    n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    '''

    
    model.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            ##sample_weight=weights,
            
            verbose=True)
    
    # -------------------------------
    # 5. Evaluate
    # -------------------------------
    ###y_pred_test = np.expm1(model.predict(test_data))
    y_pred_test = model.predict(test_data)
    errors = np.abs(target_data - y_pred_test)
    MSE = np.mean((y_pred_test - target_data)**2)
    MARD = np.mean(errors / target_data) * 100
    Bias = np.mean(y_pred_test - target_data)
    MAE = np.mean(errors) 
    RMSE = np.sqrt(MSE)


    # 測 ratio 加速：關掉逐筆 print
    # for pred, target in zip(y_pred_test, target_data):
    #     print(f"prediction: {pred:.4f} target: {target:.4f}")

    max_glucose_value = np.max(target_data)
    min_glucose_value = np.min(target_data)
    
    ##plot_importance(model, importance_type='gain')
    ##plt.show()

    '''
    importance = model.feature_importances_

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)

    shap.summary_plot(shap_values, X_train)
    '''

    return MSE, MARD, Bias, MAE, RMSE, max_glucose_value,min_glucose_value,model,y_pred_test


def _svr_manual(X_all_scaled,y_all,test_data,target_data,mode='log'):
    
    model = SVR(kernel='rbf', C=100.0, epsilon=0.1)          
    
    if(mode=='log'):
        y_all = np.log(y_all + 1e-8)
      
    model.fit(X_all_scaled, y_all)  

    # train prediction
    y_pred_train = model.predict(X_all_scaled)

    # test prediction
    y_pred_test = model.predict(test_data)     
    
    if(mode=='log'):
        y_pred_train = np.exp(y_pred_train)
        y_pred_test = np.exp(y_pred_test)

    errors = np.abs(target_data - y_pred_test)
    MSE = np.mean((y_pred_test - target_data)**2)
    MARD = np.mean(errors / target_data) * 100
    Bias = np.mean(y_pred_test - target_data)
    MAE = np.mean(errors) 
    RMSE = np.sqrt(MSE)


    # 測 ratio 加速：關掉逐筆 print（test 上千筆會嚴重拖慢 I/O）
    # for pred, target in zip(y_pred_test, target_data):
    #     print(f"prediction: {pred:.4f} target: {target:.4f}")

    max_glucose_value = np.max(target_data)
    min_glucose_value = np.min(target_data)

    
    return MSE, MARD, max_glucose_value,min_glucose_value,model,y_pred_train, y_pred_test



def _symbolic_regression(X_train, y_train, X_test, y_test):

    # 2. 定義模型
    model = PySRRegressor(
    niterations=1000,
    # 直接把自定義函數寫進這裡
    unary_operators=[
        "exp", 
        "log_abs(x) = log(abs(x) + 1f-8)",
        "sq(x) = x^2"##,      # 定義 2 次方
        ##"cube(x) = x^3"    # 定義 3 次方
        ##"if_gt_zero(x) = x > 0 ? 1f0 : 0f0" 
    ],
    binary_operators=[
        "+", "-", "*", "/" ##,
        ##"is_greater(x, y) = x > y ? 1f0 : 0f0" # <--- 二元邏輯放這裡
    ],
    # 映射關係保持不變
    extra_sympy_mappings={
        ##"if_gt_zero": lambda x: sympy.Piecewise((1.0, x > 0), (0.0, True)),
        ##"is_greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True)),
        "log_abs": lambda x: sympy.log(sympy.Abs(x) + 1e-8),
        "sq": lambda x: x**2 ##,    # 補上這行
        ##"cube": lambda x: x**3,  # 補上這行
    },
    parsimony=0.001,
    maxsize=30,
    model_selection="best"
    )

    y_train_log = np.log(y_train + 1e-8)

    # 2. 訓練模型 (使用 log 處理後的 y)
    model.fit(X_train, y_train_log)

    # 3. 預測 (此時得到的是 log 空間的預測值)
    y_pred_log = model.predict(X_test)

    # 4. 還原預測值 (使用 exp 反轉 log)
    y_pred = np.exp(y_pred_log)

    '''
    # 3. 訓練
    model.fit(X_train, y_train)

    # 4. 預測測試集
    # model.predict 會自動使用最佳 (Best) 的公式
    y_pred = model.predict(X_test)

    # 5. 計算 MSE
    ##mse = mean_squared_error(y_test, y_pred)
    '''
    errors = np.abs(y_test - y_pred)
    MSE = np.mean(errors)
    MARD = np.mean(errors / y_test) * 100


    print("\n" + "="*40)
    print(f"測試集 MARD: {MARD:.6f}")
    print(f"最佳數學運算式: {model.get_best().equation}")
    print("="*40)
    
    for i in range(len(y_test)):
        print('target:',y_test[i],' y_pred:',y_pred[i])


    max_glucose_value = np.max(y_test)
    min_glucose_value = np.min(y_test)

    
    return MSE,MARD,max_glucose_value,min_glucose_value,model


def _svr_auto(X_all_scaled, y_all, test_data, target_data, num_cv):  ###srv training(RFECV/SVR對固定輸入是確定性的，不需要重複跑多次取最好)
    
    max_glucose_value = np.max(target_data)
    min_glucose_value = np.min(target_data)

    # 暫時只跑一次（不做 Pool）
    # 確定後再改回多進程：
    # num_processes=6
    # inputs = [(X_all_scaled, y_all, test_data, target_data, num_cv) for _ in range(num_processes)]
    # with Pool(processes=num_processes) as pool:
    #     results = pool.map(_train_one_model, inputs)
    # best_result = min(results, key=lambda x: x[1])  ##based on MARD
    print('[_svr_auto] skip Pool, run _train_one_model once')

    best_result = _train_one_model((X_all_scaled, y_all, test_data, target_data, num_cv))

    Min_MSE, Min_MARD, Min_rfecv, Min_svr, Min_selected_feature_indices, Min_y_pred_train,Min_y_pred_test = best_result

    return Min_MSE, Min_MARD, max_glucose_value, min_glucose_value, Min_rfecv, Min_svr, Min_selected_feature_indices, Min_y_pred_train, Min_y_pred_test


def _train_one_model(args):

    X_all_scaled, y_all, test_data, target_data, num_cv = args

    svr = SVR(kernel='linear', C=100.0, epsilon=0.1)

    rfecv = RFECV(estimator=svr, step=1, cv=num_cv, scoring='neg_mean_absolute_percentage_error')  ##cv=7
    rfecv.fit(X_all_scaled, y_all)
    
    X_train_selected = rfecv.transform(X_all_scaled)
    svr.fit(X_train_selected, y_all)
    selected_feature_indices = np.where(rfecv.support_)[0]
    
    X_test_selected = rfecv.transform(test_data)

    # train prediction
    y_pred_train = svr.predict(X_train_selected)

    # test prediction
    y_pred_test = svr.predict(X_test_selected)

    errors = np.abs(target_data - y_pred_test)
    MSE = np.mean((y_pred_test - target_data)**2)
    MARD = np.mean(errors / target_data) * 100
    Bias = np.mean(y_pred_test - target_data)
    MAE = np.mean(errors) 
    RMSE = np.sqrt(MSE)

   
    return (MSE, MARD, rfecv, svr, selected_feature_indices, y_pred_train,y_pred_test)



##def BuildModel(uuid,basepath,srj_db_path,glucosedata_path,server_db_path=""): ##測試用

def make_downsample_run_tag(cap_ref, ratio):
    """實驗目錄名，例如 ds_median_r0.30"""
    return f"ds_{cap_ref}_r{float(ratio):.2f}"


def copy_regression_features_uuid(basepath, src_rel, dst_rel, uuid):
    """
    把完整特徵 uuid 目錄從 src_rel copy 到 dst_rel（覆蓋既有 dst）。
    路徑形如：{basepath}/{rel}/Regression_Features/{uuid}
    """
    src = os.path.join(basepath, src_rel, "Regression_Features", uuid)
    dst = os.path.join(basepath, dst_rel, "Regression_Features", uuid)
    if not os.path.isdir(src):
        return False, f"完整特徵來源不存在：{src}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[copy features] {src} → {dst}")
    return True, dst


def BuildModel(
    uuid,
    basepath,
    srj_db_path,
    glucosedata_path,
    processnum=8,
    splitting_ratio="70_30",
    run_rel=None,
    features_full_rel=None,
    downsample_ratio=0.3,
    downsample_cap_ref="median",
    skip_feature_extract=False,
    skip_normalize=False,
    do_downsample=True,
    seed=42,
):

    errorcode="0"
    message="" 
    status=-1

    features_full_rel = features_full_rel or os.path.join(splitting_ratio, "full")
    if run_rel is None:
        if do_downsample:
            run_rel = os.path.join(
                splitting_ratio,
                make_downsample_run_tag(downsample_cap_ref, downsample_ratio),
            )
        else:
            run_rel = features_full_rel

    
    DataArrangement_Obj = DataArrangement() 
    ##errorcode, message = DataArrangement_Obj.data_processing(uuid,start_time,end_time,srj_db_path,glucosedata_path,basepath,server_db_path) ##測試用
    errorcode, message = DataArrangement_Obj.data_processing(
        uuid,
        srj_db_path=srj_db_path,
        glucosedata_path=glucosedata_path,
        basepath=basepath,
        processnum=processnum,
        splitting_ratio=splitting_ratio,
        run_rel=run_rel,
        features_full_rel=features_full_rel,
        downsample_ratio=downsample_ratio,
        downsample_cap_ref=downsample_cap_ref,
        skip_feature_extract=skip_feature_extract,
        skip_normalize=skip_normalize,
        do_downsample=do_downsample,
        seed=seed,
    )
    

    if(int(errorcode)<0):
        return status, errorcode, message     
    
    
    checkedpath=os.path.join(basepath,run_rel,"Regression_Features",uuid,"Train","Low")
    filelist_low_train=os.listdir(checkedpath) if os.path.isdir(checkedpath) else []

   
    checkedpath=os.path.join(basepath,run_rel,"Regression_Features",uuid,"Test","Low")
    filelist_low_test=os.listdir(checkedpath) if os.path.isdir(checkedpath) else []
    
    
    checkedpath=os.path.join(basepath,run_rel,"Regression_Features",uuid,"Train","High")
    filelist_high_train=os.listdir(checkedpath) if os.path.isdir(checkedpath) else []

   
    checkedpath=os.path.join(basepath,run_rel,"Regression_Features",uuid,"Test","High")
    filelist_high_test=os.listdir(checkedpath) if os.path.isdir(checkedpath) else []
    
    
    checkedpath=os.path.join(basepath,run_rel,"Regression_Features",uuid,"Train","Normal")
    filelist_normal_train=os.listdir(checkedpath) if os.path.isdir(checkedpath) else []

  
    checkedpath=os.path.join(basepath,run_rel,"Regression_Features",uuid,"Test","Normal")
    filelist_normal_test=os.listdir(checkedpath) if os.path.isdir(checkedpath) else []
    
    if(len(filelist_low_train)>0 and len(filelist_low_test)>0 and len(filelist_high_train)>0 and len(filelist_high_test)>0 and len(filelist_normal_train)>0 and len(filelist_normal_test)>0):  ###有中，低和高血糖資料
        model_type_array=['Normal','High','Low']
        print('Start training normal, high and low regression models!')
        status, errorcode, message=BuildRegressionModel(uuid,basepath,run_rel,model_type_array)
        
        if(int(errorcode)>=0):
            message="Model with three classes has been built!"      
    elif(len(filelist_high_train)>0 and len(filelist_high_test)>0 and len(filelist_normal_train)>0 and len(filelist_normal_test)>0):         
        model_type_array=['Normal','High']
        print('Start training normal and high regression models!')
        status, errorcode, message=BuildRegressionModel(uuid,basepath,run_rel,model_type_array)
        
        if(int(errorcode)>=0):
            message="Model with normal and high categories has been built!"

    elif(len(filelist_low_train)>0 and len(filelist_low_test)>0 and len(filelist_normal_train)>0 and len(filelist_normal_test)>0):         
        model_type_array=['Normal','Low']
        print('Start training normal and low regression models!')
        status, errorcode, message=BuildRegressionModel(uuid,basepath,run_rel,model_type_array)
        
        if(int(errorcode)>=0):
            message="Model with normal and low categories has been built!"        
    
    elif(len(filelist_normal_train)>0 and len(filelist_normal_test)>0):         
        model_type_array=['Normal']
        print('Start training normal regression model!')
        status, errorcode, message=BuildRegressionModel(uuid,basepath,run_rel,model_type_array)
        
        if(int(errorcode)>=0):
            message="Model with normal category has been built!"  

    elif(len(filelist_high_train)>0 and len(filelist_high_test)>0):         
        model_type_array=['High']
        print('Start training high regression model!')
        status, errorcode, message=BuildRegressionModel(uuid,basepath,run_rel,model_type_array)
        
        if(int(errorcode)>=0):
            message="Model with high category has been built!" 

    else:
        errorcode="-906"
        message="An error occurs in the BuildModel function of Regression_Model_Predictor.py: No any data can be used to train regression model!"
        status=-1  
    
    
    
    return status, errorcode, message   
 

class GlucoseValuesPredictor: ###血糖預測類別

    def __init__(self,basepath,uuid,splliting_ratio,model_type):
        
        self.uuid=uuid
        self.basepath = basepath   
               
        if(model_type==0 or model_type==-1): ##-1是因為可能沒有訓練出任何類別(可能就只有normal資料無法訓練)，直接用normal的Regression model去預測
            self.model_type ='Normal'
        elif(model_type==1):
            self.model_type ='High'
        else:
            self.model_type ='Low'
        
        self.splliting_ratio=splliting_ratio
        self.features_uuid_path=""
        self.test_feature_uuid_path=""
        self.train_uuid_path=""
        self.test_uuid_path=""
        self.set_directory() ##設定路徑

    
    def set_directory(self):  ##設定輸入之預測資料raw data和特徵資料放置位置
        
        print('Setting directionary...')
        
        base_path=self.basepath       
        features_filepath=os.path.join(base_path,self.splliting_ratio,'Regression_Features')
        if not os.path.exists(features_filepath): os.makedirs(features_filepath)
                                         
        self.features_uuid_path=os.path.join(features_filepath,self.uuid) ##創建Regresion_Features/uuid 資料夾
        if not os.path.exists(self.features_uuid_path): os.makedirs(self.features_uuid_path)
        

        features_uuid_testdata_path=os.path.join(self.features_uuid_path,'TestData') ##創建Regresion_Features/uuid/TestData 資料夾
        if not os.path.exists(features_uuid_testdata_path): 
            os.makedirs(features_uuid_testdata_path)
        else: ##本來就存在，先刪除內部資料
            print('delete files in the directory:'+features_uuid_testdata_path)
            shutil.rmtree(features_uuid_testdata_path)      

       
        self.test_feature_uuid_path=os.path.join(features_uuid_testdata_path,'Test_Feature') 
        if not os.path.exists(self.test_feature_uuid_path): os.makedirs(self.test_feature_uuid_path)
            

        test_feature_uuid_type_path=os.path.join(self.test_feature_uuid_path,self.model_type) 
        if not os.path.exists(test_feature_uuid_type_path): os.makedirs(test_feature_uuid_type_path)

        self.train_uuid_path=os.path.join(features_uuid_testdata_path,'Train') ##創建Regresion_Features/uuid/TestData/Train 資料夾
        if not os.path.exists(self.train_uuid_path): os.makedirs(self.train_uuid_path)
                
        
        path_train_high=os.path.join(self.train_uuid_path,'High') ##創建Regresion_Features/uuid/Train/High 資料夾
        if not os.path.exists(path_train_high): os.makedirs(path_train_high)

        path_train_low=os.path.join(self.train_uuid_path,'Low') ##創建Regresion_Features/uuid/Train/High 資料夾
        if not os.path.exists(path_train_low): os.makedirs(path_train_low)

        path_train_normal=os.path.join(self.train_uuid_path,'Normal') ##創建Regresion_Features/uuid/Train/High 資料夾
        if not os.path.exists(path_train_normal): os.makedirs(path_train_normal)       
              
            
        self.test_uuid_path=os.path.join(features_uuid_testdata_path,'Test') ##創建Regresion_Features/uuid/TestData/Test 資料夾
        if not os.path.exists(self.test_uuid_path): os.makedirs(self.test_uuid_path)

        
        path_test_high=os.path.join(self.test_uuid_path,'High') ##創建Regresion_Features/uuid/Test/High 資料夾
        if not os.path.exists(path_test_high): os.makedirs(path_test_high)

        path_test_low=os.path.join(self.test_uuid_path,'Low') ##創建Regresion_Features/uuid/Test/Low 資料夾
        if not os.path.exists(path_test_low): os.makedirs(path_test_low)
            
        path_test_normal=os.path.join(self.test_uuid_path,'Normal') ##創建Regresion_Features/uuid/Test/Normal 資料夾
        if not os.path.exists(path_test_normal): os.makedirs(path_test_normal)  
        
    
    def majority_filtering(self,glucose_list):
        
        Q1 = np.percentile(glucose_list, 25)
        Q3 = np.percentile(glucose_list, 75)
        IQR = Q3 - Q1

        # Bounds for filtering
        lower_bound = Q1 - 1* IQR
        upper_bound = Q3 + 1* IQR

        # Remove outliers
        b = glucose_list[(glucose_list >= lower_bound) & (glucose_list <= upper_bound)]

        # Compute median of filtered array
        median_without_outliers = np.median(b)
        mean_without_outliers = np.mean(b)

        return median_without_outliers,mean_without_outliers
    


    def glucosevalue_predict(self,ecgdata): 

        errorcode="0"
        message=""

        ten_minutes_pointnum = 10 * 60 * 250
        if (len(ecgdata) > ten_minutes_pointnum):  ##長度超過10分鐘，取最近10分鐘
            ecgdata = ecgdata[len(ecgdata) - ten_minutes_pointnum:len(ecgdata)]
            print('ecg data is trimmed to 10 minutes!')
                
        segment_num=int(len(ecgdata)/2500)
        bad_quality_count=0
        good_quality_count=0
        ecgdata=baseline_remove(ecgdata)  ##新增基線拉直
        
        if(segment_num>=1): ##至少1個10秒的資料長度
            for i in range(0,segment_num):
                current_ecg_data=ecgdata[i*2500:(i+1)*2500]  ##切成10秒一個
                current_ecg_data=baseline_remove(current_ecg_data)  ##新增基線拉直
                current_ecg_data = remove_spike(current_ecg_data, spike_threshold=500.0, fs=250) 
                rpeak_array=rpeak_detection(current_ecg_data)
                result=ecg_quality_check_v3(current_ecg_data, rpeak_array[1:])
                if(result == "Normal"): ###統計訊號品質良好
                    good_quality_count=good_quality_count+1
                    ##-----write ecg file to TestData/test------------------
                    test_type_path=os.path.join(self.test_uuid_path,self.model_type) ##Regresion_Features/uuid/TestData/Test/High or Low or Noraml資料夾
                    file_path=os.path.join(test_type_path,self.uuid+'_00000_'+'0_'+str(i)+'_100.txt') ###編碼最後數字血糖值給100，是虛設的，不重要
                    with open(file_path, 'w') as file:
                        for value in current_ecg_data:
                            file.write(f"{value}\n")

                else:
                    bad_quality_count=bad_quality_count+1

            if(good_quality_count>=1): ##超過1 segments的訊號品質是好的才進行辨識
                self.regression_feature_extraction_for_prediction()  ##擷取特徵
                errorcode,message,glucosevalue_list=self.predict()
                if(errorcode=="0"):
                    print('glucosevalue_list:',glucosevalue_list)
                    predicted_value=np.median(glucosevalue_list)  
                    ##predicted_value_mean=np.mean(glucosevalue_list)  
                    ##median_without_outliers,mean_without_outliers=self.majority_filtering(glucosevalue_list) ###做majority filtering，取中間值
                    ##print('predicted_value_median:',predicted_value,' predicted_value_mean:',predicted_value_mean,' median_without_outliers:',median_without_outliers,' mean_without_outliers:',mean_without_outliers)
                    errorcode="0"
                    message="Finish predicting the glucose value"
                else:
                    predicted_value=-1

            else:
                predicted_value=-1
                errorcode="-600"
                message="An error occurs in the predict function of the GlucoseValuesPredictor class: Too many segments of ECG signals are of poor quality!"
                  
        else:           
            predicted_value=-1
            errorcode="-601" 
            message="An error occurs in the predict function of the GlucoseValuesPredictor class: Data is too short, the lenth of input ecg data is shorter than 1 minute long!"
                                       
    
        return errorcode, message, predicted_value    
     
    def regression_feature_extraction_for_prediction(self):
        
        '''
        extract features from ecg signal in text files of raw data folder, and then save results in Regression_Features folder 
        '''

        errorcode="0"
        message=""       
           
       
        globals.initialize()
        code_version = get_version()
        current_time_str = strftime("%Y_%m_%d_%H%M", time.localtime())

        ## Feature extraction        
        features_parquet_filename = "Features_{}_{}.parquet".format(code_version, current_time_str)
        rawdata_path=os.path.join(self.test_uuid_path,self.model_type)
        df_features = features_extraction.load_rawdata_extract_features_multiprocess(rawdata_path, self.train_uuid_path, self.test_uuid_path,"", features_parquet_filename,True,self.test_feature_uuid_path)  ##講特徵寫到Dataset資料夾
        
       
        return errorcode, message
    
    def predict(self):     
              
        errorcode="0"
        message=""  

        predict_list=[]       
        
        ##------load model--------
        best_model_performance_path = os.path.join(self.basepath,self.splliting_ratio,"Best_ThreeClasses_Regression_Model",self.uuid,"Historic_Best_Performance_"+self.model_type+".txt")
        if not os.path.exists(best_model_performance_path): ##不在三類模型，讀二類模型
            best_model_performance_path = os.path.join(self.basepath,self.splliting_ratio,"Best_TwoClasses_Regression_Model",self.uuid,"Historic_Best_Performance_"+self.model_type+".txt")
            if not os.path.exists(best_model_performance_path): ##也不在二類模型，回傳錯誤
                errorcode="-908"
                message="No regression model can be used to predcit glucose values!"
                return errorcode, message, [-1]
            else:
                uuid_model_classes_path=os.path.join(self.basepath,self.splliting_ratio,"Best_TwoClasses_Regression_Model",self.uuid)    
        else:
            uuid_model_classes_path=os.path.join(self.basepath,self.splliting_ratio,"Best_ThreeClasses_Regression_Model",self.uuid)    


        rfecv_name = "None"
        model_name = "None"

        with open(best_model_performance_path, 'r') as file:
            for line in file:
                if line.startswith("Rfecv_Name:"):
                    rfecv_name = line.strip().split(":", 1)[1]
                elif line.startswith("Model_Name:"):
                    model_name = line.strip().split(":", 1)[1]

        print("Rfecv_Name:", rfecv_name)
        print("Model_Name:", model_name)
    
        model_path = os.path.join(uuid_model_classes_path,model_name)
        svr_model = joblib.load(model_path)
            
        ##-----load data----------
        current_test_path=os.path.join(self.test_feature_uuid_path)   
        used_feature_array = np.array([row[0] for row in used_feature_dic])        
        testdata = Regression_ECGDataset(dir_path=current_test_path, used_feature_array=used_feature_array, type=self.model_type,method='raw')   
        
        if(len(testdata)>0):
            test_loader = DataLoader(testdata)
            data_all = []
            for data, target in test_loader:
                data_all.append(data.squeeze().numpy())  
                ##target_all.append(target.numpy())  ###target再新進來要預測的資料不重要
        
            data_all = np.vstack(data_all)
            ##target_all = np.concatenate(target_all)                
            if(np.isnan(data_all).any()):  ##擁有NaNs的數值，不做預測
                    errorcode="-909"
                    message="No good signal can be used to predcit glucose values!"
                    return errorcode, message, predict_list
            
            ##col_means = np.nanmean(data_all, axis=0)  # Calculate column means ignoring NaNs           
            
            cols_mean_txtfile_path=os.path.join(uuid_model_classes_path,"cols_mean.txt")
            col_means=[]
            with open(cols_mean_txtfile_path, 'r') as file:
                for line in file:
                    col_means.append(float(line))
            
            print('col_means:',col_means)

            nan_indices = np.where(np.isnan(data_all))  # Find the NaN indices
            data_all[nan_indices] = np.take(col_means, nan_indices[1])  # Replace NaNs with column means

            scaler = StandardScaler()
            data_all_scaled = scaler.fit_transform(data_all)
            
            if(rfecv_name=="None"):
                print('Manual mode')
                X_test_selected=data_all_scaled
            else:
                print('Auto model...')
                
                rfecv_path= os.path.join(uuid_model_classes_path,rfecv_name)
                rfecv = joblib.load(rfecv_path)
                selected_features = np.where(rfecv.support_ == True)[0]  ## Get selected feature indices from RFECV
                X_test_selected = rfecv.transform(data_all_scaled)        
                ##X_test_selected = X_test_selected[:, selected_features]

            ##-------predict---------
            predict_list = svr_model.predict(X_test_selected)
        else:
            errorcode="-909"
            message="No good signal can be used to predcit glucose values!"

        return errorcode, message, predict_list
        

if __name__ == "__main__":
    

    user_information = [
                        # ['2197','20250212','20250225'],       ##T1003(績效良，但可再訓練)
                        # ['2200','20250212','20250226'],       ##T1005(績效優)  
                        ['2133','20250212','20250225'],       ##T1006(績效可，70%左右)  
                        # ['2131','20250212','20250225'],       ##T1007(績效良，但可再訓練)  
                        # ['2208','20250217','20250302'],       ##T1011(績效良，但可再訓練)         
                        ['2205','20250217','20250302'], ##5   ##T1014(績效良，但可再訓練，注意血糖值有到500的情況)
                        # ['2202','20250217','20250302'],       ##T1015(績效良，但可再訓練)                     
                        # ['2215','20250219','20250304'],       ##T1020(績效優)                                                                                         
                        # ['2223','20250226','20250311'],       ##T1026(績效良)                      
                        # ['2230','20250226','20250311'],       ##T1028(績效超級優)                                    
                        # ['2235','20250312','20250326'], ##10  ##T1032(績效普通)
                        # ['2246','20250312','20250320'],       ##T1034(績效良，可在訓練)
                        # ['2249','20250313','20250326'],       ##T1035(績效普通，需再訓練)                                                   
                        ['2253','20250320','20250402'],       ##T1037(績效普通，要再訓練)                     
                        # ['2261','20250401','20250414'],       ##T1038(績效尚可，還可再訓練)                   
                        # ['2262','20250408','20250421'], ##15  ##T1039(績效普通，train and test各有2筆血糖資料,已使用special move測試)       ##待測試                                    
                        # ['2257','20250408','20250421'],       ##T1040(績效普通，可再訓練)


                        ['2199','20250212','20250225'],         ##T1001(績效尚可,使用special move後有進步) 
                        # ['2198','20250212','20250225'],         ##T1004(績效極度不平衡)                                  ##待測試
                        # ['2196','20250212','20250226'], ##19    ##T1002(高血糖只有2筆,未能訓練)
                        # ['2206','20250217','20250228'],         ##T1008(test只有兩筆，績效尚可有點不平衡可再訓練或觀察)
                        # ['2201','20250217','20250302'],         ##T1009  ##可能只做到2/28(無高、低血糖資料)
                        # ['2204','20250217','20250228'],         ##T1010(test high 只有一筆，績效不好可再訓練或觀察，spcial move有進步)
                        # ['2203','20250217','20250302'],         ##T1012(train中的高血糖資料只有兩筆，績效普通，可再訓練或觀察)              ##可再次測試   
                        # ['2207','20250217','20250302'],         ##T1013(special move後績效依然不好)      
                        # ['2210','20250217','20250302'], ##25    ##T1016(test的高血糖只有1筆，績效不好，已經測試2次)              
                        # ['2209','20250219','20250304'],         ##T1017(test的高血糖只有2天4筆，test and train個只有2筆, 已經執行過4次訓練) 
                        ['2218','20250219','20250304'],         ##T1018(高血糖資料很多，但績效不好過度失衡，還沒使用specail move測試過)
                        # ['2219','20250219','20250304'],         ##T1021(只有中、低血糖，無高血糖資料，已訓練回歸)                                       ##要能練low and normal
                        # ['2217','20250219','20250304'],         ##T1022(只有中、低血糖，無高血糖資料，已訓練回歸)                                       ##要能練low and normal
                        ['2216','20250219','20250304'], ##30    ##T1023(績效依然不好，special move測試多次) 
                        # ['2214','20250219','20250304'],         ##T1024(高血糖資料train 只有2筆，test沒有資料，已使用specail move，low血糖很多)  ##要能練low and normal 
                        # ['2226','20250226','20250312'],         ##T1025(高血糖只有2筆且在同一天,未能處理) 
                        ['2224','20250226','20250311'],         ##T1027(高血糖資料很少只有3筆，績效不好)                                    ##待測試
                        # ['2227','20250226','20250312'],         ##T1029(高血糖資料很多，但績效差，不平衡，使用specail move測試) 
                        # ['2225','20250226','20250312'], ##35    ##T1030(高血糖只有1筆、低血糖資料只有4筆)                                  ##待測試
                        # ['2236','20250310','20250323'],         ##T1031(高血糖資料只有1筆)                                                 ##待測試
                        # ['2248','20250312','20250325'],         ##T1033(績效不好，需再訓練，更新後績效變差，要特別注意)                        ##待測試
                        # ['2251','20250319','20250401'],         ##T1036(績效極不平衡，需再訓練)                                              ##待測試
                        # ['2221','20250219','20250220'],         ##提早退出

                        
                        # ['2279','20250426','20250509'], ##40    ##T1041  
                        # ['2276','20250426','20250509'],         ##T1042
                        ['2278','20250426','20250509'],         ##T1043
                        # ['2281','20250502','20250515'],         ##T1044
                        ['2286','20250509','20250522'],         ##T1045
                        # ['2285','20250509','20250522'], ##45    ##T1046
                        ['2294','20250520','20250531'],         ##T1047
                        # ['2293','20250520','20250602'],         ##T1048
                        # ['2295','20250522','20250604'],         ##T1049
                        # ['2291','20250527','20250610'],         ##T1050
                        ['2298','20250604','20250617'],  ##50   ##T1051(19歲，小於20歲)
                        # ['2300','20250604','20250617'],         ##T1052
                        # ['2299','20250604','20250617'],         ##T1053  ##訓練到一半出問題
                        # ['2304','20250606','20250619'],         ##T1054
                        # ['2305','20250606','20250619'],         ##T1055
                        # ['2306','20250606','20250619'],  ##55   ##T1056
                        # ['2130','20250623','20250706'],         ##T1057  ##test沒有高血糖
                        # ['2132','20250623','20250706'],         ##T1058
                        # ['2322','20250623','20250706'],  ##58   ##T1059
                        # ['2329','20250701','20250713'],         ##T1060
                        ['2352','20250731','20250813'],         ##T1061 
                        ['2378','20251022','20251104'],         ##T1062 
                        # ['2397','20251129','20251213'],         ##T1063 


                        # ['91', '20240731', '20250319'],
                        # ['91', '20240812', '20250319'],
                        # ['91', '20241108', '20250719'],
                        # ['2283','20260214','20260227'],
                        # ['2423','20260214','20260227'],
                        # ['798','20260227','20260312'],
                        # ['746','20250619','20250626']##David資料測試 
                        ]         
                                       
    

    ##basepath='D:\\Dennis Project\\GlucoseMdoel_Regression\\Model'  ###血糖數值對應之ECG資料，分析得到特徵檔案會存放在此路徑
    ##glucosedata_path='D:\\Dennis Project\\GlucoseMdoel_Regression\\GlucoseDataCSV' ###APP收到的血糖數值，整理成CSV檔後存放路徑
    ##server_db_path='G:\\.shortcut-targets-by-id\\1Mc_sTYrGzDau1JPki2AQDpV4FeX5tKu3\\SWM_DataCenter\\Health_Server'   ###ECG資料所在雲端路徑(Dennis測試用)

    mother_path=Path(__file__).resolve().parent   ###血糖數值對應之ECG資料，分析得到特徵檔案會存放在此路徑
    base_path=mother_path/"Model_exp_1"
    base_path=str(base_path)
    ##glucosedata_path=r'D:\Dennis Project\Glucose_CSV\BGM_CSV\filtered_BGM'
    glucosedata_path=r"D:\stella\server_program_2.1.1\CGM_CSV"
    glucosedata_path=str(glucosedata_path)
    server_db_path='G:\\.shortcut-targets-by-id\\1Mc_sTYrGzDau1JPki2AQDpV4FeX5tKu3\\SWM_DataCenter\\Health_Server_Script\\_rawdata_download'

    parser = argparse.ArgumentParser(
        description=(
            "個人化回歸訓練（方案A）：完整特徵抽一次存 70_30/full，"
            "各 downsample ratio 複製到 70_30/ds_{cap}_r{xx} 再砍樣、訓練。"
        )
    )
    parser.add_argument("--data-split", default="70_30", help="GlucoseData 所在切分目錄，預設 70_30")
    parser.add_argument("--downsample-ratio", type=float, default=0.3,
                        help="每血糖值上限比例 r；median: ceil(median*r)，max: floor(max*r)")
    parser.add_argument("--cap-ref", choices=["median", "max"], default="median",
                        help="downsample 上限參考：median 或 max")
    parser.add_argument("--skip-feature-extract", action="store_true",
                        help="略過抽特徵，直接用既有 70_30/full/Regression_Features（需已建立）")
    parser.add_argument("--skip-normalize", action="store_true",
                        help="略過 RawData→NormalizedData（通常搭配 --skip-feature-extract）")
    parser.add_argument("--no-downsample", action="store_true",
                        help="不做 downsample，直接在 70_30/full 上訓練")
    parser.add_argument("--seed", type=int, default=42, help="downsample 隨機種子")
    parser.add_argument("--processnum", type=int, default=16)
    parser.add_argument("--skip-re-add", action="store_true", help="略過開頭的 GlucoseData re_add")
    parser.add_argument(
        "--legacy-balanced",
        action="store_true",
        help=(
            "原始 data_balanced 路徑：直接用 Model/{data-split}/Regression_Features "
            "（含既有 Remove_Data、不 re-add、不抽特徵、不 downsample）。"
            "等同自動開啟 --no-downsample --skip-feature-extract --skip-normalize --skip-re-add"
        ),
    )
    args = parser.parse_args()

    data_split = args.data_split

    if args.legacy_balanced:
        # 原始方式：70_30/Regression_Features（已 IQR/data_balanced，Remove 留在 Remove_Data）
        args.no_downsample = True
        args.skip_feature_extract = True
        args.skip_normalize = True
        args.skip_re_add = True
        features_full_rel = data_split  # → Model/70_30/Regression_Features/{uuid}
        run_rel = data_split
        do_downsample = False
    else:
        features_full_rel = os.path.join(data_split, "full")
        do_downsample = not args.no_downsample
        if do_downsample:
            run_tag = make_downsample_run_tag(args.cap_ref, args.downsample_ratio)
            run_rel = os.path.join(data_split, run_tag)
        else:
            run_rel = features_full_rel

    print(
        f"[run config] data_split={data_split}, full={features_full_rel}, run={run_rel}, "
        f"downsample={do_downsample}, ratio={args.downsample_ratio}, cap_ref={args.cap_ref}, "
        f"skip_feature_extract={args.skip_feature_extract}, skip_normalize={args.skip_normalize}, "
        f"legacy_balanced={args.legacy_balanced}, skip_re_add={args.skip_re_add}"
    )

    def re_add(uuid_list, re_add_base):
        for uuid in uuid_list:
            removed_data_path = os.path.join(re_add_base, uuid, "Remove_Data", "Train")
            if not os.path.exists(removed_data_path):
                print(f"[re_add] uuid={uuid} 沒有 Remove_Data/Train，略過")
                continue

            removed_data_files = os.listdir(removed_data_path)
            added_count = 0
            skipped_count = 0
            for file in removed_data_files:
                try:
                    bg_value = int(file.split("_")[-1].split(".")[0])
                except ValueError:
                    print(f"[re_add] 檔名格式異常，跳過: {file}")
                    continue

                if bg_value < 85:
                    bg_level = "Low"
                elif bg_value <= 170:
                    bg_level = "Normal"
                else:
                    bg_level = "High"

                source_file = os.path.join(removed_data_path, file)
                target_dir = os.path.join(re_add_base, uuid, "Train", bg_level)
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, file)

                if os.path.exists(target_file):
                    skipped_count += 1
                    continue

                shutil.copy2(source_file, target_file)
                added_count += 1

            print(
                f"[re_add] uuid={uuid}: 從 Remove_Data/Train 加回 {added_count} 筆、跳過已存在 {skipped_count} 筆，"
                f"共 {len(removed_data_files)} 筆到 Train/{{Low,Normal,High}}"
            )

    if not args.skip_re_add:
        re_add_uuid_list = ['2133', '2205', '2253', '2199','2218','2216','2224','2278','2286','2294','2298','2352','2378']
        re_add(re_add_uuid_list, os.path.join(base_path, data_split, "GlucoseData"))
    else:
        print("Skip re_add (--skip-re-add)")

    os.makedirs(os.path.join(base_path, run_rel), exist_ok=True)
    performance_csv_path = os.path.join(base_path, run_rel, "performance_table.csv")

    for i, user_info in enumerate(user_information):
        uuid = user_info[0]
        start_time = user_info[1]
        end_time = user_info[2]
        print('index:', i, ' uuid:', uuid)
        srj_db_path = r'D:\DataDB' + os.sep + uuid

        status, errorcode, message = BuildModel(
            uuid,
            base_path,
            srj_db_path,
            glucosedata_path,
            processnum=args.processnum,
            splitting_ratio=data_split,
            run_rel=run_rel,
            features_full_rel=features_full_rel,
            downsample_ratio=args.downsample_ratio,
            downsample_cap_ref=args.cap_ref,
            skip_feature_extract=args.skip_feature_extract,
            skip_normalize=args.skip_normalize,
            do_downsample=do_downsample,
            seed=args.seed,
        )
        print('status:', str(status), ' error code:', errorcode, ' message:', message)
        print(performance_table)

        with open(performance_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["uuid", "MSE", "MARD", "Bias", "MAE", "RMSE", "largest_value", "lowest_value"])
            writer.writerows(performance_table)
        print(f"performance_table 已存到: {performance_csv_path}")
