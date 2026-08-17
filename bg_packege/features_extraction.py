# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:53:10 2023

author: SWM-Jared
collaborator: SWM-Benjamin, SWM-Wayne

Revise Record:
    # 20230908 版本是因應修改資料儲存格式,以 UUID為母資料夾, 實驗編號為子資料夾
    # 20230908 特徵擷取方法修改
    # 20231012 加入峰值偵測穩定度檢查
    # 20231027 新增/修改特徵
    # 20231110 放寬 峰值穩定度 (剔除絕對值誤差過大之離群值) 避免一兩個peak打壞進而丟掉整段訊號
               新增 peak to peak slope 特徵
    # 20231213 限縮峰值穩定度 少量打壞的點影響整段訊號
    # 20231221 修改sharpness計算方法, 修正沒找到peak時的數值會跟r_value一樣
    # 20240408 新增 HRV 特徵(暫緩)
    # 20240416 修改每個 ECG 片段第一拍 RRI 計算方式
    # 20240527 刪除重複性質特徵
    # 20240906 修改每個ECG片段邊界打點錯誤的問題 用median補植
    # 20241106: 改成Training也不做Quality check (已改回)
            parquet檔2024_11_06是Training set沒有做Quality Check
    # 20241107: Training Test都做Quality Check
        find_latest_file 可以設定閾值 取2024_11_01前最後一個檔
    # 20241112: 修改計算特徵穩定性 提高穩定度避免容易被Quality Check刪除整個片段

Note:
    # 使用過多核心(num_cores)執行 可能導致記憶體不足
"""

import json


import os
import psutil
import sys
import time
import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from scipy import signal
from multiprocessing import Process, Pool, Lock, cpu_count

from bg_packege import data_management
from bg_packege import waves_detection
from SWMlib.srj import dl_data
import globals

wave_info = waves_detection.WaveDetector()


def detect_peak_stability(loc_list):
    """
    :param loc_list: ECG 峰值位置清單 ex:p波位置
    :return: 位置標準差(穩定度),  新的位置差 (用median補nan值 list元素為int)
    """

    new_peaks_list = np.array(loc_list)
    peak_num = len(new_peaks_list.nonzero()[0])  # 不含空值"0"

    if peak_num > 3:  # 數量太少不計算std
        idx_arr = np.array(new_peaks_list, dtype=float).copy()

        # Original from BJM
        # idx_arr[idx_arr == 0] = np.nan
        # std_page = np.NaN if np.isnan(idx_arr).all() else np.nanstd(idx_arr)

        # 用median補nan值 (Wayne)(20240905)
        idx_arr[idx_arr == 0] = np.nanmedian(idx_arr)
        std_page = np.nanstd(idx_arr)
        new_peaks_list = idx_arr.astype(int)

        return round(std_page, 2), new_peaks_list

    return np.NaN, loc_list  # 峰值數量過少


def detect_peak_stability_del_outlier(idx_arr):
    """
    Deleted one standard deviation (Wayne)(20241111)
    Parameters
    ----------
    idx_arr

    Returns
    -------
    位置標準差(穩定度)
    新的位置差 (用median補nan值 list元素為int)
    """

    std = np.nanstd(idx_arr)
    mean = np.nanmean(idx_arr)
    new_peaks_list = np.where((idx_arr >= mean-std) & (idx_arr <= mean+std), idx_arr, 0).astype(int)
    std_page = new_peaks_list[new_peaks_list!=0].std()

    return round(std_page, 2), new_peaks_list


def detect_peak_stability_original(loc_list):
    """
    :param loc_list: ECG 峰值位置清單
    :return: 位置標準差(穩定度)
    """

    new_peaks_list = np.array(loc_list)

    peak_num = len(new_peaks_list.nonzero()[0])  # 不含空值"0"

    if peak_num > 3:  # 數量太少不計算std
        idx_arr = np.array(new_peaks_list, dtype=float).copy()

        idx_arr[idx_arr == 0] = np.nan  # Original from BJM
        std_page = np.NaN if np.isnan(idx_arr).all() else np.nanstd(idx_arr)  # Original from BJM
        return round(std_page, 2), new_peaks_list  # Original from BJM

        # idx_arr[idx_arr == 0] = np.nanmedian(idx_arr)  # Wayne 20240905 adjusted
        # std_page = np.nanstd(idx_arr)  # Wayne 20240905 adjusted
        # new_peaks_list = idx_arr.astype(int)  # Wayne
        # return round(std_page, 2), new_peaks_list  # Wayne 20240910 adjusted

    return np.NaN, loc_list  # 峰值數量過少


def check_irregular(rris):
    if len(rris) < 2:
        return 0
    for i, rri in enumerate(rris[:-1]):
        if (rri-rris[i+1])/rri > 0.2 or (rri-rris[i+1])/rri < -0.2:
            return 0  # Irregular
    return 1  # Regular


def pattern_clustering_pscore(ecg, ridxs, th=0.75):
    if 40 > len(ridxs) > 4:

        ecgs_list = []
        V = []
        ecg = np.array(ecg, dtype='int32')
        ridxs = np.array(ridxs, dtype='int32')

        RRI = np.diff(ridxs)
        RRI = [rri for rri in RRI if 2000 >= 1000*rri/250 >= 250]
        if len(RRI) < 1:
            return 0
        rri_q1 = int(np.percentile(RRI, 25))

        before_r = int(0.334*0.8*rri_q1)
        after_r = int(0.667*0.8*rri_q1)

        for rpeak in ridxs:
            n_rpeak_before = int(rpeak-before_r)
            n_rpeak_after = int(rpeak+after_r)
            if n_rpeak_before >= 0 and n_rpeak_after < len(ecg):
                ecgs_list.append(list(map(float, ecg[n_rpeak_before:n_rpeak_after])))

        if len(ecgs_list) <= 4:
            return 0

        ecgs_coeff = np.corrcoef(ecgs_list)

        for ecg_coeff in ecgs_coeff:
            ecgs_th = []

            for i_c, coeff in enumerate(ecg_coeff):
                if coeff > th:
                    ecgs_th.append(i_c)

            V.append(ecgs_th)

        V = sorted(V, key=len, reverse=True)

      
        Relation = []

        for i in range(len(V)):
            if len(Relation) == 0:
                Relation.append(V[i])

            else:
                for j in range(len(V)):
                    inter_num = len(set(V[i]) & set(V[j]))

                    if inter_num > 0:

                        if len(Relation) == j:
                            Relation.append(V[i])
                        else:
                            Relation[j] = list(set(V[i]) | set(Relation[j]))
                        break
                    elif inter_num == 0 and j == len(Relation):
                        Relation.append(V[i])
                    else:
                        continue

        Relation = sorted([list(t) for t in set(tuple(element) for element in Relation)], key=len, reverse=True)

        Percent = np.full((len(ecgs_list),), 0, dtype=np.float64)
        s = 0
        for index in range(len(Relation)):
            Percent[index] = (len(Relation[index]) / len(ecgs_list))
            if (Percent[index] >= 0.2):
                s = s + Percent[index]

        ##s = len(Relation[0])/len(ecgs_list)*100

        return s*100

    else:

        return 0


def cal_area_ratio(ecg, ridxs):
    '''
    Response the score based on areas of detected beats.

    Calculate areas of beats by given R peaks and uncovered areas.
    Check the ratio between uncovered areas and covered areas of beats.
    The score is higher if all areas are fully covered.

    Parameters:
      ecg: list
          A list of input ecg signal.
      ridxs: list
          A list of indices of R peaks in ecg.
    Returns:
      out: float
          The score of how detected beats cover areas in signal.
    '''

    # Calculate the median of RRI (unit:samples)
    RRIs = np.diff(ridxs)
    RRIs = [rri for rri in RRIs if 2000 >= 1000*rri/250 >= 250]  # filter with ms
    if len(RRIs) < 1:
        return 0

    medRRI = np.median(RRIs)
    forward = int(0.4 * medRRI)
    backward = int(0.6 * medRRI)

    prevEndIdx = 0  # prevent from overlapping
    beatAreas = []
    missedAreas = []

    for i in range(0, len(ridxs)):
        # Set start point
        startIdx = int(ridxs[i] - forward)
        # Left Boundary
        if startIdx < 0:
            startIdx = int(0)
        # Prevent from overlapping
        if startIdx < prevEndIdx:
            startIdx = prevEndIdx

        # Set end point
        endIdx = int(ridxs[i] + backward)
        # Right boundary
        if endIdx >= len(ecg):
            endIdx = int(len(ecg))

        # Calculate Areas
        beat = np.absolute(ecg[startIdx:endIdx + 1])
        beatArea = np.trapz(beat, dx=1/250)
        beatAreas.append(beatArea)
        if startIdx > prevEndIdx > 0:
            miss = np.absolute(ecg[prevEndIdx:startIdx + 1])
            missArea = np.trapz(miss, dx=1/250)
            missedAreas.append(missArea)

        # Update last end point
        prevEndIdx = endIdx

    totalArea = np.trapz(np.absolute(ecg), dx=1/250)
    ideaArea = sum(beatAreas) + sum(missedAreas)
    R1 = ideaArea / totalArea
    ratio2 = max(missedAreas) / np.median(beatAreas) if len(missedAreas) > 0 and len(beatAreas) > 0 else 0
    R2 = 1 - ratio2 if ratio2 < 1 else 0

    return float(R1*R2)


def compute_wave_(r, a, b, rri, ecgs, mode, key_name, input):
    """  Feature: Duration and Amplitude  """
    if mode == "R":
        duration = abs(a) * 4 if a != 0 else np.nan
        amplitude = ecgs[r] - ecgs[r + a] if a != 0 else np.nan
    else:
        duration = abs(a - b) * 4 if a != 0 and b != 0 else np.nan
        amplitude = abs(ecgs[r + a] - ecgs[r + b]) if a != 0 and b != 0 else np.nan

    """  Feature: Distance  """
    distance = (duration*duration + amplitude*amplitude) ** 0.5 if ~np.isnan(duration) and ~np.isnan(
        amplitude) else np.nan

    """  Feature: Direction  """
    direction = np.arctan2(amplitude, duration) * 180 / np.pi if ~np.isnan(duration) and ~np.isnan(
        amplitude) else np.nan
    direction = float(direction)

    """ Feature: peak to peak slope """  # 2023.11.15 新增
    slope = amplitude / duration if ~np.isnan(duration) and ~np.isnan(
        amplitude) else np.nan

    """ Feature: Heart Rate correction version 1 : 來源待證實 (不使用)"""
    # correction1 = distance / (rri**0.33) if rri < 1000 else distance / (rri**0.5) if ~np.isnan(distance) else np.nan
    # correction1 = float(correction1)

    """ Feature: Heart Rate correction version 2 : Bazett and Fridericia formula (不使用) """
    # correction2 = duration / ((rri/1000)**0.33333) if (rri > 1000) or (rri < 600) else duration / ((rri/1000)**0.5) if ~np.isnan(distance) else np.nan
    # correction2 = float(correction2)

    """ Feature: Heart Rate correction version 3 : Rautaharju formula. Edited at 2023.09.19"""
    correction3 = duration * (120+(60000/rri)) / 180
    correction3 = float(correction3)

    """ Feature: Heart Rate correction version 4 : 
    以HR80為基準,區間等比例縮放(D發想>>效果沒有比較好,且沒有理論支持). Edited at 2023.09.25 (不使用)"""
    # correction4 = duration / rri * (60000/80)
    # correction4 = float(correction4)

    # assert data type
    assert type(direction) is float, f'JSON dtype could not be {type(direction)}'
    assert type(correction3) is float, f'JSON dtype could not be {type(correction3)}'

    # save
    input[f"{key_name}_duration"].append(duration)
    input[f"{key_name}_amplitude"].append(amplitude)
    input[f"{key_name}_distances"].append(distance)
    input[f"{key_name}_directions"].append(direction)
    input[f"{key_name}_slope"].append(slope)  # 2023.11.15 新增
    input[f"{key_name}_corrections3"].append(correction3)
    return input


def compute_curve_(r, a, ecgs, wave_range, mode, key_name, input):
   
    if mode == "R":
        wave_ecgs = ecgs[r-wave_range: r+wave_range+1]

    elif a != 0:
        wave_ecgs = ecgs[r+a-wave_range: r+a+wave_range+1]

    else:  # 找不到峰值(a=0)
        input[f"{key_name}_left_slope"].append(None)
        input[f"{key_name}_right_slope"].append(None)
        input[f"{key_name}_left_sharp"].append(None)
        input[f"{key_name}_right_sharp"].append(None)
        input[f"{key_name}_tilt"].append(None)
        return input

    wave_peak = wave_ecgs[wave_range]
    wave_left_ecgs = wave_ecgs[:wave_range + 1]
    wave_right_ecgs = wave_ecgs[wave_range:]

    wave_left_mean_ecgs = np.nanmean(wave_left_ecgs)
    wave_right_mean_ecgs = np.nanmean(wave_right_ecgs)

    """  Feature: Slope  """
    # # slope  舊版本
    # left_slope = float(wave_peak - wave_left_mean_ecgs)
    # right_slope = float(wave_peak - wave_right_mean_ecgs)

    # slope  # 2023.10.26 修改
    #left_slope = float((wave_peak - wave_left_mean_ecgs) / (wave_range / 2))
    #right_slope = float((wave_right_mean_ecgs - wave_peak) / (wave_range / 2))

    left_slope =float((wave_peak - wave_ecgs[0]) / wave_range)    # Jane 修改

    right_slope =float((wave_ecgs[-1] - wave_peak) / wave_range)

    """ 測試用 畫圖 
    plt.figure()
    plt.plot(wave_ecgs)
    plt.plot([0, wave_range], [wave_ecgs[wave_range] - left_slope * wave_range, wave_ecgs[wave_range]])
    plt.plot([wave_range, 2 * wave_range], [wave_ecgs[wave_range], wave_ecgs[wave_range] + right_slope * wave_range])
    ---測試用 畫圖 """

    """  Feature: Peak Sharpness  2023.10.26新增  """
    # Sharpness
    left_sharp = 0
    right_sharp = 0

    for idx in range(1, wave_range + 1):
        #left_sharp += abs(wave_ecgs[wave_range - idx]) - abs(wave_ecgs[wave_range] - left_slope * idx)
        #right_sharp += abs(wave_ecgs[wave_range + idx]) - abs(wave_ecgs[wave_range] + right_slope * idx)
        left_line = wave_ecgs[wave_range] - left_slope * idx
        right_line = wave_ecgs[wave_range] + right_slope * idx

        left_sharp += abs(wave_ecgs[wave_range - idx] - left_line)
        right_sharp += abs(wave_ecgs[wave_range + idx] - right_line)

    left_sharp = left_sharp / wave_range
    right_sharp = right_sharp / wave_range

    """  Feature: Curve  """  """ 2024.05.27刪除 (與slope特徵相似)"""
    # # curve
    # left_curve = 0
    # for i, wave_left_ecg in enumerate(np.flip(wave_left_ecgs)):
    #     left_curve += wave_peak - 4 * (i + 1) - wave_left_ecg
    # left_curve = float(left_curve)
    #
    # right_curve = 0
    # for i, wave_right_ecg in enumerate(wave_right_ecgs):
    #     right_curve += wave_peak - 4 * (i + 1) - wave_right_ecg
    # right_curve = float(right_curve)

    """  Feature: Tilt  """
    # tilt
    try:
        wave_tilt = abs(left_slope / right_slope)
    except ZeroDivisionError:
        wave_tilt = None
        # wave_tilt = float(np.nan)

    assert type(left_slope) is float, f'JSON dtype could not be {type(left_slope)}'
    assert type(right_slope) is float, f'JSON dtype could not be {type(right_slope)}'

    # save
    input[f"{key_name}_left_slope"].append(left_slope)
    input[f"{key_name}_right_slope"].append(right_slope)
    input[f"{key_name}_left_sharp"].append(left_sharp)
    input[f"{key_name}_right_sharp"].append(right_sharp)
    input[f"{key_name}_tilt"].append(wave_tilt)
    return input


def plot_wave_detection(ecgs, pidxs, qidxs, ridxs, sidxs, tidxs,  stable_p, stable_q, stable_s, stable_t):

    ridxs = np.array(ridxs) + 250

    plt.figure(figsize=(6, 3))
    plt.plot(ecgs, c='black')
    plt.scatter(pidxs + ridxs, np.array(ecgs)[pidxs + ridxs], label='p')
    plt.scatter(qidxs + ridxs, np.array(ecgs)[qidxs + ridxs], label='q')
    plt.scatter(sidxs + ridxs, np.array(ecgs)[sidxs + ridxs], label='s')
    plt.scatter(tidxs + ridxs, np.array(ecgs)[tidxs + ridxs], label='t')
    plt.scatter(ridxs, np.array(ecgs)[ridxs], label='r')
    plt.legend()
    plt.title("p score={}, q score={}, s score={}, t_score={}".format(stable_p, stable_q, stable_s, stable_t))
    plt.show()


def plot_wave_detection_compared(ecgs, pidxs, qidxs, ridxs, sidxs, tidxs):
    ridxs = np.array(ridxs) + 250

    # Plot and check
    stable_p_ori, re_pidxs_ori = detect_peak_stability_original(pidxs)
    stable_q_ori, re_qidxs_ori = detect_peak_stability_original(qidxs)
    stable_s_ori, re_sidxs_ori = detect_peak_stability_original(sidxs)
    stable_t_ori, re_tidxs_ori = detect_peak_stability_original(tidxs)

    stable_p, re_pidxs = detect_peak_stability(pidxs)
    stable_q, re_qidxs = detect_peak_stability(qidxs)
    stable_s, re_sidxs = detect_peak_stability(sidxs)
    stable_t, re_tidxs = detect_peak_stability(tidxs)

    plt.figure(figsize=(12, 4))
    # plt.figure(figsize=(6, 6))

    plt.subplot(2, 1, 1)
    plt.plot(ecgs, c='black')
    plt.scatter(re_pidxs_ori + ridxs, np.array(ecgs)[re_pidxs_ori + ridxs], label='p')
    plt.scatter(re_qidxs_ori + ridxs, np.array(ecgs)[re_qidxs_ori + ridxs], label='q')
    plt.scatter(ridxs, np.array(ecgs)[ridxs], label='r')
    plt.scatter(re_sidxs_ori + ridxs, np.array(ecgs)[re_sidxs_ori + ridxs], label='s')
    plt.scatter(re_tidxs_ori + ridxs, np.array(ecgs)[re_tidxs_ori + ridxs], label='t')
    plt.legend()
    plt.title(
        "p score={}, q score={}, s score={}, t_score={}".format(stable_p_ori, stable_q_ori, stable_s_ori, stable_t_ori)
    )

    plt.subplot(2, 1, 2)
    plt.plot(ecgs, c='black')
    plt.scatter(re_pidxs + ridxs, np.array(ecgs)[re_pidxs + ridxs], label='p')
    plt.scatter(re_qidxs + ridxs, np.array(ecgs)[re_qidxs + ridxs], label='q')
    plt.scatter(ridxs, np.array(ecgs)[ridxs], label='r')
    plt.scatter(re_sidxs + ridxs, np.array(ecgs)[re_sidxs + ridxs], label='s')
    plt.scatter(re_tidxs + ridxs, np.array(ecgs)[re_tidxs + ridxs], label='t')
    plt.legend()
    plt.title(
        "p score={}, q score={}, s score={}, t_score={}".format(stable_p, stable_q, stable_s, stable_t)
    )

    plt.tight_layout()
    plt.show()


    # ----------特徵擷取-------------
    # Wayne adjusted


# main function
def extract_features(dict_wave_info): ##, quality_count):
    
    """
    input:
          dict_wave_info: 10秒ECG片段各項資訊
          範例 dict_wave_info = {'uuid': '67', 'type': 0, 'p': [[0, -47, 0, 0, -41, -45,...]], 'q': [[-8, -7, -7, -7, -8, -7,...]], 
           r': [[52, 192, 333, 474, 614, 756, ...]], 's': [[7, 7, 7, 7, 8, 7, ...]], 't': [[61, 63, 63, 63, 63, 61, 62, ...]],
           'ecg': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,...]],'BG':'164'}
    
    output:
          1. uuid_data:是這10秒ECG片段擷取出的各特徵值
          2. generate_features_flag: 此片段是否符合各種品質條件分析，若是，uuid_data才會有特徵值
    """


    p_quality_threshold = 6
    q_quality_threshold = 1.5
    s_quality_threshold = 2
    t_quality_threshold = 5

    uuid_data = {
        "type": [], "rr_interval": [], "hr": [], "a_score": [], "p_score": [], "r_score": [],
        "p_stability": [], "q_stability": [], "s_stability": [], "t_stability": [],
        "p_value": [], "q_value": [], "r_value": [], "s_value": [], "t_value": [],
        "pr_duration": [], "pr_amplitude": [], "pr_distances": [], "pr_directions": [],
        "pr_slope": [], "pr_corrections3": [],
        "qr_duration": [], "qr_amplitude": [], "qr_distances": [], "qr_directions": [],
        "qr_slope": [], "qr_corrections3": [],
        "rs_duration": [], "rs_amplitude": [], "rs_distances": [], "rs_directions": [],
        "rs_slope": [], "rs_corrections3": [],
        "rt_duration": [], "rt_amplitude": [], "rt_distances": [], "rt_directions": [],
        "rt_slope": [], "rt_corrections3": [],
        "pq_duration": [], "pq_amplitude": [], "pq_distances": [], "pq_directions": [],
        "pq_slope": [], "pq_corrections3": [],
        "ps_duration": [], "ps_amplitude": [], "ps_distances": [], "ps_directions": [],
        "ps_slope": [], "ps_corrections3": [],
        "pt_duration": [], "pt_amplitude": [], "pt_distances": [], "pt_directions": [],
        "pt_slope": [], "pt_corrections3": [],
        "qs_duration": [], "qs_amplitude": [], "qs_distances": [], "qs_directions": [],
        "qs_slope": [], "qs_corrections3": [],
        "qt_duration": [], "qt_amplitude": [], "qt_distances": [], "qt_directions": [],
        "qt_slope": [], "qt_corrections3": [],
        "st_duration": [], "st_amplitude": [], "st_distances": [], "st_directions": [],
        "st_slope": [], "st_corrections3": [],
        "p_left_slope": [], "p_right_slope": [], "p_left_sharp": [], "p_right_sharp": [], "p_tilt": [],
        "r_left_slope": [], "r_right_slope": [], "r_left_sharp": [], "r_right_sharp": [], "r_tilt": [],
        "t_left_slope": [], "t_right_slope": [], "t_left_sharp": [], "t_right_sharp": [], "t_tilt": [],
        "qrs_area":[], "st_area":[], "t_wave_cog":[] 

    }

    ecgs = dict_wave_info["ecg"][0]
    pidxs = dict_wave_info["p"][0]
    qidxs = dict_wave_info["q"][0]
    ridxs = dict_wave_info["r"][0]
    sidxs = dict_wave_info["s"][0]
    tidxs = dict_wave_info["t"][0]
    
   
    if(len(pidxs)>0):
        pidxs[pidxs == np.nan] = 0
    
    if(len(qidxs)>0):
        qidxs[qidxs == np.nan] = 0
    
    if(len(sidxs)>0):
        sidxs[sidxs == np.nan] = 0
    
    if(len(tidxs)>0):
        tidxs[tidxs == np.nan] = 0

    # """Add for test"""
    # stable_p, re_pidxs = detect_peak_stability_original(pidxs)  # 調整後的p_idxs (含median補點)
    # stable_q, re_qidxs = detect_peak_stability_original(qidxs)
    # stable_s, re_sidxs = detect_peak_stability_original(sidxs)
    # stable_t, re_tidxs = detect_peak_stability_original(tidxs)
    # generate_features = (len(ridxs) > 5) and (
    #             stable_p < 5 and stable_q < 1.5 and stable_s < 2 and stable_t < 5)  # from BJM
    # if not generate_features:
    #     plot_wave_detection_compared(ecgs, pidxs, qidxs, ridxs, sidxs, tidxs)

    """  計算PQRST的穩定度  """
    # 峰值偵測穩定度, 空值用median補點
    stable_p, pidxs_impulse = detect_peak_stability(pidxs)
    stable_q, qidxs_impulse = detect_peak_stability(qidxs)
    stable_s, sidxs_impulse = detect_peak_stability(sidxs)
    stable_t, tidxs_impulse = detect_peak_stability(tidxs)
    # plot_wave_detection(ecgs, pidxs_impulse, qidxs_impulse, ridxs,
    #                     sidxs_impulse, tidxs_impulse, stable_p, stable_q,
    #                     stable_s, stable_t)

    # 因為10秒片段中ECG的特徵不會變動大於outlier (Q1-1.5*iqr, Q3+1.5*iqr) 避免outlier影響穩定度計算 把outlier刪除
    # 為了SVM模型做的 未來資料夠多就可以整段segment刪除 不須執行下面程式判斷
    if stable_p >= p_quality_threshold:
        stable_p, pidxs_impulse = detect_peak_stability_del_outlier(pidxs_impulse)
    if stable_q >= q_quality_threshold:
        stable_q, qidxs_impulse = detect_peak_stability_del_outlier(qidxs_impulse)
    if stable_s >= s_quality_threshold:
        stable_s, sidxs_impulse = detect_peak_stability_del_outlier(sidxs_impulse)
    if stable_t >= t_quality_threshold:
        stable_t, tidxs_impulse = detect_peak_stability_del_outlier(tidxs_impulse)

    ##print("stable_p:",stable_p," stable_q:",stable_q," stable_s:",stable_s," stable_t:",stable_t)
    ## 計算特徵條件
    generate_features_flag = ((len(ridxs) > 5) and (stable_p < p_quality_threshold and stable_q < q_quality_threshold and stable_s < s_quality_threshold and stable_t < t_quality_threshold))

   
    ## plot_wave_detection(ecgs, pidxs_impulse, qidxs_impulse, ridxs, sidxs_impulse, tidxs_impulse, stable_p, stable_q, stable_s, stable_t)  # 觀察wave的圖

    """  開始計算特徵  """
    # 是否要生成features
    if generate_features_flag:
        ###quality_count += 1
        ridxs = np.array(ridxs) + 250
        rris = np.diff(ridxs) * 4
        # mean_rris = np.add(np.insert(rris, 0, rris[0]), np.insert(rris, -1, rris[-1])) / 2  # 2023

        """推算第一拍rri，目前方法會犧牲第一拍RRI準度，未來可精進"""
        first_rri = rris[0]  # 20240416 修改: 直接假設與下一拍RRI相同
        mean_rris = np.insert(rris, 0, first_rri)  # 20240325 修改

        wear_type = int(dict_wave_info["type"])  # 1 for strap, 2 for patch
        uuid_data["type"].extend([wear_type for i in range(len(ridxs))])

        a_score = float(round(cal_area_ratio(ecgs, ridxs), 3))
        uuid_data["a_score"].extend([a_score for i in range(len(ridxs))])

        p_score = float(round(pattern_clustering_pscore(ecgs, ridxs, th=0.9), 3))
        uuid_data["p_score"].extend([p_score for i in range(len(ridxs))])

        r_score = int(check_irregular(rris))  # 0 for Irregular, 1 for Regular
        uuid_data["r_score"].extend([r_score for i in range(len(ridxs))])

        uuid_data["p_stability"].extend([stable_p for i in range(len(ridxs))])
        uuid_data["q_stability"].extend([stable_q for i in range(len(ridxs))])
        uuid_data["s_stability"].extend([stable_s for i in range(len(ridxs))])
        uuid_data["t_stability"].extend([stable_t for i in range(len(ridxs))])

        for p, q, r, s, t, rri in zip(pidxs_impulse, qidxs_impulse, ridxs, sidxs_impulse, tidxs_impulse, mean_rris):

            uuid_data["rr_interval"].append(rri)  # 20240325 新增
            uuid_data["hr"].append((60 * 1000) / rri)  # 20240927 Wayne新增HR特徵
            uuid_data = compute_wave_(r, p, 0, rri, ecgs, "R", key_name='pr', input=uuid_data)
            uuid_data = compute_wave_(r, q, 0, rri, ecgs, "R", key_name='qr', input=uuid_data)
            uuid_data = compute_wave_(r, s, 0, rri, ecgs, "R", key_name='rs', input=uuid_data)
            uuid_data = compute_wave_(r, t, 0, rri, ecgs, "R", key_name='rt', input=uuid_data)
            uuid_data = compute_wave_(r, p, q, rri, ecgs, "N", key_name='pq', input=uuid_data)
            uuid_data = compute_wave_(r, p, s, rri, ecgs, "N", key_name='ps', input=uuid_data)
            uuid_data = compute_wave_(r, p, t, rri, ecgs, "N", key_name='pt', input=uuid_data)
            uuid_data = compute_wave_(r, q, s, rri, ecgs, "N", key_name='qs', input=uuid_data)
            uuid_data = compute_wave_(r, q, t, rri, ecgs, "N", key_name='qt', input=uuid_data)
            uuid_data = compute_wave_(r, s, t, rri, ecgs, "N", key_name='st', input=uuid_data)

            uuid_data = compute_curve_(r, p, ecgs, 5, "N", key_name='p', input=uuid_data)
            uuid_data = compute_curve_(r, 0, ecgs, 5, "R", key_name='r', input=uuid_data)
            uuid_data = compute_curve_(r, t, ecgs, 7, "N", key_name='t', input=uuid_data)
            
            ## dennis 新增
            
            uuid_data = calculate_qrs_area_(q+r, s+r, ecgs, key_name='qrs', input=uuid_data)
            uuid_data = calculate_st_area_(s+r, t+r, ecgs, key_name='st', input=uuid_data)
            uuid_data = calculate_t_wave_cog_(t+r, ecgs, key_name='t', input=uuid_data,fs=250)

            ## 紀錄peak index 資訊
            if p != 0:
                uuid_data["p_value"].append(ecgs[r + p])
            else:
                uuid_data["p_value"].append(np.nan)
            if q != 0:
                uuid_data["q_value"].append(ecgs[r + q])
            else:
                uuid_data["q_value"].append(np.nan)
            uuid_data["r_value"].append(ecgs[r])
            if s != 0:
                uuid_data["s_value"].append(ecgs[r + s])
            else:
                uuid_data["s_value"].append(np.nan)
            if t != 0:
                uuid_data["t_value"].append(ecgs[r + t])
            else:
                uuid_data["t_value"].append(np.nan)

        uuid_data["uuid"] = dict_wave_info["uuid"]
        # uuid_data["epoch"] = dict_wave_info["epoch"]

        return uuid_data,generate_features_flag ##, quality_count

    return uuid_data, generate_features_flag ###quality_count


def calculate_qrs_area_(xq, xs, signal, key_name, input,fs=250,):  ##Dennis新增 2026.05.12
    """
    計算 QRS 面積
    :param signal: 原始心電圖訊號數組 (array)
    :param xq: Q點的索引 (index)
    :param xr: R點的索引 (index)
    :param xs: S點的索引 (index)
    :param fs: 採樣頻率 (Hz)，預設 500Hz
    :return: QRS 面積 (mV*s)
    """
    # 1. 提取 QRS 段的訊號 (包含 S 點)
    qrs_segment = signal[xq : xs+1]
    
    # 2. 基準校正 (Baseline Correction)
    # 通常以 Q 點的電壓作為零點基準，避免基線漂移干擾面積計算
    baseline = signal[xq]
    corrected_segment = qrs_segment - baseline
    
    # 3. 計算時間步長 (dx)
    dx = 1.0 / fs
    
    # 4. 使用梯形法計算面積
    # 我們取絕對值積分，因為 Q 波和 S 波是向下的，如果不取絕對值會與 R 波抵消
    # 在偵測血鉀時，通常關注的是總體能量變化，因此建議取絕對值
    area = np.trapz(np.abs(corrected_segment), dx=dx)

    input[f"{key_name}_area"].append(area)
    
    return input


def calculate_st_area_(s,t,ecg,key_name,input):  ##2026.05.19 Agon新增
    
    """計算ST連線與心電圖曲線之間的封閉面積"""

    st = ecg[s:t+1]
    area_under_signal=np.trapz(st)
    y_start = ecg[s]
    y_end=ecg[t]
    width=t-s
    area_under_line = (y_start+y_end)* width/2
    st_area=abs(area_under_signal - area_under_line)
    st_area = st_area  
    input[f"{key_name}_area"].append(st_area)
    
    return input


def calculate_t_wave_cog_(t, ecg, key_name,input,fs=250):

    """
    計算 T 波的重心 (COG)
    t_segment: 時間數組
    v_segment: 電壓數組 (需先進行基線校正，確保最小值為 0)
    """  
    
    # 確保電壓值為正數（針對 T 波向上成長的部分）
    start_idx=t-40
    end_idx=t+40
        
    if(start_idx<0 or end_idx>=len(ecg)):
        input[f"{key_name}_wave_cog"].append(None)
        ##print('t_wave_cog:None')
    else:            
        v_segment=ecg[start_idx:end_idx]           
        actual_indices = np.arange(max(0,start_idx),min(len(ecg),end_idx))
        t_segment = actual_indices/fs
        v_min = np.min(v_segment)
        v_shifted = v_segment - v_min
        area = np.sum(v_shifted)

        if area == 0:
            ##return None,"T wave is smooth"
            input[f"{key_name}_wave_cog"].append(None)
            ##print('t_wave_cog:None')
        else:
            # 計算時間重心 (X軸)
            t_cog = np.sum(t_segment * v_shifted) / area
            
            # 計算電壓重心 (Y軸)
            # 基於幾何面積重心的公式：(1/2 * sum(v^2)) / sum(v)
            v_cog = (0.5 * np.sum(v_shifted**2) / area)+v_min
            
            # 找出峰值點作為參考

            t_peak = t/fs
            v_peak = ecg[t]
                            
            #1. 時間偏移(秒) ->反映橫向對稱性
            # 值為正:重心在峰值右邊(通常代表前陡後緩)
            # 值為負:重心在峰值左邊(通常代表前緩後陡，正常ECG多為此類)
            t_dist = t_cog - t_peak

            #2. 電壓差值(高度差)
            v_dist = v_cog - v_peak

            #直線距離
            twave_cog = np.sqrt(t_dist**2+v_dist**2)
                
            input[f"{key_name}_wave_cog"].append(twave_cog)        

    return input 

'''
def get_mean_in_window(dict_features):

    # df_features_mean = pd.DataFrame()
    features = ["rr_interval", "hr", "a_score", "p_score", "r_score",
                "p_stability", "q_stability", "s_stability", "t_stability",
                "p_value", "q_value", "r_value", "s_value", "t_value",
                "pr_duration", "pr_amplitude", "pr_distances", "pr_directions",
                "pr_slope", "pr_corrections3",
                "qr_duration", "qr_amplitude", "qr_distances", "qr_directions",
                "qr_slope", "qr_corrections3",
                "rs_duration", "rs_amplitude", "rs_distances", "rs_directions",
                "rs_slope", "rs_corrections3",
                "rt_duration", "rt_amplitude", "rt_distances", "rt_directions",
                "rt_slope", "rt_corrections3",
                "pq_duration", "pq_amplitude", "pq_distances", "pq_directions",
                "pq_slope", "pq_corrections3",
                "ps_duration", "ps_amplitude", "ps_distances", "ps_directions",
                "ps_slope", "ps_corrections3",
                "pt_duration", "pt_amplitude", "pt_distances", "pt_directions",
                "pt_slope", "pt_corrections3",
                "qs_duration", "qs_amplitude", "qs_distances", "qs_directions",
                "qs_slope", "qs_corrections3",
                "qt_duration", "qt_amplitude", "qt_distances", "qt_directions",
                "qt_slope", "qt_corrections3",
                "st_duration", "st_amplitude", "st_distances", "st_directions",
                "st_slope", "st_corrections3",
                "p_left_slope", "p_right_slope", "p_left_sharp", "p_right_sharp", "p_tilt",
                "r_left_slope", "r_right_slope", "r_left_sharp", "r_right_sharp", "r_tilt",
                "t_left_slope", "t_right_slope", "t_left_sharp", "t_right_sharp", "t_tilt",
                "qrs_area","st_area","t_wave_cog"
                ]

    df_features = pd.DataFrame(dict_features)
    df_features_mean = df_features[features].mean().to_frame().T
    df_features_mean['uuid'] = dict_features['uuid']
    # df_features_mean['epoch'] = dict_features['epoch']
    # df_features_mean['time'] = df_features['time'].iloc[0]
    # df_features_mean['BG'] = df_features['BG'].iloc[0]
    df_features_mean['type'] = df_features['type'].iloc[0]
    df_features_mean = df_features_mean[['uuid', 'type'] + features]

    return df_features_mean
'''


# stable-window aggregation
def stable_window_mean(values, window_size=3, cv_threshold=0.15, window_mean_cv_threshold=0.15, epsilon=1e-6):
    values = pd.Series(values).dropna()

    if len(values) < window_size:
        return values.mean()
    
    # 把這些值照數值大小排序
    values_sorted = values.sort_values().reset_index(drop=True)

    window_means = []

    for i in range(len(values_sorted) - window_size + 1):

        # 取出視窗內的三個數值
        window_values = values_sorted.iloc[i:i + window_size]

        mean_value = window_values.mean()
        std_value = window_values.std()

        cv = std_value / max(abs(mean_value), epsilon)

        if cv < cv_threshold:
            window_means.append(mean_value)

    if len(window_means) == 0:
        return values.mean()

    if len(window_means) == 1:
        return window_means[0]

    window_means = pd.Series(window_means)

    window_mean_cv = window_means.std() / max(abs(window_means.mean()), epsilon)

    # 如果所有 stable window 的平均值彼此也很接近
    # 代表沒有明顯分群，直接取所有 stable windows 的平均
    if window_mean_cv < window_mean_cv_threshold:
        return window_means.mean()
        
    # stable windows 內部穩，但彼此差異大：
    # 用最大 gap 把 window means 分成兩群
    sorted_means = window_means.sort_values().reset_index(drop=True)
    gaps = sorted_means.diff().abs()

    split_idx = gaps.iloc[1:].idxmax()

    group1 = sorted_means.iloc[:split_idx]
    group2 = sorted_means.iloc[split_idx:]

    # 選 window 數比較多的那群
    if len(group1) > len(group2):
        selected_group = group1
    elif len(group2) > len(group1):
        selected_group = group2
    else:
        # 如果兩群一樣多，沒有明確主要群，回到 window means 的中位數
        selected_group = window_means

    return selected_group.median()


# Jane 修改
def get_mean_in_window(dict_features, mode='mean', position_threshold=0.5,amplitude_threshold=0.5,position_criterion=1):
    
    '''
    迴圈每一特徵，將所有拍的特徵值取平均，得出一段訊號的平均特徵

    Input: 
    
    dict_features: 一段訊號逐拍的特徵值字典
    mode: 可選擇模式，取平均(mode='mean')，或是QC後取平均(mode='qc')
    position_threshold: 與中位數的相對距離超過多少，position_threshold=0.5表示與中位數的相對位置超過50%
    amplitude_threshold: 與中位數的相對距離超過多少
    position_criterion: 至少要有幾個位置特徵異常

    Output: 平均後的特徵

    '''

    # df_features_mean = pd.DataFrame()
    features = ["rr_interval", "hr", "a_score", "p_score", "r_score",
                "p_stability", "q_stability", "s_stability", "t_stability",
                "p_value", "q_value", "r_value", "s_value", "t_value",
                "pr_duration", "pr_amplitude", "pr_distances", "pr_directions",
                "pr_slope", "pr_corrections3",
                "qr_duration", "qr_amplitude", "qr_distances", "qr_directions",
                "qr_slope", "qr_corrections3",
                "rs_duration", "rs_amplitude", "rs_distances", "rs_directions",
                "rs_slope", "rs_corrections3",
                "rt_duration", "rt_amplitude", "rt_distances", "rt_directions",
                "rt_slope", "rt_corrections3",
                "pq_duration", "pq_amplitude", "pq_distances", "pq_directions",
                "pq_slope", "pq_corrections3",
                "ps_duration", "ps_amplitude", "ps_distances", "ps_directions",
                "ps_slope", "ps_corrections3",
                "pt_duration", "pt_amplitude", "pt_distances", "pt_directions",
                "pt_slope", "pt_corrections3",
                "qs_duration", "qs_amplitude", "qs_distances", "qs_directions",
                "qs_slope", "qs_corrections3",
                "qt_duration", "qt_amplitude", "qt_distances", "qt_directions",
                "qt_slope", "qt_corrections3",
                "st_duration", "st_amplitude", "st_distances", "st_directions",
                "st_slope", "st_corrections3",
                "p_left_slope", "p_right_slope", "p_left_sharp", "p_right_sharp", "p_tilt",
                "r_left_slope", "r_right_slope", "r_left_sharp", "r_right_sharp", "r_tilt",
                "t_left_slope", "t_right_slope", "t_left_sharp", "t_right_sharp", "t_tilt", 
                "qrs_area","st_area","t_wave_cog"
                ]

    # 用來判斷位置是否壞掉
    position_features = [
        "rt_duration",
        "qt_duration",
        "pt_duration",
        "rr_interval"
    ]

    # 用來判斷振幅是否壞掉
    amplitude_feature_mapping = {
        "t_left_slope": "t_value",
        "t_right_slope": "t_value",
        "t_left_sharp": "t_value",
        "t_right_sharp": "t_value",
        "st_area":"st_area",
        "t_wave_cog":"t_value",
        "q_value": "q_value",    
        "r_value": "r_value",
        "s_value": "s_value",
        "t_value": "t_value",
        "qr_amplitude":"qr_amplitude",
        "qs_amplitude":"qs_amplitude",
        "rt_amplitude":"rt_amplitude",
        "st_amplitude":"st_amplitude",
        "st_corrections3":"st_corrections3"
    }

    df_features = pd.DataFrame(dict_features)
  
    if mode == "mean":
        df_features_mean = df_features[features].mean().to_frame().T
    

    elif mode == "qc":

        # Step 1: position QC
        position_outlier_count = pd.Series(0, index=df_features.index)

        for feature in position_features:
            values = df_features[feature]

            median_value = values.median()
            relative_distance = (values - median_value).abs() / median_value

            # 逐拍判斷是否離群
            outlier_mask = relative_distance >= position_threshold
            position_outlier_count += outlier_mask.astype(int)

        position_error_mask = position_outlier_count >= position_criterion
  
        # 刪掉位置錯誤的拍
        df_clean = df_features.loc[~position_error_mask].copy()

        # 避免整條訊號被刪到太多拍
        n_beats = len(df_features)
        n_removed = position_error_mask.sum()

        if n_beats <= 7:
            max_removed = 1
        elif n_beats <= 10:
            max_removed = 2
        else:
            max_removed = 3

        if n_removed > max_removed:
            df_clean = df_features.copy()
        
        # Step 2: amplitude QC
        amplitude_error_masks = {}
        epsilon = 1e-6
        amplitude_abs_threshold = 20    # 與中位數的絕對位置相差10

        for feature in set(amplitude_feature_mapping.values()):
            values = df_clean[feature]
            median_value = values.median()

            if abs(median_value) < epsilon:    # 如果中位數=0，就用絕對位置判斷
                outlier_mask = (values - median_value).abs() >= amplitude_abs_threshold
            else:
                relative_distance = (values - median_value).abs() / abs(median_value)    # 振幅中位數有可能是0，所以取絕對值
                outlier_mask = relative_distance >= amplitude_threshold

            amplitude_error_masks[feature] = outlier_mask
        

        # 計算最後的平均特徵
        result = {}

        for feature in features:
            
            if feature in amplitude_feature_mapping:

                qc_feature = amplitude_feature_mapping[feature]
                mask = amplitude_error_masks[qc_feature]

                values = df_clean.loc[~mask, feature]

                # 保護機制
                if values.empty:
                    values = df_clean[feature]
                

            else:    # 如果不是振幅相關特徵，直接取平均
                values = df_clean[feature]
            
          
            # Step 3
            result[feature] = stable_window_mean(
                values,
                window_size=3,
                cv_threshold=0.10,
                window_mean_cv_threshold=0.05
            )

        df_features_mean = pd.DataFrame([result])

    else:
        raise ValueError("mode must be 'mean' or 'qc'")

        

    df_features_mean['uuid'] = dict_features['uuid']
    # df_features_mean['epoch'] = dict_features['epoch']
    # df_features_mean['time'] = df_features['time'].iloc[0]
    # df_features_mean['BG'] = df_features['BG'].iloc[0]
    df_features_mean['type'] = df_features['type'].iloc[0]
    df_features_mean = df_features_mean[['uuid', 'type'] + features]

    return df_features_mean



# Wayne added
def concat_features_label(dataset_index_path, raw_ecg_path, bg_level, dataset_type,features_uuid_path="",save_txt_flag=False,test_feature_uuid_path=""):
    
    """
    Parameters
    ----------
    dataset_index_path
    raw_ecg_path
    bg_level[str]: 'Low', 'Normal', 'High'
    dataset_type[str]: 'Train', 'Test'

    Returns
    -------

    """

    good_quality_count = 0
    completed_files_count = 0

    df_dataset_the_bg_level = pd.DataFrame()
    
    dataset_index_txtpath = os.path.join(dataset_index_path, bg_level)
    
    ##dataset_index_txtpath = os.path.join(raw_ecg_path, bg_level)    
    
    if os.path.exists(dataset_index_txtpath):
    
        print('Extracting features for {} BG Dataset'.format(bg_level))
        

        for name in os.listdir(dataset_index_txtpath):

            target_rawdata_file_url = os.path.join(raw_ecg_path, name)  ##到RawData資料夾去找到原始ECG訊號檔案
            
            if os.path.exists(target_rawdata_file_url):
                ## 已設定好input的ecg是10秒長度
                ecg = data_management.load_txt_file(target_rawdata_file_url)
                try:
                    uuid = name.split('_')[0]
                    bg = (name.split('_')[4]).replace('.txt', '')
                except IndexError:
                    globals.errorcode = -702  ##原本定義為-706
                    globals.message = (
                        "An error occurred in the concat_features_label from bg_package.features_extraction: "
                        "Incorrectly formatted filename ECG data."
                        )
                    return df_dataset_the_bg_level, good_quality_count

                # measure_type 0: not setting, 1: strap, 2: patch
                dict_waves_detection = wave_info.generate_waves_info(ecg, uuid, measure_type=0)  ##取得p,q,r,s,t waves, 其中p,q,s,t是相對於r的位置, p,q是負值,r,s,t是正值
                dict_waves_detection["BG"] = bg
                               
                ##dict_features_extraction, good_quality_count = extract_features(dict_waves_detection, good_quality_count)
                dict_features_extraction, generate_features_flag = extract_features(dict_waves_detection)
               
                if(generate_features_flag):
                    good_quality_count+=1                   
                    

                if dict_features_extraction['type']:  # 有擷取到features
                    #df_feature_extraction_resample = get_mean_in_window(dict_features_extraction)  # 取平均
                    df_feature_extraction_resample = get_mean_in_window(dict_features_extraction, mode='qc',position_threshold=0.5,amplitude_threshold=0.5,position_criterion=1)
                    
                    df_feature_extraction_resample["Dataset"] = dataset_type
                    df_feature_extraction_resample["BG_Level"] = bg_level
                    
                    ###---------dennis新增----------                    
                    if(save_txt_flag):                         
                        
                        if(test_feature_uuid_path !=""): ##預測血糖時的feature extraction階段，寫出路徑不同                          
                            save_txt_path=os.path.join(test_feature_uuid_path,bg_level,name)                           
                        else:
                            save_txt_path=os.path.join(features_uuid_path, dataset_type, bg_level, name) 
                                              

                        with open(save_txt_path, 'w') as f:
                            for row in df_feature_extraction_resample.itertuples(index=False):
                                for col_name, value in zip(df_feature_extraction_resample.columns, row):
                                    f.write(f"{col_name}\t{value}\n")
                           
                            ##------Dennis新增特徵 2025.04------------
                            if(df_feature_extraction_resample["qr_amplitude"].iloc[0]!=0):
                                feature_value=(df_feature_extraction_resample["qs_amplitude"]/df_feature_extraction_resample["qr_amplitude"]).tolist()
                                f.write("dif_qs_amp/dif_qr_amp\t"+str(feature_value[0])+"\n")
                            else:
                                f.write("dif_qs_amp/dif_qr_amp\t"+str(np.nan)+"\n")
                            
                            if(df_feature_extraction_resample["st_amplitude"].iloc[0]!=0):
                                feature_value=(df_feature_extraction_resample["rt_amplitude"]/df_feature_extraction_resample["st_amplitude"]).tolist()
                                f.write("dif_rt_amp/dif_st_amp\t"+str(feature_value[0])+"\n")
                            else:
                                f.write("dif_rt_amp/dif_st_amp\t"+str(np.nan)+"\n")

                            if(df_feature_extraction_resample["r_value"].iloc[0]!=0):
                                feature_value=(df_feature_extraction_resample["t_value"]/df_feature_extraction_resample["r_value"]).tolist()
                                f.write("t/r_amp\t"+str(feature_value[0])+"\n")
                            else:
                                f.write("t/r_amp\t"+str(np.nan)+"\n")

                            if(df_feature_extraction_resample["t_value"].iloc[0]!=0): 
                                feature_value=(df_feature_extraction_resample["s_value"]/df_feature_extraction_resample["t_value"]).tolist()
                                f.write("s/t_amp\t"+str(feature_value[0]))
                            else:    
                                f.write("s/t_amp\t"+str(np.nan))
                    ##-------------------------------------

                    df_dataset_the_bg_level = pd.concat([df_dataset_the_bg_level, df_feature_extraction_resample])
                    

            if (completed_files_count % 50) == 0:
                print("Completed files: {}/{}".format(completed_files_count,len(os.listdir(dataset_index_txtpath))))

            completed_files_count += 1

            if(completed_files_count==len(os.listdir(dataset_index_txtpath))):
                print("Completed files: {}/{}".format(completed_files_count,len(os.listdir(dataset_index_txtpath))))

        # 資料集訊號全部都是差的 且要有資料
        if (good_quality_count == 0) and (len(os.listdir(dataset_index_txtpath))>0):
            ## Future: 若要判斷筆數是否過少 加在這
            ##globals.errorcode = -202
            ##globals.message = ("An error occurred in the concat_features_label function  of the bg_package.features_extraction: "
            ##                   "Quality of {} blood glucose ECG signals are bad in the {} dataset".format(bg_level, dataset_type))
            return df_dataset_the_bg_level
        
    return df_dataset_the_bg_level


# Wayne added
##def load_rawdata_extract_features(rawdata_path, train_index_path, test_index_path, features_filepath, parquet_filename,save_txt_flag=False):
def load_rawdata_extract_features(rawdata_path, train_index_path, test_index_path, features_filepath, parquet_filename,save_txt_flag=False,test_feature_uuid_path=""):

        
    ##saved_parquet_filepath = os.path.join(features_filepath, parquet_filename)
    df_features = pd.DataFrame()

    print("Start extracting features from training set..")
    df_normalbg_train = concat_features_label(train_index_path, rawdata_path, "Normal", "Train", features_filepath, save_txt_flag,test_feature_uuid_path)    
   
    if globals.errorcode < 0:
        return

    df_lowbg_train = concat_features_label(train_index_path, rawdata_path, "Low", "Train", features_filepath, save_txt_flag,test_feature_uuid_path)
   
    if globals.errorcode < 0:
        return

    df_highbg_train = concat_features_label(train_index_path, rawdata_path, "High", "Train", features_filepath, save_txt_flag,test_feature_uuid_path)
   
    if globals.errorcode < 0:
        return

    df_features = pd.concat([df_features, df_lowbg_train, df_normalbg_train, df_highbg_train])

    print("Start extracting features from testing set..")
    df_lowbg_test = concat_features_label(test_index_path, rawdata_path, "Low", "Test", features_filepath, save_txt_flag,test_feature_uuid_path)
   
    if globals.errorcode < 0:
        return

    df_normalbg_test = concat_features_label(test_index_path, rawdata_path, "Normal", "Test", features_filepath, save_txt_flag,test_feature_uuid_path)
   
    if globals.errorcode < 0:
        return

    df_highbg_test = concat_features_label(test_index_path, rawdata_path, "High", "Test", features_filepath, save_txt_flag,test_feature_uuid_path)
   
    if globals.errorcode < 0:
        return

    df_features = pd.concat([df_features, df_lowbg_test, df_normalbg_test, df_highbg_test])

    if(features_filepath!=""): ##若為空，是預測血糖值時候的feature extraction 階段，不需要寫出檔案
        saved_parquet_filepath = os.path.join(features_filepath, parquet_filename)
        table = pa.Table.from_pandas(df_features)
        pq.write_table(table, saved_parquet_filepath)
        print('saved_parquet_filepath:',saved_parquet_filepath)

    return df_features




import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

def load_rawdata_extract_features_multiprocess(rawdata_path, train_index_path, test_index_path, features_filepath, parquet_filename, save_txt_flag=False, test_feature_uuid_path=""):
    
    df_features = pd.DataFrame()
    
    # ----------------------------------------------------
    # 階段 1：平行處理 Training Set (Normal, High, Low)
    # ----------------------------------------------------
    print("Start extracting features from training set in parallel..")
    train_tasks = ["Normal", "Low", "High"]
    train_results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 同時提交 3 個 Train 任務
        futures = {
            executor.submit(
                concat_features_label, 
                train_index_path, rawdata_path, bg_type, "Train", features_filepath, save_txt_flag, test_feature_uuid_path
            ): bg_type for bg_type in train_tasks
        }
        
        # 收集結果
        for future in futures:
            bg_type = futures[future]
            try:
                df_res = future.result()
                train_results.append(df_res)
            except Exception as e:
                print(f"Error extracting Train - {bg_type}: {e}")
                globals.errorcode = -1
                
    # 檢查 Train 階段是否有任一錯誤或全域錯誤碼
    if globals.errorcode < 0:
        print("Error detected during Training feature extraction. Aborting.")
        return

    # 合併 Train 的結果
    df_train_all = pd.concat(train_results, ignore_index=True)
    df_features = pd.concat([df_features, df_train_all], ignore_index=True)


    # ----------------------------------------------------
    # 階段 2：平行處理 Testing Set (Normal, High, Low)
    # ----------------------------------------------------
    print("Start extracting features from testing set in parallel..")
    test_tasks = ["Low", "Normal", "High"]
    test_results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 同時提交 3 個 Test 任務
        futures = {
            executor.submit(
                concat_features_label, 
                test_index_path, rawdata_path, bg_type, "Test", features_filepath, save_txt_flag, test_feature_uuid_path
            ): bg_type for bg_type in test_tasks
        }
        
        # 收集結果
        for future in futures:
            bg_type = futures[future]
            try:
                df_res = future.result()
                test_results.append(df_res)
            except Exception as e:
                print(f"Error extracting Test - {bg_type}: {e}")
                globals.errorcode = -1

    # 檢查 Test 階段是否有任一錯誤或全域錯誤碼
    if globals.errorcode < 0:
        print("Error detected during Testing feature extraction. Aborting.")
        return

    # 合併 Test 的結果
    df_test_all = pd.concat(test_results, ignore_index=True)
    df_features = pd.concat([df_features, df_test_all], ignore_index=True)


    # ----------------------------------------------------
    # 階段 3：寫出檔案與回傳
    # ----------------------------------------------------
    if features_filepath != "":  ## 若為空，是預測血糖值時候的feature extraction 階段，不需要寫出檔案
        saved_parquet_filepath = os.path.join(features_filepath, parquet_filename)
        table = pa.Table.from_pandas(df_features)
        pq.write_table(table, saved_parquet_filepath)
        print('saved_parquet_filepath:', saved_parquet_filepath)

    return df_features

