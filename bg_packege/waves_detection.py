# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:58:15 2023

author: SWM-Jared
collaborator: SWM-Benjamin

2024.07.17 Benjamin更新: 調整P波偵測

"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, butter, filtfilt
import neurokit2 as nk
# from bg_packege import features_extraction
from SWMlib.ecg import baseline
from SWMlib.common import filters
from bg_packege.splitted_T_wave_library_version import splitted_t_wave_detector
import plotly.graph_objects as go
# ------- 訊號處理 ---------
def fix_scale(ecgs):
    new_ecgs = []

    ecg_length = len(ecgs)

    ecg_number = int(ecg_length // 2500) 

    for i in range(ecg_number):

        ecg_segment = ecgs[i * 2500:i * 2500 + 2500] ##切成每2500個點為一片段 

        ecg_max = max(ecg_segment)  ##每一片段中的2500個點找最大值

        scale = 1

        if 500 >= ecg_max > 10:  ##如果最大值介於10~500之間
            scale = int(500 / ecg_max) ##計算放大倍數

        ecg = ecg_segment * scale  ##此片段放大scale倍

        new_ecgs.extend(ecg)

    return np.array(new_ecgs).astype('int')

def find_max_median_index(arr):  # original is def median_index_of_max
    # Find the maximum value in the array
    max_value = np.max(arr)

    # Find all occurrences of the maximum value
    max_indices = np.where(arr == max_value)[0]

    # Find the median index among these occurrences
    median_index = np.median(max_indices)

    return int(median_index)

def delete_impulse_(ecgs):
    ecgs_diff = np.diff(ecgs)
    diff_len = len(ecgs_diff) - 1

    j = 0

    for i, (ecg, ecg_diff) in enumerate(zip(ecgs[:-1], ecgs_diff)):
        if j == 0:
            if ecg_diff < -500:
                j = 1
                try:
                    while ecgs_diff[i + j] == 0:
                        j += 1
                except:
                    j -= 1
                if ecgs_diff[i + j] > 500:
                    for k in range(i + 1, i + j + 1): ecgs[k] = (ecgs[i] + ecgs[i + j + 1]) / 2
            elif ecg_diff > 700:
                j = 1
                try:

                    while ecgs_diff[i + j] == 0:
                        j += 1

                except:
                    j -= 1

                if ecgs_diff[i + j] < -700:

                    for k in range(i + 1, i + j + 1): ecgs[k] = (ecgs[i] + ecgs[i + j + 1]) / 2
        else:

            j = j - 1

    return ecgs

def cal_pattern_clustering_corr(corr, th):
    '''
    Clustering patterns in groups by correlation coefficients.

    Parameters:
        corr (ndarray): square correlation coefficient matrix between patterns.

        th (float): threshold of correlation coefficient to determine whether two pattern are similar.

    Return:
        Relation (list): Indices (ndarray) of groups.
    '''

    V = []
    for i in range(0, len(corr)):
        # % 相關係數大於th視為相似
        idx = np.argwhere(corr[:, i] > th)
        if len(idx) > 0:
            V.append(idx[:, 0])

    # 將V由大至小排列，目的:減少交集比對
    V.sort(key=len, reverse=True)

    # 整理分群關係
    Relation = []
    for i in range(0, len(V)):
        if len(Relation) == 0:
            Relation.append(V[i])
        else:
            for j in range(0, len(Relation)):
                InterNum = len(np.intersect1d(V[i], Relation[j]))
                if InterNum > 0:
                    Relation[j] = np.union1d(V[i], Relation[j])
                    break
                elif InterNum == 0 and j == len(Relation) - 1:
                    Relation.append(V[i])
                else:
                    continue

    # 確認群跟群之間沒有重複
    for i in range(0, len(Relation)):
        for j in range(0, len(Relation)):
            if i == j:
                continue

            C, ix, iy = np.intersect1d(Relation[i], Relation[j], return_indices=True)
            Px = len(C) / len(Relation[i])
            Py = len(C) / len(Relation[j])
            if Px > Py:
                np.delete(Relation[j], iy)
            else:
                np.delete(Relation[i], ix)

    Relation.sort(key=len, reverse=True)
    return Relation

def mean_filter_1d(data, kernel_size):
    """
    Apply a one-dimensional mean filter to a list of data.

    Parameters:
    - data: The input list or one-dimensional array of data points.

    Returns:
    - filtered_data: The smoothed data after applying the mean filter.
    """
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    # Calculate the amount of padding needed on each side
    pad_size = int(kernel_size // 2)

    # Pad the data on both sides
    padded_data = np.full(int(len(data) + 2 * pad_size), np.nan)
    padded_data[:pad_size] = data[0]
    padded_data[pad_size:-pad_size] = data
    padded_data[-pad_size:] = data[-1]
    # Initialize the output list for the filtered data
    filtered_data = []

    # Apply the mean filter
    for i in range(len(data)):
        # Calculate the mean for the current window and append it to the output list
        window = padded_data[i:i + kernel_size]
        filtered_data.append(sum(window) / kernel_size)

    return kernel_size, np.array(filtered_data)

def check_r_peak(ecg_data, rpeaks):
    updated_rpeaks = []
    gap = 7
    for rpeak in rpeaks:
        start = max(0, rpeak - gap)
        end = min(len(ecg_data), rpeak + gap)
        ecg_subset = ecg_data[int(start):int(end)]

        # find peak
        try:
            index = int(find_peaks(ecg_subset)[0])
        except TypeError:  # 沒有peak
            # tmp = max(ecg_subset)
            # index = list(ecg_subset).index(tmp)
            index = find_max_median_index(ecg_subset)
        peak_index = index + start

        if peak_index != rpeak:
            rpeak = peak_index
        updated_rpeaks.append(int(rpeak))
    return updated_rpeaks


# ---------找waves----------
class WaveDetector():

    def __init__(self):
        pass

    def find_valley_v3(self, x):
        """
        修改找波谷方法
        1. 找最低點視為波谷, 若不符合 1(找最低點) 之條件, 則找
        2. 正向最大加速度的點

        note 拐點為二階導數符號變化的位置
        """
        if len(x) < 4:  # idx 0~2 主程式會剔除 不算gradient
            return None

        # 梯度
        grad_x = np.gradient(x)  # 梯度 (一階導數)
        grad_xx = np.gradient(grad_x)  # (二階導數)

        # 找波谷
        valley_idx, _ = signal.find_peaks(x * -1)
        if len(valley_idx) > 0:  # 有波谷
            idx = valley_idx[np.argmax(grad_xx[valley_idx])]
            return int(idx)
            """  
            # Plot valley
            plt.figure(figsize=(3, 5))
            plt.plot(x, label='ECG', linewidth=2)
            plt.scatter(idx, x[idx], c='r', label='Valley')
            plt.scatter(0, x[0], c='green', label='R peak')
            plt.legend()
            plt.tight_layout()
            plt.show()
            """

        else:
            # ---------------grad x-----------------
            # 找拐點：二階導數符號變化點
            # sign_changes = np.where(np.diff(np.sign(grad_xx)))[0]

            idx = np.argmin(abs(grad_x))
            idx = np.argmin(abs(grad_x[3:])) + 3  # 2023.10.11:  idx 0~2 主程式會剔除, 不如在這裡先做
            if x[idx] >= (x[0] / 2):  # 剔除 valley 比r-peak 一半還高
                return None
            """  
            # Plot min gradient 
            plt.figure(figsize=(3, 5))
            plt.plot(x, label='ECG', linewidth=2)
            plt.scatter(idx, x[idx], c='r', label='minimum gradient')
            plt.scatter(0, x[0], c='green', label='R peak')
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
            return idx

    def find_waves(self, X, P1, P2, target=None):
        '''
        Use two different width of mean filter to detect outstanding waves.

        Parameters:
            X (ndarray): 1D signal.

            P1 (float): narrower width of mean filter.

            P2 (float): wider width of mean filter.

            target (str): If the purpose is to find the T wave, then set target to "T".

        Return:
            peaks (ndarray): 2D array.
                1st column: the magnitude of peaks.
                2nd column: the index of peak in signal X
                3rd column: the standard deviation represents the level of wave
        '''
        # 以 P1 做平均濾波保留波的起伏
        # X = np.nan_to_num(X)  # 帶測
        # W1, MApeak = mean_filter(X, P1)
        W1, MApeak = mean_filter_1d(X, P1)

        # 以 P2 做平均濾波得到飄移基線
        # W2, MAbase = mean_filter(X, P2)
        W2, MAbase = mean_filter_1d(X, P2)

        # Block of interest: MApeak > MAbase表示波型有明顯起伏
        blocks = np.zeros(X.shape)
        for i in range(0, len(X)):
            if ~np.isnan(MApeak[i]) and ~np.isnan(MAbase[i]):
                if target == "T" and abs(MApeak[i]) > abs(MAbase[i]):  # 考慮t導致 要取絕對值
                    blocks[i] = 200
                elif MApeak[i] > MAbase[i] and MApeak[i] > -50:  # MApeak[i] > -50 考慮有些p波會小於0但又不過小於50
                    blocks[i] = 200

        """
        plt.figure()
        plt.plot(X, label='ECG')
        plt.plot(MApeak, label='Filter_peak')
        plt.plot(MAbase, label='Filter_base')
        plt.plot(blocks, label='Blocks of interest')
        s_idx = np.where(~np.isnan(X))[0][0]
        q_idx = np.where(~np.isnan(X))[0][-1]
        plt.scatter(s_idx, X[s_idx], label='S point', c='b')
        plt.scatter(q_idx, X[q_idx], label='Q point', c='gold')
        plt.legend()
        plt.show()  
        """

        # 查看各個blocks的連續性
        # 對blocks作微分，在blocks的開始和結束處才會有值
        Dif = np.diff(blocks)
        start_flag = 0
        peaks = []
        for i in range(0, len(Dif)):
            d = Dif[i]

            # block 的起點
            if d == 200:
                start_flag = 1
                start_idx = i

            # block 的終點
            if d == -200 and start_flag:
                block_len = i - start_idx

                # block寬度大於 W1 才找峰值
                # if block_len < W1:
                if block_len < (W1 - 2):  # benjamin revise
                    blocks[start_idx + 1:i + 1] = 0
                else:
                    # 擷取該block對應的X值
                    block = np.full(X.shape, np.nan)
                    block[start_idx + 1:i + 1] = X[start_idx + 1:i + 1]

                    # 尋找block中的峰值
                    if target == "T":
                        block = abs(block)
                    M = np.nanmax(block)
                    Imax = np.nanargmax(block)

                    # 計算兩個平均濾波的差:MAdiff
                    MAdiff = MApeak[start_idx + 1:i + 1] - MAbase[start_idx + 1:i + 1]
                    STD = np.std(MAdiff)

                    peaks.append([M, Imax, STD])  # M:峰直高度, Imax:峰值INDEX

                # reset
                start_flag = 0

        peaks = np.array(peaks)
        """
        plt.figure()
        plt.plot(X, label='ECG')
        plt.plot(MApeak, label='Filter_peak')
        plt.plot(MAbase, label='Filter_base')
        plt.plot(blocks, label='Blocks of interest')
        s_idx = np.where(~np.isnan(X))[0][0]
        q_idx = np.where(~np.isnan(X))[0][-1]
        plt.scatter(s_idx, X[s_idx], label='S point', c='b')
        plt.scatter(q_idx, X[q_idx], label='Q point', c='gold')
        for ii, target in enumerate(peaks):
            tar_i = int(target[1])
            plt.scatter(tar_i, X[tar_i], label=f'Target', c='red')
        plt.legend()
        plt.show()  
        """
        return peaks

    def detect_pqrst_waves_v3(self, ecg, Fs, Ridx, mode='ori', FalseR=False):  # used to be "features_generate_v3"
        """
        2024.03.04修改
        Generate Fiducial Features from ECGs
        Arguments:
            ecg (ndarray): The 1st column is the value of ecg, and the 2nd column is Rpeak locations.
            Fs (int): The sampling rate of ecg.
            Ridx (ndarray): R peak location indices of ecg.
        Returns:
            new_Ridx (ndarray): The modified Ridx after false R detection.
            avgHR (float): Averaged Heart Rate (BPM)
            avgPR (float): Averaged PR interval (ms)
            avgQRS (float): Averaged QRS interval (ms)
            avgQT (float): Averaged QT interval (ms)
            avgQTc (float): Averaged Corrected QT interval (ms)
        """

        Qidx = np.full(Ridx.shape, np.nan)
        Sidx = np.full(Ridx.shape, np.nan)
        Tidx = np.full(Ridx.shape, np.nan)
        Pidx = np.full(Ridx.shape, np.nan)

        #  QRS detection ######
        if len(Ridx) > 1:
            # 重新標記 Rpeak 位置
            ecg = np.array(ecg)

            # butterworth bandpass filter design
            fs = 0.12  # 0.12
            fc = 0.3
            b, a = signal.butter(2, [fs, fc], btype='bandpass', output='ba')

            # 尋找 Q 波和 S 波
            Sf = 50
            Sb = 99
            Beta = 0.17
            W1 = 20
            Beats = []
            plt.close()
            for i in range(0, len(Ridx)):
                idx = int(Ridx[i])
                if idx - Sf >= 0 and idx + Sb + 1 <= len(ecg):

                    # extract a beat X from ecg
                    X = ecg[idx - Sf: idx + Sb + 1, 0]
                    Beats.append(X)

                    # bandpass filter on X
                    Xf = signal.filtfilt(b, a, X)
                    Xsq = Xf * Xf

                    # Moving Average with W1 as window size to detect QRS segment
                    MAqrs = np.zeros(len(Xsq))
                    for j in range(0, len(Xsq)):
                        if j - W1 / 2 >= 0 and j + W1 / 2 < len(Xsq):
                            window = Xsq[int(j - W1 / 2): int(j + W1 / 2 + 1)]
                            MAqrs[j] = np.mean(window)

                    # block of interest (QRS)
                    z = np.mean(Xsq)
                    thr1 = Beta * z
                    block = np.zeros(len(MAqrs))
                    block[MAqrs > thr1] = 1

                    # 擷取出 block 對應的 X
                    Xblock = np.full(len(X), np.nan)
                    Xblock[block == 1] = X[block == 1]

                    # 找Xblock前的第一個波谷為 Q ----------------------------------------------------------------------------
                    """
                    Q peak lead R peak 最多約 28ms (7個點)
                    https://litfl.com/q-wave-ecg-library/
                    Q Wave width ~= 40 ms (10個點)
                    peak 約在寬的一半 ~= 5個點
                    考慮要偵測病患放寬標準 ~=8個點
                    # 2023.12.15 範圍修正 56ms(14個點)
                    """
                    XbeforeR = np.flip(Xblock[:Sf + 1])
                    XbeforeR = XbeforeR[~np.isnan(XbeforeR)]
                    Qloc = self.find_valley_v3(XbeforeR[:15])  # 2024.03.07   scipy to find peak

                    # 找Xblock後的第一個波谷為 S ----------------------------------------------------------------------------
                    """
                    S peak 延遲 R peak 最多約 32ms (8個點)  貼片有些會超過8個點,所以修正成12
                    S peak 延遲 R peak 最多約 48ms (12個點)  2023.10.04 
                    S peak 延遲 R peak 最多約 68ms (17個點)  2023.12.15 
                    S peak 延遲 R peak 最多約 104ms (26個點)  2023.12.26
                    S peak 延遲 R peak 最多約 92ms (23個點)  2024.03.04 dennis說20pt
                    # According to "QRS Peaks, P and T Waves Identification in ECG", 
                    S Peak is behind R Peak about 3 ~ 69 ms  >> 1 ~ 17 sample points

                    """

                    XafterR = Xblock[Sf: Sf + 30]  # 2024.03.05
                    XafterR = XafterR[~np.isnan(XafterR)]
                    Sloc = self.find_valley_v3(XafterR)  # 2024.03.07   scipy to find peak

                    # 修改時用 --------------------------------------------------------------
                    # if Sloc:  # 修改時用
                    #     plt.plot(XafterR[:13], c='lightgray', linewidth=0.7)
                    #     # plt.scatter(Sloc, XafterR[:9][Sloc], label=str(i), s=7)
                    #     plt.scatter(Sloc, XafterR[:13][Sloc], label=str(i), s=7)  # 2023.10.04 Benjamin
                    # 修改時用 --------------------------------------------------------------

                    if (FalseR):
                        # False R Detection
                        if Qloc is None and Sloc is None:
                            Ridx[i] = np.nan

                    # plt.figure()
                    # plt.plot(X)
                    # plt.scatter(Sloc+50, X[Sloc+50], s=9, c='r')

                else:
                    Qloc = None
                    Sloc = None

                # 寫入找到的  Peak index
                if Qloc:
                    if Qloc >= 3:
                        Qidx[i] = idx - Qloc

                if Sloc:
                    if Sloc >= 3:
                        if (idx + Sloc) < 2750:
                            Sidx[i] = idx + Sloc

            # 找出打錯的R，並把連帶影響的元素刪除
            Beats = np.array(Beats)
            false_R = np.argwhere(np.isnan(Ridx))
            if len(false_R) != 0:
                Ridx = np.delete(Ridx, false_R)
                Qidx = np.delete(Qidx, false_R)
                Sidx = np.delete(Sidx, false_R)
                Beats = np.delete(Beats, false_R, 0)

            ##### 計算 Beats 相似性並分群 ######
            # 計算相關係數
            th = 0.9
            if len(Beats) > 1:
                corr = np.corrcoef(Beats)
            else:
                corr = []

            # 利用相關係數分群
            Relation = cal_pattern_clustering_corr(corr, th)

            # 標記不合群的 Beats
            igno_Ridx = []
            if len(Relation) > 1:
                Len = np.zeros(len(Relation))
                for i in range(0, len(Relation)):
                    Len[i] = len(Relation[i])
                Per = Len / sum(Len)

                for c in np.argwhere(Per < 0.1):
                    ignore_list = Relation[int(c)]
                    igno_Ridx.append(Ridx[ignore_list][0])

            # 計算 RRI & HR
            RRIs = np.full(Ridx.shape, np.nan)
            RRIs[1:] = np.diff(Ridx)
            avgRRI = np.nanmean(RRIs)

            # Paper參數 from:
            # Fast T Wave Detection Calibrated by Clinical Knowledge with Annotation of P and T Waves
            P1 = 0.07 * avgRRI
            P2 = 0.14 * avgRRI

            # Based on MIT-BIH test >>>> Dmin = 0.17, Dmax = 0.5
            Dmin = 0.17
            Dmax = 0.58  # benjamin revise
            RiTmin = np.ceil(Dmin * avgRRI)
            RiTmax = np.ceil(Dmax * avgRRI)

            # benjamin revise------
            # if RiTmin < 40:
            #     RiTmin = 40
            if RiTmax > 100:
                RiTmax = 100
            # --------------------

            ##### 尋找 T 波和 P 波 #####
            # Jane新增
            if mode == 'new':
                ecg_data = np.array(ecg[:, 0])  ###dennis修改
                Tidx = splitted_t_wave_detector(ecg_data, rpeak=Ridx)
                #print("Tidx", Tidx)
                #Tidx_raw = Ridx[:-1] + Tloc
                #mask = ~np.isnan(Tidx_raw)
                #Tidx = Tidx_raw[mask].astype(int)

            for i in range(0, len(Ridx) - 1):
                idx1 = int(Ridx[i])
                idx2 = int(Ridx[i + 1])

                '''
                if idx1 in igno_Ridx:
                    continue
                '''

                # 避免漏打Rpeak造成誤判
                medRRI = int(np.nanmedian(RRIs))
                if idx2 - idx1 > 1.5 * medRRI:  # 有漏r peak
                    # 擷取兩個 R 波之間的訊號
                    XX = np.copy(ecg[idx1:(idx1 + medRRI + 1), 0])  # 2023.10.11
                else:
                    # 擷取兩個 R 波之間的訊號
                    XX = np.copy(ecg[idx1:idx2 + 1, 0])

                # 兩個R波太靠近，無法找T波和P波
                if len(XX) < P2:
                    continue

                # 從當前S波到下一個Q波中尋找T波和P波
                Sloc1 = Sidx[i] - Ridx[i]
                Qloc2 = Qidx[i + 1] - Ridx[i + 1]

                if ~np.isnan(Sloc1):  # 有偵測到s
                    XX[:int(Sloc1 + 1)] = np.nan
                else:  # 沒偵測到s
                    XX[:int(20 + 1)] = np.nan

                if ~np.isnan(Qloc2):
                    XX[int(-1 + Qloc2):] = np.nan
                    XX = XX[:int(-1 + Qloc2)]  # 20240716
                    XX[-3:] = np.nan  # 避免移動平均被深Q波影響 # 20240716
                    # XX[int(-15):] = np.nan  # 避免Moving Average 受 Q peak 訊號驟降影響
                    # XX[int(-10):] = np.nan  # 避免Moving Average 受 Q peak 訊號驟降影響 20240308
                else:  # 沒偵測到q
                    XX = XX[:int(-14)]

                # P1 & P2 為尋找T波的兩個mean filter寬度，產生blocks of interest in Twave
                peaks = self.find_waves(XX, P1, P2, "T")

                # Jane新增
                if mode == 'ori':
                    # 若T的位置落在paper定義的範圍內[RiTmin, RiTmax)，視為候選
                    Tloc_candidate = []
                    Tloc = []
                    for j in peaks:
                        if RiTmax > j[1] > RiTmin:
                            Tloc_candidate.append(j)
                    Tloc_candidate = np.array(Tloc_candidate)

                    if len(Tloc_candidate) > 0:
                        # 從候選T中挑選強度最高的為Twave
                        Imax = np.argmax(abs(Tloc_candidate[:, 0]))
                        Tloc = Tloc_candidate[Imax][1]
                        Tidx[i] = Ridx[i] + Tloc
             
                # =========== Find P peak================
                if idx2 - idx1 > 1.5 * medRRI:  # 2  # 有漏R的情況 2023.10.11
                    XX = np.copy(ecg[(idx2 - medRRI):idx2, 0])  # 2023.10.11
                if idx2 + Sb <= len(ecg):
                    # # 在視窗的後1/3內尋找對應下個Rpeak的P波
                    # XX[:int(np.ceil(0.667 * len(XX)))] = np.nan
                    # 在相對於R 位置-50~Q之間找P # 20240716
                    XX[:-50] = np.nan

                    XX = np.nan_to_num(XX)  # 2023.10.11   # 避免 moving avg 計算為空值 使 block len 縮短被剔除

                    Ploc = []

                    # 0.03*avgRRI, 0.06*avgRRI 為自定義的參數，參考P duration < 80ms
                    Ppeaks = self.find_waves(XX, 0.04 * avgRRI, 0.08 * avgRRI)

                    # 如果P index 相對於R peak 介於 -50 ~ -17 間 則保留   benjamin revise
                    try:
                        Ppeaks = Ppeaks[np.logical_and((Ppeaks[:, 1] - len(XX)) < -17, (Ppeaks[:, 1] - len(XX)) > -50)]

                    except IndexError:
                        pass

                    if len(Ppeaks) > 0:
                        # 將Ppeaks對應的wave起伏程度 / 對應R波的振幅 = Pscore , 以避免整個beat強度過小導致找不到Pwave
                        if ~np.isnan(Qidx[i + 1]) and ~np.isnan(Sidx[i + 1]):
                            Qmag = abs(ecg[int(Qidx[i + 1]), 0])
                            Rmag = abs(ecg[int(Ridx[i + 1]), 0])
                            Smag = abs(ecg[int(Sidx[i + 1]), 0])
                            maxMag = np.max([Qmag, Rmag, Smag])
                            Pscore = 0 if Rmag == 0 else 100 * Ppeaks[:, 2] / maxMag
                            # Pscore = 100 * Ppeaks[:, 2] / maxMag
                        else:
                            Rmag = abs(ecg[int(Ridx[i + 1]), 0])
                            Pscore = 0 if Rmag == 0 else 100 * Ppeaks[:, 2] / Rmag
                            # Pscore = 100 * Ppeaks[:, 2] / Rmag

                        Ip = np.argmax(Pscore)
                        Mp = np.max(Pscore)

                        # 如果大於自定義的閥值TH，視為Pwave
                        th = 0.1  # 0.2
                        if Mp > th:
                            Ploc = Ppeaks[Ip, 1]

                    if Ploc:
                        Pidx[i + 1] = Ridx[i] + Ploc
                        if idx2 - idx1 > 1.5 * medRRI:  # 2  # 有漏R的情況 2023.10.11
                            Pidx[i + 1] = Ridx[i + 1] - medRRI + Ploc

        Dict = {'Pidx': Pidx,
                'Qidx': Qidx,
                'Ridx': Ridx,
                'Sidx': Sidx,
                'Tidx': Tidx,
                }
        return Dict

    # 取得PQRST 的位置
    def generate_waves_info(self, ecg, uuid, measure_type, plot_dir=None,file_name=None):
        """
        input-
            ecg: ecg signal with length larger than 2500 data points

        output-
            dict_waves_info: {"uuid": uuid, "type": measure_type, "p": pidxs, "q": qidxs, "r": ridxs, "s": sidxs, "t": tidxs, "ecg": ecgs_3000}      
        """
        # ECG signal processing
        srj_ecgs = delete_impulse_(ecg)
        srj_ecgs_vg = nk.ecg_clean(srj_ecgs, sampling_rate=250, method='vg')  # 20240221 新增
        srj_ecgs_nk_ = nk.ecg_clean(srj_ecgs, sampling_rate=250,method="neurokit")  # 20231031 新增  nk.ecg_clean 已經處理基線飄移基線準位略小於0
        srj_ecgs_nk = baseline.baseline_remove(srj_ecgs_nk_,processnum=1)  # 重新校正基線以0為準 # Wayne: 改用swmlib (datatype不同 float變int32)
        
        # Jane新增：用處理後的 ECG 判斷低振幅
        # ecg_max = np.max(srj_ecgs_vg)

        # if ecg_max < 80:
        #     print(f"Skip low amplitude ECG after cleaning: {uuid}, max={ecg_max}")
        #     return None

        #srj_ecgs_vg = np.array(fix_scale(srj_ecgs_vg)).astype('int')
        srj_ecgs_vg = np.array(srj_ecgs_vg).astype('int')
        ecgs = srj_ecgs_vg.reshape(-1, 2500)
        ecgs_3000_vg = np.zeros((len(ecgs), 3000))  # ECG顯示: 前一段250點訊號+目前2500點訊號+後一段250點訊號 (目的是改善邊界問題)
        ecgs_3000_vg[1:, :250] = ecgs[:-1, 2250:]
        ecgs_3000_vg[:, 250:-250] = ecgs
        ecgs_3000_vg[:-1, 2750:] = ecgs[1:, :250]

        # Jane新增：用處理後的 ECG 判斷低振幅
        # ecg_max = np.max(srj_ecgs_nk)

        # if ecg_max < 80:
        #     print(f"Skip low amplitude ECG after cleaning: {uuid}, max={ecg_max}")
        #     return None

        #srj_ecgs_nk = np.array(fix_scale(srj_ecgs_nk)).astype('int')  ##切分成2500個點的ECG片段，片段中最大值介於10~500者，此片段訊號值將被放大到0~500，太小或太大的片段維持不變
        srj_ecgs_nk = np.array(srj_ecgs_nk).astype('int')
        ecgs2 = srj_ecgs_nk.reshape(-1, 2500)  ##改為每2500點為一個片段結構
        ecgs_3000 = np.zeros((len(ecgs2), 3000)) ## create a structure that have the same number of rows as in ecgs2, but have 3000 columns
        ecgs_3000[1:, :250] = ecgs2[:-1, 2250:]  ## Adds a 250-point buffer before each ECG segment, taken from the end of the previous segment.
        ecgs_3000[:, 250:-250] = ecgs2           ## copy the main 2500-sample ECG segments into the center of each 3000-point row in ecgs_3000
        ecgs_3000[:-1, 2750:] = ecgs2[1:, :250]  ## Adds a 250-point buffer after each ECG segment, taken from the start of the next segment.

        # Detect Peak Location
        rpeaks = []
        for ecg_nk, ecg_vg in zip(ecgs_3000, ecgs_3000_vg):
            try:
                # 20240222 vg filter + nk r-peak detect
                _, info_vg_nk = nk.ecg_peaks(ecg_vg, sampling_rate=250, correct_artifacts=False, show=False)
                rpeak_vg_nk = check_r_peak(ecg_nk, info_vg_nk['ECG_R_Peaks'])
                rpeaks.append(np.array(rpeak_vg_nk).astype('int'))

                # plot and check
                # plt.figure(figsize=(8, 3))
                # plt.plot(ecgs_3000.T, label='Raw ECG', c='black')
                # plt.plot(ecg_vg, label='vg', c='green')
                # plt.plot(ecg_nk, label='nk', c='blue')
                # plt.scatter(rpeaks, ecg_vg[(rpeaks)], label='vg', c='green')
                # plt.scatter(info_vg_nk['ECG_R_Peaks'], ecg_nk[(info_vg_nk['ECG_R_Peaks'])], label='nk', marker='x', c='blue')
                # plt.legend()
                # plt.show()

            except IndexError:  # 沒有r peak
                # errorcode = -202
                # message = ''
                # print("No r peak detection")
                rpeaks.append(np.array([]).astype('int'))

        # Peaks record
        pidxs = []
        qidxs = []
        ridxs = []
        sidxs = []
        tidxs = []

        for i, (ecg, rpeak) in enumerate(zip(ecgs_3000, rpeaks)):
            pidx = []
            qidx = []
            ridx = []
            sidx = []
            tidx = []

            ecgs_for_waves = [np.array(ecg), np.zeros(len(ecg))]
            for rpeak_i in rpeak:
                ecgs_for_waves[1][rpeak_i] = 1
  
            waves_dict = self.detect_pqrst_waves_v3(np.array(ecgs_for_waves).T, 250, np.array(rpeak), mode='new', FalseR=False)
            
            for p, q, r, s, t in zip(waves_dict['Pidx'], waves_dict['Qidx'], waves_dict['Ridx'], waves_dict['Sidx'],
                                     waves_dict['Tidx']):
                if 250 <= r < 2750:
                    pidx.append(int(p - r) if str(p) != "nan" else 0)
                    qidx.append(int(q - r) if str(q) != "nan" else 0)
                    sidx.append(int(s - r) if str(s) != "nan" else 0)
                    tidx.append(int(t - r) if str(t) != "nan" else 0)
                    ridx.append(int(r - 250))
            
            # Jane新增
            # if len(ridx) > 2:
            #     pidx = pidx[1:]
            #     qidx = qidx[1:]
            #     ridx = ridx[1:]
            #     sidx = sidx[1:]
            #     tidx = tidx[1:]

            pidxs.append(pidx)
            qidxs.append(qidx)
            ridxs.append(ridx)
            sidxs.append(sidx)
            tidxs.append(tidx)

            '''
            # Jane測試
            # Plot R/T detection result
            # Plot R/T detection result as interactive HTML
            if plot_dir is not None:
                os.makedirs(plot_dir, exist_ok=True)

                curr_ecg = np.array(ecg[250:2750])
                x = np.arange(len(curr_ecg))

                r_abs = np.array(ridx, dtype=int)
                t_abs = r_abs + np.array(tidx, dtype=int)
                p_abs = r_abs + np.array(pidx, dtype=int)
                q_abs = r_abs + np.array(qidx, dtype=int)
                s_abs = r_abs + np.array(sidx, dtype=int)

                # keep valid points only
                r_valid = (r_abs >= 0) & (r_abs < len(curr_ecg))
                t_valid = (t_abs >= 0) & (t_abs < len(curr_ecg)) & (np.array(tidx) != 0)
                p_valid = (p_abs >= 0) & (p_abs < len(curr_ecg)) & (np.array(pidx) != 0)
                q_valid = (q_abs >= 0) & (q_abs < len(curr_ecg)) & (np.array(qidx) != 0)
                s_valid = (s_abs >= 0) & (s_abs < len(curr_ecg)) & (np.array(sidx) != 0)


                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=x,
                    y=curr_ecg,
                    mode='lines',
                    name='ECG'
                ))

                fig.add_trace(go.Scatter(
                    x=r_abs[r_valid],
                    y=curr_ecg[r_abs[r_valid]],
                    mode='markers',
                    name='R peak',
                    marker=dict(size=8, color='red'),
                    hovertemplate='R index: %{x}<br>Amplitude: %{y}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=t_abs[t_valid],
                    y=curr_ecg[t_abs[t_valid]],
                    mode='markers',
                    name='T peak',
                    marker=dict(size=8, color='blue'),
                    hovertemplate='T index: %{x}<br>Amplitude: %{y}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=p_abs[p_valid],
                    y=curr_ecg[p_abs[p_valid]],
                    mode='markers',
                    name='p peak',
                    marker=dict(size=8, color='orange'),
                    hovertemplate='p index: %{x}<br>Amplitude: %{y}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=q_abs[q_valid],
                    y=curr_ecg[q_abs[q_valid]],
                    mode='markers',
                    name='q peak',
                    marker=dict(size=8, color='green'),
                    hovertemplate='q index: %{x}<br>Amplitude: %{y}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=s_abs[s_valid],
                    y=curr_ecg[s_abs[s_valid]],
                    mode='markers',
                    name='s peak',
                    marker=dict(size=8, color='purple'),
                    hovertemplate='s index: %{x}<br>Amplitude: %{y}<extra></extra>'
                ))

                fig.update_layout(
                    title=f'{uuid}_{measure_type}_segment_{i} | R/T detection',
                    xaxis_title='Sample index',
                    yaxis_title='Amplitude',
                    hovermode='closest'
                )
                
                save_path = os.path.join(
                    plot_dir,
                    f'{file_name.replace(".txt","")}_RT_check.html'
                )
               
                fig.write_html(save_path)
            '''
        ecgs_3000 = [[i for i in ecg] for ecg in ecgs_3000]

        # Plot and check
        # stable_p, p_idxs = features_extraction.detect_peak_stability(pidxs)
        # stable_q, q_idxs = features_extraction.detect_peak_stability(qidxs)
        # stable_s, s_idxs = features_extraction.detect_peak_stability(sidxs)
        # stable_t, t_idxs = features_extraction.detect_peak_stability(tidxs)
        #
        # plt.figure(figsize=(12, 4))
        # curr_ecg = np.array(ecgs_3000[0][250:])
        # plt.plot(curr_ecg, c='black')
        # plt.scatter(np.array(pidx)+np.array(ridx), curr_ecg[np.array(pidx)+np.array(ridx)], label='p')
        # plt.scatter(np.array(qidx)+np.array(ridx), curr_ecg[np.array(qidx)+np.array(ridx)], label='q')
        # plt.scatter(ridx, curr_ecg[ridx], label='r')
        # plt.scatter(np.array(sidx)+np.array(ridx), curr_ecg[np.array(sidx)+np.array(ridx)], label='s')
        # plt.scatter(np.array(tidx)+np.array(ridx), curr_ecg[np.array(tidx)+np.array(ridx)], label='t')
        # plt.legend()
        # plt.title("p score={}, q score={}, s score={}, t_score={}".format(stable_p, stable_q, stable_s, stable_t))
        # plt.show()

        dict_waves_info = {"uuid": uuid, "type": measure_type, "p": pidxs, "q": qidxs, "r": ridxs, "s": sidxs, "t": tidxs, "ecg": ecgs_3000}
    
        
        return dict_waves_info

