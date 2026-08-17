# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 15:02:27 2025

@author: jane
"""

from scipy import signal
import os, gc
import glob
import numpy as np
import pandas as pd

import sys
import shutil
from SWMlib.ecg import baseline
from SWMlib.ecg.rpeak import rpeak_detection 
from scipy.ndimage import gaussian_filter1d



       
def _roi(ecg, rpeak, alpha_start, fs=250, sigma_gauss=0, threshold=0):

    """ 
    input---
      ecg         : 傳入1-D ecg array
      fs          : 取樣頻率
      alpha_start : ROI的開始
      sigma_gauss : 高斯濾波的參數
      threshold   : 用來調整ROI範圍
      
    output---
      roi_df_all : 逐拍結果，一拍一列 
      sig_df_all : 整段訊號，一段一列 
    """
   
    alpha_end=0.50
    
    # 1) 準備ECG: stem(訊號欄位名稱)、ecg_data(10秒訊號)
    df = pd.DataFrame([{
        "stem"     : "single",
        ##"ecg_data" : np.array(ecg[:, 0])  ###dennis修改
        "ecg_data" : np.asarray(ecg, dtype=float).ravel()
        }])
    
    
    all_roi_rows = []   # 收集所有 stem 的逐拍表
    all_sig_rows = []   # 收集每個 stem 的整段訊號
    
    
    for _, row in df.iterrows():
        stem = row["stem"]
        ecg_raw = np.asarray(row["ecg_data"], dtype=float).ravel()              

        # 2) 將訊號高斯濾波、基線拉直
        if(sigma_gauss != 0):
            ecg_filtered = _gaussian_smooth_1d(ecg_raw, fs=fs, sigma=sigma_gauss)
            ecg_filtered = np.asarray(ecg_filtered, dtype=float).ravel()
            ecg_filtered = baseline.baseline_remove(ecg_filtered, processnum=1)  # 重新校正基線以0為準
        else:            
            ecg_filtered = ecg_raw

        n_filter   = len(ecg_filtered)
        

        # 3) 設定threshold: 振幅太低不可能會是T波的ROI區域，因此設定一個threshold來限縮ROI
        if(threshold !=0):
            threshold_filter = threshold * np.mean(ecg_filtered)   
        
        ## 先存起來
        all_sig_rows.append({
            "stem"       : stem,
            "ecg_filter" : ecg_filtered,
        })
        
         
        # 4) 如果R peak為空，則偵測R peak並微調rpeaks，否則直接微調rpeaks
        
        if rpeak is None or len(rpeak) == 0:
            rpeak_array  = rpeak_detection(ecg_filtered,measuring_mode='strap',method_type='vg',mode='original') 
            rpeaks_filtered       = rpeak_array[1:].astype(int)  #rpeak_detection第一個位置是r波個數，之後才是r波位置
            rpeaks_filtered       = _revised_rpeaks(ecg_filtered, rpeak_array[1:].astype(int),radius=10) ##rpeaks_filtered, radius=10)
        
        else:            
            rpeaks_filtered = _revised_rpeaks(ecg_filtered, rpeak, radius=10)
               
        
       
        # 5) 框出ROI(因為要逐拍處理)
        ## 透過R peak找出RRI，並接著框出ROI
        n_beats = max(0, len(rpeaks_filtered) - 1)    
            
        for j in range(n_beats):
            r0_f = int(rpeaks_filtered[j])
            r1_f = int(rpeaks_filtered[j+1])
            RR_f = r1_f - r0_f
        
            start = r0_f + int(alpha_start)
            end   = r0_f + int(alpha_end * RR_f)
            start = max(0, start)
            end   = min(n_filter, end)
            
            if end <= start:
                continue
            
            seg_filtered = ecg_filtered[start:end]
           
            
            ## 限縮ROI範圍: 設定threshold
            if threshold!=0 and seg_filtered.size > 0:
                mask_f = (seg_filtered >= threshold_filter)
                 
                if np.any(mask_f):
                    idxf         = np.where(mask_f)[0]
                    base         = start_filter
                    start_filter = base + int(idxf[0])
                    end_filter   = base + int(idxf[-1]) + 1
            
            
            all_roi_rows.append({
               'stem'                : stem,
               'beat_idx'            : j,
               'r0_filter'           : int(r0_f),
               'r1_filter'           : int(r1_f), 
               'start_filter'        : int(start),
               'end_filter'          : int(end),
  
                }) 
       

    ## 串所有 stem 
    if not all_roi_rows:
        roi_df_all = pd.DataFrame()
    else:
        roi_df_all = pd.DataFrame(all_roi_rows)

    sig_df_all = pd.DataFrame(all_sig_rows)

    return roi_df_all, sig_df_all


def _revised_rpeaks(ecg, rpeaks, radius=10):
    
    """ 
    微調 R peak 位置，使其為局部最大值所在位置

    input---
        ecg    : ecg
        rpeaks : rpeak的位置
        radius : 設定要微調的範圍（±radius）
        
    output---
        revised_rpeaks: 調整後的rpeak     
    """
    
    # 1) 先抓出rpeak位置
    n = len(ecg)
    rpeaks = np.asarray(rpeaks, dtype=int).ravel()
    revised = np.empty_like(rpeaks)    # 預先建立一個與rpeaks同長度的陣列，用來裝調整後的新R位置

    # 2) 分別將每個rpeak左右框出一個視窗，判斷視窗內是否有最大值
    for k, r in enumerate(rpeaks):    # 可以同時取得元素的索引和值。k 是第幾個 R（for 迴圈的索引），r 是當前 R 的樣本位置。
        ## 這樣就會得到一段長度約2*radius+1的小窗，包住原本的r點
        a = max(0, r - radius)
        b = min(n, r + radius + 1)  # 右界不含
        segment = ecg[a:b]
        if segment.size == 0:    # 如果因為極端邊界導致片段是空的，就維持原 R，不做微調，直接處理下一個 R
            revised[k] = r
            continue
        j = int(np.argmax(segment))    # 找到最大值
        revised[k] = a + j

    return revised

    
def _gaussian_smooth_1d(ecg, sigma, fs, mode="reflect"):
    
    """
    高斯濾波
    
    input---
        ecg   : ecg_filter
        sigma : 標準差（單位=樣本）
        fs    : 取樣頻率
        mode  : 遇到邊界的處理方式
    
    output---
        濾波完的整條ecg signal
    """
    
    ecg = np.asarray(ecg, dtype=float).ravel()
    gaussian_filter = gaussian_filter1d(ecg, sigma=float(sigma), mode=mode)
    
    return gaussian_filter


def _max4(seg, window, min_frac):
    
    """  
    在每一個ROI裡面判斷高峰，並找出候選T peak
    
    input---
        seg      : segment(ROI)
        window   : 設定window用來找候選peak
        min_frac : 設定振幅門檻，避免找到的峰值太小
    
    output--- 
        list[int]，長度為1或2的tpeak index
    """
    
    y = np.asarray(seg, dtype=float).ravel()
    n = y.size
    
    if n == 0:
        raise ValueError("seg is empty")
    
    # if n <= 2:
    #     raise ValueError("seg error")
        
    # 1) 先找seg局部最大
    cand_idx = []
    for i in range(n):
        left  = max(0, i - window)
        right = min(n, i + window + 1)  # 右界不含
        local_max = np.max(y[left:right]) # 這一段的最大值
        
        ## 如果這個點就是區段裡的最大值，就當作候選peak
        if y[i] == local_max:
            cand_idx.append(i)
            
    if len(cand_idx)  == 0:
        ## 找不到局部最大，就退回整段最大
        return [int(np.argmax(y))]
    
    cand_idx  = np.asarray(cand_idx, dtype=int)
    cand_amps = y[cand_idx]
       
    # 2) 振幅門檻: 用前4大振幅的平均(可以避免T peak打在太前面，太前面會導致和R的距離在這一拍突然縮短)
    order_desc = np.argsort(cand_amps)[::-1]  # 降冪排序振幅
    k          = min(4, cand_amps.size)       # 取前4個，不夠就全部
    topk_amps  = cand_amps[order_desc[:k]]    # 前k大的振幅
    base_amp   = float(np.mean(topk_amps))    # 這些的平均當作基準振幅
    thr        = base_amp * float(min_frac)  
    keep       = cand_amps >= thr
    
    cand_idx = cand_idx[keep]
    if cand_idx.size == 0:
        ## 全部都被門檻刪掉，那就退回整段最大
        return [int(np.argmax(y))]
    
    # 3) 按時間排序，取前兩個峰，並作為兩個候選T peak
    cand_idx = np.sort(cand_idx)
    if cand_idx.size >= 2:
        return [int(cand_idx[0]), int(cand_idx[1])]
    else:
        return [int(cand_idx[0])]   
          
            
def _local_area(y, center, area_win):    

    """
    計算高峰的面積
    """
    
    # 以高峰為中心，左右框出一個視窗，並加總視窗範圍內的所有絕對值振幅
    left  = max(0, center - area_win)
    right = min(y.size, center + area_win + 1)
    return float(np.sum(np.abs(y[left:right])))    


def _pipeline(ecg, start, end, fs):
    
    """
    這是基本的pipeline，對每一ROI進行計算，決定t peak
    
    input---
        ecg   : ecg_filter
        start : seg的開始
        end   : seg的結束
        fs    : 取樣頻率
    
    output---
        tpeak_abs : tpeak的絕對位置
        tpeak_amp : tpeak的振幅
        rule      : 偵測tpeak的規則
    """
    
    ## 先擋掉 None / NaN
    if start is None or end is None:
        return None
    
    ## 有些情況是 float('nan')
    if (isinstance(start, float) and np.isnan(start)) or (isinstance(end, float) and np.isnan(end)):
        return None
    
    # 1) 找到該ECG的ROI
    ecg = np.asarray(ecg, dtype=float).ravel()
    n = len(ecg)
   
    start = max(0, int(start))
    end   = min(n, int(end))
    
    if end <= start:
        return None
    
    seg = ecg[start:end]
    #print(len(seg), seg)

    if seg.size < 3:
        print(f"seg<3無法做gradient，因此跳過此拍{ecg}")
        return None
    
    # 2) 將ROI計算曲率並正規化
    d1 = np.gradient(seg, 1/fs)
    d2 = np.gradient(d1, 1/fs)
    k  = d2 / (1 + d1**2)**1.5
    eps = 1e-12
    k_norm = k / (np.max(np.abs(k)) + eps)
    
    
    # 3) 在ROI內找到兩個最高峰作為候選T peak                
    peaks = _max4(seg, window=2, min_frac=0.9)    
    peaks = np.asarray(peaks, dtype=int)    
    #print("peaks:", peaks)  
        
    if peaks.size == 0:
        ## 理論上 _max4 至少會給 1 個，但保險起見
        tpeak_roi = int(np.argmax(seg))
        rule      = "no_peak_instead_max"
    else:    
        amps = seg[peaks]    # 高峰
        #print("amps:", amps)
        order = np.argsort(amps)[::-1]    # 將高峰排序由大到小
        #print("order:", order)
              
        
        # 4) 如果有高峰的話，執行以下判斷決定T peak要打在former或max      
        if order.size >= 2:
            ## 取振幅前兩大，再依時間排序
            top2 = peaks[order[:2]]
            top2.sort()
            p1, p2 = int(top2[0]), int(top2[1])
            #print("p1, p2:", p1, p2)        
              
            ## 面積判斷(為避免兩峰的面積差太多，導致T peak與R距離不穩定)
            ## 例如: 高峰1面積<<高峰2面積，T peak如果偵測在former(高峰1)的話，與R的距離會縮短
            area1 = _local_area(seg, p1, area_win=5)
            area2 = _local_area(seg, p2, area_win=5)
            #print("area1, area2:", area1, area2)
        
            ratio = np.inf
            if area1 > 0 and area2 > 0:
                ratio = area2 / area1
        
            ## 判斷兩峰面積是否差不多
            if (0.6 <= ratio <= 1.5): ## 表示面積相近 
                pos = (k_norm > 0)    ## 如果有正曲率
                if np.any(pos):               
                    ## 找到最大正曲率索引(可能不只一個)
                    max_val    = np.max(k_norm[pos])  
                    kmax_idxs  = np.where(k_norm == max_val)[0]
                else:
                    kmax_idxs  = np.array([], dtype=int)                   
            
                ## 偵測T peak: 如果是高峰-波谷-高峰 -> former，否則最高峰
                if (area1 >= area2) and (kmax_idxs.size > 0) and np.any((kmax_idxs > p1) & (kmax_idxs < p2)):
                    tpeak_roi = p1
                    #print(tpeak_roi)
                    rule = "former_notch"
                else:
                    tpeak_roi = int(peaks[order[0]])
                    rule = "max_wave_area_balanced"
            else:    ## 面積差很多，就不硬打前峰，直接最高峰
                tpeak_roi = int(peaks[order[0]])
                rule = "max_wave_area_unbalanced"                
                 
        else:  # 如果peak<2，那就選擇最高峰
             tpeak_roi = int(peaks[order[0]])
             rule = "single_peak"
            
    tpeak_abs = start + int(tpeak_roi)    # 轉回絕對索引
    tpeak_abs = int(min(max(0, tpeak_abs), n - 1))    # 邊界保護
    tpeak_amp = float(ecg[tpeak_abs])    # 取出振幅
         
    return tpeak_abs, tpeak_amp, rule     


def _pipeline_force(ecg, start, end, fs, mode):
    
    """
    這是強制執行的pipeline，強制將T peak偵測在former和max
        
    input---
        ecg   : ecg_filter
        start : seg的開始
        end   : seg的結束
        fs    : 取樣頻率 
        mode  : 強制T的選擇，former/max
        
    output---
        tpeak_abs : tpeak的絕對位置
        tpeak_amp : tpeak的振幅
        rule      : 偵測tpeak的規則
    """
    
    ## 先擋掉 None / NaN
    if start is None or end is None:
        return None
    
    ## 有些情況是 float('nan')
    if (isinstance(start, float) and np.isnan(start)) or (isinstance(end, float) and np.isnan(end)):
        return None
    
    # 1) 將ROI抓出來
    ecg = np.asarray(ecg, dtype=float).ravel()
    n = len(ecg)
    start = max(0, int(start))
    end   = min(n, int(end))
    
    if end <= start:
        return None
    
    seg = ecg[start:end]
    if seg.size == 0:
        return None
    
    
    if seg.size < 3:
        print(f"seg<3，因此跳過此拍{ecg}")
        return None
    

    # 2) ROI內兩個最高峰                
    peaks = _max4(seg, window=2, min_frac=0.9)    # 找到時間框內的高峰
    peaks = np.asarray(peaks, dtype=int)
    #print("force_peaks:", peaks)  
        
    if peaks.size == 0:
        # 理論上 _max4 至少會給 1 個，但保險起見
        tpeak_roi = int(np.argmax(seg))
    else: 
        if mode == "former":   # 兩峰選時間比較早那個；只有一峰就用那一峰
            tpeak_roi = int(np.min(peaks))
        elif mode == "max":
            tpeak_roi = int(np.argmax(seg))
        else:
            raise ValueError(f"Unknown mode for pipeline_force: {mode}")
    
            
    tpeak_abs = start + int(tpeak_roi)    # 轉回絕對索引
    tpeak_abs = int(min(max(0, tpeak_abs), n - 1))    # 邊界保護
    tpeak_amp = float(ecg[tpeak_abs])    # 取出振幅
    
    # 3) 區分為former/max    
    if mode == "former":
        rule = "no_matter_former"
    else:
        rule = "no_matter_max"
    
    return tpeak_abs, tpeak_amp, rule 


def _t_peak(sig_df, roi_df, fs=250):
    
    """
    這是基本的偵測T peak
    對每一個ROI，去偵測T peak
    
    input---
        sig_df : 一個訊號
        roi_df : 一拍一列的訊號
        fs     : 取樣頻率
        
    output---
        每拍一列，包含
          - filter: tpeak_abs_filter, tpeak_amp_filter, rule_filter
        以及原本的 ROI 資訊 (r0_filter, start_filter, ...)
    """
    
    # 1) 建立字典    
    filter_map = {row["stem"]: np.asarray(row["ecg_filter"], dtype=float).ravel()
               for _, row in sig_df.iterrows()}
    
    
    out = []
    for _, rr in roi_df.iterrows():
        stem            = rr["stem"]
        beat_idx        = int(rr["beat_idx"])
        ecg_filter_arr  = filter_map.get(stem)
        
        
        ## 如果拿不到訊號就跳過這拍
        if ecg_filter_arr is None:
            continue
         
            
        ## 有些欄位可能是nan，要確保下一步不會拿到nan
        def _safe_get_int(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return int(x)
        
        
        ## 把這一拍的filter讀出來
        r0_filter = _safe_get_int(rr.get("r0_filter"))
        r1_filter = _safe_get_int(rr.get("r1_filter"))

        s_filter  = _safe_get_int(rr.get("start_filter"))
        e_filter  = _safe_get_int(rr.get("end_filter"))
        
        
        ## 一拍的基本輸出欄位
        row_out = {
           "stem"          : stem,
           "beat_idx"      : beat_idx,
           "r0_filter"     : np.nan if r0_filter is None else int(r0_filter),
           "r1_filter"     : np.nan if r1_filter is None else int(r1_filter),
           "start_filter"  : np.nan if s_filter  is None else int(s_filter),
           "end_filter"    : np.nan if e_filter  is None else int(e_filter),

           "tpeak_abs_filter": np.nan,
            "tpeak_amp_filter": np.nan,
            "rule_filter"     : None
       }
        
        
        # 2) 偵測T peak   
        ret_filter = _pipeline(ecg_filter_arr, s_filter, e_filter, fs)
        if ret_filter is not None:
            t_abs, t_amp, rule = ret_filter
            
            row_out.update({
                "tpeak_abs_filter": int(t_abs),
                "tpeak_amp_filter": float(t_amp),
                "rule_filter"     : rule                
                })
            
        out.append(row_out)
        
    return pd.DataFrame(out)


def _t_peak_force(sig_df, roi_df, mode, fs = 250):
    
    """
    這是強制執行的偵測T peak
    (逐拍重新偵測T peak)
    
    input---
        sig_df : 一個訊號
        roi_df : 一拍一列的訊號
        mode   : former/ max
        fs     : 取樣頻率
        
    output---
        每拍一列，包含
          - filter: tpeak_abs_filter, tpeak_amp_filter, rule_filter
        以及原本的 ROI 資訊 (r0_filter, start_filter, ...)
    """
    
    ## 建立字典   
    filter_map = {row["stem"]: np.asarray(row["ecg_filter"], dtype=float).ravel()
               for _, row in sig_df.iterrows()}
    
    
    out = []
    for _, rr in roi_df.iterrows():
        stem            = rr["stem"]
        beat_idx        = int(rr["beat_idx"])
        ecg_filter_arr  = filter_map.get(stem)
        
        row_out = {
            "stem": stem,
            "beat_idx": beat_idx,

            "tpeak_abs_filter": np.nan,
            "tpeak_amp_filter": np.nan,
            "rule_filter"     : None

        }
        
            
        ## 有些欄位可能是nan，要確保下一步不會拿到nan
        def _safe_get_int(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return int(x)
        
        
        if ecg_filter_arr is not None:
            s_filter = _safe_get_int(rr.get("start_filter"))
            e_filter = _safe_get_int(rr.get("end_filter")) 
            ret_filter = _pipeline_force(
                ecg_filter_arr, s_filter, e_filter, fs,
                mode=mode)
            
            if ret_filter is not None:
                t_abs, t_amp, rule = ret_filter
                row_out.update({
                    "tpeak_abs_filter" : int(t_abs),
                    "tpeak_amp_filter" : float(t_amp),
                    "rule_filter"      : rule                
                    })
            
        out.append(row_out)
        
    return pd.DataFrame(out)


def _compute_rt_distances(tpeaks_df, fs):
    
    """
    逐拍計算R-T水平距離，單位是豪秒，再做統計
    
    input--- 
        tpeaks_df : 每一拍的rt相關資訊
        fs       : 取樣頻率
        
    output---
      1) df_with_dt：逐拍結果（tpeaks_df 加上 dt_filter_ms 欄位）
      2) summary_dt：每個 stem 的統計表:有效拍數、平均值、標準差
    """
    
    df = tpeaks_df.copy()
   
    # 1) 計算rt距離
    m_flt = (~pd.isna(df.get("tpeak_abs_filter"))) & (~pd.isna(df.get("r0_filter")))
   
    df["dt_filter_sec"] = np.where(
        m_flt,
        (df["tpeak_abs_filter"].astype(float) - df["r0_filter"].astype(float)) / float(fs), np.nan
        )      
    
    ## 換成毫秒
    df["dt_filter_ms"] = df["dt_filter_sec"] * 1000.0
    
    # 2) summarize: 把逐拍合併成一個10秒訊號，才可算平均值與標準差
    grp = df.groupby("stem", dropna=False)    # 依照stem去分組
    summary = pd.DataFrame({
        
      ## 有效拍數
      "rt_filter_n":     grp["dt_filter_ms"].count(),
      
      ## 平均與標準差
      "rt_filter_mean_ms" : grp["dt_filter_ms"].mean(),
      "rt_filter_std_ms"  : grp["dt_filter_ms"].std(),
    }).reset_index()

    return df, summary


def _rename_rt_cols(df, suffix):
    
    """
    合併 beats: 三個版本(baseline/former/max)，可以去看逐拍的T位置、振福、偵測結果以及RT距離
    """
    
    col_map = {}
    for base in [
        "tpeak_abs_filter", "tpeak_amp_filter", "rule_filter", "dt_filter_ms",
    ]:
        if base in df.columns:
            col_map[base] = f"{base}_{suffix}"
    return df.rename(columns=col_map)


def _rename_summary_cols(df, suffix):
    
    """
    合併訊號：三版本(baseline/former/max)的 mean/std，可以去看一個10秒訊號的RT平均值與標準差
    """
    
    col_map = {}
    for base in [
        "rt_filter_n",
        "rt_filter_mean_ms", "rt_filter_std_ms",
    ]:
        if base in df.columns:
            col_map[base] = f"{base}_{suffix}"
    return df.rename(columns=col_map)




def splitted_t_wave_detector(ecg, fs=250, rpeak=[]):
    
    """
    進行ECG訊號的T wave peak偵測(T波可能有分裂之情況)，確保偵測出來的T peak與R peak距離是更stable (用RT距離的標準差來判斷)
         
    input---
        ecg (ndarray): The 1st column is the value of ecg, and the 2nd column is Rpeak locations.
        fs  : 取樣頻率
        
    output---
        t相對於r的位置的array
    """
    
    std_thr=18.0
    alpha_start = (fs / 250) * 20
    
    # 1) 針對輸入的心電圖進行RR interval中間感興趣ROI偵測，幫助T波位置偵測時考慮正確範圍
    roi_df, sig_df = _roi(ecg, rpeak, alpha_start=alpha_start,fs=fs)
   
    # 2) 針對每個ROI範圍內，偵測T peak位置，並計算RT距離
    tpeaks_base = _t_peak(sig_df, roi_df, fs=fs)
        
    df_base, summary_base = _compute_rt_distances(tpeaks_base,fs=fs)
   
    # 3) 透過比較std (標準差)去決定是否stable。如果RT距離不stable，就會需要強制產生兩個版本(former/max)
    bad_filter_stems = summary_base.loc[summary_base["rt_filter_std_ms"] > std_thr, "stem"].astype(str).tolist()

    bad_filter_set = set(bad_filter_stems)
   
    final_source = "baseline" # 預設使用 baseline
      
    if bad_filter_set:
        # 3A) 強制 former 版本
        tpeaks_former = tpeaks_base.copy()
        roi_bad_filter = roi_df[roi_df["stem"].isin(bad_filter_set)]
        #print("roi_bad_filter", roi_bad_filter)
        forced_filter_former = _t_peak_force(
            sig_df=sig_df,
            roi_df=roi_bad_filter,
            mode="former",
            fs=fs
        )

        tpeaks_former = tpeaks_former.merge(
            forced_filter_former[["stem", "beat_idx",
                                  "tpeak_abs_filter", "tpeak_amp_filter", "rule_filter"]],
            on=["stem", "beat_idx"],
            how="left",
            suffixes=("", "_forced_filter_former"),
        )
        m = tpeaks_former["tpeak_abs_filter_forced_filter_former"].notna()
        for col in ["tpeak_abs_filter", "tpeak_amp_filter", "rule_filter"]:
            forced_col = col + "_forced_filter_former"
            tpeaks_former.loc[m, col] = tpeaks_former.loc[m, forced_col]
            tpeaks_former.drop(columns=[forced_col], inplace=True)
        
        df_former, summary_former = _compute_rt_distances(tpeaks_former, fs=fs)

        # 3B) 強制 max 版本
        tpeaks_max    = tpeaks_base.copy()
        roi_bad_filter = roi_df[roi_df["stem"].isin(bad_filter_set)]
        forced_filter_max = _t_peak_force(
            sig_df=sig_df,
            roi_df=roi_bad_filter,
            mode="max",
            fs=fs
        )

        tpeaks_max = tpeaks_max.merge(
            forced_filter_max[["stem", "beat_idx",
                               "tpeak_abs_filter", "tpeak_amp_filter", "rule_filter"]],
            on=["stem", "beat_idx"],
            how="left",
            suffixes=("", "_forced_filter_max"),
        )
        m = tpeaks_max["tpeak_abs_filter_forced_filter_max"].notna()
        for col in ["tpeak_abs_filter", "tpeak_amp_filter", "rule_filter"]:
            forced_col = col + "_forced_filter_max"
            tpeaks_max.loc[m, col] = tpeaks_max.loc[m, forced_col]
            tpeaks_max.drop(columns=[forced_col], inplace=True)
        df_max,    summary_max    = _compute_rt_distances(tpeaks_max,    fs=fs)
    
    
        # 4) 逐拍合併三個版本
        ## 幫欄位名稱加上suffix(_base/ _former/ _max)
        ## baseline
        base_cols = [
            "stem", "beat_idx",
            "r0_filter",
            "tpeak_abs_filter", "tpeak_amp_filter", "rule_filter", "dt_filter_ms",
        ]
        base_cols = [c for c in base_cols if c in df_base.columns]
        df_base_sel = df_base[base_cols].copy()
        df_base_sel = _rename_rt_cols(df_base_sel, "base")  # 不會 rename r0，因為r0都不變

        ## former / max
        df_former_sel = _rename_rt_cols(
            df_former[[
                "stem", "beat_idx",
                "tpeak_abs_filter", "tpeak_amp_filter", "rule_filter", "dt_filter_ms",
            ]],
            "former",
        )

        df_max_sel = _rename_rt_cols(
            df_max[[
                "stem", "beat_idx",
                "tpeak_abs_filter", "tpeak_amp_filter", "rule_filter", "dt_filter_ms",
            ]],
            "max",
        )
        
        ## 把baseline/ former/ max合併成一張beats表
        beats_merged = df_base_sel.merge(
            df_former_sel, on=["stem", "beat_idx"], how="left"
        ).merge(
            df_max_sel, on=["stem", "beat_idx"], how="left"
        )
        #print("beats_merged", beats_merged)
        '''
        ## 把「沒有被 force 的 stem」的 former / max 欄位改成 NaN (因為前面是用baseline來複製的)
        good_filter_stems = set(summary_base["stem"].astype(str)) - bad_filter_set

        ## 這些 stem 的 former/max filter 欄位設為 NaN
        mask_good_filter = beats_merged["stem"].astype(str).isin(good_filter_stems)
        flt_former_cols = [c for c in beats_merged.columns
                        if c.startswith("tpeak_abs_filter_former")
                        or c.startswith("tpeak_amp_filter_former")
                        or c.startswith("rule_filter_former")
                        or c.startswith("dt_filter_ms_former")]
        flt_max_cols = [c for c in beats_merged.columns
                        if c.startswith("tpeak_abs_filter_max")
                        or c.startswith("tpeak_amp_filter_max")
                        or c.startswith("rule_filter_max")
                        or c.startswith("dt_filter_ms_max")]
        if flt_former_cols:
            beats_merged.loc[mask_good_filter, flt_former_cols] = np.nan
        if flt_max_cols:
            beats_merged.loc[mask_good_filter, flt_max_cols] = np.nan
        '''
        
        # 5) 合併三版本訊號的mean/std 
        sum_base   = _rename_summary_cols(summary_base,  "base")
        sum_former = _rename_summary_cols(summary_former, "former")
        sum_max    = _rename_summary_cols(summary_max,    "max")

        summary_merged = sum_base.merge(
            sum_former, on="stem", how="left"
        ).merge(
            sum_max, on="stem", how="left"
        )
      
        '''
        ## 也把沒有被 force 的 stem 的 summary former/max 改成 NaN

        mask_good_filter = summary_merged["stem"].astype(str).isin(good_filter_stems)
        flt_sum_cols = [c for c in summary_merged.columns
                        if "rt_filter_" in c and (c.endswith("_former") or c.endswith("_max"))]
        if flt_sum_cols:
            summary_merged.loc[mask_good_filter, flt_sum_cols] = np.nan
        '''
        
        # 6) 呼叫build_final_summary: 決定每個訊號最終採用的版本
        final_summary = build_final_summary(summary_merged)
        stem_name = sig_df["stem"].iloc[0]
        final_row = final_summary.loc[final_summary["stem"] == stem_name]
        final_source = final_row["final_filter_source"].iloc[0]
        #print("final_source", final_source)
        
        ## 根據final_source選rt距離(毫秒)
        dt_col_map = {
            "baseline" : "tpeak_abs_filter_base",
            "former"   : "tpeak_abs_filter_former",
            "max"      : "tpeak_abs_filter_max"
            }
        
        dt_col      = dt_col_map.get(final_source)
        t_abs       = beats_merged[dt_col].to_numpy()
    else:
        #print("No bad segments. Using baseline results directly.")
        t_abs = df_base["tpeak_abs_filter"].to_numpy()

    
    return t_abs


def build_final_summary(summary_merged: pd.DataFrame) -> pd.DataFrame:
    
    """
    決定每個訊號最後要用 baseline或former或max，並輸出一張新的 summary
    
    input---
        summary_merged: 所有tpeak的每拍結果，包含 base / former / max
    
    output---
        stem
        final_filter_source  : "baseline" / "former" / "max"
        final_filter_n
        final_filter_mean_ms
        final_filter_std_ms
      
    """

    df = summary_merged.copy()

    # 1) 先把需要的欄位拿出來
    flt_n_base   = df.get("rt_filter_n_base")
    flt_n_for    = df.get("rt_filter_n_former")
    flt_n_max    = df.get("rt_filter_n_max")

    flt_mean_base = df.get("rt_filter_mean_ms_base")
    flt_mean_for  = df.get("rt_filter_mean_ms_former")
    flt_mean_max  = df.get("rt_filter_mean_ms_max")

    flt_std_base  = df.get("rt_filter_std_ms_base")
    flt_std_for   = df.get("rt_filter_std_ms_former")
    flt_std_max   = df.get("rt_filter_std_ms_max")


    mask_force_flt    = flt_std_for.notna() & flt_std_max.notna()
    mask_no_force_flt = flt_std_for.isna() & flt_std_max.isna() & flt_std_base.notna()

    final_filter_source = np.empty(len(df), dtype=object)
    final_filter_source[:] = None

    # 2) 沒 force → baseline
    final_filter_source[mask_no_force_flt.values] = "baseline"

    # 3) 有 force → 比較 former和max誰的標準差比較小，就用哪個偵測方式
    mask_flt_former_better = mask_force_flt & (flt_std_for <= flt_std_max)
    mask_flt_max_better    = mask_force_flt & (flt_std_max   <  flt_std_for)

    final_filter_source[mask_flt_former_better.values] = "former"
    final_filter_source[mask_flt_max_better.values]    = "max"

    final_filter_n    = np.where(final_filter_source == "baseline", flt_n_base,
                           np.where(final_filter_source == "former", flt_n_for,
                           np.where(final_filter_source == "max",    flt_n_max,    np.nan)))
    final_filter_mean = np.where(final_filter_source == "baseline", flt_mean_base,
                           np.where(final_filter_source == "former", flt_mean_for,
                           np.where(final_filter_source == "max",    flt_mean_max, np.nan)))
    final_filter_std  = np.where(final_filter_source == "baseline", flt_std_base,
                           np.where(final_filter_source == "former", flt_std_for,
                           np.where(final_filter_source == "max",    flt_std_max,  np.nan)))

    # 4) 每個訊號只會有一個判斷方式，並組成 final_summary
    final_summary = pd.DataFrame({
        "stem": df["stem"],
        "final_filter_source": final_filter_source, # baseline / former / max
        "final_filter_n":      final_filter_n,
        "final_filter_mean_ms":final_filter_mean,
        "final_filter_std_ms": final_filter_std,
    })

    return final_summary



# %%


if __name__ == "__main__":
    
    ecg  = np.loadtxt(r"C:\Users\jane\OneDrive\Desktop\SWM\251008\給dennis\data\2236_20250310082600_0_43_91.txt")    
    #rt_index_array = splitted_t_wave_detector(ecg, rpeak=[])
    t_index_array = splitted_t_wave_detector(ecg, rpeak=[])
    print(t_index_array)
  

