# ============================================================
# 共用函式 - M1 / M2 / M3 都會 import 這個檔案
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.fftpack import fft
import pywt

# ─── 可用回歸特徵字典（flag=1 表示啟用）──────────────────────────────────────
'''
USED_FEATURE_DIC = [
    (0,'uuid'),(0,'type'),(1,'rr_interval'),(1,'hr'),(1,'a_score'),(1,'p_score'),(1,'r_score'),
    (1,'p_stability'),(1,'q_stability'),(1,'s_stability'),(1,'t_stability'),
    (1,'p_value'),(1,'q_value'),(1,'r_value'),(1,'s_value'),(1,'t_value'),
    (1,'pr_duration'),(1,'pr_amplitude'),(1,'pr_distances'),(1,'pr_directions'),(1,'pr_slope'),(1,'pr_corrections3'),
    (1,'qr_duration'),(1,'qr_amplitude'),(1,'qr_distances'),(1,'qr_directions'),(1,'qr_slope'),(1,'qr_corrections3'),
    (1,'rs_duration'),(1,'rs_amplitude'),(1,'rs_distances'),(1,'rs_directions'),(1,'rs_slope'),(1,'rs_corrections3'),
    (1,'rt_duration'),(1,'rt_amplitude'),(1,'rt_distances'),(1,'rt_directions'),(1,'rt_slope'),(1,'rt_corrections3'),
    (1,'pq_duration'),(1,'pq_amplitude'),(1,'pq_distances'),(1,'pq_directions'),(1,'pq_slope'),(1,'pq_corrections3'),
    (1,'ps_duration'),(1,'ps_amplitude'),(1,'ps_distances'),(1,'ps_directions'),(1,'ps_slope'),(1,'ps_corrections3'),
    (1,'pt_duration'),(1,'pt_amplitude'),(1,'pt_distances'),(1,'pt_directions'),(1,'pt_slope'),(1,'pt_corrections3'),
    (1,'qs_duration'),(1,'qs_amplitude'),(1,'qs_distances'),(1,'qs_directions'),(1,'qs_slope'),(1,'qs_corrections3'),
    (1,'qt_duration'),(1,'qt_amplitude'),(1,'qt_distances'),(1,'qt_directions'),(1,'qt_slope'),(1,'qt_corrections3'),
    (1,'st_duration'),(1,'st_amplitude'),(1,'st_distances'),(1,'st_directions'),(1,'st_slope'),(1,'st_corrections3'),
    (1,'p_left_slope'),(1,'p_right_slope'),(1,'p_left_sharp'),(1,'p_right_sharp'),(1,'p_tilt'),
    (1,'r_left_slope'),(1,'r_right_slope'),(1,'r_left_sharp'),(1,'r_right_sharp'),(1,'r_tilt'),
    (1,'t_left_slope'),(1,'t_right_slope'),(1,'t_left_sharp'),(1,'t_right_sharp'),(1,'t_tilt'),
    (0,'Dataset'),(0,'BG_Level'),(1,'dif_qs_amp/dif_qr_amp'),(1,'dif_rt_amp/dif_st_amp'),(1,'t/r_amp'),(1,'s/t_amp'),
]
'''

USED_FEATURE_DIC = [
    (0,'uuid'),(0,'type'),(1,'rr_interval'),(0,'hr'),(0,'a_score'),(0,'p_score'),(0,'r_score'),
    (0,'p_stability'),(0,'q_stability'),(0,'s_stability'),(0,'t_stability'),
    (0,'p_value'),(0,'q_value'),(0,'r_value'),(0,'s_value'),(0,'t_value'),
    (0,'pr_duration'),(0,'pr_amplitude'),(0,'pr_distances'),(0,'pr_directions'),(0,'pr_slope'),(0,'pr_corrections3'),
    (0,'qr_duration'),(0,'qr_amplitude'),(1,'qr_distances'),(1,'qr_directions'),(1,'qr_slope'),(0,'qr_corrections3'),
    (0,'rs_duration'),(0,'rs_amplitude'),(1,'rs_distances'),(1,'rs_directions'),(1,'rs_slope'),(0,'rs_corrections3'),
    (1,'rt_duration'),(0,'rt_amplitude'),(0,'rt_distances'),(0,'rt_directions'),(0,'rt_slope'),(0,'rt_corrections3'),
    (0,'pq_duration'),(0,'pq_amplitude'),(0,'pq_distances'),(0,'pq_directions'),(0,'pq_slope'),(0,'pq_corrections3'),
    (0,'ps_duration'),(0,'ps_amplitude'),(0,'ps_distances'),(0,'ps_directions'),(0,'ps_slope'),(0,'ps_corrections3'),
    (1,'pt_duration'),(0,'pt_amplitude'),(0,'pt_distances'),(0,'pt_directions'),(0,'pt_slope'),(0,'pt_corrections3'),
    (1,'qs_duration'),(0,'qs_amplitude'),(0,'qs_distances'),(0,'qs_directions'),(0,'qs_slope'),(0,'qs_corrections3'),
    (1,'qt_duration'),(0,'qt_amplitude'),(1,'qt_distances'),(0,'qt_directions'),(0,'qt_slope'),(0,'qt_corrections3'),
    (0,'st_duration'),(0,'st_amplitude'),(0,'st_distances'),(0,'st_directions'),(0,'st_slope'),(1,'st_corrections3'),
    (0,'p_left_slope'),(0,'p_right_slope'),(0,'p_left_sharp'),(0,'p_right_sharp'),(0,'p_tilt'),
    (0,'r_left_slope'),(0,'r_right_slope'),(0,'r_left_sharp'),(0,'r_right_sharp'),(0,'r_tilt'),
    (1,'t_left_slope'),(1,'t_right_slope'),(1,'t_left_sharp'),(1,'t_right_sharp'),(0,'t_tilt'),(1,'qrs_area'),(1,'st_area'),(1,'t_wave_cog'),
    (0,'Dataset'),(0,'BG_Level'),
    (1,'dif_qs_amp/dif_qr_amp'),(1,'dif_rt_amp/dif_st_amp'),(1,'t/r_amp'),(1,'s/t_amp')
]




REGRESSION_FEATURE_NAMES = [name for flag, name in USED_FEATURE_DIC if flag == 1]
# 原本使用的 14 features: rr_interval, rt_duration, pt_duration, qt_duration, qt_distances,
#              st_corrections3, t_left_slope, t_right_slope, t_left_sharp, t_right_sharp,
#              dif_qs_amp/dif_qr_amp, dif_rt_amp/dif_st_amp, t/r_amp, s/t_amp


class ECGDataset(Dataset):
    def __init__(self, dir_path, method='time', classes=3):
        self.dir_path = os.path.abspath(dir_path)
        self.method = method
        self.data_len = 150
        self.channel = 1

        # 修改為同時讀取血糖值和檔名
        if(classes==2):
            normalSigs, normalGlucose, normalFilenames = self.read_files_with_glucose('Normal')
            normalLabels = np.zeros(len(normalSigs))

            abnormSigs, abnormGlucose, abnormFilenames = self.read_files_with_glucose('High')
            abnormLabels = np.ones(len(abnormSigs))

            self.Signals = torch.cat((normalSigs, abnormSigs), 0)
            self.Labels = np.concatenate((normalLabels, abnormLabels), axis = 0).reshape(-1, 1)
            self.Labels = torch.from_numpy(self.Labels)
            self.GlucoseValues = normalGlucose + abnormGlucose  
            self.Filenames = normalFilenames + abnormFilenames  

        else:
            normalSigs, normalGlucose, normalFilenames = self.read_files_with_glucose('Normal')
            normalLabels = np.zeros(len(normalSigs))

            abnormSigs, abnormGlucose, abnormFilenames = self.read_files_with_glucose('High')
            abnormLabels =  np.ones(len(abnormSigs))

            lowSigs, lowGlucose, lowFilenames = self.read_files_with_glucose('Low')
            lowLabels = 2*np.ones(len(lowSigs))

            self.Signals = torch.cat((normalSigs, abnormSigs,lowSigs), 0)
            self.Labels = np.concatenate((normalLabels, abnormLabels,lowLabels), axis = 0).reshape(-1, 1)
            self.Labels = torch.from_numpy(self.Labels)
            self.GlucoseValues = normalGlucose + abnormGlucose + lowGlucose  
            self.Filenames = normalFilenames + abnormFilenames + lowFilenames  

        # 推導 Category_Features 路徑
        abs_dir = os.path.abspath(dir_path)
        if 'Category_Features' in abs_dir:
            self.reg_features_path = abs_dir
        elif 'GlucoseData' in abs_dir:
            self.reg_features_path = abs_dir.replace('GlucoseData', 'Category_Features')
        else:
            self.reg_features_path = None

        # 載入回歸特徵
        if self.reg_features_path and os.path.isdir(self.reg_features_path):
            self.RegFeatures = self._load_all_reg_features()
        else:
            self.RegFeatures = torch.zeros(len(self.Labels), len(REGRESSION_FEATURE_NAMES), dtype=torch.float32)

    def _load_all_reg_features(self):
        feat_list = []
        for fname in self.Filenames:
            reg_path = os.path.join(self.reg_features_path, fname)
            feat = self.load_reg_features(reg_path)
            feat_list.append(feat)
        tensor = torch.FloatTensor(feat_list)
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def load_reg_features(self, filepath):
        values = {name: 0.0 for name in REGRESSION_FEATURE_NAMES}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        key, val = parts[0], parts[1]
                        if key in values:
                            try:
                                v = float(val)
                                values[key] = 0.0 if (v != v or v == float('inf') or v == float('-inf')) else v
                            except ValueError:
                                values[key] = 0.0
        except (FileNotFoundError, OSError):
            pass
        return [values[name] for name in REGRESSION_FEATURE_NAMES]
    
    def get_reg_feature_stats(self):
        mean = self.RegFeatures.mean(dim=0)
        std = self.RegFeatures.std(dim=0)
        std = torch.clamp(std, min=1e-8)
        return mean, std

    def apply_reg_zscore(self, mean, std):
        self.RegFeatures = (self.RegFeatures - mean) / std

    def __getitem__(self, index):
        signal = self.Signals[index]
        reg_feat = self.RegFeatures[index]
        label = self.Labels[index]
        filename = self.Filenames[index]
        return signal, reg_feat, label, filename

    def get_glucose(self, index):
        """取得指定索引的血糖值"""
        return self.GlucoseValues[index]

    def get_filename(self, index):
        """取得指定索引的檔名"""
        return self.Filenames[index]

    def __len__(self):
        return len(self.Labels)

    # Parsing files in folder
    def read_files(self, foldername):
        Sig = []
        for filename in os.listdir(os.path.join(self.dir_path, foldername)):
            file_path = os.path.join(self.dir_path, foldername, filename)
            Sig.append(self.load_data(file_path))
        Sig = torch.FloatTensor(Sig)
        if self.channel == 1:
            Sig = Sig.unsqueeze(1)

        return Sig

    # 新增讀取檔案並解析血糖值的方法
    def read_files_with_glucose(self, foldername):
        Sig = []
        glucose_values = []
        filenames = []

        for filename in os.listdir(os.path.join(self.dir_path, foldername)):
            file_path = os.path.join(self.dir_path, foldername, filename)
           
            Sig.append(self.load_data(file_path))
            filenames.append(os.path.join(foldername, filename))
            # 從檔名取得血糖值 (格式: uuid_time_flag_index_glucose.txt)
            try:
                glucose_str = filename.replace('.txt', '').split('_')[-1]
                glucose_values.append(int(glucose_str))
            except:
                glucose_values.append(-1)  # 無法解析時設為 -1
        
        
        Sig = torch.FloatTensor(Sig)

        if self.channel == 1:
            Sig = Sig.unsqueeze(1)

        return Sig, glucose_values, filenames

    # Read data in files & Preprocessing
    def load_data(self, filepath):
        
        try:
            values = np.genfromtxt(filepath, delimiter='')
            if values.ndim != 1:
                raise ValueError
        except Exception:
            return np.zeros(self.data_len).tolist()

        sig = np.zeros(self.data_len)
        if len(values) >= self.data_len:
            sig[:self.data_len] = values[:self.data_len]
        else:
            sig[:len(values)] = values

        self.channel = 1
        
        if self.method == 'raw':
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
    

def initialize_weights(model):              
        
    for m in model.modules():            
        if isinstance(m, nn.Conv1d):  # 判断是否為Conv1d
            ##torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)                
            
        if isinstance(m, nn.Linear):               
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None: ## 加上偏差初始化
                torch.nn.init.zeros_(m.bias.data)  
        
        elif isinstance(m, nn.BatchNorm1d):              
            m.weight.data.fill_(1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
'''
def _majority_filtering(glucose_list):
        
        majority_result=0 ##先視為正常血糖
        category_array=[0, 0, 0]
        
        for i in range(len(glucose_list)):
            current_category=glucose_list[i]
            category_array[current_category]=category_array[current_category] + 1

        maxindex_array=[i for i,val in enumerate(category_array) if (val==max(category_array))]   
        if(len(maxindex_array)==1): ###只有一個類別佔多數
            majority_result=maxindex_array[0] ##直接輸出
        else: ###有多個類別佔多數
            for k in range(len(maxindex_array)): ###檢查是否有低或高血糖
                if(maxindex_array[k]==2):   ##低血糖priority最高
                    majority_result=2
                    break
                elif(maxindex_array[k]==1): ###高血糖次之
                    majority_result=1
        
        return majority_result
'''

class FocalLoss(nn.Module):
    """
    自動支援：
    1. Binary classification: logits shape [N] or [N,1]
    2. Multi-class classification: logits shape [N,C], C >= 2

    參數：
    - alpha:
        * binary: 可給 float，例如 0.25
        * multi-class: 可給 tensor/list，例如 [1.0, 2.0, 2.0]
    - gamma: focal loss 的 focusing parameter
    - reduction: "mean" / "sum" / "none"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        logits:
            - binary: [N] or [N,1]
            - multi-class: [N,C]

        targets:
            - binary: [N] or [N,1], values in {0,1}
            - multi-class: [N], values in {0,1,...,C-1}
        """
        # -------------------------
        # Case 1: Binary classification
        # logits.shape = [N] or [N,1]
        # -------------------------
        if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
            logits = logits.view(-1)
            targets = targets.float().view(-1)
            
            if self.label_smoothing > 0:
                # 將 1 變為 1 - smoothing/2 (例如 0.95)，0 變為 smoothing/2 (例如 0.05)
                targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            pt = torch.exp(-bce)

            if self.alpha is None:
                alpha_t = 1.0
            else:
                # binary alpha: 建議給 float，例如 0.25
                alpha = float(self.alpha)
                alpha_t = torch.where(
                    targets > 0.5,
                    torch.full_like(targets, alpha),
                    torch.full_like(targets, 1 - alpha)
                )

            fl = alpha_t * ((1 - pt) ** self.gamma) * bce

        # -------------------------
        # Case 2: Multi-class classification
        # logits.shape = [N,C], C>=2
        # -------------------------
        elif logits.ndim == 2 and logits.size(1) >= 2:
            targets = targets.view(-1).long()

            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            fl = ((1 - pt) ** self.gamma) * ce

            if self.alpha is not None:
                if isinstance(self.alpha, (list, tuple)):
                    alpha = torch.tensor(self.alpha, dtype=logits.dtype, device=logits.device)
                elif isinstance(self.alpha, torch.Tensor):
                    alpha = self.alpha.to(logits.device, dtype=logits.dtype)
                else:
                    raise TypeError("For multi-class, alpha must be list, tuple, or torch.Tensor.")

                fl = alpha[targets] * fl

        else:
            raise ValueError(
                f"Unsupported logits shape: {logits.shape}. "
                "Expected [N], [N,1], or [N,C]."
            )

        # -------------------------
        # Reduction
        # -------------------------
        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        elif self.reduction == "none":
            return fl
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class BalancedBatchSampler:
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)
        self.n_samples_per_class = batch_size // self.n_classes
        self.batch_size = self.n_samples_per_class * self.n_classes
        
        self.class_indices = {
            cls: np.where(self.labels == cls)[0].tolist()
            for cls in self.classes
        }
        
        self.n_batches = len(labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for cls, indices in self.class_indices.items():
                sampled = np.random.choice(indices, self.n_samples_per_class, replace=True)
                batch.extend(sampled)
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches

def train_valid_split_indices(labels_array, valid_ratio=0.1, seed=10):
    """
    依照類別分層切分 train / valid，避免 valid 幾乎沒有 minority class
    labels_array: 1D numpy array
    """
    rng = np.random.RandomState(seed)
    
    train_idx_all = []
    val_idx_all = []

    unique_classes = np.unique(labels_array)
    for cls in unique_classes:
        cls_idx = np.where(labels_array == cls)[0]

        n_val = int(len(cls_idx) * valid_ratio)
        # 至少保留 1 筆到 validation（若該類別本來有資料）
        if len(cls_idx) > 1:
            n_val = max(1, n_val)
        else:
            n_val = 0

        val_idx_all.extend(cls_idx[:n_val].tolist())
        train_idx_all.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_idx_all)
    rng.shuffle(val_idx_all)

    return train_idx_all, val_idx_all
