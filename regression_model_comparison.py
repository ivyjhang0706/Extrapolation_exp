"""
比較不同回歸模型的外推(extrapolation)能力。

延續 SVR_model.py 的實驗設計(依 ratio 窄化訓練範圍 -> 用固定 test 集算 Intra/Extra 誤差)，
但這裡同時跑多個模型(SVR-linear / SVR-rbf / LinearRegression / Ridge)，
用「外推/內插誤差比 = Extra_MAE / Intra_MAE」當核心指標比較誰的外推能力比較不會隨訓練範圍窄化而崩壞。

之所以用「比值」而不是直接比較 Extra_MAE 的絕對值，是因為不同模型整體準確度本來就不一樣
(例如某模型整體就是比較準)，比值可以把「整體準不準」跟「外推相對變差多少」這兩件事解耦，
比較的才是我們真正關心的「外推能力」本身。

SVR-rbf 刻意放進來當對照組：RBF kernel 離訓練資料越遠，決策函數會趨近一個常數，
不會延伸訓練資料的趨勢，預期外推能力會明顯比 linear 系列的模型差，用來當一個「不太能外推」的參照點。
"""

import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from SVR_expansion_exp_pre import get_ratio_bounds
from SVR_model import (
    _compute_metrics,
    evaluate_intra_extra,
    load_features_and_labels,
    _print_metrics,
)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def fit_and_evaluate(model, model_name: str, X_all_scaled, y_all, test_data, target_data, mode='log',
                      lower_bound: float | None = None, upper_bound: float | None = None,
                      verbose: bool = True):
    """
    通用版的「訓練 + 評估」流程，邏輯跟 SVR_model.py 的 svr_manual 相同，
    差別只在於 model 是外部傳入的任意 sklearn regressor，
    這樣才能讓不同模型走同一套前處理/評估流程，確保比較時「除了模型本身，其他條件都一樣」。
    """
    if mode == 'log':
        y_all = np.log(y_all + 1e-8)

    print(f"[{model_name}] 開始 fit：{X_all_scaled.shape[0]} 筆樣本、{X_all_scaled.shape[1]} 個 feature ...", flush=True)
    fit_start = time.time()
    model.fit(X_all_scaled, y_all)
    print(f"[{model_name}] fit 完成，耗時 {time.time() - fit_start:.1f} 秒", flush=True)

    y_pred_train = model.predict(X_all_scaled)
    y_pred_test = model.predict(test_data)

    test_clipped_mask = np.zeros_like(target_data, dtype=bool)

    if mode == 'log':
        # 跟 svr_manual 一樣的保護：外推時任何模型都可能在 log 空間給出誇張的值(尤其是
        # LinearRegression/Ridge 沒有 SVR 的 margin 概念，極端外推時反而可能發散得更嚴重)，
        # exp() 還原前先示警、還原後再夾住上限，避免 inf 汙染整體指標。
        LOG_WARN_THRESHOLD = 20.0
        n_extreme_train = int(np.sum(np.abs(y_pred_train) > LOG_WARN_THRESHOLD))
        n_extreme_test = int(np.sum(np.abs(y_pred_test) > LOG_WARN_THRESHOLD))
        if n_extreme_train or n_extreme_test:
            max_abs_log = max(
                np.max(np.abs(y_pred_train)) if len(y_pred_train) else 0.0,
                np.max(np.abs(y_pred_test)) if len(y_pred_test) else 0.0,
            )
            print(f"[{model_name}] 警告：log 空間預測值超出合理範圍(|value|>{LOG_WARN_THRESHOLD})，"
                  f"train 有 {n_extreme_train} 筆、test 有 {n_extreme_test} 筆，"
                  f"裁切前最大絕對值={max_abs_log:.2f}")

        with np.errstate(over="ignore"):
            y_pred_train = np.exp(y_pred_train)
            y_pred_test = np.exp(y_pred_test)

        GLUCOSE_CLIP_MAX = 300.0
        test_clipped_mask = y_pred_test > GLUCOSE_CLIP_MAX
        n_clipped_train = int(np.sum(y_pred_train > GLUCOSE_CLIP_MAX))
        n_clipped_test = int(np.sum(test_clipped_mask))
        if n_clipped_train or n_clipped_test:
            print(f"[{model_name}] 血糖預測值夾在 <= {GLUCOSE_CLIP_MAX}：train 夾住 {n_clipped_train} 筆、"
                  f"test 夾住 {n_clipped_test} 筆(注意：這會讓這些離群預測的誤差看起來變小，只是避免 inf 汙染整體指標)")
        y_pred_train = np.clip(y_pred_train, None, GLUCOSE_CLIP_MAX)
        y_pred_test = np.clip(y_pred_test, None, GLUCOSE_CLIP_MAX)

    overall_metrics = _compute_metrics(y_pred_test, target_data)

    intra_metrics, extra_metrics = None, None
    if lower_bound is not None and upper_bound is not None:
        intra_metrics, extra_metrics = evaluate_intra_extra(y_pred_test, target_data, lower_bound, upper_bound)
        if verbose:
            print(f"--- [{model_name}] 內插(intra) vs 外推(extra)誤差 (訓練範圍=[{lower_bound}, {upper_bound}]) ---")
            _print_metrics("overall", overall_metrics)
            _print_metrics("intra", intra_metrics)
            _print_metrics("extra", extra_metrics)

    return overall_metrics, intra_metrics, extra_metrics, y_pred_test, test_clipped_mask


def build_models() -> dict:
    """
    要比較的模型清單(皆為「理論上可以外推」的模型)：
        SVR_linear       - 決策函數是線性的，外推時會延伸訓練資料的趨勢(可能延伸過頭)
        SVR_rbf          - 對照組，RBF kernel 離訓練資料越遠決策函數越趨近常數，預期外推能力較差
        LinearRegression - 最單純的線性外推基準，沒有正則化
        Ridge            - 線性 + L2 正則化，外推行為理論上比 LinearRegression 溫和一些
    """
    return {
        "SVR_linear": SVR(kernel='linear', C=100.0, epsilon=0.1, verbose=True),
        "SVR_rbf": SVR(kernel='rbf', C=100.0, epsilon=0.1, verbose=True),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
    }


if __name__ == "__main__":
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

    # 這裡要回答的問題是「同一個訓練範圍下，哪個模型架構的外推能力比較好」，
    # 跟 SVR_model.py「窄化程度對(同一個模型的)外推能力有沒有影響」是不同問題，
    # 不需要整個 ratio_list 掃過一輪(否則 4 個模型 x 7 個 ratio，SVR 系列的訓練時間會被放大好幾倍)，
    # 固定一個有代表性的訓練範圍即可。
    # ratio=0.2 -> train range=[102, 153]：範圍不會太窄(intra 樣本還夠、Intra_MAE 分母穩定)，
    # 外推區也留了足夠大的範圍可以評估外推。想換其他範圍，改這個常數即可。
    comparison_ratio = 0.2
    uuid = "2261"
    glucose_type = "Normal"
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results", uuid, "regression_model_comparison")
    os.makedirs(results_dir, exist_ok=True)

    used_feature_array = np.array([row[0] for row in used_feature_dic])

    # 測試集固定用完整的 Test，不隨 model 改變，搬到迴圈外面只讀取一次
    X_test_raw, y_test, _ = load_features_and_labels(
        os.path.join(base_dir, "dataset", uuid, "Test", glucose_type), used_feature_array
    )

    lower_bound, upper_bound = get_ratio_bounds(comparison_ratio)
    print(f"比較用的固定訓練範圍：ratio={comparison_ratio:.2f} -> [{lower_bound}, {upper_bound}]")

    X_train, y_train, _ = load_features_and_labels(
        os.path.join(base_dir, "dataset", "EXP", uuid, "Train", glucose_type, f"ratio_{comparison_ratio:.2f}"),
        used_feature_array,
    )

    # Handle NaN values by replacing with column means
    col_means = np.nanmean(X_train, axis=0)
    nan_indices = np.where(np.isnan(X_train))
    X_train[nan_indices] = np.take(col_means, nan_indices[1])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test = X_test_raw.copy()
    nan_indices = np.where(np.isnan(X_test))
    X_test[nan_indices] = np.take(col_means, nan_indices[1])
    X_test_scaled = scaler.transform(X_test)

    intra_mask = (y_test >= lower_bound) & (y_test <= upper_bound)
    extra_mask = ~intra_mask

    summary_rows = []       # 完整彙整(model x Overall/Intra/Extra)，格式跟 SVR_model_summary.csv 一致，多一個 model 欄位
    comparison_rows = []    # 核心比較表：每個 model 一列，含外推/內插誤差比

    models = build_models()

    for model_name, base_model in models.items():
        model_start = time.time()
        print(f"\n===== model={model_name} 開始 (train range=[{lower_bound}, {upper_bound}]) =====", flush=True)

        # clone：每個模型都要用全新的實例，避免沿用到別的模型已經 fit 過的參數
        model_instance = clone(base_model)

        overall_metrics, intra_metrics, extra_metrics, y_pred_test, test_clipped_mask = fit_and_evaluate(
            model_instance, model_name, X_train_scaled, y_train, X_test_scaled, y_test,
            lower_bound=lower_bound, upper_bound=upper_bound,
        )

        for group_name, metrics, group_mask in (
            ("Overall", overall_metrics, np.ones_like(y_test, dtype=bool)),
            ("Intra", intra_metrics, intra_mask),
            ("Extra", extra_metrics, extra_mask),
        ):
            if metrics is None:
                continue
            summary_rows.append({
                "model": model_name,
                "ratio": comparison_ratio,
                "group": group_name,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "n_clipped": int(np.sum(test_clipped_mask[group_mask])),
                **metrics,
            })

        # -------- 外推/內插誤差比：本次比較的核心指標 --------
        # 用比值而不是絕對值，才能把「模型整體準不準」跟「外推相對變差多少」解耦
        extrap_ratio = None
        if intra_metrics is not None and extra_metrics is not None and intra_metrics["MAE"] > 0:
            extrap_ratio = extra_metrics["MAE"] / intra_metrics["MAE"]
        comparison_rows.append({
            "model": model_name,
            "ratio": comparison_ratio,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "Intra_n": intra_metrics["n"] if intra_metrics else 0,
            "Extra_n": extra_metrics["n"] if extra_metrics else 0,
            "Intra_MAE": intra_metrics["MAE"] if intra_metrics else None,
            "Extra_MAE": extra_metrics["MAE"] if extra_metrics else None,
            "extrap_ratio": extrap_ratio,
        })

        print(f"===== model={model_name} 完成，耗時 {time.time() - model_start:.1f} 秒 =====", flush=True)

    # -------- 完整彙整結果存成 CSV(跟 SVR_model_summary.csv 同樣格式，方便對照) --------
    summary_csv_path = os.path.join(results_dir, "regression_model_comparison_summary.csv")
    fieldnames = ["model", "ratio", "group", "lower_bound", "upper_bound", "n", "n_clipped", "MSE", "MARD", "Bias", "MAE", "RMSE"]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"完整彙整結果已存到: {summary_csv_path}")

    # -------- 核心比較表：外推/內插誤差比 --------
    comparison_csv_path = os.path.join(results_dir, "regression_model_comparison_ratio.csv")
    comparison_fieldnames = ["model", "ratio", "lower_bound", "upper_bound", "Intra_n", "Extra_n", "Intra_MAE", "Extra_MAE", "extrap_ratio"]
    with open(comparison_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"外推/內插誤差比已存到: {comparison_csv_path}")

    # -------- 長條圖：外推/內插誤差比，每個模型一根柱子 --------
    # 這張圖直接回答「固定同一個訓練範圍，哪個模型的外推能力比較好」(越低越好，1.0=外推跟內插一樣準)
    model_names = [r["model"] for r in comparison_rows]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(model_names))]

    plt.figure(figsize=(8, 6))
    ratios_plot = [r["extrap_ratio"] if r["extrap_ratio"] is not None else 0 for r in comparison_rows]
    plt.bar(model_names, ratios_plot, color=colors)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Extra=Intra (無外推損失)")
    plt.ylabel("外推/內插誤差比 (Extra MAE / Intra MAE)")
    plt.title(f"外推能力比較 (train range=[{lower_bound}, {upper_bound}])")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "regression_model_comparison_ratio.png"))
    plt.close()
    print(f"外推/內插誤差比圖已存到: {os.path.join(results_dir, 'regression_model_comparison_ratio.png')}")

    # -------- 長條圖：Intra_MAE vs Extra_MAE 並排比較 --------
    # 只看比值可能忽略「絕對誤差規模」，例如某模型比值低但 Extra MAE 本身就很大，這張圖補上這個視角
    intra_maes = [r["Intra_MAE"] if r["Intra_MAE"] is not None else 0 for r in comparison_rows]
    extra_maes = [r["Extra_MAE"] if r["Extra_MAE"] is not None else 0 for r in comparison_rows]
    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, intra_maes, width, label="Intra MAE", color="#4C72B0")
    plt.bar(x + width / 2, extra_maes, width, label="Extra MAE", color="#C44E52")
    plt.xticks(x, model_names)
    plt.ylabel("MAE")
    plt.title(f"Intra vs Extra MAE(絕對值) (train range=[{lower_bound}, {upper_bound}])")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "regression_model_comparison_extra_mae.png"))
    plt.close()
    print(f"Intra/Extra MAE 圖已存到: {os.path.join(results_dir, 'regression_model_comparison_extra_mae.png')}")
