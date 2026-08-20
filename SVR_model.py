import numpy as np
from sklearn.svm import SVR
import os
import csv
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from SVR_expansion_exp_pre import data_preprocessing, get_ratio_bounds, GLUCOSE_BOUNDS, resolve_glucose_bounds

# 讓 matplotlib 能正常顯示中文，避免圖上的中文字變成缺字方框
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def svr_manual(X_all_scaled, y_all, test_data, target_data, mode='log',lower_bound=None, upper_bound=None,
               verbose=True):
    """
    若有傳入 lower_bound/upper_bound(訓練時實際的血糖範圍)，
    除了整體指標外，還會另外印出/回傳「內插」與「外推」誤差，
    用來評估 SVR 對訓練範圍外血糖值的外推能力。

    overall_metrics/intra_metrics/extra_metrics 三者是同一種格式的字典
    {n, MSE, MARD, Bias, MAE, RMSE}，方便彙整成同一張表比較，不用外面再算一次。
    """

    # kernel='linear' 用 libsvm 實作，樣本數大(上萬筆)時可能非常慢(數十分鐘等級)。
    # verbose=True 會讓 libsvm 訓練過程中印出內部訊息，用來確認它還在跑、沒有卡死。
    model = SVR(kernel='linear', C=100.0, epsilon=0.1, verbose=verbose) # linear is better for 外推

    if(mode=='log'):
        y_all = np.log(y_all + 1e-8)

    print(f"[svr_manual] 開始 fit：{X_all_scaled.shape[0]} 筆樣本、{X_all_scaled.shape[1]} 個 feature ...", flush=True)
    fit_start = time.time()
    model.fit(X_all_scaled, y_all)
    print(f"[svr_manual] fit 完成，耗時 {time.time() - fit_start:.1f} 秒", flush=True)

    # train prediction
    y_pred_train = model.predict(X_all_scaled)

    # test prediction
    y_pred_test = model.predict(test_data)

    test_clipped_mask = np.zeros_like(target_data, dtype=bool)  # 記錄哪些 test 樣本被 GLUCOSE_CLIP_MAX 夾住過

    if(mode=='log'):
        # linear SVR 對某些樣本可能給出很誇張的 log 決策值(離群點/外推時線性決策函式沒有飽和上限)，
        # 印出裁切前的最大絕對值方便觀察：float64 要 >709 exp() 才會真的溢位變成 inf。
        LOG_WARN_THRESHOLD = 20.0  # exp(20)≈4.85億，明顯超出生理合理範圍就先示警
        n_extreme_train = int(np.sum(np.abs(y_pred_train) > LOG_WARN_THRESHOLD))
        n_extreme_test = int(np.sum(np.abs(y_pred_test) > LOG_WARN_THRESHOLD))
        if n_extreme_train or n_extreme_test:
            max_abs_log = max(
                np.max(np.abs(y_pred_train)) if len(y_pred_train) else 0.0,
                np.max(np.abs(y_pred_test)) if len(y_pred_test) else 0.0,
            )
            print(f"[svr_manual] 警告：log 空間預測值超出合理範圍(|value|>{LOG_WARN_THRESHOLD})，"
                  f"train 有 {n_extreme_train} 筆、test 有 {n_extreme_test} 筆，"
                  f"裁切前最大絕對值={max_abs_log:.2f}")

        with np.errstate(over="ignore"):  # log 值太大時 exp() 會 overflow 成 inf，屬於預期內狀況，不用讓它印警告訊息
            y_pred_train = np.exp(y_pred_train)
            y_pred_test = np.exp(y_pred_test)

        # 血糖預測值(還原成原始單位後)直接夾在生理上合理的上限，不管是 exp 後的極大值還是真的 inf 都會被壓下來
        GLUCOSE_CLIP_MAX = 300.0
        test_clipped_mask = y_pred_test > GLUCOSE_CLIP_MAX
        n_clipped_train = int(np.sum(y_pred_train > GLUCOSE_CLIP_MAX))
        n_clipped_test = int(np.sum(test_clipped_mask))
        if n_clipped_train or n_clipped_test:
            print(f"[svr_manual] 血糖預測值夾在 <= {GLUCOSE_CLIP_MAX}：train 夾住 {n_clipped_train} 筆、"
                  f"test 夾住 {n_clipped_test} 筆(注意：這會讓這些離群預測的誤差看起來變小，只是避免 inf 汙染整體指標，不代表模型真的預測得準)")
        y_pred_train = np.clip(y_pred_train, None, GLUCOSE_CLIP_MAX)
        y_pred_test = np.clip(y_pred_test, None, GLUCOSE_CLIP_MAX)

    overall_metrics = _compute_metrics(y_pred_test, target_data)

    max_glucose_value = np.max(target_data)
    min_glucose_value = np.min(target_data)

    intra_metrics, extra_metrics = None, None
    if lower_bound is not None and upper_bound is not None:
        intra_metrics, extra_metrics = evaluate_intra_extra(
            y_pred_test, target_data, lower_bound, upper_bound
        )
        print(f"--- 內插(intra) vs 外推(extra)誤差 (訓練範圍=[{lower_bound}, {upper_bound}]) ---")
        _print_metrics("overall", overall_metrics)
        _print_metrics("intra", intra_metrics)
        _print_metrics("extra", extra_metrics)

    return (overall_metrics, max_glucose_value, min_glucose_value, model,
            y_pred_train, y_pred_test, intra_metrics, extra_metrics, test_clipped_mask)
def _compute_metrics(y_pred, y_true):
    """計算單一群組(內插或外推)的誤差指標，樣本數為 0 時回傳 None。"""
    if len(y_true) == 0:
        return None

    errors = np.abs(y_true - y_pred)
    mse = np.mean((y_pred - y_true) ** 2)
    mard = np.mean(errors / y_true) * 100
    bias = np.mean(y_pred - y_true)
    mae = np.mean(errors)
    rmse = np.sqrt(mse)

    return {"n": len(y_true), "MSE": mse, "MARD": mard, "Bias": bias, "MAE": mae, "RMSE": rmse}


def evaluate_intra_extra(y_pred, y_true, lower_bound, upper_bound):
    """
    依照訓練時的血糖範圍 [lower_bound, upper_bound]，把 test 樣本分成兩組分別算誤差：
        intra (內插): lower_bound <= y_true <= upper_bound（落在訓練範圍內）
        extra (外推): y_true < lower_bound 或 y_true > upper_bound（落在訓練範圍外）

    回傳 (intra_metrics, extra_metrics)，兩者皆為 _compute_metrics 的回傳結果。
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    intra_mask = (y_true >= lower_bound) & (y_true <= upper_bound)
    extra_mask = ~intra_mask

    intra_metrics = _compute_metrics(y_pred[intra_mask], y_true[intra_mask])
    extra_metrics = _compute_metrics(y_pred[extra_mask], y_true[extra_mask])

    return intra_metrics, extra_metrics


def compute_metrics_by_bin(y_pred, y_true, bin_values, bin_edges: np.ndarray) -> list[dict]:
    """
    用固定的 bin_edges(跟 ratio 無關)把 test 樣本依 bin_values 分箱，逐箱計算誤差。

    bin_values 可以是真實血糖值本身(絕對值分箱：同一把尺看不同 ratio 在同一個血糖區間表現如何)，
    也可以是「離訓練邊界的距離」(距離分箱：同一把尺看不同 ratio 在同樣的外推難度下表現如何)。

    只要 bin_edges 固定不變，不同 ratio 訓練出來的模型就是用「同一把尺」評分，
    才能公平比較，不會像 Intra/Extra 那樣因為分組邊界跟著 ratio 變動，評分的樣本組成也跟著變。
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    bin_values = np.asarray(bin_values)

    rows = []
    for bin_low, bin_high in zip(bin_edges[:-1], bin_edges[1:]):
        bin_mask = (bin_values >= bin_low) & (bin_values < bin_high)
        metrics = _compute_metrics(y_pred[bin_mask], y_true[bin_mask])
        rows.append({
            "bin_low": bin_low,
            "bin_high": bin_high,
            "bin_center": (bin_low + bin_high) / 2,
            **(metrics if metrics is not None else {"n": 0, "MSE": None, "MARD": None, "Bias": None, "MAE": None, "RMSE": None}),
        })
    return rows


def _print_metrics(name, metrics):
    if metrics is None:
        print(f"[{name}] 無樣本，略過")
        return
    print(f"[{name}] n={metrics['n']}, MSE={metrics['MSE']:.4f}, MARD={metrics['MARD']:.4f}, "
          f"Bias={metrics['Bias']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

def load_features_and_labels(dir_path: str, used_feature_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_indices = np.where(used_feature_array == 1)[0]
    x, y, file_list = [], [], []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.getsize(file_path) == 0:
            print(f"[load_features_and_labels] 跳過空檔案: {file_path}")
            continue
        try:
            values = np.genfromtxt(file_path, delimiter='')[:, 1]
        except IndexError:
            print(f"[load_features_and_labels] 檔案格式異常，跳過: {file_path}")
            continue
        x.append(values[feature_indices])
        y.append(int(file.split("_")[-1].split(".")[0]))
        file_list.append(file)
    return np.array(x), np.array(y), file_list

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

    expansion_ratio_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    uuid = "2295"
    glucose_type = "Normal"
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results", uuid, glucose_type)
    os.makedirs(results_dir, exist_ok=True)
    # 逐一 ratio 的散佈圖只是健檢用的診斷圖(主要結論看分箱圖就夠了)，另外放子資料夾，results_dir 才不會被塞滿
    diagnostics_dir = os.path.join(results_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    if glucose_type not in GLUCOSE_BOUNDS:
        raise ValueError(f"Invalid glucose type: {glucose_type}，可選: {list(GLUCOSE_BOUNDS)}")
    class_lower, class_upper = resolve_glucose_bounds(uuid, glucose_type, base_dir=base_dir)

    # 依 ratio 限縮並複製到 dataset/EXP/.../ratio_*（已存在可註解掉，不必每次重跑）
    data_preprocessing(expansion_ratio_list, uuid, glucose_type)
    used_feature_array = np.array([row[0] for row in used_feature_dic])

    # 測試集固定用完整的 Test，不隨 ratio 改變，搬到迴圈外面只讀取一次(省時間，也確保每個 ratio 用的是同一批樣本)
    X_test_raw, y_test, _ = load_features_and_labels(
        os.path.join(base_dir, "dataset", uuid, "Test", glucose_type), used_feature_array
    )

    # 固定分箱區間(跟 ratio 無關)：每 10 mg/dL 一格，涵蓋整個 test 集的血糖範圍，
    # 讓不同 ratio 訓練出來的模型可以用同一把尺公平比較(見下方 compute_metrics_by_bin)
    BIN_WIDTH = 10
    bin_start = np.floor(y_test.min() / BIN_WIDTH) * BIN_WIDTH
    bin_end = np.ceil(y_test.max() / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH
    bin_edges = np.arange(bin_start, bin_end, BIN_WIDTH)

    # 固定的「離訓練邊界距離」分箱(跟 ratio 無關)：用 class lower 當正規化基準，
    # 距離越遠代表要求模型外推得越遠，這樣才能公平比較「不同 ratio 在同樣外推難度下的表現」，
    # 而不是像整組 Extra 平均那樣，範圍越窄的 ratio 天生就要背負更遠、更難的外推點。
    LOWER_BASE = class_lower

    DIST_BIN_WIDTH_PCT = 5
    narrowest_lower, narrowest_upper = get_ratio_bounds(
        max(expansion_ratio_list), class_lower, class_upper
    )
    max_dist = max(narrowest_lower - y_test.min(), y_test.max() - narrowest_upper, 0)
    max_dist_pct = np.ceil(max_dist / LOWER_BASE * 100 / DIST_BIN_WIDTH_PCT) * DIST_BIN_WIDTH_PCT
    dist_bin_edges = np.arange(0, max_dist_pct + DIST_BIN_WIDTH_PCT, DIST_BIN_WIDTH_PCT)

    summary_rows = []  # 彙整所有 ratio 的 Intra/Extra 結果(注意：Intra/Extra 的樣本組成會隨 ratio 改變，只能粗略參考)
    bin_summary_rows = []  # 彙整所有 ratio 在「固定絕對值分箱」下的結果，公平比較同一個血糖值誰預測得準
    dist_summary_rows = []  # 彙整所有 ratio 在「固定距離分箱」下的結果，公平比較同樣外推距離誰預測得準

    for ratio in expansion_ratio_list:
        ratio_start = time.time()
        lower_bound, upper_bound = get_ratio_bounds(ratio, class_lower, class_upper)
        print(f"\n===== ratio={ratio:.2f} 開始 (train range=[{lower_bound}, {upper_bound}]) =====", flush=True)

        # 載入訓練集(1D array)；資料夾由上方 data_preprocessing 產生
        X_train, y_train, file_list = load_features_and_labels(
            os.path.join(base_dir, "dataset", "EXP", uuid, "Train", glucose_type, f"ratio_{ratio:.2f}"),
            used_feature_array,
        )

        # Handle NaN values by replacing with column means
        col_means = np.nanmean(X_train, axis=0)  # Calculate column means ignoring NaNs
        nan_indices = np.where(np.isnan(X_train))  # Find the NaN indices
        X_train[nan_indices] = np.take(col_means, nan_indices[1])  # Replace NaNs with column means
        ## Scale the features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 每個 ratio 用自己訓練集的 col_means 填補 test 的 NaN，所以要用複本，不能直接改到 X_test_raw
        X_test = X_test_raw.copy()
        nan_indices = np.where(np.isnan(X_test))  # Find the NaN indices
        X_test[nan_indices] = np.take(col_means, nan_indices[1])  # Replace NaNs with column means(來自訓練集的 mean)
        # Scale the features using StandardScaler(使用訓練集的 scaler)
        X_test_scaled = scaler.transform(X_test)

        (overall_metrics, max_glucose_value, min_glucose_value, model,
         y_pred_train, y_pred_test, intra_metrics, extra_metrics, test_clipped_mask) = svr_manual(
            X_train_scaled, y_train, X_test_scaled, y_test,
            lower_bound=lower_bound, upper_bound=upper_bound,
        )

        intra_mask = (y_test >= lower_bound) & (y_test <= upper_bound)
        extra_mask = ~intra_mask

        for group_name, metrics, group_mask in (
            ("Overall", overall_metrics, np.ones_like(y_test, dtype=bool)),
            ("Intra", intra_metrics, intra_mask),
            ("Extra", extra_metrics, extra_mask),
        ):
            if metrics is None:  # 某個 ratio 的 test 裡剛好沒有外推樣本時會是 None
                continue
            summary_rows.append({
                "ratio": ratio,
                "group": group_name,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "n_clipped": int(np.sum(test_clipped_mask[group_mask])),
                **metrics,
            })

        # -------- 固定絕對值分箱評分：同一把尺，公平比較不同 ratio 的模型 --------
        bin_rows = compute_metrics_by_bin(y_pred_test, y_test, y_test, bin_edges)
        for row in bin_rows:
            is_intra_bin = (row["bin_center"] >= lower_bound) and (row["bin_center"] <= upper_bound)
            bin_summary_rows.append({
                "ratio": ratio,
                "is_intra": is_intra_bin,
                **row,
            })

        # -------- 固定距離分箱評分：只看 Extra(外推)樣本，依「離該 ratio 自己邊界的距離」分箱 --------
        # 距離 0 代表 Intra，一律排除，只留下真正在外推的樣本才有意義比較外推難度
        dist_to_boundary = np.where(
            y_test < lower_bound, lower_bound - y_test,
            np.where(y_test > upper_bound, y_test - upper_bound, 0.0),
        )
        dist_pct = dist_to_boundary / LOWER_BASE * 100
        is_truly_extra = dist_to_boundary > 0
        dist_rows = compute_metrics_by_bin(
            y_pred_test[is_truly_extra], y_test[is_truly_extra], dist_pct[is_truly_extra], dist_bin_edges
        )
        for row in dist_rows:
            dist_summary_rows.append({
                "ratio": ratio,
                "dist_pct_low": row.pop("bin_low"),
                "dist_pct_high": row.pop("bin_high"),
                "dist_pct_center": row.pop("bin_center"),
                **row,
            })

        # -------- 視覺化：預測 vs 真實，內插/外推分色 + 對角線 --------
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test[intra_mask], y_pred_test[intra_mask], alpha=0.6, label="Intra (train range內)")
        plt.scatter(y_test[extra_mask], y_pred_test[extra_mask], alpha=0.6, color="red", label="Extra (train range外)")

        lim_min = min(y_test.min(), y_pred_test.min())
        lim_max = max(y_test.max(), y_pred_test.max())
        plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", label="Perfect prediction (y=x)")

        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"SVR Model Expansion EXP: ratio={ratio:.2f} (train range=[{lower_bound}, {upper_bound}])")
        plt.legend()
        plt.savefig(os.path.join(diagnostics_dir, f"SVR_model_ratio_{ratio:.2f}.png"))
        plt.close()

        print(f"===== ratio={ratio:.2f} 完成，總耗時 {time.time() - ratio_start:.1f} 秒 =====", flush=True)

    # -------- 把所有 ratio 的結果彙整成一份 CSV--------
    summary_csv_path = os.path.join(results_dir, "SVR_model_summary.csv")
    fieldnames = ["ratio", "group", "lower_bound", "upper_bound", "n", "n_clipped", "MSE", "MARD", "Bias", "MAE", "RMSE"]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"彙整結果已存到: {summary_csv_path}")

    # -------- 固定分箱結果也存成 CSV，是跨 ratio 公平比較用的主要依據 --------
    bin_csv_path = os.path.join(results_dir, "SVR_model_summary_by_bin.csv")
    bin_fieldnames = ["ratio", "is_intra", "bin_low", "bin_high", "bin_center", "n", "MSE", "MARD", "Bias", "MAE", "RMSE"]
    with open(bin_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bin_fieldnames)
        writer.writeheader()
        writer.writerows(bin_summary_rows)

    print(f"固定分箱結果已存到: {bin_csv_path}")

    # -------- 畫出「MAE vs 血糖分箱」，每個 ratio 一條線，同一張圖上直接比較外推能力 --------
    plt.figure(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    for i, ratio in enumerate(expansion_ratio_list):
        rows = [r for r in bin_summary_rows if r["ratio"] == ratio and r["n"] > 0]
        centers = [r["bin_center"] for r in rows]
        maes = [r["MAE"] for r in rows]
        intra_flags = [r["is_intra"] for r in rows]
        color = cmap(i / max(len(expansion_ratio_list) - 1, 1))
        plt.plot(centers, maes, marker="o", color=color, label=f"ratio={ratio:.2f}")
        # 用實心點標示該模型「自己訓練範圍內」的分箱，空心點代表該模型是在外推
        for c, m, is_intra in zip(centers, maes, intra_flags):
            if not is_intra:
                plt.scatter([c], [m], facecolors="none", edgecolors=color, s=60, zorder=3)

    plt.xlabel("True glucose value (bin center)")
    plt.ylabel("MAE")
    plt.title("MAE by fixed glucose bin, across ratios (實心=該模型的訓練範圍內, 空心=外推)")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "SVR_model_MAE_by_bin.png"))
    plt.close()

    # -------- 固定距離分箱結果存成 CSV：回答「訓練範圍窄化本身，是否傷害外推能力」的主要依據 --------
    dist_csv_path = os.path.join(results_dir, "SVR_model_summary_by_distance.csv")
    dist_fieldnames = ["ratio", "dist_pct_low", "dist_pct_high", "dist_pct_center", "n", "MSE", "MARD", "Bias", "MAE", "RMSE"]
    with open(dist_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dist_fieldnames)
        writer.writeheader()
        writer.writerows(dist_summary_rows)

    print(f"固定距離分箱結果已存到: {dist_csv_path}")

    # -------- 畫出「MAE vs 離訓練邊界距離(%)」，每個 ratio 一條線 --------
    # 這張圖才是直接回答「同樣要求外推這麼遠，訓練範圍窄的模型是不是比較差」的關鍵證據，
    # 不會像整組 Extra 平均那樣，被「範圍窄的 ratio 天生就要外推更遠」這個混淆因子污染。
    plt.figure(figsize=(9, 6))
    for i, ratio in enumerate(expansion_ratio_list):
        rows = [r for r in dist_summary_rows if r["ratio"] == ratio and r["n"] > 0]
        if not rows:  # ratio=0(訓練範圍涵蓋整個 test 集)完全沒有外推樣本，不會有線
            continue
        centers = [r["dist_pct_center"] for r in rows]
        maes = [r["MAE"] for r in rows]
        color = cmap(i / max(len(expansion_ratio_list) - 1, 1))
        plt.plot(centers, maes, marker="o", color=color, label=f"ratio={ratio:.2f}")

    plt.xlabel(f"離訓練邊界的距離 (% of lower_base={LOWER_BASE})")
    plt.ylabel("MAE")
    plt.title("MAE by distance-from-boundary bin, across ratios(只看外推樣本)")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "SVR_model_MAE_by_distance.png"))
    plt.close()