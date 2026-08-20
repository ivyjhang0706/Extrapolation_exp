from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

import Regression_Model_Predictor_meta_v1 as base


CURRENT_N_THRESHOLD = 1200
NUM_CV = 3
PROCESSNUM = 16
DATA_SPLIT = "70_30"
MODEL_EXP_SRC = "Model_exp_1"
MODEL_EXP_DST = "Model_exp_2"
UUID_LIST = [
    "2133",
    "2205",
    "2253",
    "2199",
    "2218",
    "2216",
    "2224",
    "2278",
    "2286",
    "2294",
    "2298",
    "2352",
    "2378",
]

# 讓 v1 內的 BuildModel/print 可沿用同一份全域表
performance_table = []
base.performance_table = performance_table


def _reset_performance_table():
    performance_table.clear()
    base.performance_table = performance_table


def BuildRegressionModel(uuid, basepath, splitting_ratio, model_type_array):
    status = 1
    errorcode = "0"
    message = "ok"
    model_current_best_performance_txtfile = []
    model_historic_best_performance_txtfile = []
    classes_num = len(model_type_array)
    current_time_str = time.strftime("%Y_%m_%d_%H%M", time.localtime())
    code_version = base.get_version()

    if classes_num == 1:
        regression_model_output_folder = os.path.join(
            basepath, splitting_ratio, "Best_OneClass_Regression_Model", uuid
        )
        os.makedirs(regression_model_output_folder, exist_ok=True)
        model_current_best_performance_txtfile.append(
            os.path.join(
                regression_model_output_folder,
                "Performance_" + code_version + "_" + current_time_str + "_" + model_type_array[0] + ".txt",
            )
        )
        model_historic_best_performance_txtfile.append(
            os.path.join(
                regression_model_output_folder,
                "Historic_Best_Performance_" + model_type_array[0] + ".txt",
            )
        )
    elif classes_num == 2:
        regression_model_output_folder = os.path.join(
            basepath, splitting_ratio, "Best_TwoClasses_Regression_Model", uuid
        )
        os.makedirs(regression_model_output_folder, exist_ok=True)
        if model_type_array[0] == "Normal" and model_type_array[1] == "High":
            model_current_best_performance_txtfile.extend(
                [
                    os.path.join(regression_model_output_folder, "Performance_" + code_version + "_" + current_time_str + "_Normal.txt"),
                    os.path.join(regression_model_output_folder, "Performance_" + code_version + "_" + current_time_str + "_High.txt"),
                ]
            )
            model_historic_best_performance_txtfile.extend(
                [
                    os.path.join(regression_model_output_folder, "Historic_Best_Performance_Normal.txt"),
                    os.path.join(regression_model_output_folder, "Historic_Best_Performance_High.txt"),
                ]
            )
        else:
            model_current_best_performance_txtfile.extend(
                [
                    os.path.join(regression_model_output_folder, "Performance_" + code_version + "_" + current_time_str + "_Normal.txt"),
                    os.path.join(regression_model_output_folder, "Performance_" + code_version + "_" + current_time_str + "_Low.txt"),
                ]
            )
            model_historic_best_performance_txtfile.extend(
                [
                    os.path.join(regression_model_output_folder, "Historic_Best_Performance_Normal.txt"),
                    os.path.join(regression_model_output_folder, "Historic_Best_Performance_Low.txt"),
                ]
            )
    else:
        regression_model_output_folder = os.path.join(
            basepath, splitting_ratio, "Best_ThreeClasses_Regression_Model", uuid
        )
        os.makedirs(regression_model_output_folder, exist_ok=True)
        model_current_best_performance_txtfile.extend(
            [
                os.path.join(regression_model_output_folder, "Performance_" + code_version + "_" + current_time_str + "_Normal.txt"),
                os.path.join(regression_model_output_folder, "Performance_" + code_version + "_" + current_time_str + "_High.txt"),
                os.path.join(regression_model_output_folder, "Performance_" + code_version + "_" + current_time_str + "_Low.txt"),
            ]
        )
        model_historic_best_performance_txtfile.extend(
            [
                os.path.join(regression_model_output_folder, "Historic_Best_Performance_Normal.txt"),
                os.path.join(regression_model_output_folder, "Historic_Best_Performance_High.txt"),
                os.path.join(regression_model_output_folder, "Historic_Best_Performance_Low.txt"),
            ]
        )

    feature_method = "raw"
    MODE_N_THRESHOLD = CURRENT_N_THRESHOLD
    num_cv = NUM_CV
    used_feature_array = np.array([row[0] for row in base.used_feature_dic])
    false_count = 0

    for i, model_type in enumerate(model_type_array):
        print("It is dealing with " + model_type + " model...")

        current_path = os.path.join(basepath, splitting_ratio, "Regression_Features", uuid, "Train")
        current_test_path = os.path.join(basepath, splitting_ratio, "Regression_Features", uuid, "Test")
        train_dir = os.path.join(current_path, model_type)
        test_dir = os.path.join(current_test_path, model_type)
        filelist_model_type = os.listdir(train_dir) if os.path.isdir(train_dir) else []
        filelist_test_model_type = os.listdir(test_dir) if os.path.isdir(test_dir) else []

        if len(filelist_model_type) < num_cv:
            continue
        if len(filelist_model_type) == 0 or len(filelist_test_model_type) == 0:
            print("Runing the BuildRegressionModel function of Regression_Model_Predictor.py: No training or testing data!")
            false_count += 1
            continue

        n_train = len(filelist_model_type)
        training_mode = "manual" if n_train < MODE_N_THRESHOLD else "meta"
        print(f"[mode] uuid={uuid} {model_type} n_train={n_train} → {training_mode} (threshold={MODE_N_THRESHOLD})")

        dataset = base.Regression_ECGDataset(
            dir_path=current_path, used_feature_array=used_feature_array, type=model_type, method=feature_method
        )
        train_loader = base.DataLoader(dataset, shuffle=False)
        testdata = base.Regression_ECGDataset(
            dir_path=current_test_path, used_feature_array=used_feature_array, type=model_type, method=feature_method
        )
        test_loader = base.DataLoader(testdata)

        X_all, y_all, data_all, target_all, file_names_all = [], [], [], [], []
        for batch_x, batch_y in train_loader:
            X_all.append(batch_x.squeeze().numpy())
            y_all.append(batch_y.numpy())
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        col_means = np.nanmean(X_all, axis=0)
        nan_indices = np.where(np.isnan(X_all))
        X_all[nan_indices] = np.take(col_means, nan_indices[1])

        for idx, (data, target) in enumerate(test_loader):
            data_all.append(data.squeeze().numpy())
            target_all.append(target.numpy())
            file_names_all.append(testdata.file_list[idx])
        data_all = np.vstack(data_all)
        target_all = np.concatenate(target_all)
        nan_idx_test = np.where(np.isnan(data_all))
        data_all[nan_idx_test] = np.take(col_means, nan_idx_test[1])

        n_feat_before = X_all.shape[1]
        X_all, keep_idx = dataset.remove_collinearity(X_all, y_all)
        data_all = data_all[:, keep_idx]
        col_means = np.asarray(col_means)[keep_idx]
        print(f"[collinearity] {model_type}: kept {len(keep_idx)}/{n_feat_before} features, keep_idx={keep_idx}")

        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        data_all_scaled = scaler.transform(data_all)

        selected_feature_indices = None
        Bias = MAE = RMSE = 0.0

        if training_mode == "manual":
            print("manual mode (SVR-RBF, no RFECV / no XGB)")
            (
                MSE,
                MARD,
                max_glucose_value,
                min_glucose_value,
                model,
                _y_pred_train,
                y_pred_test,
            ) = base._svr_manual(X_all_scaled, y_all, data_all_scaled, target_all)
            errors = np.abs(target_all - y_pred_test)
            Bias = float(np.mean(y_pred_test - target_all))
            MAE = float(np.mean(errors))
            RMSE = float(np.sqrt(MSE))
            selected_feature_indices = None
            current_svr_model_name = "SVR_Model_" + current_time_str + "_" + model_type + ".pkl"
            joblib.dump(model, os.path.join(regression_model_output_folder, current_svr_model_name))
            best_y_pred = y_pred_test
        else:
            print("meta mode (RFECV-linear + XGB, no manual compare)")
            (
                _MSE_auto,
                _MARD_auto,
                max_glucose_value_auto,
                min_glucose_value_auto,
                rfecv,
                svr,
                selected_feature_indices,
                y_pred_train_auto,
                y_pred_test_auto,
            ) = base._svr_auto(X_all_scaled, y_all, data_all_scaled, target_all, num_cv)
            current_rfecv_name = "Rfecv_Model_" + current_time_str + "_" + model_type + ".pkl"
            current_svr_model_name = "SVR_Model_" + current_time_str + "_" + model_type + ".pkl"
            joblib.dump(rfecv, os.path.join(regression_model_output_folder, current_rfecv_name))
            joblib.dump(svr, os.path.join(regression_model_output_folder, current_svr_model_name))

            X_train_residual = np.column_stack((X_all_scaled, y_pred_train_auto))
            X_test_residual = np.column_stack((data_all_scaled, y_pred_test_auto))
            (
                MSE,
                MARD,
                Bias,
                MAE,
                RMSE,
                max_glucose_value,
                min_glucose_value,
                xgb_model,
                y_pred,
            ) = base._xgboost(X_train_residual, y_all, X_test_residual, target_all)
            current_meta_model_name = "Meta_Model_" + current_time_str + "_" + model_type + ".pkl"
            meta_model_path = os.path.join(regression_model_output_folder, current_meta_model_name)
            meta_model_package = {
                "mode": "meta",
                "feature_scaler": scaler,
                "collinearity_keep_idx": keep_idx,
                "svr_model": svr,
                "rfecv": rfecv,
                "selected_feature_indices": selected_feature_indices,
                "xgb_model": xgb_model,
                "model_type": model_type,
            }
            joblib.dump(meta_model_package, meta_model_path)
            xgb_model.save_model(os.path.join(regression_model_output_folder, "xGB_Model_" + current_time_str + "_" + model_type + ".json"))
            best_y_pred = y_pred
            print("Meta model saved:", meta_model_path)

        base.save_predictions_csv(
            file_names_all, best_y_pred, target_all, regression_model_output_folder, current_time_str, model_type
        )
        print(
            f"MSE:{MSE:.4f}, MARD:{MARD:.4f}, Bias:{Bias:.4f}, MAE:{MAE:.4f}, RMSE:{RMSE:.4f}, "
            f"largest_value:{max_glucose_value:.1f}, lowest_value:{min_glucose_value:.1f}"
        )

        with open(model_current_best_performance_txtfile[i], "w") as f:
            f.write(f"largest_value:{max_glucose_value:.1f}\n")
            f.write(f"lowest_value:{min_glucose_value:.1f}\n")
            if selected_feature_indices is None:
                f.write("used feature array: all\n")
            else:
                f.write("used feature array: " + str(selected_feature_indices) + "\n")
            f.write(f"MSE:{MSE:.4f}\n")
            f.write(f"MARD:{MARD:.4f}\n")
            f.write(f"Bias:{Bias:.4f}\n")
            f.write(f"MAE:{MAE:.4f}\n")
            f.write(f"RMSE:{RMSE:.4f}\n")

        performance_table.append(
            [
                uuid,
                model_type,
                str(n_train),
                training_mode,
                str(MODE_N_THRESHOLD),
                f"{MSE:.4f}",
                f"{MARD:.4f}",
                f"{Bias:.4f}",
                f"{MAE:.4f}",
                f"{RMSE:.4f}",
                f"{max_glucose_value:.1f}",
                f"{min_glucose_value:.1f}",
            ]
        )

    if false_count == len(model_type_array):
        status = -1
        errorcode = "-907"
        message = "No any regression model have been built!"
    return status, errorcode, message


base.BuildRegressionModel = BuildRegressionModel


def ensure_model_exp2_baseline(project_root: Path):
    src = project_root / MODEL_EXP_SRC / DATA_SPLIT / "Regression_Features"
    dst = project_root / MODEL_EXP_DST / DATA_SPLIT / "Regression_Features"
    if dst.exists():
        print(f"[baseline] 已存在：{dst}")
        return
    if not src.exists():
        raise FileNotFoundError(f"baseline Regression_Features 不存在：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"[baseline] 複製 {src} → {dst}")


def re_add_features(uuid_list, model_exp2_root: Path):
    base_dir = model_exp2_root / DATA_SPLIT / "Regression_Features"
    for uuid in uuid_list:
        removed_root = base_dir / uuid / "Remove_Data" / "Train"
        train_root = base_dir / uuid / "Train"
        if not removed_root.exists():
            print(f"[re_add] uuid={uuid} 沒有 Remove_Data/Train，略過")
            continue
        added_count = 0
        skipped_count = 0
        for level in ("Low", "Normal", "High"):
            src_dir = removed_root / level
            dst_dir = train_root / level
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            for file in src_dir.iterdir():
                if not file.is_file():
                    continue
                target = dst_dir / file.name
                if target.exists():
                    skipped_count += 1
                    continue
                shutil.copy2(file, target)
                added_count += 1
        print(f"[re_add] uuid={uuid}: 加回 {added_count} 筆、跳過已存在 {skipped_count} 筆")


def remove_outliers_after_readd(uuid_list, model_exp2_root: Path):
    """re-add 之後再跑 data_balanced（IQR outlier remove）。"""
    arranger = base.DataArrangement()
    for uuid in uuid_list:
        print(f"[data_balanced] uuid={uuid} start")
        errorcode, message = arranger.data_balanced(
            uuid=uuid,
            txtfile_path=str(model_exp2_root),
            splitting_ratio=DATA_SPLIT,
        )
        print(f"[data_balanced] uuid={uuid} errorcode={errorcode} message={message}")


def ensure_readme(model_exp2_root: Path):
    readme_path = model_exp2_root / "README.md"
    content = """# Model_exp_2

這一版固定資料順序為：複製 baseline → re-add → data_balanced（remove outlier）。

## 固定資料前處理
- baseline 來源：`Model_exp_1/70_30/Regression_Features`
- 複製到：`Model_exp_2/70_30/Regression_Features`
- 先對 `Remove_Data/Train/{Low,Normal,High}` 做 re-add 回 `Train/{Low,Normal,High}`
- 再跑 `data_balanced` 砍 outlier
- `n_train` 以 **re-add + remove outlier 後** 的筆數計算

## 模式切換
- `n_train < n_threshold` → `manual`
  - `remove_collinearity`
  - `SVR-RBF`（共線性後全特徵，不跑 RFECV / XGB）
- `n_train >= n_threshold` → `meta`
  - `remove_collinearity`
  - `SVR-linear + RFECV (cv=3)`
  - `SVR pred` 當特徵
  - `XGB`

## 執行方式
```bash
python Regression_Model_Predictor_meta_v2.py --n-threshold 900
python Regression_Model_Predictor_meta_v2.py --n-threshold 1200
python Regression_Model_Predictor_meta_v2.py --n-threshold 1500
```

## 輸出
- 各 threshold 自己的模型結果：`Model_exp_2/70_30/n{threshold}/...`
- 單次摘要：`Model_exp_2/70_30/n{threshold}/threshold_summary.csv`
- 跨 threshold 比較：`Model_exp_2/70_30/threshold_compare.csv`
"""
    readme_path.write_text(content, encoding="utf-8")


def write_threshold_summary(model_exp2_root: Path, threshold: int):
    run_dir = model_exp2_root / DATA_SPLIT / f"n{threshold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "threshold_summary.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["uuid", "class", "n_train", "mode", "threshold", "MSE", "MARD", "Bias", "MAE", "RMSE", "largest_value", "lowest_value"]
        )
        writer.writerows(performance_table)
    print(f"[summary] 已寫入 {out}")


def update_threshold_compare(model_exp2_root: Path, threshold: int):
    compare_path = model_exp2_root / DATA_SPLIT / "threshold_compare.csv"
    rows = {}
    if compare_path.exists():
        with compare_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows[(row["uuid"], row["class"])] = row

    for row in performance_table:
        uuid, model_type, n_train, mode, _th, _mse, mard, *_rest = row
        key = (uuid, model_type)
        record = rows.setdefault(
            key,
            {
                "uuid": uuid,
                "class": model_type,
                "n_train": n_train,
                "mode_900": "",
                "MARD_900": "",
                "mode_1200": "",
                "MARD_1200": "",
                "mode_1500": "",
                "MARD_1500": "",
            },
        )
        record["n_train"] = n_train
        record[f"mode_{threshold}"] = mode
        record[f"MARD_{threshold}"] = mard

    fieldnames = [
        "uuid",
        "class",
        "n_train",
        "mode_900",
        "MARD_900",
        "mode_1200",
        "MARD_1200",
        "mode_1500",
        "MARD_1500",
    ]
    with compare_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])
    print(f"[compare] 已更新 {compare_path}")


def main():
    global CURRENT_N_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Model_exp_2 threshold experiment: re-add + data_balanced + manual/meta compare"
    )
    parser.add_argument("--n-threshold", type=int, required=True, help="mode 切換門檻，例如 900 / 1200 / 1500")
    args = parser.parse_args()

    CURRENT_N_THRESHOLD = int(args.n_threshold)
    project_root = Path(__file__).resolve().parent
    model_exp2_root = project_root / MODEL_EXP_DST
    ensure_readme(model_exp2_root)
    ensure_model_exp2_baseline(project_root)
    re_add_features(UUID_LIST, model_exp2_root)
    remove_outliers_after_readd(UUID_LIST, model_exp2_root)
    _reset_performance_table()

    run_rel = os.path.join(DATA_SPLIT, f"n{CURRENT_N_THRESHOLD}")
    features_full_rel = DATA_SPLIT

    for i, uuid in enumerate(UUID_LIST):
        print("index:", i, " uuid:", uuid)
        srj_db_path = r"D:\DataDB" + os.sep + uuid
        status, errorcode, message = base.BuildModel(
            uuid,
            str(model_exp2_root),
            srj_db_path,
            "",
            processnum=PROCESSNUM,
            splitting_ratio=DATA_SPLIT,
            run_rel=run_rel,
            features_full_rel=features_full_rel,
            skip_feature_extract=True,
            skip_normalize=True,
            do_downsample=False,
            seed=42,
        )
        print("status:", str(status), " error code:", errorcode, " message:", message)

    write_threshold_summary(model_exp2_root, CURRENT_N_THRESHOLD)
    update_threshold_compare(model_exp2_root, CURRENT_N_THRESHOLD)


if __name__ == "__main__":
    main()
