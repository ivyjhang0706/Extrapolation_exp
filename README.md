# SVR 外推 / downsample ratio 實驗

## 目前設定

- **資料根目錄**：`Model_exp_1/`
- **自動訓練模式**（依 uuid×class 當下 Train 筆數）：
  - **n < 1200** → `manual`：`remove_collinearity` → **SVR-RBF** → MARD（不跑 RFECV / XGB）
  - **n ≥ 1200** → `meta`：`remove_collinearity` → **SVR-linear + RFECV** → SVR pred 當特徵 → **XGB** → MARD（**不再**做 manual vs auto bakeoff）
- 共線性：Pearson `|r|>0.95` 衝突保留與血糖相關較高者；再 VIF 砍到 `<15`（至少留 5 維）
- `_svr_manual` 已關掉逐筆 prediction print（避免 test 上千筆拖慢）
- 現在meta or svr only是看資料筆數判定 所以沒有training_mode='meta'

## 目錄（方案 A）

- `pre/Model_re_add/`：底稿（勿覆寫）
- `Model_exp_1/70_30/full/Regression_Features/`：從 pre 複製的完整特徵
- `Model_exp_1/70_30/ds_max_r{xx}/`：各 ratio 實驗（copy full → downsample → 訓練）



## 建議跑法（見 `balance_ratio.sh`）

```bash
# 1) full 當 reference
python Regression_Model_Predictor_meta.py --no-downsample --skip-feature-extract --skip-normalize --skip-re-add

# 2) max × ratio
python Regression_Model_Predictor_meta.py --downsample-ratio 0.9 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
python Regression_Model_Predictor_meta.py --downsample-ratio 0.8 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
python Regression_Model_Predictor_meta.py --downsample-ratio 0.6 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
python Regression_Model_Predictor_meta.py --downsample-ratio 0.4 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
python Regression_Model_Predictor_meta.py --downsample-ratio 0.2 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
```



### ratio 說明

- `cap_ref=max`：`cap = floor(max_count * ratio)`；`count≤cap` 的血糖值全留，只砍尖峰



## CLI 常用參數


| 參數                                 | 說明                                             |
| ---------------------------------- | ---------------------------------------------- |
| `--skip-feature-extract`           | 不重抽 ECG 特徵，用既有 `70_30/full`                    |
| `--skip-normalize`                 | 跳過 Raw→Normalized                              |
| `--no-downsample`                  | 直接在 full 上訓練                                   |
| `--downsample-ratio` / `--cap-ref` | downsample 強度                                  |
| `--skip-re-add`                    | 跳過開頭 GlucoseData re_add                        |
| `--legacy-balanced`                | 用既有 `70_30/Regression_Features`（data_balanced） |


