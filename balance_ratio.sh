# 建議順序：full reference → max × ratio
# base_path = Model_exp_1（需已有 Model_exp_1/70_30/full/Regression_Features；從 pre 複製）
# 訓練模式自動：uuid×class Train n<1200 → manual(SVR-RBF)；n≥1200 → meta(RFECV-linear+XGB)
# 共線性：remove_collinearity（Pearson|r|>0.95 → VIF<15）後再進 scaler / 模型

# ---------- 之前實驗（先註解）----------
# # 1) full 當 reference
# python Regression_Model_Predictor_meta.py --no-downsample --skip-feature-extract --skip-normalize --skip-re-add

# # 2) median × 較大 ratio
# python Regression_Model_Predictor_meta.py --downsample-ratio 1.0 --cap-ref median --skip-feature-extract --skip-normalize --skip-re-add
# python Regression_Model_Predictor_meta.py --downsample-ratio 0.9 --cap-ref median --skip-feature-extract --skip-normalize --skip-re-add
# python Regression_Model_Predictor_meta.py --downsample-ratio 0.8 --cap-ref median --skip-feature-extract --skip-normalize --skip-re-add
# python Regression_Model_Predictor_meta.py --downsample-ratio 0.6 --cap-ref median --skip-feature-extract --skip-normalize --skip-re-add

# # 3) max × 較大 ratio（舊：含 r=1.0）
# python Regression_Model_Predictor_meta.py --downsample-ratio 1.0 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
# python Regression_Model_Predictor_meta.py --downsample-ratio 0.9 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
# python Regression_Model_Predictor_meta.py --downsample-ratio 0.8 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
# python Regression_Model_Predictor_meta.py --downsample-ratio 0.6 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
# python Regression_Model_Predictor_meta.py --downsample-ratio 0.4 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
# python Regression_Model_Predictor_meta.py --downsample-ratio 0.2 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add

# ---------- 本次實驗 ----------
# 1) full reference
python Regression_Model_Predictor_meta.py --no-downsample --skip-feature-extract --skip-normalize --skip-re-add

# 2) max × ratio（0.9 / 0.8 / 0.6 / 0.4 / 0.2）
python Regression_Model_Predictor_meta.py --downsample-ratio 0.9 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
python Regression_Model_Predictor_meta.py --downsample-ratio 0.8 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
python Regression_Model_Predictor_meta.py --downsample-ratio 0.6 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
python Regression_Model_Predictor_meta.py --downsample-ratio 0.4 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
python Regression_Model_Predictor_meta.py --downsample-ratio 0.2 --cap-ref max --skip-feature-extract --skip-normalize --skip-re-add
