def initialize():

    global errorcode
    global message

    errorcode = 0
    message = ' '

"""

# Error Code
# Error Message


# -502
"An error occurred the set_path function of the SVM_Model_Builder_Predictor.py: "
"No training data in path: {}.".format(trainindex_path)
# 讀取arrangement完的training dataset路徑錯誤

# -503
"An error occurred in the set_path function of the SVM_Model_Builder_Predictor.py: 
"No testing data in path: {}.".format(testindex_path))."
# 讀取arrangement完的testing dataset路徑錯誤

# -504
"An error occurred in the set_path function of the SVM_Model_Builder_Predictor.py: 
No ECG raw data in path: {}.format(rawdata_path))"
# 讀取arrangement完的ECG訊號路徑錯誤

# -------Extract Features------------

# -202
"An error occurred in the concat_features_label function  of the SVM_Model_Builder_Predictor.py"
"of the bg_package.features_extraction: "
"Quality of {High/Normal/Low} blood glucose ECG signals are bad in the {Train/Test} dataset during model building or predicting."
# 特徵擷取時ECG訊號皆是差的 (差的標準是wave stable高於閾值 無法進行特徵擷取)  

# -------Load Data------------

# -505
"An error occurred in the decide_classify_level function of the SVM_Model_Builder_Predictor.py"
"from svmmodel_prediction.py: 
Only or no normal data." 
# 只有中血糖資料 無法進行分類
                                                  
# -------Build/Predict SVM Model------------

# -506
"An error occurred in the build_model function of the SVM_Model_Builder_Predictor.py
from bg_package.swmmodel_prediction: "
"fail to read the result of features parquet file"
# 建立模型時無法讀取特徵擷取的parquet檔

# -507
"An error occurred in the build_svm_model function of the SVM_Model_Builder_Predictor.py from "
"bg_package.swmmodel_prediction: "
"No features extraction in Train/Test dataset"
# parquet檔中缺少training或testing其一

# -603
"An error occurred in the predict_bg function of the SVM_Model_Builder_Predictor.py"
"from bg_package.svmmodel_prediction: "
"No built model for uuid {} exists".format(self.uuid)"
# Predict時無法讀取模型
                               

"""