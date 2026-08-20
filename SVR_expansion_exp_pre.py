import numpy as np
import os
import shutil
from sklearn.svm import SVR
import time
from time import strftime
import math


# 各血糖類別的預設訓練範圍。
# High 的 upper 為 None：改由該 uuid 實際最高血糖決定（見 resolve_glucose_bounds）。
GLUCOSE_BOUNDS = {
    "Normal": (85, 170),
    "High": (180, None),
}


def _glucose_from_filename(filename: str):
    try:
        return int(filename.split("_")[-1].split(".")[0])
    except ValueError:
        return None


def _max_glucose_in_dir(dir_path: str):
    if not os.path.isdir(dir_path):
        return None
    vals = []
    for name in os.listdir(dir_path):
        g = _glucose_from_filename(name)
        if g is not None:
            vals.append(g)
    return max(vals) if vals else None


def resolve_glucose_bounds(uuid: str, glucose_type: str, base_dir: str = None) -> tuple[int, int]:
    """
    回傳該 uuid×glucose_type 用於限縮的 [lower, upper]。
    - Normal：固定 (85, 170)
    - High：lower 固定 180；upper = 該 uuid 在 Train（若有 Test 則一併納入）的最高血糖
    """
    if glucose_type not in GLUCOSE_BOUNDS:
        raise NotImplementedError(
            f"glucose_type={glucose_type} 尚未支援，目前可選: {list(GLUCOSE_BOUNDS)}"
        )
    class_lower, class_upper = GLUCOSE_BOUNDS[glucose_type]
    if class_upper is not None:
        return int(class_lower), int(class_upper)

    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    train_dir = os.path.join(base_dir, "dataset", uuid, "Train", glucose_type)
    test_dir = os.path.join(base_dir, "dataset", uuid, "Test", glucose_type)
    train_max = _max_glucose_in_dir(train_dir)
    test_max = _max_glucose_in_dir(test_dir)
    candidates = [v for v in (train_max, test_max) if v is not None]
    if not candidates:
        raise FileNotFoundError(
            f"無法推得 {uuid}/{glucose_type} 的 upper_bound：Train/Test 都沒有可解析的血糖檔名"
        )
    resolved_upper = max(candidates)
    if resolved_upper <= class_lower:
        raise ValueError(
            f"{uuid}/{glucose_type} 最高血糖={resolved_upper} <= lower={class_lower}，無法形成有效範圍"
        )
    print(
        f"[resolve_glucose_bounds] {uuid}/{glucose_type}: "
        f"lower={class_lower}, upper={resolved_upper} "
        f"(train_max={train_max}, test_max={test_max})"
    )
    return int(class_lower), int(resolved_upper)


def get_ratio_bounds(ratio: float, lower_base: int = 85, upper_base: int = 170) -> tuple[int, int]:
    """
    依照 ratio 算出訓練資料被限縮後的血糖範圍 [lower_bound, upper_bound]。
    抽成獨立函式，讓 data_preprocessing 跟後續評估(SVR_model.py)用同一份公式，
    避免兩邊各寫一次、日後改動時算出不一致的範圍。

    兩端各往內縮 ceil(ratio * span)，span = upper_base - lower_base。
    （Normal 的 span=85，與舊版 ceil(ratio * lower_base) 等價。）
    """
    span = upper_base - lower_base
    ratio_value = math.ceil(ratio * span)
    return lower_base + ratio_value, upper_base - ratio_value


def _align_to_min_count(ratio_matched_files: dict, seed: int = 42) -> dict:
    """
    把每個 ratio 的檔案清單都「下採樣」到筆數最少的那個 ratio，
    避免『訓練範圍縮小』跟『訓練樣本數變少』兩個效應混在一起，
    讓後面比較各 ratio 的表現時，只有「範圍」這一個變因不同。
    """
    min_count = min(len(files) for files in ratio_matched_files.values())
    rng = np.random.default_rng(seed)

    aligned = {}
    for ratio, files in ratio_matched_files.items():
        if len(files) > min_count:
            chosen = rng.choice(np.array(files, dtype=object), size=min_count, replace=False).tolist() # 隨機選擇 min_count 筆
        else:
            chosen = files # 如果筆數已經足夠，則不進行選擇
        aligned[ratio] = chosen

    print(f"對齊訓練筆數：所有 ratio 都下採樣到 {min_count} 筆（原始各 ratio 筆數："
          f"{ {r: len(f) for r, f in ratio_matched_files.items()} }）")
    return aligned # 每個 ratio最終選擇的檔案列表


def data_preprocessing(expansion_ratio_list: list[float], uuid: str, glucose_type: str,
                        align_to_min_count: bool = True, seed: int = 42) -> dict:
    """
    依照 expansion_ratio_list，把原始 Train 資料集依血糖值範圍限縮，
    複製到各自獨立的資料夾，用來測試 SVR 對訓練範圍外血糖值的外推(extrapolation)能力。

    以 Normal 類別為例，原始範圍為 [85, 170]：
        ratio=0.00 -> [85, 170]  (不縮減)
        ratio=0.05 -> [90, 165]
        ratio=0.10 -> [94, 161]
        ...
    High：lower 固定 180，upper 取該 uuid 實際最高血糖，再套同樣限縮公式。

    align_to_min_count=True 時，所有 ratio 最終複製的筆數都會對齊到
    「筆數最少的那個 ratio」，避免樣本數差異干擾外推能力的比較。

    回傳 dict[ratio] = (lower_bound, upper_bound)，供之後評估內插/外推誤差使用。
    """
    base_dir = os.path.dirname(__file__)
    class_lower, class_upper = resolve_glucose_bounds(uuid, glucose_type, base_dir=base_dir)

    raw_train_dir = os.path.join(base_dir, "dataset", uuid, "Train", glucose_type)
    if not os.path.isdir(raw_train_dir):
        raise FileNotFoundError(f"找不到原始訓練資料夾: {raw_train_dir}")
    raw_train_file_list = os.listdir(raw_train_dir)

    ratio_bounds = {}
    ratio_matched_files = {}

    for ratio in expansion_ratio_list:
        lower_bound, upper_bound = get_ratio_bounds(ratio, class_lower, class_upper)
        ratio_bounds[ratio] = (lower_bound, upper_bound)

        matched_files = []
        for file in raw_train_file_list:
            glucose_value = _glucose_from_filename(file)
            if glucose_value is None:
                print(f"skip file with unexpected name format: {file}")
                continue

            if lower_bound <= glucose_value <= upper_bound:
                matched_files.append(file)

        ratio_matched_files[ratio] = matched_files

    if align_to_min_count:
        ratio_matched_files = _align_to_min_count(ratio_matched_files, seed=seed)

    for ratio, matched_files in ratio_matched_files.items():
        lower_bound, upper_bound = ratio_bounds[ratio]
        new_train_dir = os.path.join(base_dir, "dataset", "EXP", uuid, "Train", glucose_type, f"ratio_{ratio:.2f}")
        # 清掉舊檔再複製，避免先前固定 upper 留下的殘檔混進來
        if os.path.isdir(new_train_dir):
            shutil.rmtree(new_train_dir)
        os.makedirs(new_train_dir, exist_ok=True)

        for file in matched_files:
            source_file_path = os.path.join(raw_train_dir, file)
            shutil.copy2(source_file_path, new_train_dir)

        print(f"[ratio={ratio:.2f}] range=[{lower_bound}, {upper_bound}] -> "
              f"{len(matched_files)} files copied to {new_train_dir}")

    return ratio_bounds



