"""
1) 把 Regression_Features_IQR/{uuid}/Remove_Data/Train/{Low,Normal,High}
   搬回同 uuid 的 Train/{level}（已存在則跳過）
2) 若 full/Regression_Features/{uuid} 不存在，把整包 IQR uuid copy 過去
   若已存在則不覆蓋（避免蓋掉昨晚重抽的 full）
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\ivy_j\projects\SVR_外推exp\Model\70_30")
IQR = ROOT / "Regression_Features_IQR"
FULL = ROOT / "full" / "Regression_Features"
LEVELS = ("Low", "Normal", "High")


def count_txt(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.glob("*.txt"))


def re_add_remove(uuid_dir: Path) -> tuple[int, int]:
    removed_root = uuid_dir / "Remove_Data" / "Train"
    added, skipped = 0, 0
    if not removed_root.is_dir():
        return added, skipped

    for level in LEVELS:
        src_dir = removed_root / level
        if not src_dir.is_dir():
            continue
        dst_dir = uuid_dir / "Train" / level
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.txt"):
            dst = dst_dir / src.name
            if dst.exists():
                skipped += 1
                continue
            shutil.move(str(src), str(dst))
            added += 1
    return added, skipped


def main() -> None:
    FULL.mkdir(parents=True, exist_ok=True)
    print(f"{'uuid':6} {'iqr_train':>10} {'remove':>8} {'full':>10} | action")

    for uuid_dir in sorted(p for p in IQR.iterdir() if p.is_dir()):
        uuid = uuid_dir.name
        rem_before = sum(count_txt(uuid_dir / "Remove_Data" / "Train" / lv) for lv in LEVELS)
        train_before = sum(count_txt(uuid_dir / "Train" / lv) for lv in LEVELS)
        full_before = sum(count_txt(FULL / uuid / "Train" / lv) for lv in LEVELS)

        added, skipped = re_add_remove(uuid_dir)
        train_after = sum(count_txt(uuid_dir / "Train" / lv) for lv in LEVELS)
        rem_after = sum(count_txt(uuid_dir / "Remove_Data" / "Train" / lv) for lv in LEVELS)

        dest = FULL / uuid
        if dest.exists():
            action = f"keep existing full (train={full_before}); re_add +{added}/skip {skipped}"
        else:
            # copy Train/Test (+ empty Remove_Data ok); exclude nothing critical
            shutil.copytree(uuid_dir, dest)
            action = f"copied IQR → full; re_add +{added}/skip {skipped}"

        print(
            f"{uuid:6} {train_before:10} {rem_before:8} {full_before:10} | "
            f"iqr_train_now={train_after} rem_now={rem_after} | {action}"
        )


if __name__ == "__main__":
    main()
