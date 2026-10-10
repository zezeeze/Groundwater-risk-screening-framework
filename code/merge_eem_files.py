from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import re
import pandas as pd

def clean_sheet_name(filename):
    """清洗工作表名称，适配Excel规则"""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[\\/*?:\[\]]', '_', name)
    name = name.replace("'", "")
    return name[:31] or "Sheet"


def merge_eem_to_excel(input_folder, output_path):
    """
    批量EEM‑csv合并成一个xlsx，每个csv一个工作表，**无索引、无表头、纯数据矩阵**
    """
    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"⚠️ 文件夹中没有 CSV 文件: {input_folder}")
        return

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    used_names = set()

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for csv_file in csv_files:
            sheet_name = clean_sheet_name(csv_file)
            # 处理重名
            base, n = sheet_name, 1
            while sheet_name in used_names:
                suffix = f"_{n}"
                sheet_name = base[:31 - len(suffix)] + suffix
                n += 1
            used_names.add(sheet_name)

            # 读取：无表头，第一列也不作为索引（全部当成光谱数值）
            df = pd.read_csv(os.path.join(input_folder, csv_file), header=None)

            # to_excel关键参数：index=False → 去掉行号；header=False → 去掉列标题
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            print(f"✅ 已写入工作表: {sheet_name}  ({df.shape[0]}行 × {df.shape[1]}列)")

    print(f"✨ 完成! 共合并 {len(csv_files)} 个文件，保存到:\n{output_path}")


if __name__ == "__main__":
    # ============在这里改成你自己的路径============
    INPUT_FOLDER = PROJECT_ROOT / "data/eem/blank_corrected"
    OUTPUT_PATH = PROJECT_ROOT / "data/eem/eem_blank_corrected_merged.xlsx"

    merge_eem_to_excel(INPUT_FOLDER, OUTPUT_PATH)
