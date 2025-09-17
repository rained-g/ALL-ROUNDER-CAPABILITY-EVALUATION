import pandas as pd
import os

# 设置绝对路径
base_dir = "/mnt/sda/ymji/Report Generation"
input_file = os.path.join(base_dir, "datasets/MIMIC/CSV/openi/Test.csv")
output_dir = os.path.join(base_dir, "evaluation-metric/origin_report/openi")
output_file = os.path.join(output_dir, "Test.csv")

# 创建保存目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 检查输入文件是否存在
if not os.path.exists(input_file):
    raise FileNotFoundError(f"输入文件不存在: {input_file}")

# 加载 CSV 文件
df = pd.read_csv(input_file)

# 将列名 "text" 改为 "Report Impression"
if "text" not in df.columns:
    raise KeyError("输入文件中未找到 'text' 列！")
df = df.rename(columns={"text": "Report Impression"})

# 保存修改后的文件
df.to_csv(output_file, index=False)
print(f"新文件已保存为: {output_file}")



# relative_path = os.path.join(os.path.dirname(__file__), "../datasets/MIMIC/CSV/mimic/Test.csv")
# print("Absolute path to the file:", os.path.abspath(relative_path))