import pandas as pd
import os

# 读取总表   dataset_download.py中从yahoo拉取到的数据集
df = pd.read_csv("../nasdaq100/nasdaq100_latest_raw_data.csv", parse_dates=["date"])

output_dir = "./csv_data"
os.makedirs(output_dir, exist_ok=True)

count = 0  # 记录拆分出的股票文件数量

for code, group in df.groupby("code"):
    # 只保留 OHLCV 列，去重排序
    stock_data = group[["date", "open", "high", "low", "close", "volume", "amount"]]
    stock_data = stock_data.drop_duplicates("date").sort_values("date")

    # 将雅虎代码转换为 Qlib 格式（小写）
    if ".SZ" in code:
        qlib_code = "sz" + code.replace(".SZ", "")
    elif ".SS" in code:
        qlib_code = "sh" + code.replace(".SS", "")
    else:
        qlib_code = code  # 美股或其他

    file_name = f"{qlib_code.lower()}.csv"
    file_path = os.path.join(output_dir, file_name)
    
    stock_data.to_csv(file_path, index=False)
    count += 1

print(f"✅ 拆分完成，共生成 {count} 个股票文件，存放在 {output_dir}/")




# 进入qlib目录下   然后用bin生成与qlib兼容的数据集,生成bin文件
# git clone https://github.com/microsoft/qlib.git
# cd qlib
# python3.10 scripts/dump_bin.py dump_all --data_path ../dataset/csv_data_500(上面生成的csv数据的目录) --qlib_dir ./qlib_data --include_fields open,high,low,close,volume,amount