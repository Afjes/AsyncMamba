import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 数据集根目录统一从 dataset_paths.py 读取（默认 <repo_root>/dataset）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_paths import DATASET_ROOT, PROCESSED_DATASETS_DIR

# 处理bin
# python scripts/dump_bin.py dump_all --data_path ../dataset/csv_data_500 --qlib_dir ./qlib_data --include_fields open,high,low,close,volume,amount
# 可以加载qlib的bin数据，并生成因子和tnf（这里加载的是所有股票，没有按照时间筛选，适合全部的时间范围）
# ==================== 配置 ====================
CONFIG = {
    "market": "cn",
    "index_name": "CSI300",
    "freq": "day",
    "start_date": "2010-01-01",
    "end_date": "2026-01-01",               # 根据实际数据覆盖范围调整
    "qlib_data_dir": os.path.join(DATASET_ROOT, "cn_data_latest"),   # Qlib 二进制数据目录
    "factor_set": "alpha158",
    "save_dir": PROCESSED_DATASETS_DIR,
}

# ==================== 生成因子 + TNF 输出 ====================
def generate_factors_and_tnf():
    import qlib
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158, Alpha360

    # 初始化
    qlib.init(provider_uri=CONFIG["qlib_data_dir"], freq="day")
    instruments = D.instruments(market="all")
    print(f"✅ 使用本地 Qlib 数据，成分股数量: {len(instruments)}")

    # 生成因子
    handler = Alpha158(instruments=instruments,
                       start_time=CONFIG["start_date"],
                       end_time=CONFIG["end_date"],
                       freq="day") if CONFIG["factor_set"] == "alpha158" else \
              Alpha360(instruments=instruments,
                       start_time=CONFIG["start_date"],
                       end_time=CONFIG["end_date"],
                       freq="day")
    factors = handler.fetch().astype(np.float32)

    print(f"✅ 因子生成完成，共 {len(factors.columns)} 列")
    print(f"   其中：{len(factors.columns)-1} 个因子 + 1个标签(LABEL0)")


    # 获取收盘价
    close = D.features(instruments=instruments,
                       fields=["$close"],
                       start_time=CONFIG["start_date"],
                       end_time=CONFIG["end_date"],
                       freq="day")["$close"].astype(np.float32)
    close = close.swaplevel(0, 1).sort_index()

    # ✅ 现在再执行学术标准股票筛选（同时筛选factors和close）
    print("\n✅ 执行学术标准股票筛选")
    initial_count = factors.index.get_level_values(1).nunique()

    # 1. 计算每只股票的有效数据比例
    stock_valid_ratio = factors.groupby(level=1).apply(lambda x: 1 - x.isna().sum().sum() / (len(x) * len(x.columns)))
    valid_stocks = stock_valid_ratio[stock_valid_ratio >= 0.9].index  # 有效数据≥90%

    # 2. 同时筛选factors和close
    factors = factors.loc[factors.index.get_level_values(1).isin(valid_stocks)]
    close = close.loc[close.index.get_level_values(1).isin(valid_stocks)]

    final_count = factors.index.get_level_values(1).nunique()
    print(f"✅ 筛选完成：{initial_count} → {final_count} 只股票")


    # 转换为 TNF
    times = factors.index.get_level_values(0).unique().sort_values()
    stocks = factors.index.get_level_values(1).unique().sort_values()
    T, N, F = len(times), len(stocks), len(factors.columns)
    tnf_data = np.zeros((T, N, F), dtype=np.float32)
    tnf_close = np.zeros((T, N), dtype=np.float32)

    for i, t in enumerate(tqdm(times, desc="转换格式")):
        t_factors = factors.loc[t]
        t_close = close.loc[t]
        for j, s in enumerate(stocks):
            if s in t_factors.index:
                tnf_data[i, j] = t_factors.loc[s].values
                tnf_close[i, j] = t_close.loc[s]

    tnf_data = np.nan_to_num(tnf_data, nan=0.0)
    tnf_close = np.nan_to_num(tnf_close, nan=0.0)

    # 保存
    save_path = f"{CONFIG['save_dir']}/{CONFIG['index_name']}_{CONFIG['factor_set']}"
    os.makedirs(save_path, exist_ok=True)
    np.save(f"{save_path}/tnf_data.npy", tnf_data)
    np.save(f"{save_path}/tnf_close.npy", tnf_close)
    np.save(f"{save_path}/times.npy", times.astype(str))
    np.save(f"{save_path}/stocks.npy", stocks)
    with open(f"{save_path}/dataset_config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)


    # 保存配置
    config = CONFIG.copy()
    config["n_stocks"] = N
    config["n_features"] = F
    config["n_time_steps"] = T
    config["has_label"] = True  # 标记包含LABEL0标签
    with open(f"{save_path}/dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 7. 最终验证
    print(f"\n{'='*60}")
    print(f"🎉 纯Qlib官方数据处理完成！")
    print(f"{'='*60}")
    print(f"数据集目录: {save_path}")
    print(f"股票数量: {N}")
    print(f"特征数量: {F} (含LABEL0标签)")
    print(f"时间步: {T}")
    print(f"\n训练时直接使用: prepare_data(data_path='{save_path}')")

# ==================== 主函数 ====================
def main():
    generate_factors_and_tnf()

if __name__ == "__main__":
    main()