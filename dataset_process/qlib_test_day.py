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
# 这是从2010年到2020年qlib官方数据下载到本地后，经过alpha158因子计算得到的数据(也可以加载选择日期的新增的，注意本代码加了筛选条件)
# qlib   a股数据
# 如果要做全量历史数据，需要下载其他的近期的成分股数据，可在链接
# https://github.com/SunsetWolf/qlib_dataset/releases/   qlib数据但目前仅到24年
# https://github.com/chenditc/investment_data/releases  下载的bin形式qlib数据(持续更新)，可直接生成因子和NTF/TNF数据
# 中获取

# ==============================================================================
#                          唯一需要修改的配置
# ==============================================================================
CONFIG = {
    "qlib_data_dir": os.path.join(DATASET_ROOT, "cn_data"),  # 本地Qlib日频数据目录
    "index_name": "csi500",        # 指数：csi300 / sz50 / csi500 /  nasdaq100    sp500
    "start_date": "2010-01-01",
    "end_date": "2020-01-01",
    "factor_set": "alpha158",      # alpha158 / alpha360
    "save_dir": PROCESSED_DATASETS_DIR
}

# ==============================================================================
#                          纯Qlib官方标准流程（无任何错误）
# ==============================================================================
def main():
    # 1. 初始化Qlib
    import qlib
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158, Alpha360
    
    print(f"✅ 初始化Qlib，数据目录: {CONFIG['qlib_data_dir']}")
    qlib.init(provider_uri=CONFIG["qlib_data_dir"], freq="day")
    
    # 2. 获取Qlib内置的官方指数成分股（历史并集，无生存者偏差）
    print(f"\n✅ 获取{CONFIG['index_name']}历史成分股")
    instrument_config = D.instruments(market=CONFIG["index_name"])
    instruments = D.list_instruments(
        instruments=instrument_config,
        start_time=CONFIG["start_date"],
        end_time=CONFIG["end_date"],
        as_list=True
    )
    # 转换为小写，匹配本地sh600000/sz000001格式
    instruments = [code.lower() for code in instruments]
    print(f"✅ 共获取到 {len(instruments)} 只历史成分股（无生存者偏差）")
    print(f"   前5只: {instruments[:5]}")
    
    # 3. 生成官方Alpha158因子（严格158个因子+1个标签=159列）
    print(f"\n✅ 生成Qlib官方{CONFIG['factor_set']}因子")
    if CONFIG["factor_set"] == "alpha158":
        handler = Alpha158(
            instruments=instruments,
            start_time=CONFIG["start_date"],
            end_time=CONFIG["end_date"],
            freq="day"
        )
    else:
        handler = Alpha360(
            instruments=instruments,
            start_time=CONFIG["start_date"],
            end_time=CONFIG["end_date"],
            freq="day"
        )
    
    factors = handler.fetch().astype(np.float32)
    print(f"✅ 因子生成完成，共 {len(factors.columns)} 列")
    print(f"   其中：{len(factors.columns)-1} 个因子 + 1个标签(LABEL0)")

    # ✅ 修正：先获取收盘价并交换索引，再进行筛选
    print("\n✅ 获取收盘价数据")
    close = D.features(
        instruments=instruments,
        fields=["$close"],
        start_time=CONFIG["start_date"],
        end_time=CONFIG["end_date"],
        freq="day"
    )["$close"].astype(np.float32)

    # 交换close的索引顺序，变成和factors一样的 (datetime, instrument)
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

    # 5. 转换为标准TNF格式
    print("\n✅ 转换为TNF标准格式")
    times = factors.index.get_level_values(0).unique().sort_values()
    stocks = factors.index.get_level_values(1).unique().sort_values()
    
    T, N, F = len(times), len(stocks), len(factors.columns)
    print("TNF是:")
    print(T)
    print(N)
    print(F)
    tnf_data = np.zeros((T, N, F), dtype=np.float32)
    tnf_close = np.zeros((T, N), dtype=np.float32)
    
    for i, t in enumerate(tqdm(times, desc="转换格式")):
        t_factors = factors.loc[t]
        t_close = close.loc[t]
        for j, stock in enumerate(stocks):
            if stock in t_factors.index:
                tnf_data[i, j] = t_factors.loc[stock].values
                tnf_close[i, j] = t_close.loc[stock]
    
    tnf_data = np.nan_to_num(tnf_data, nan=0.0)
    tnf_close = np.nan_to_num(tnf_close, nan=0.0)
    
    # 6. 保存所有文件
    save_path = f"{CONFIG['save_dir']}/{CONFIG['index_name']}_qlib_official"
    os.makedirs(save_path, exist_ok=True)
    
    np.save(f"{save_path}/tnf_data.npy", tnf_data)
    np.save(f"{save_path}/tnf_close.npy", tnf_close)
    np.save(f"{save_path}/times.npy", times.astype(str))
    np.save(f"{save_path}/stocks.npy", stocks)
    
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

if __name__ == "__main__":
    main()


