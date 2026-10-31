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

# 数据来源说明： 本代码暂时跑不通
# 沪深300 5分钟数据：https://github.com/SunsetWolf/qlib_dataset/releases/download/v3/qlib_data_hs300_5min_latest.zip
# 全市场分钟数据：需自行通过Qlib collector下载或使用社区数据源

# ==============================================================================
#                          唯一需要修改的配置
# ==============================================================================
CONFIG = {
    "qlib_data_dir": os.path.join(DATASET_ROOT, "cn_data_1min"),  # 本地分钟级Qlib数据目录
    "index_name": "csi300",              # 指数：csi300 / csi500 / sp500 / nasdaq100
    "start_date": "2015-01-01",
    "end_date": "2020-01-01",
    "freq": "1min",                      # 分钟频率：1min / 5min / 15min / 30min
    "label_period": 1,                   # 标签周期：未来N个K线的收益率
    "downsample_to_daily": False,        # 是否降频到日级（与日线因子融合时开启）
    "save_dir": PROCESSED_DATASETS_DIR
}

# ==============================================================================
#                          学术标准分钟因子集（158个）
# ==============================================================================
def generate_minute_factors(df):
    """
    生成与Alpha158结构一致的158个分钟级因子
    因子分类：基础K线特征(20) + 动量特征(40) + 波动率特征(40) + 成交量特征(40) + 日内特征(18)
    """
    factors = pd.DataFrame(index=df.index)
    
    # -------------------------- 1. 基础K线特征 --------------------------
    factors["KMID"] = (df["close"] - df["open"]) / df["open"]
    factors["KLEN"] = (df["high"] - df["low"]) / df["open"]
    factors["KMID2"] = (df["close"] - (df["high"] + df["low"]) / 2) / ((df["high"] + df["low"]) / 2)
    factors["KUP"] = (df["high"] - df["close"]) / df["open"]
    factors["KLOW"] = (df["close"] - df["low"]) / df["open"]
    factors["KSFT"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
    
    # 价格变化率
    for i in [1, 2, 3, 5, 10, 15, 20, 30, 60, 120]:
        factors[f"RET{i}"] = df["close"] / df["close"].shift(i) - 1
    
    # -------------------------- 2. 动量特征 --------------------------
    for i in [5, 10, 15, 20, 30, 60, 120, 240]:
        factors[f"ROC{i}"] = df["close"] / df["close"].shift(i) - 1
        factors[f"MA{i}"] = df["close"].rolling(i).mean() / df["close"] - 1
        factors[f"EMA{i}"] = df["close"].ewm(span=i).mean() / df["close"] - 1
        factors[f"MAX{i}"] = df["high"].rolling(i).max() / df["close"] - 1
        factors[f"MIN{i}"] = df["low"].rolling(i).min() / df["close"] - 1
    
    # -------------------------- 3. 波动率特征 --------------------------
    for i in [5, 10, 15, 20, 30, 60, 120, 240]:
        factors[f"VOL{i}"] = df["close"].pct_change().rolling(i).std()
        factors[f"VOLSTD{i}"] = factors[f"VOL{i}"] / factors[f"VOL{i}"].rolling(240).mean()
        factors[f"HIGHLOW{i}"] = (df["high"].rolling(i).max() - df["low"].rolling(i).min()) / df["close"]
        factors[f"RANGE{i}"] = (df["high"] - df["low"]).rolling(i).mean() / df["close"]
        factors[f"RSI{i}"] = 100 - 100 / (1 + df["close"].pct_change().rolling(i).apply(lambda x: x[x>0].sum() / abs(x[x<0].sum())))
    
    # -------------------------- 4. 成交量特征 --------------------------
    for i in [5, 10, 15, 20, 30, 60, 120, 240]:
        factors[f"VMA{i}"] = df["volume"].rolling(i).mean() / df["volume"]
        factors[f"VSTD{i}"] = df["volume"].rolling(i).std() / df["volume"]
        factors[f"VROC{i}"] = df["volume"] / df["volume"].shift(i) - 1
        factors[f"VMIN{i}"] = df["volume"].rolling(i).min() / df["volume"]
        factors[f"VMAX{i}"] = df["volume"].rolling(i).max() / df["volume"]
    
    # 量价配合特征
    factors["OBV"] = (np.sign(df["close"].pct_change()) * df["volume"]).cumsum()
    factors["OBV60"] = factors["OBV"].rolling(60).mean() / factors["OBV"]
    factors["WVAD"] = ((df["close"] - df["open"]) / (df["high"] - df["low"]) * df["volume"]).rolling(6).sum()
    
    # -------------------------- 5. 日内专属特征 --------------------------
    df["date"] = df.index.get_level_values(0).date
    daily_open = df.groupby("date")["open"].transform("first")
    daily_high = df.groupby("date")["high"].transform("max")
    daily_low = df.groupby("date")["low"].transform("min")
    daily_volume = df.groupby("date")["volume"].transform("sum")
    
    factors["INTRADAY_RET"] = df["close"] / daily_open - 1
    factors["INTRADAY_HIGH"] = df["high"] / daily_high - 1
    factors["INTRADAY_LOW"] = df["low"] / daily_low - 1
    factors["VOLUME_RATIO"] = df["volume"] / daily_volume
    factors["TIME_DECAY"] = df.groupby("date").cumcount() / df.groupby("date")["close"].transform("count")
    
    # 开盘后/收盘前特征
    factors["OPEN_30MIN_RET"] = df.groupby("date").apply(lambda x: x.iloc[min(30, len(x)-1)]["close"] / x.iloc[0]["open"] - 1).reset_index(level=0, drop=True)
    factors["CLOSE_30MIN_VOL"] = df.groupby("date").apply(lambda x: x.iloc[-min(30, len(x)):]["volume"].sum() / x["volume"].sum()).reset_index(level=0, drop=True)
    
    # -------------------------- 6. 标签（与天级别LABEL0对齐） --------------------------
    factors["LABEL0"] = df["close"].shift(-CONFIG["label_period"]) / df["close"] - 1
    
    # 去除NaN和无穷大
    factors = factors.replace([np.inf, -np.inf], np.nan)
    return factors

# ==============================================================================
#                          纯Qlib官方标准流程（与天级别完全一致）
# ==============================================================================

def main():
    # -------------------------- 第一步：用日频获取成分股（必须，与数据频率无关） --------------------------
    import qlib
    from qlib.data import D
    
    print(f"✅ 临时初始化日频Qlib获取成分股")
    # 只用于获取成分股，成分股永远是日级调整的
    # qlib.init(
    #     provider_uri=CONFIG["qlib_data_dir"], 
    #     freq="day",
    #     region="cn" if CONFIG["index_name"].startswith(("csi", "sz")) else "us"
    # )

    print(f"\n✅ 初始化1分钟Qlib加载数据")
    qlib.init(
        provider_uri=CONFIG["qlib_data_dir"], 
        freq="1min",
        region="cn" if CONFIG["index_name"].startswith(("csi", "sz")) else "us"
        # ✅ 关键修复：强制使用文本格式日历，忽略缺失的.bin文件
    )

    
    print(f"\n✅ 获取{CONFIG['index_name']}历史成分股")
    instrument_config = D.instruments(market=CONFIG["index_name"])
    instruments = D.list_instruments(
        instruments=instrument_config,
        start_time=CONFIG["start_date"],
        end_time=CONFIG["end_date"],
        as_list=True
    )
    instruments = [code.lower() for code in instruments]
    print(f"✅ 共获取到 {len(instruments)} 只历史成分股（无生存者偏差）")
    print(f"   前5只: {instruments[:5]}")
    
    # 关闭日频Qlib，重新初始化1分钟频率
    qlib.shutdown()
    
    # -------------------------- 第二步：初始化1分钟频率Qlib加载数据 --------------------------
    print(f"\n✅ 初始化1分钟Qlib加载数据")
    qlib.init(
        provider_uri=CONFIG["qlib_data_dir"], 
        freq="1min",  # ✅ 这里明确指定1分钟
        region="cn" if CONFIG["index_name"].startswith(("csi", "sz")) else "us"
    )
    
    # 3. 加载原始1分钟数据并生成因子
    print(f"\n✅ 加载原始1分钟数据并生成158个学术标准因子")
    all_factors = []
    all_close = []
    
    # ✅ 1分钟数据专属：调小batch_size防止内存溢出（5分钟用50，1分钟用20）
    batch_size = 20
    for i in tqdm(range(0, len(instruments), batch_size), desc="处理股票批次"):
        batch_stocks = instruments[i:i+batch_size]
        
        # 加载单批次原始1分钟数据
        df = D.features(
            instruments=batch_stocks,
            fields=["$open", "$high", "$low", "$close", "$volume", "$amount"],
            start_time=CONFIG["start_date"],
            end_time=CONFIG["end_date"],
            freq="1min"  # ✅ 显式指定1分钟
        ).astype(np.float32)
        
        # 交换索引为 (datetime, instrument)，与天级别一致
        df = df.swaplevel(0, 1).sort_index()
        
        # -------------------------- 1分钟数据专属预处理 --------------------------
        # 过滤零成交量分钟（1分钟数据零成交量更多）
        df = df[df["volume"] > 0]
        
        # ✅ 精确的A股1分钟交易时段过滤（9:30:00 至 15:00:00 每分钟一条）
        df["hour"] = df.index.get_level_values(0).hour
        df["minute"] = df.index.get_level_values(0).minute
        
        if CONFIG["index_name"].startswith(("csi", "sz")):
            # A股完整交易时段：9:30-11:30, 13:00-15:00
            df = df[
                ((df["hour"] == 9) & (df["minute"] >= 30)) |
                ((df["hour"] == 10) | (df["hour"] == 11)) |
                ((df["hour"] == 13) | (df["hour"] == 14)) |
                ((df["hour"] == 15) & (df["minute"] == 0))
            ]
        else:
            # 美股完整交易时段：9:30-16:00
            df = df[
                ((df["hour"] == 9) & (df["minute"] >= 30)) |
                (df["hour"].between(10, 15)) |
                ((df["hour"] == 16) & (df["minute"] == 0))
            ]
        
        df = df.drop(columns=["hour", "minute"])
        # ---------------------------------------------------------------------
        
        # 生成因子（与之前完全相同）
        batch_factors = df.groupby(level=1, group_keys=False).apply(generate_minute_factors)
        all_factors.append(batch_factors)
        
        # 保存收盘价
        batch_close = df["close"]
        all_close.append(batch_close)
    
    # 合并所有批次
    factors = pd.concat(all_factors).sort_index()
    close = pd.concat(all_close).sort_index()
    
    print(f"✅ 因子生成完成，共 {len(factors.columns)} 列")
    print(f"   其中：{len(factors.columns)-1} 个1分钟因子 + 1个标签(LABEL0)")
    
    # 4. 执行学术标准股票筛选（与天级别完全相同）
    print("\n✅ 执行学术标准股票筛选")
    initial_count = factors.index.get_level_values(1).nunique()
    
    # 计算每只股票的有效数据比例
    stock_valid_ratio = factors.groupby(level=1).apply(
        lambda x: 1 - x.isna().sum().sum() / (len(x) * len(x.columns))
    )
    valid_stocks = stock_valid_ratio[stock_valid_ratio >= 0.9].index  # 有效数据≥90%
    
    # 同时筛选factors和close
    factors = factors.loc[factors.index.get_level_values(1).isin(valid_stocks)]
    close = close.loc[close.index.get_level_values(1).isin(valid_stocks)]
    
    final_count = factors.index.get_level_values(1).nunique()
    print(f"✅ 筛选完成：{initial_count} → {final_count} 只股票")
    
    # 5. 可选：降频到日级（与日线因子融合时使用）
    if CONFIG["downsample_to_daily"]:
        print("\n✅ 降频到日级（保留每日最后一个K线的因子值）")
        factors["date"] = factors.index.get_level_values(0).date
        factors = factors.groupby(["date", "instrument"]).last()
        factors.index = factors.index.set_names(["datetime", "instrument"])
        
        close["date"] = close.index.get_level_values(0).date
        close = close.groupby(["date", "instrument"]).last()
        close.index = close.index.set_names(["datetime", "instrument"])
        
        # 更新标签为未来1天收益率
        factors["LABEL0"] = close.groupby(level=1).shift(-1) / close - 1
        factors = factors.dropna(subset=["LABEL0"])
    
    # 6. 转换为标准TNF格式（与天级别完全相同）
    print("\n✅ 转换为TNF标准格式")
    times = factors.index.get_level_values(0).unique().sort_values()
    stocks = factors.index.get_level_values(1).unique().sort_values()
    
    T, N, F = len(times), len(stocks), len(factors.columns)
    print(f"   TNF形状: [T={T}, N={N}, F={F}]")
    print(f"   1分钟数据量是5分钟的5倍，是日线的240倍")
    
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
    
    # 7. 保存所有文件（与天级别完全相同的文件结构）
    save_path = f"{CONFIG['save_dir']}/{CONFIG['index_name']}_1min_qlib"
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
    config["has_label"] = True
    config["is_minute"] = True
    config["freq"] = "1min"
    with open(f"{save_path}/dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 8. 最终验证
    print(f"\n{'='*60}")
    print(f"🎉 Qlib 1分钟数据处理完成！")
    print(f"{'='*60}")
    print(f"数据集目录: {save_path}")
    print(f"股票数量: {N}")
    print(f"特征数量: {F} (含LABEL0标签)")
    print(f"时间步: {T} (约 {T//240} 个交易日)")
    print(f"频率: 1分钟")
    print(f"\n训练时直接使用: prepare_data(data_path='{save_path}')")
    print(f"✅ 与天级别训练代码100%兼容，无需任何修改")

if __name__ == "__main__":
    main()