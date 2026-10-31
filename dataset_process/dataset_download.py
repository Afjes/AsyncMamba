import os
import time
import random
import akshare as ak
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 数据集根目录统一从 dataset_paths.py 读取（默认 <repo_root>/dataset）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_paths import PROCESSED_DATASETS_DIR

# 但是会拉取最新的成分股数据，如果要做全量历史数据，需要下载其他的近期的成分股数据，可在链接
# https://github.com/SunsetWolf/qlib_dataset/releases/   qlib数据但目前仅到24年
# https://github.com/chenditc/investment_data/releases  下载的bin形式qlib数据(持续更新)，可直接生成因子和NTF/TNF数据
# 中获取

# ==================== 配置 ====================
CONFIG = {
    "market": "us",               # cn / us
    "index_code": "^NDX",       # 指数代码
    "index_name": "nas100",       # 简称
    "freq": "day",
    "start_date": "2020-01-01",
    "end_date": "2026-01-01",
    "save_dir": PROCESSED_DATASETS_DIR,
    "batch_fetch_size": 1,
    "retry_count": 3,
}

INDEX_CONFIG = {
    "cn": {
        "000300": {"name": "沪深300"}, "000905": {"name": "中证500"}, "000906": {"name": "中证800"},
        "000016": {"name": "上证50"}, "399006": {"name": "创业板指"}, "000688": {"name": "科创50"},
        "399001": {"name": "深证成指"}, "000852": {"name": "中证1000"}
    },
    "us": {
        "^GSPC": {"name": "标普500"}, "^NDX": {"name": "纳斯达克100"}, "^DJI": {"name": "道琼斯工业平均"},
        "^RUT": {"name": "罗素2000"}, "^IXIC": {"name": "纳斯达克综合"}
    }
}

# ==================== 工具函数 ====================
def create_dirs():
    dataset_dir = f"{CONFIG['save_dir']}/{CONFIG['index_name']}_{CONFIG['freq']}"
    os.makedirs(f"{dataset_dir}/raw", exist_ok=True)
    os.makedirs(f"{dataset_dir}/code_map", exist_ok=True)
    return dataset_dir

# ==================== 成分股获取 ====================
def fetch_index_stocks(dataset_dir):
    print(f"\n步骤1：拉取{INDEX_CONFIG[CONFIG['market']][CONFIG['index_code']]['name']}最新成分股")
    stocks_df = None

    if CONFIG["market"] == "cn":
        try:
            print("尝试接口1：ak.index_stock_cons_csindex")
            stocks_df = ak.index_stock_cons_csindex(symbol=CONFIG['index_code'])
            stocks_df = stocks_df.rename(columns={
                "成分券代码": "code",
                "成分券名称": "name",
                "纳入日期": "list_date",
                "行业": "industry",
                "权重": "weight"
            })
            stocks_df = stocks_df[~stocks_df["name"].str.contains(r"ST|\*ST", case=False, na=False)]
            print(f"✅ 接口1成功，获取到{len(stocks_df)}只成分股")
        except Exception as e:
            print(f"接口1失败: {e}")
            try:
                print("尝试接口2：ak.index_stock_cons")
                stocks_df = ak.index_stock_cons(symbol=CONFIG['index_code'])
                stocks_df = stocks_df.rename(columns={
                    "品种代码": "code",
                    "品种名称": "name",
                    "纳入日期": "list_date"
                })
                stocks_df["industry"] = "未知"
                stocks_df["weight"] = 0.0
                stocks_df = stocks_df[~stocks_df["name"].str.contains(r"ST|\*ST", case=False, na=False)]
                print(f"✅ 接口2成功，获取到{len(stocks_df)}只成分股")
            except Exception as e2:
                print(f"接口2失败: {e2}")
                raise Exception("所有成分股接口都失败")

        # 代码标准化
        stocks_df["code"] = stocks_df["code"].astype(str).str.zfill(6)
        stocks_df["qlib_code"] = stocks_df["code"].apply(
            lambda x: f"sh{x}" if x.startswith("6") else f"sz{x}"
        )
        stocks_df["yahoo_code"] = stocks_df["code"].apply(
            lambda x: f"{x}.SS" if x.startswith("6") else f"{x}.SZ"
        )


    elif CONFIG["market"] == "us":
        # 手动解析维基百科，兼容所有 pandas/beautifulsoup 版本
        try:
            if CONFIG["index_code"] == "^GSPC":
                url = "https://zh.wikipedia.org/wiki/标普500指数成分股列表"
                ticker_col = 0   # 第一列
                name_col = 1     # 第二列
            elif CONFIG["index_code"] == "^NDX":
                url = "https://zh.wikipedia.org/wiki/纳斯达克100指数"
                ticker_col = 1   # 第二列（第一列是序号）
                name_col = 2
            else:
                raise Exception("暂不支持该指数，请使用 ^GSPC 或 ^NDX")

            import requests
            from bs4 import BeautifulSoup

            print(f"📊 从维基百科获取 {CONFIG['index_code']} 成分股...")
            resp = requests.get(url, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            # 找到页面中第一个 wikitable sortable 表格
            table = soup.find("table", {"class": "wikitable sortable"})
            if not table:
                raise ValueError("未找到成分股表格")

            rows = table.find_all("tr")[1:]  # 跳过表头
            tickers = []
            names = []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= max(ticker_col, name_col) + 1:
                    ticker = cols[ticker_col].text.strip()
                    name = cols[name_col].text.strip()
                    tickers.append(ticker)
                    names.append(name)

            stocks_df = pd.DataFrame({"code": tickers, "name": names})
            stocks_df["industry"] = "未知"
            stocks_df["weight"] = 0.0
            print(f"✅ 获取成功，共 {len(stocks_df)} 只成分股")

        except Exception as e:
            print(f"成分股获取失败: {e}")
            raise Exception("无法获取美股成分股列表，请检查网络或手动提供 CSV")

        stocks_df["qlib_code"] = stocks_df["code"]
        stocks_df["yahoo_code"] = stocks_df["code"]

    index_self_codes = ["000300", "000905", "000016", "000688", "399006", "000852"]
    stocks_df = stocks_df[stocks_df["code"] != stocks_df["name"]]
    stocks_df = stocks_df[~stocks_df["code"].isin(index_self_codes)]

    code_map_path = f"{dataset_dir}/code_map/code_map.csv"
    stocks_df.to_csv(code_map_path, index=False, encoding="utf-8-sig")
    print(f"✅ 代码映射表已保存到 {code_map_path}")
    return stocks_df


# ==================== 行情拉取（yfinance 优先） ====================
def fetch_market_data(stocks_df, dataset_dir):
    print(f"\n步骤2：拉取{CONFIG['freq']}行情数据")
    raw_data_path = f"{dataset_dir}/raw/raw_data.csv"

    # 断点续存
    all_data = []
    if os.path.exists(raw_data_path):
        try:
            existing_df = pd.read_csv(raw_data_path)
            existing_codes = set(existing_df["code"].unique())
            stocks_df = stocks_df[~stocks_df["qlib_code"].isin(existing_codes)]
            all_data.append(existing_df)
            print(f"发现已有数据，跳过已拉取的{len(existing_codes)}只股票")
        except Exception:
            os.remove(raw_data_path)

    if len(stocks_df) == 0:
        return pd.concat(all_data, ignore_index=True) if all_data else None

    # 尝试 yfinance 批量下载
    tickers_all = stocks_df["yahoo_code"].tolist()
    yf_success = False
    if tickers_all:
        for attempt in range(2):
            try:
                data = yf.download(tickers=tickers_all,
                                   start=CONFIG['start_date'],
                                   end=CONFIG['end_date'],
                                   auto_adjust=False,
                                   progress=True,
                                   group_by='ticker')
                if not data.empty:
                    # 拆分
                    for _, row in tqdm(stocks_df.iterrows(), total=len(stocks_df), desc="拆分数据"):
                        yahoo_code = row["yahoo_code"]
                        qlib_code = row["qlib_code"]
                        if yahoo_code not in data.columns.levels[0]:
                            continue
                        stock_data = data[yahoo_code].copy()
                        stock_data = stock_data.dropna(how='all')
                        if stock_data.empty:
                            continue
                        stock_data = stock_data.reset_index()
                        stock_data.columns = [col.lower() for col in stock_data.columns]
                        stock_data = stock_data.rename(columns={
                            "date": "date",
                            "open": "open", "high": "high", "low": "low",
                            "close": "close", "volume": "volume"
                        })
                        stock_data["amount"] = stock_data["close"] * stock_data["volume"]
                        stock_data = stock_data[["date", "open", "high", "low", "close", "volume", "amount"]]
                        stock_data["date"] = pd.to_datetime(stock_data["date"])
                        stock_data["code"] = qlib_code
                        all_data.append(stock_data)
                    yf_success = True
                    break
            except Exception as e:
                if "rate" in str(e).lower():
                    print(f"限流，等待60秒后重试...")
                    time.sleep(60)
                else:
                    print(f"yfinance 失败: {e}")
                    break

    if yf_success:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df.sort_values(["code", "date"]).reset_index(drop=True)
        full_df.to_csv(raw_data_path, index=False)
        print(f"✅ 拉取完成 (yfinance): {full_df['code'].nunique()} 只股票, {len(full_df)} 条记录")
        return full_df
    else:
        # 回退到 akshare 逐只拉取（此处省略，可保留原有多重接口）
        print("yfinance 未成功，请检查网络或手动运行 akshare 分支")
        return None

# ==================== 主函数 ====================
def main():
    dataset_dir = create_dirs()
    stocks_df = fetch_index_stocks(dataset_dir)
    _ = fetch_market_data(stocks_df, dataset_dir)

if __name__ == "__main__":
    main()