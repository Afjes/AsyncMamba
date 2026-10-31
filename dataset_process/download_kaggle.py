import kagglehub
import pandas as pd
# Download latest version
# path = kagglehub.dataset_download("kpatel528/nasdaq-100-daily-stock-data-19852025")
# print("Path to dataset files:", path)

import pandas as pd

# 请将文件路径替换为你本地的实际路径
raw_data_path = "../nasdaq100/nasdaq100_latest_raw_data.csv"          # 历史行情
current_symbols_path = "../nasdaq100/nasdaq100_latest_symbols.csv"    # 当前成分股

# ==============================================================================
# 1. 读取历史行情数据（正确列名：Company）
# ==============================================================================
print("="*60)
print("📊 读取历史行情数据...")
print("="*60)

# 只提取股票代码列，加快读取速度
raw_df = pd.read_csv(raw_data_path, usecols=['Company'])
all_hist_tickers = raw_df['Company'].unique()
num_hist = len(all_hist_tickers)

print(f"✅ 历史行情文件读取成功")
print(f"   总记录数: {len(raw_df):,} 条")
print(f"   历史出现过的股票总数: {num_hist} 只")
print(f"   前10只股票代码: {list(all_hist_tickers[:10])}")
print()

# ==============================================================================
# 2. 读取当前成分股列表（正确列名：Ticker）
# ==============================================================================
print("="*60)
print("📋 读取当前纳斯达克100成分股...")
print("="*60)

try:
    current_df = pd.read_csv(current_symbols_path)
    
    # 明确指定列名，不再自动识别（避免歧义）
    current_tickers = current_df['Ticker'].tolist()
    current_names = current_df['Company_Name'].tolist()
    num_current = len(current_tickers)
    
    print(f"✅ 当前成分股文件读取成功")
    print(f"   当前纳斯达克100成分股数量: {num_current} 只")
    print(f"   前10只成分股: {list(current_tickers[:10])}")
    print()

    # ==============================================================================
    # 3. 生存者偏差验证（核心）
    # ==============================================================================
    print("="*60)
    print("🔍 生存者偏差验证结果")
    print("="*60)

    hist_set = set(all_hist_tickers)
    current_set = set(current_tickers)

    # 历史出现过但已被剔除的股票
    removed_tickers = hist_set - current_set
    num_removed = len(removed_tickers)

    # 当前成分股但历史数据中没有的股票（新上市/新纳入）
    new_tickers = current_set - hist_set
    num_new = len(new_tickers)

    print(f"📌 核心结论：你的数据存在明显的生存者偏差！")
    print(f"   历史上共有 {num_hist} 只股票曾经进入过纳斯达克100指数")
    print(f"   但当前只剩下 {num_current} 只成分股")
    print(f"   有 {num_removed} 只股票已经被剔除，你的历史数据中包含了它们！")
    print()

    if num_removed > 0:
        print(f"📋 已被剔除的股票列表（前20只）：")
        for ticker in list(removed_tickers)[:20]:
            print(f"   - {ticker}")
        print()

    if num_new > 0:
        print(f"📋 新纳入但历史数据缺失的股票（前10只）：")
        for ticker in list(new_tickers)[:10]:
            print(f"   - {ticker}")
        print()

    # 计算生存者偏差率
    survival_bias_rate = (num_hist - num_current) / num_hist * 100
    print(f"📈 生存者偏差率: {survival_bias_rate:.2f}%")
    print(f"   这意味着如果你只用当前的100只股票做回测，你的收益率会被高估约 {survival_bias_rate:.2f}%")

except FileNotFoundError:
    print("❌ 错误：未找到当前成分股文件，请确认文件名和路径是否正确")
except KeyError as e:
    print(f"❌ 错误：找不到列名 {e}")
    print("   请检查你的CSV文件列名是否为：Ticker, Company_Name")