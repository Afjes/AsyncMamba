"""数据集路径统一配置（本地跑通入口）。

所有训练 / 数据处理脚本都从这里取数据集路径，避免在多份代码里重复硬编码。
数据根目录默认为仓库根目录下的 ``dataset/``：

    <repo_root>/dataset

可通过环境变量 ``DATASET_ROOT`` 覆盖（换机器 / 换数据盘时只改这一处）：

    export DATASET_ROOT=/path/to/dataset

当前数据根目录下已有的目录结构：

    dataset/
    ├── processed_datasets/
    │   ├── csi300_qlib_official/{tnf_data.npy, tnf_close.npy, stocks.npy, times.npy, dataset_config.json}
    │   ├── csi500_qlib_official/...
    │   ├── nasdaq100_qlib_official/...
    │   └── sp500_qlib_official/...
    ├── cn_data/              # 日频 Qlib bin（A股）
    ├── cn_data_1min/         # 分钟频 Qlib bin（A股）
    ├── cn_data_latest/       # 最新 Qlib bin（A股）
    ├── us_data_yahoo/        # 美股 Qlib bin
    ├── csi300_alpha158_alpha360.pt
    └── nas100_alpha158_alpha360.pt
"""

import os

# AsyncMamba/ 的上一级即仓库根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集根目录：默认 <repo_root>/dataset，可用环境变量覆盖
DATASET_ROOT = os.path.abspath(
    os.environ.get("DATASET_ROOT", os.path.join(PROJECT_ROOT, "dataset"))
)

# Qlib 官方流程产出的已处理数据集根目录
PROCESSED_DATASETS_DIR = os.path.join(DATASET_ROOT, "processed_datasets")

# 支持的市场（与 dataset/processed_datasets 下的目录一一对应）
MARKETS = ("csi300", "csi500", "nasdaq100", "sp500")


def _check(path):
    """检查文件是否存在，不存在时给出可操作的报错信息。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到数据集文件: {path}\n"
            f"当前 DATASET_ROOT = {DATASET_ROOT}\n"
            "请确认 dataset 目录结构是否完整，"
            "或设置环境变量 DATASET_ROOT 指向正确的数据根目录。"
        )
    return path


def market_dir(data_name, check=True):
    """返回某个市场的已处理数据目录。

    例: market_dir("csi300") -> <DATASET_ROOT>/processed_datasets/csi300_qlib_official
    """
    path = os.path.join(PROCESSED_DATASETS_DIR, f"{data_name}_qlib_official")
    return _check(path) if check else path


def tnf_path(data_name="csi300", check=True):
    """返回模型输入特征文件 tnf_data.npy 的路径（[T, N, F]，最后一列为 LABEL0）。"""
    path = os.path.join(market_dir(data_name, check=False), "tnf_data.npy")
    return _check(path) if check else path


def close_path(data_name="csi300", check=False):
    """返回未标准化收盘价文件 tnf_close.npy 的路径（回测用，可选）。"""
    path = os.path.join(market_dir(data_name, check=False), "tnf_close.npy")
    return _check(path) if check else path


def stocks_path(data_name="csi300", check=True):
    """返回股票代码列表文件 stocks.npy 的路径。"""
    path = os.path.join(market_dir(data_name, check=False), "stocks.npy")
    return _check(path) if check else path


def times_path(data_name="csi300", check=False):
    """返回交易日序列文件 times.npy 的路径。"""
    path = os.path.join(market_dir(data_name, check=False), "times.npy")
    return _check(path) if check else path


def raw_qlib_dir(name="cn_data"):
    """返回原始 Qlib bin 目录（cn_data / cn_data_1min / cn_data_latest / us_data_yahoo）。"""
    return os.path.join(DATASET_ROOT, name)


def pt_file(name="csi300_alpha158_alpha360.pt"):
    """返回历史 .pt 数据集文件（alpha158/alpha360 版）的路径。"""
    return os.path.join(DATASET_ROOT, name)


if __name__ == "__main__":
    print(f"PROJECT_ROOT        = {PROJECT_ROOT}")
    print(f"DATASET_ROOT        = {DATASET_ROOT}")
    print(f"PROCESSED_DATASETS  = {PROCESSED_DATASETS_DIR}")
    for _market in MARKETS:
        try:
            print(f"  [{_market}] tnf_data.npy -> {tnf_path(_market)}")
        except FileNotFoundError as _err:
            print(f"  [{_market}] 缺失: {_err}")
    for _dir in ("cn_data", "cn_data_1min", "cn_data_latest", "us_data_yahoo"):
        _p = raw_qlib_dir(_dir)
        print(f"  [raw] {_dir}: {'存在' if os.path.isdir(_p) else '缺失'} -> {_p}")
