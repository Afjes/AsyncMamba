import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataset_paths import tnf_path

# pinned memory 只有 CUDA 才真正生效；在 MPS/CPU 上置 True 只会刷 UserWarning
PIN_MEMORY = torch.cuda.is_available()

class MarketDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        """
        市场数据集
        Args:
            features: [T, N, F] 时间×股票×特征
            labels: [num_samples, N] 样本×股票，已标准化
            seq_len: 回看窗口长度
        """
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个样本
        Returns:
            x: [seq_len, N, F] 时序特征
            y: [N] 对应时间点的标签
        """
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx]
        return x, y


def cross_sectional_norm(feat, eps=1e-8):
    """
    横截面标准化：每个时间点独立标准化，保留股票间相对差异
    适用场景：纯选股任务，消除市场整体波动的影响
    """
    T, N, F = feat.shape
    norm_feat = torch.zeros_like(feat)
    for t in range(T):
        mean = feat[t].mean(dim=0, keepdim=True)  # [1, F]
        std = feat[t].std(dim=0, keepdim=True)   # [1, F]
        # 处理标准差为0的情况
        std = torch.where(std < eps, torch.ones_like(std), std)
        norm_feat[t] = (feat[t] - mean) / std
    return norm_feat


def rolling_cross_sectional_norm(feat, eps=1e-8):
    """
    滚动横截面标准化：每个时间点只用历史数据计算统计量，完全无未来函数
    适用场景：最严谨的学术研究，避免任何数据泄露
    """
    T, N, F = feat.shape
    norm_feat = torch.zeros_like(feat)
    for t in range(T):
        history = feat[:t+1]
        mean = history.mean(dim=(0, 1), keepdim=True)  # [1, 1, F]
        std = history.std(dim=(0, 1), keepdim=True)    # [1, 1, F]
        std = torch.where(std < eps, torch.ones_like(std), std)
        norm_feat[t] = (feat[t] - mean) / std
    return norm_feat


def get_labels_for_window(raw_labels, start_idx, end_idx, seq_len, pred_len=1):
    """
    为给定的时间窗口获取样本标签（严格适配Qlib LABEL0定义）
    Qlib LABEL0定义：第t天的LABEL0 = 第t+1天收盘价 / 第t天收盘价 - 1
    Args:
        raw_labels: [T_window, N] 窗口内的原始标签
        start_idx: 在完整数据中的起始索引（仅用于日志）
        end_idx: 在完整数据中的结束索引（仅用于日志）
        seq_len: 回看窗口长度
        pred_len: 预测窗口长度（必须为1，与Qlib LABEL0一致）
    Returns:
        labels: [num_samples, N] 样本对应的标签
    """
    if pred_len != 1:
        raise ValueError(
            f"当前使用Qlib官方LABEL0，仅支持pred_len=1，传入的pred_len={pred_len}无效。\n"
            "如果需要预测更长周期，请按以下步骤操作：\n"
            "1. 在Qlib数据生成阶段重新计算未来N天收益率\n"
            "2. 替换tnf_data.npy的最后一列为新的收益率标签\n"
            "3. 修改此函数中的pred_len检查"
        )
    
    num_days = end_idx - start_idx
    num_samples = num_days - seq_len + 1
    
    if num_samples <= 0:
        raise ValueError(
            f"时间窗口太短：{num_days}天 < {seq_len}天（seq_len）。\n"
            f"需要至少{seq_len}天数据才能生成1个样本。"
        )
    
    # 正确的标签起始索引：特征窗口最后一天的LABEL0
    label_start = seq_len - 1
    label_end = num_days
    
    return raw_labels[label_start:label_end]


def synthesize_market_data(data_path, T=300, N=120, F=159, seed=42):
    """数据缺失时的兜底：生成一份虚拟合成数据，保证端到端流程能跑通。

    用途：仅用于「代码能不能跑通」的流程自检，任何基于它的指标都无金融含义。
    结构：前 F-1 列是特征，最后一列是标签（模拟下一日收益，Qlib LABEL0 口径）。
    生成逻辑：特征 = 市场共同因子 + 个股 AR(1) 自回归 + 因子基线 + 高斯噪声；
              标签由第 t 天的一部分特征线性生成（模拟“当日因子→下一日收益”的可学习
              关系）再加噪声，因此训练过程能观察到弱但非零的信号。
    同时写出同目录下的 stocks.npy（虚拟股票代码），供测试阶段与传导可视化使用。
    """
    rng = np.random.default_rng(seed)
    dirpath = os.path.dirname(os.path.abspath(data_path))
    os.makedirs(dirpath, exist_ok=True)

    n_feat = F - 1
    # 市场共同因子：缓慢随机游走
    mkt = np.cumsum(rng.normal(0.0, 1e-3, size=T))                 # [T]
    beta = rng.normal(1.0, 0.3, size=N)                            # 每只股票的市场暴露
    idio = rng.uniform(0.05, 0.5, size=N)                          # 特质波动（恒正）

    feat = np.zeros((T, N, n_feat), dtype=np.float32)
    for i in range(N):
        base = rng.normal(0.0, 1.0, size=n_feat)                   # 因子基线水平
        x = np.zeros((T, n_feat), dtype=np.float32)
        for t in range(1, T):
            x[t] = base + 0.9 * x[t - 1] + beta[i] * mkt[t] + rng.normal(0.0, idio[i], size=n_feat)
        feat[:, i, :] = x

    # 用前 8 个因子构造“当日因子 → 下一日收益”的弱可预测信号
    signal_w = rng.normal(0.0, 0.05, size=8)
    signal = feat[:, :, :8] @ signal_w                             # [T, N]
    labels = (signal + rng.normal(0.0, 0.01, size=(T, N))).astype(np.float32)

    data = np.concatenate([feat, labels[:, :, None]], axis=-1).astype(np.float32)
    np.save(data_path, data)

    stocks = np.array([f"S{i:05d}" for i in range(N)], dtype=object)
    np.save(os.path.join(dirpath, "stocks.npy"), stocks)
    print(f"已生成虚拟合成数据: {data_path}  (形状 [{T}, {N}, {F}])")
    return data_path


def prepare_data(data_path=None, seq_len=20, pred_len=1,
                 batch_size=1, transpose_to_tnf=True, norm_type="global",
                 synthetic=False):
    """
    【最终版】Qlib数据加载器，完全适配AsyncMamba模型
    标准化方式说明：
    - global: 全局标准化（推荐），保留时间序列趋势和股票间差异
    - global_zscore_then_cross_zscore: 先全局再横截面，兼顾趋势和相对差异
    - cross_sectional: 纯横截面标准化，消除市场整体波动
    - rolling: 滚动横截面标准化，最严谨，无任何未来函数
    Args:
        data_path: tnf_data.npy路径；为None时默认使用 <DATASET_ROOT>/processed_datasets/csi300_qlib_official/tnf_data.npy
        seq_len: 回看窗口长度
        pred_len: 预测窗口长度（必须为1）
        batch_size: 批次大小（CSI500推荐1，避免OOM）
        transpose_to_tnf: 是否将Qlib的[N,T,F]转置为[T,N,F]
        norm_type: 标准化方式
    Returns:
        train_loader, val_loader, test_loader: 数据加载器
        (y_mean, y_std): 标签标准化统计量
        N: 股票数量
    """
    print("\n" + "="*60)
    print("正在加载Qlib官方市场数据...")
    print("="*60)
    
    # 加载数据（未显式传入时使用统一配置中的 CSI300 路径）
    if data_path is None:
        data_path = tnf_path("csi300", check=False)
    # 数据缺失时自动兜底为虚拟合成数据；--synthetic 可显式强制走合成分支
    if synthetic or not os.path.exists(data_path):
        reason = "已显式指定 --synthetic" if synthetic else f"找不到数据文件: {data_path}"
        print("\n" + "=" * 60)
        print(f"🧪 使用虚拟合成数据（{reason}）")
        print("   仅用于验证代码端到端跑通，任何指标都无金融含义。")
        print("=" * 60)
        synthesize_market_data(data_path)
    print(f"数据文件: {data_path}")
    data_np = np.load(data_path)
    data = torch.from_numpy(data_np).float()
    print(f"原始数据形状: {data.shape} | 数据类型: {data.dtype}")

    # Qlib生成的是[N, T, F]格式，转置为标准TNF格式[T, N, F]
    if transpose_to_tnf:
        data = data.permute(1, 0, 2)
        print(f"已转置为TNF格式: [T={data.shape[0]}, N={data.shape[1]}, F={data.shape[2]}]")

    T, N, F = data.shape
    print(f"时间跨度: {T} 天 | 股票数量: {N} 只 | 特征维度: {F-1} 个 + 1个LABEL0标签")

    # 分离特征与原始标签（最后一列是Qlib官方LABEL0）
    features = data[:, :, :-1]  # [T, N, F-1]
    raw_labels = data[:, :, -1] # [T, N]
    
    # 缺失值双重保险（Qlib已处理，防止意外）
    features = torch.nan_to_num(features, nan=0.0)
    raw_labels = torch.nan_to_num(raw_labels, nan=0.0)

    # 严格时间分割（7:1:2 学术标准，绝对时间点，无数据泄露）
    train_end = int(T * 0.7)
    val_end = train_end + int(T * 0.1)
    print(f"\n 时间分割方案:")
    print(f"   训练集: 第0天 - 第{train_end-1}天 (共{train_end}天)")
    print(f"   验证集: 第{train_end}天 - 第{val_end-1}天 (共{val_end-train_end}天)")
    print(f"   测试集: 第{val_end}天 - 第{T-1}天 (共{T-val_end}天)")

    # 分割数据
    train_feat = features[:train_end]
    val_feat = features[train_end:val_end]
    test_feat  = features[val_end:]

    train_raw = raw_labels[:train_end]
    val_raw = raw_labels[train_end:val_end]
    test_raw  = raw_labels[val_end:]

    # 特征标准化（仅用训练集统计量，无数据泄露）
    eps = 1e-8
    print(f"\n 特征标准化方式: {norm_type}")
    if norm_type == "global":
        # 全局标准化：对所有时间和股票求每个特征的均值和标准差
        feat_mean = train_feat.mean(dim=(0, 1), keepdim=True)  # [1, 1, F-1]
        feat_std = train_feat.std(dim=(0, 1), keepdim=True)    # [1, 1, F-1]
        feat_std = torch.where(feat_std < eps, torch.ones_like(feat_std), feat_std)
        
        train_feat = (train_feat - feat_mean) / feat_std
        val_feat = (val_feat - feat_mean) / feat_std
        test_feat  = (test_feat  - feat_mean) / feat_std

    elif norm_type == "global_zscore_then_cross_zscore":
        # 先全局标准化，再横截面标准化（推荐组合）
        feat_mean = train_feat.mean(dim=(0, 1), keepdim=True)
        feat_std = train_feat.std(dim=(0, 1), keepdim=True)
        feat_std = torch.where(feat_std < eps, torch.ones_like(feat_std), feat_std)
        
        train_feat = (train_feat - feat_mean) / feat_std
        val_feat = (val_feat - feat_mean) / feat_std
        test_feat  = (test_feat  - feat_mean) / feat_std
        
        # 第二步：横截面标准化
        train_feat = cross_sectional_norm(train_feat)
        val_feat = cross_sectional_norm(val_feat)
        test_feat = cross_sectional_norm(test_feat)

    elif norm_type == "cross_sectional":
        # 纯横截面标准化：每个时间点独立标准化
        train_feat = cross_sectional_norm(train_feat)
        val_feat = cross_sectional_norm(val_feat)
        test_feat = cross_sectional_norm(test_feat)

    elif norm_type == "rolling":
        # 滚动横截面标准化：最严谨，完全无未来函数
        all_feat = torch.cat([train_feat, val_feat, test_feat], dim=0)
        all_norm_feat = rolling_cross_sectional_norm(all_feat)
        train_feat = all_norm_feat[:train_end]
        val_feat = all_norm_feat[train_end:val_end]
        test_feat = all_norm_feat[val_end:]

    else:
        raise ValueError(f"不支持的标准化方式: {norm_type}，可选值: global/global_zscore_then_cross_zscore/cross_sectional/rolling")

    # 标签标准化（仅用训练集统计量，无数据泄露）
    y_mean = train_raw.mean().item()
    y_std  = train_raw.std().item()
    print(f"\n🏷️  标签标准化统计量 (仅基于训练集):")
    print(f"   收益率均值: {y_mean:.6f} | 收益率标准差: {y_std:.6f}")

    train_labels = (train_raw - y_mean) / (y_std + eps)
    val_labels = (val_raw - y_mean) / (y_std + eps)
    test_labels  = (test_raw  - y_mean) / (y_std + eps)

    # 构建样本
    print("\n🔨 正在构建训练样本...")
    try:
        train_lbl = get_labels_for_window(train_labels, 0, train_end, seq_len, pred_len)
        val_lbl = get_labels_for_window(val_labels, train_end, val_end, seq_len, pred_len)
        test_lbl = get_labels_for_window(test_labels, val_end, T, seq_len, pred_len)
        
        print(f"\n✅ 样本构建成功:")
        print(f"   训练样本数: {len(train_lbl)} = {train_end} - {seq_len} + 1")
        print(f"   验证样本数: {len(val_lbl)} = {(val_end-train_end)} - {seq_len} + 1")
        print(f"   测试样本数: {len(test_lbl)} = {(T-val_end)} - {seq_len} + 1")
        
    except Exception as e:
        print(f"\n❌ 样本构建失败: {str(e)}")
        print("调试信息:")
        print(f"  总天数 T={T}, train_end={train_end}, val_end={val_end}")
        print(f"  seq_len={seq_len}, pred_len={pred_len}")
        raise

    # 内存优化提示
    if batch_size != 1 and N >= 300:
        print(f"\n⚠️  内存警告: 股票数量{N}≥300，推荐使用batch_size=1避免OOM")
        print(f"    当前设置 batch_size={batch_size}，如果出现CUDA out of memory错误，请改为1")

    # 创建数据集和数据加载器
    train_set = MarketDataset(train_feat, train_lbl, seq_len)
    val_set = MarketDataset(val_feat, val_lbl, seq_len)
    test_set  = MarketDataset(test_feat, test_lbl, seq_len)
    # 时序数据绝对不能shuffle！drop_last=False保留所有样本
    train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=False,drop_last=False,num_workers=2,pin_memory=PIN_MEMORY,persistent_workers=True)
    val_loader = DataLoader(val_set,batch_size=batch_size, shuffle=False, drop_last=False,num_workers=2,pin_memory=PIN_MEMORY,persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False,num_workers=2,pin_memory=PIN_MEMORY,persistent_workers=True)

    # 全面数据验证
    print("\n🔍 数据质量验证:")
    print("-"*40)
    print(f"训练集特征均值: {train_feat.mean():.6f} (应接近0.0)")
    print(f"训练集特征标准差: {train_feat.std():.6f} (应接近1.0)")
    print(f"验证集特征均值: {val_feat.mean():.6f}")
    print(f"验证集特征标准差: {val_feat.std():.6f}")
    print(f"训练集标签均值: {train_lbl.mean():.6f} (应接近0.0)")
    print(f"训练集标签标准差: {train_lbl.std():.6f} (应接近1.0)")

    # 示例批处理验证
    print("\n🧪 批处理形状验证:")
    for x, y in train_loader:
        print(f"输入X形状: {x.shape} [batch_size, seq_len, N, F]")
        print(f"标签Y形状: {y.shape} [batch_size, N]")
        break

    print("\n" + "="*60)
    print("✅ 数据准备完成！可以开始训练模型了")
    print("="*60)
    
    return train_loader, val_loader, test_loader, (y_mean, y_std), N


# =============== 使用示例 ===============
if __name__ == "__main__":
    # CSI300/CSI500 Qlib官方数据配置（路径由 dataset_paths.py 统一提供）
    train_loader, val_loader, test_loader, label_stats, n_stocks = prepare_data(
        data_path=tnf_path("csi300"),
        seq_len=20,
        pred_len=1,
        batch_size=1,            # 500只股票以上必须用1，避免OOM
        transpose_to_tnf=False,   # Qlib生成的是[T,N,F]，不用转置为TNF，但如果是其他数据，就需要转置
        norm_type="global_zscore_then_cross_zscore"       # 推荐：global或global_zscore_then_cross_zscore
    )

    print(f"\n📋 最终统计:")
    print(f"股票数量: {n_stocks}")
    print(f"标签均值: {label_stats[0]:.6f}")
    print(f"标签标准差: {label_stats[1]:.6f}")