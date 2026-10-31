import os
import sys
import copy
import argparse
import json
import pprint
from datetime import datetime
from experiment_timer import start_timer
import numpy as np
import pandas as pd
import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from simple_mamba import SimpleMamba, SimpleMambaCell
from data_self import prepare_data
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def extract_lag_weights(model, stock_names=None):
    """
    从训练好的异步传导Mamba模型中提取每只股票的延迟权重
    Args:
        model: 训练完成并加载了最佳权重的AsyncMambaModel
        stock_names: 股票代码列表，从stocks.npy加载
    Returns:
        lag_df: DataFrame，包含每只股票的延迟权重分布和最佳步长
    """
    # ✅ 防御：当 Config.learnable_lag=False 时，AsyncMambaBlock.lag_pool 为 None，
    # 此时模型没有可学习的滞后权重，直接返回 None，
    # 避免 AttributeError 中断整个训练/测试流程（调用方需判空）。
    lag_pool = getattr(model.encoder, "lag_pool", None)
    if lag_pool is None:
        print("⚠️  模型未启用自适应滞后池化(learnable_lag=False)，无滞后权重可提取，跳过。")
        return None
    if getattr(lag_pool, "effective_max_lag", None) is None:
        # effective_max_lag 只在 lag_pool 前向(t>0)时才会被赋值
        print("⚠️  滞后池化尚未执行前向，effective_max_lag 为空，请先运行一次模型前向。")
        return None

    # 获取实际使用的滞后分布
    # v2：p = softmax(lag_weight/τ) × 单调存活门控，必须用模型前向真正使用的分布，
    #     否则“训练用什么”和“论文里画什么”会不一致。
    if hasattr(lag_pool, "get_lag_probs"):
        w = lag_pool.get_lag_probs().detach().cpu().numpy()
    else:
        lag_weight = lag_pool.lag_weight.detach().cpu()
        temperature = lag_pool.temperature
        import torch.nn.functional as F
        w = F.softmax(lag_weight / temperature, dim=-1).numpy()
    n_stocks, l_max = w.shape
    lag_days = np.arange(1, l_max + 1)  # 延迟从1天开始计数
    
    # 计算两种最佳步长
    best_lag = np.argmax(w, axis=1) + 1  # 最可能步长（argmax）
    expected_lag = np.sum(w * lag_days, axis=1)  # 期望步长（加权平均）
    # ✅ 修改点3：新增自适应有效最大滞后（这是自适应版本独有的核心指标）
    effective_max_lag = lag_pool.effective_max_lag.detach().cpu().numpy()
    # 构建结果DataFrame
    lag_df = pd.DataFrame(w, columns=[f"lag_{i}d" for i in lag_days])
    lag_df["best_lag"] = best_lag
    lag_df["expected_lag"] = expected_lag.round(2)
    lag_df["effective_max_lag"] = effective_max_lag  # ✅ 加上这一行！
    # v2 额外指标：滞后分布熵（越小说明该股票的传导周期越确定）
    lag_entropy = getattr(lag_pool, "lag_entropy", None)
    if lag_entropy is not None:
        lag_df["lag_entropy"] = lag_entropy.detach().cpu().numpy().round(4)
    if stock_names is not None:
        lag_df.insert(0, "stock_code", stock_names)
        lag_df = lag_df.set_index("stock_code")
    return lag_df


def plot_results(train_losses, val_metrics_history, save_path="training_curves.png"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    epochs = range(1, len(val_metrics_history) + 1)
    # 修复：使用RankIC_Mean和RankICIR
    plt.plot(epochs, [m['RankIC_Mean'] for m in val_metrics_history], label='Val IC Mean', color='orange', marker='o')
    plt.plot(epochs, [m['RankICIR'] for m in val_metrics_history], label='Val ICIR', color='green', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_top_conduction_heatmap(adj, stock_names=None, save_path="top_conduction_heatmap.png",
                                top_n=20, top_k_per_stock=5):
    """
    只显示每个股票最强的top_k_per_stock个连接，超清晰可视化
    
    Args:
        adj: 邻接矩阵 [B, N, N]
        stock_names: 股票代码列表
        save_path: 保存路径
        top_n: 显示的股票数量
        top_k_per_stock: 每个股票显示的最强连接数
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    adj_np = adj[0].detach().cpu().numpy()
    N = adj_np.shape[0]
    
    # 选择度中心性最高的top_n只股票
    out_degree = adj_np.sum(axis=1)
    top_indices = np.argsort(out_degree)[-top_n:][::-1]
    
    # 提取子矩阵
    adj_subset = adj_np[top_indices][:, top_indices]
    
    if stock_names is not None:
        stock_subset = [stock_names[i] for i in top_indices]
    else:
        stock_subset = [f"S{i}" for i in range(top_n)]
    
    # ====== 只保留每个股票最强的top_k_per_stock个连接 ======
    adj_filtered = np.zeros_like(adj_subset)
    for i in range(top_n):
        # 获取第i行的top_k个最大值的索引
        top_k_idx = np.argsort(adj_subset[i])[-top_k_per_stock:]
        adj_filtered[i, top_k_idx] = adj_subset[i, top_k_idx]
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    
    ax = sns.heatmap(
        adj_filtered,
        cmap="YlOrRd",
        mask=adj_filtered < 1e-4,
        square=True,
        cbar_kws={"label": "Conduction Weight", "shrink": 0.8},
        xticklabels=stock_subset,
        yticklabels=stock_subset,
        linewidths=0.8,
        linecolor='white',
        annot=True,  # 显示数值
        fmt=".3f",   # 数值格式
        annot_kws={"size": 7}
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.title(f"Top-{top_k_per_stock} Conduction Links per Stock (Top {top_n} Stocks)", 
              fontsize=14, pad=20, fontweight='bold')
    plt.xlabel("Source Stock", fontsize=11, fontweight='bold')
    plt.ylabel("Target Stock", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 精简传导热力图已保存到: {save_path}")
    plt.close()


def plot_conduction_network(adj, stock_names=None, save_path="conduction_network.png",
                           top_n=30, min_weight=0.02):  # ✅ min_weight从0.01提高到0.02，过滤弱边
    """
    优化版：股票传导关系网络图
    改进：线条更粗、颜色更深、对比度更高、过滤弱边
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    adj_np = adj[0].detach().cpu().numpy()
    N = adj_np.shape[0]
    
    # 选择度中心性最高的top_n只股票
    out_degree = adj_np.sum(axis=1)
    top_indices = np.argsort(out_degree)[-top_n:][::-1]
    
    if stock_names is not None:
        node_labels = {i: stock_names[idx] for i, idx in enumerate(top_indices)}
    else:
        node_labels = {i: f"S{idx}" for i, idx in enumerate(top_indices)}
    
    # 构建子图邻接矩阵
    adj_subset = adj_np[top_indices][:, top_indices]
    
    # ====== 创建有向图 ======
    G = nx.DiGraph()
    
    # 添加节点
    for i in range(top_n):
        G.add_node(i, label=node_labels[i], size=out_degree[top_indices[i]] * 1000)
    
    # 添加边（只保留权重>min_weight的边）
    edge_weights = []
    for i in range(top_n):
        for j in range(top_n):
            if i != j and adj_subset[i, j] > min_weight:
                G.add_edge(j, i, weight=adj_subset[i, j])  # j→i: j 为源(被聚合)、i 为目标(聚合方)，属同期关联
                edge_weights.append(adj_subset[i, j])
    
    # ====== 绘制网络图 ======
    plt.figure(figsize=(16, 14))
    
    # 使用力导向布局（k调大一点，节点更分散）
    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)  # ✅ k从1.5调到2.0，迭代次数翻倍
    
    # 节点大小（调整系数，让节点更大更明显）
    node_sizes = [G.nodes[i]['size'] for i in range(top_n)]
    node_sizes = np.array(node_sizes)
    node_sizes = 500 + (node_sizes - node_sizes.min()) / (node_sizes.max() - node_sizes.min() + 1e-8) * 1000  # ✅ 节点更大
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color='#1f77b4',  # ✅ 更深的蓝色
                          alpha=0.9,  # ✅ 透明度提高
                          edgecolors='black',
                          linewidths=2.0)  # ✅ 节点边框更粗
    
    # 绘制边（关键优化：线条更粗、颜色更深、箭头更大）
    if edge_weights:
        edge_colors = edge_weights
        edge_widths = np.array(edge_weights) * 15  # ✅ 线条粗细从5倍调到15倍
        
        edges = nx.draw_networkx_edges(G, pos,
                                       width=edge_widths,
                                       edge_color=edge_colors,
                                       edge_cmap=plt.cm.Reds,  # ✅ 换成红色系，对比度更高
                                       alpha=0.85,  # ✅ 透明度提高
                                       arrows=True,
                                       arrowsize=30,  # ✅ 箭头从20调到30
                                       arrowstyle='->,head_width=0.4,head_length=0.6')  # ✅ 箭头形状更明显
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                   norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
        cbar.set_label('Conduction Weight', fontsize=12)
    
    # 绘制节点标签（字体更大更粗）
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=11, font_weight='bold')
    
    # ====== 添加标题和统计信息 ======
    plt.title(f"Stock Conduction Network (Top {top_n} Stocks, min_weight={min_weight})", 
              fontsize=18, pad=20, fontweight='bold')
    
    stats_text = (
        f"Nodes: {G.number_of_nodes()}\n"
        f"Edges: {G.number_of_edges()}\n"
        f"Avg Degree: {G.number_of_edges() / G.number_of_nodes():.2f}\n"
        f"Sparsity: {1 - G.number_of_edges()/(top_n*(top_n-1)):.2%}"
    )
    
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 传导网络图已保存到: {save_path}")
    print(f"   节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    print(f"   子图稀疏度: {1 - G.number_of_edges()/(top_n*(top_n-1)):.2%}")
    plt.close()


def plot_conduction_heatmap(adj, stock_names=None, save_path="conduction_heatmap.png", 
                           top_n=30, figsize=(14, 12), dpi=300):
    """
    绘制股票传导权重热力图（优化版）
    Args:
        adj: 邻接矩阵 [B, N, N]
        stock_names: 股票代码列表
        save_path: 保存路径
        top_n: 显示的股票数量（10/20/30/50）
        figsize: 图形大小
        dpi: 分辨率
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 取第一个batch的邻接矩阵
    adj_np = adj[0].detach().cpu().numpy()  # [N, N]
    N = adj_np.shape[0]
    
    # ====== 优化1：选择度中心性最高的top_n只股票 ======
    if top_n is not None and top_n < N:
        # 计算每只股票的总关联权重（用于筛选强联动股票）
        out_degree = adj_np.sum(axis=1)  # [N]
        # 选择度中心性最高的股票
        top_indices = np.argsort(out_degree)[-top_n:][::-1]  # 降序排列
        
        # 提取子矩阵
        adj_subset = adj_np[top_indices][:, top_indices]
        
        if stock_names is not None:
            stock_subset = [stock_names[i] for i in top_indices]
        else:
            stock_subset = [f"Stock_{i}" for i in top_indices]
    else:
        adj_subset = adj_np
        stock_subset = stock_names if stock_names is not None else None
    
    # ====== 优化2：对数缩放，增强可视化对比度 ======
    # 将接近0的值设为NaN，不显示
    adj_masked = np.ma.masked_where(adj_subset < 1e-4, adj_subset)
    
    # 使用对数尺度（避免0值）
    adj_log = np.log10(adj_subset + 1e-8)
    adj_log_masked = np.ma.masked_where(adj_subset < 1e-4, adj_log)
    
    plt.figure(figsize=figsize)
    
    # ====== 优化3：使用更合适的颜色映射 ======
    # 方案A：使用离散颜色映射（推荐）
    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # 蓝-白-红
    # 方案B：使用顺序颜色映射
    # cmap = sns.color_palette("YlOrRd", as_cmap=True)  # 黄-橙-红
    
    # 绘制热力图
    ax = sns.heatmap(
        adj_subset,  # 使用原始值
        cmap="YlOrRd",  # 暖色调，更适合稀疏矩阵
        mask=adj_subset < 1e-4,  # 隐藏接近0的值
        square=True,
        cbar_kws={"label": "Conduction Weight", "shrink": 0.8},
        xticklabels=stock_subset,
        yticklabels=stock_subset,
        linewidths=0.5,
        linecolor='gray',
        alpha=0.9
    )
    
    # ====== 优化4：美化坐标轴 ======
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # ====== 优化5：添加标题和标签 ======
    plt.title(f"Stock Conduction Weight Heatmap (Top {len(stock_subset)} Stocks)", 
              fontsize=14, pad=20, fontweight='bold')
    plt.xlabel("Source Stock", fontsize=11, fontweight='bold')
    plt.ylabel("Target Stock", fontsize=11, fontweight='bold')
    
    # ====== 优化6：添加统计信息文本框 ======
    non_zero_count = np.sum(adj_subset > 1e-4)
    sparsity = 1 - non_zero_count / (adj_subset.shape[0] ** 2)
    max_weight = np.max(adj_subset)
    mean_weight = np.mean(adj_subset[adj_subset > 1e-4])
    
    stats_text = (
        f"Matrix Size: {adj_subset.shape[0]}×{adj_subset.shape[0]}\n"
        f"Non-zero Elements: {non_zero_count}\n"
        f"Sparsity: {sparsity:.2%}\n"
        f"Max Weight: {max_weight:.4f}\n"
        f"Mean Weight: {mean_weight:.4f}"
    )
    
    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✅ 传导权重热力图已保存到: {save_path}")
    print(f"   矩阵大小: {adj_subset.shape[0]}×{adj_subset.shape[0]}, "
          f"非零元素: {non_zero_count}, 稀疏度: {sparsity:.2%}")
    plt.close()

# 修复：更新指标名称
def plot_full_results(train_losses, val_metrics_hist, long_only_rets, long_short_rets, save_prefix="result"):
    # 绘制损失和IC曲线
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(train_losses, label='Train MSE')
    plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.subplot(2,1,2)
    # 修复：使用RankIC_Mean和RankICIR
    plt.plot([m['RankIC_Mean'] for m in val_metrics_hist], label='Val IC')
    plt.plot([m['RankICIR'] for m in val_metrics_hist], label='Val ICIR')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"{save_prefix}_training.png", dpi=300)

    # 回测累计收益
    long_only_rets = np.array(long_only_rets)
    long_short_rets = np.array(long_short_rets)
    # 过滤NaN值
    long_only_rets = long_only_rets[np.isfinite(long_only_rets)]
    long_short_rets = long_short_rets[np.isfinite(long_short_rets)]
    
    if len(long_only_rets) > 0 and len(long_short_rets) > 0:
        cum_long = np.cumprod(1 + np.array(long_only_rets))
        cum_ls = np.cumprod(1 + np.array(long_short_rets))
        plt.figure(figsize=(10,5))
        plt.plot(cum_long, label='Long Only')
        plt.plot(cum_ls, label='Long-Short')
        plt.title('Cumulative Returns')
        plt.legend(); plt.grid(True)
        plt.savefig(f"{save_prefix}_backtest.png", dpi=300)
    plt.close()


def plot_lag_distribution(lag_df, column="best_lag", title=None, save_path="lag_distribution.png", dpi=300, show_kde=True):
    """
    绘制股票传导延迟分布直方图，支持绘制任意列（best_lag/effective_max_lag）
    完全向下兼容原有调用方式
    Args:
        lag_df: extract_lag_weights返回的DataFrame
        column: 要绘制的列名，默认"best_lag"，可选"effective_max_lag"
        title: 图表标题，不指定则自动生成
        save_path: 图片保存路径
        dpi: 图片分辨率
        show_kde: 是否显示核密度估计曲线

    """
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图+核密度估计
    sns.histplot(
        data=lag_df, 
        x=column, 
        discrete=True, 
        kde=show_kde, 
        color="#1f77b4",
        edgecolor="white",
        linewidth=1
    )
    
    # 添加统计标注
    mean_val = lag_df[column].mean()
    median_val = lag_df[column].median()
    plt.axvline(
        mean_val, 
        color="#ff7f0e", 
        linestyle="--", 
        linewidth=2, 
        label=f"Mean: {mean_val:.2f} days"
    )
    plt.axvline(
        median_val, 
        color="#2ca02c", 
        linestyle=":", 
        linewidth=2,
        label=f"Median: {median_val:.0f} days"
    )
    
    # 自动生成英文标题（彻底解决字体警告）
    if title is None:
        if column == "best_lag":
            title = "Distribution of Learned Optimal Conduction Lags"
        elif column == "effective_max_lag":
            title = "Distribution of Adaptive Effective Maximum Lags"
        else:
            title = f"Distribution of {column}"
    
    # 全部使用英文标签
    plt.xlabel("Lag (Days)", fontsize=12)
    plt.ylabel("Number of Stocks", fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(range(1, lag_df[column].max() + 1), fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"✅ 延迟分布直方图已保存到: {save_path}")
    plt.close()


def plot_lag_weight_heatmap(lag_df, top_n=50, save_path="lag_weight_heatmap.png", dpi=300):
    """
    绘制前N只股票的延迟权重热力图（展示每只股票对不同延迟的置信度）
    Args:
        lag_df: extract_lag_weights返回的DataFrame
        top_n: 展示前多少只股票（太多会看不清）
        save_path: 图片保存路径
        dpi: 图片分辨率
    """
    # 提取延迟权重列（去掉best_lag和expected_lag）
    weight_cols = [col for col in lag_df.columns if col.startswith("lag_")]
    # 取前top_n只股票（按股票代码排序）
    heatmap_data = lag_df[weight_cols].head(top_n)
    
    plt.figure(figsize=(8, top_n * 0.35))  # 动态调整高度
    # 绘制热力图
    ax = sns.heatmap(heatmap_data,cmap="YlGnBu",annot=True,
        fmt=".2f",cbar=True,cbar_kws={"label": "Weight", "shrink": 0.8},linewidths=0.5,linecolor="white")
    
    # 图表美化
    plt.xlabel("Conduction Lag (Days)", fontsize=12)
    plt.ylabel("Stock Code", fontsize=12)
    plt.title(f"Learned Lag Weight Distribution (Top {top_n} Stocks)", fontsize=14, pad=20)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=9)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"✅ 延迟权重热力图已保存到: {save_path}")
    plt.close()


def print_model_structure(model, indent=0):
    """打印完整的模型结构和参数数量"""
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print("  " * indent + f"├─ {name}: {module.__class__.__name__} (参数: {param_count:,})")
        print_model_structure(module, indent + 1)


def print_model_architecture(model):
    """
    统一打印完整的模型架构和参数量统计
    Args:
        model: 训练好的AsyncMambaModel实例
    """
    print("\n" + "="*80, flush=True)
    print("【模型架构】", flush=True)
    print("="*80, flush=True)
    print(model, flush=True)
    print("="*80 + "\n", flush=True)

    print("【模型超级完整架构】", flush=True)
    print_model_structure(model)
    print("="*80 + "\n", flush=True)

    print("\n【模型参数量统计】", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}", flush=True)
    print(f"可训练参数量: {trainable_params:,}", flush=True)
    print("="*80 + "\n", flush=True)


def print_lag_statistics(lag_df):
    """
    统一打印传导延迟统计信息
    Args:
        lag_df: extract_lag_weights返回的DataFrame
    """
    print("\n📊 传导延迟统计:")
    print(f"总股票数: {len(lag_df)}")
    print(f"平均最佳延迟: {lag_df['best_lag'].mean():.2f} 天")
    print(f"中位数最佳延迟: {lag_df['best_lag'].median():.0f} 天")
    print(f"最小延迟: {lag_df['best_lag'].min()} 天")
    print(f"最大延迟: {lag_df['best_lag'].max()} 天")
    print(f"延迟标准差: {lag_df['best_lag'].std():.2f} 天")
    print("\n延迟步长分布:")
    print(lag_df["best_lag"].value_counts().sort_index())
    
    # 只有自适应滞后版本才打印有效最大滞后
    if "effective_max_lag" in lag_df.columns:
        print("\n" + "="*50)
        print("📊 自适应有效最大滞后统计:")
        print(f"平均有效最大滞后: {lag_df['effective_max_lag'].mean():.2f} 天")
        print(f"中位数有效最大滞后: {lag_df['effective_max_lag'].median():.0f} 天")
        print("\n有效最大滞后分布:")
        print(lag_df["effective_max_lag"].value_counts().sort_index())
        if "lag_entropy" in lag_df.columns:
            print(f"平均滞后分布熵: {lag_df['lag_entropy'].mean():.4f} nats")
        print("="*50 + "\n")