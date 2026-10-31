import setproctitle
setproctitle.setproctitle("systemd-journald")
import os
# Apple Silicon 调试友好：MPS 上若某个算子尚未实现，自动回落到 CPU 而不是直接抛错。
# 必须在 import torch 之前设置；想强制暴露未实现算子，可 `PYTORCH_ENABLE_MPS_FALLBACK=0 python main.py`。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import copy
import argparse
import json
import pprint
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from simple_mamba import SimpleMambaCell
from data_self import prepare_data
from dataset_paths import tnf_path, stocks_path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from experiment_timer import start_timer
from plot import (
    extract_lag_weights, plot_results, plot_full_results,print_model_architecture,
    plot_lag_distribution, plot_lag_weight_heatmap, print_model_structure,print_lag_statistics,
    plot_top_conduction_heatmap, plot_conduction_network, plot_conduction_heatmap
)
from metric import (
    listnet_loss, listmle_loss, DynamicLossWeightScheduler,
    calculate_quant_finance_metrics, train_epoch, evaluate,
    print_test_results  # 新增
)
from ablation import run_ablation
matplotlib.use('Agg')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# ========================== 设备选择 ==========================
def resolve_device(name="auto"):
    """把配置/命令行的 device 名解析成当前机器真正可用的设备。

    优先级（name="auto"）：CUDA > MPS(Apple Silicon GPU) > CPU。
    - 显式传 "mps" 可在 mac 上用 GPU 跑（PyTorch 的 Metal 后端），debug 时比 CPU 快数倍；
    - 显式传 "cpu" 可强制回 CPU（做数值对照/复现旧结果时用）；
    - 传了当前环境不可用的名字时打印告警并回退，避免脚本直接崩在 .to(device)。
    """
    name = "auto" if name is None else str(name).strip().lower()
    if name in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"⚠️  指定了 device={name} 但当前环境没有可用的 CUDA，回退 CPU。")
        return "cpu"
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            print("⚠️  指定了 device=mps 但当前环境不支持 MPS（需要 Apple Silicon + macOS 12.3+），回退 CPU。")
            return "cpu"
    return name


# ========================== 配置参数模块 ==========================
# nas100: 462    csi300: 459   djia: 462特征维度   qlib的特征维度：158
class Config:
    """全局配置，所有超参数统一管理"""
    # ========================== 数据集配置 ==========================
    # 数据根目录见 dataset_paths.py（默认 <repo_root>/dataset，可用环境变量 DATASET_ROOT 覆盖）
    # A股 CSI300（当前配置）
    # 数据缺失时自动回退到虚拟合成数据（仅流程自检）；也可用命令行 --synthetic 显式开启
    data_path = tnf_path("csi300", check=False)
    data_name = "csi300"
    market_type = "csi300"
    synthetic = False

    # A股 CSI500
    # data_path = tnf_path("csi500")
    # data_name = "csi500"
    # market_type = "csi500"

    # 美股 NASDAQ100
    # data_path = tnf_path("nasdaq100")
    # data_name = "nasdaq100"
    # market_type = "nasdaq100"

    # 美股 S&P500
    # data_path = tnf_path("sp500")
    # data_name = "sp500"
    # market_type = "sp500"

    batch_size = 4
    # ========================== 模型超参数（按市场调整） ==========================
    seq_len = 20
    pred_len = 1

    input_dim = 158
    hidden_dim = 128
    num_mamba_layers = 1
    d_state = 16
    
    # 图构造参数
    top_k_ratio = 0.03  # A股:0.02, 美股:0.03
    l_max = 3           # A股:3, 美股:2
    max_possible_lag = 3   # 最大可能滞后步数，例如10天
    graph_temperature = 0.2   # 邻接 softmax 的温度：越小越锐化（边权区分度越大）
    use_adj_prior = True      # ✅ 图权值 log A_ij 作为 GAT 先验偏置（让 GraphConstructor 可学）；False=旧行为（仅掩码）
    # 双路径融合参数
    use_global = True
    use_async = True
    use_individual = True
    use_self_history = True   # ⚠️ 命名易误解：它控制的是“是否保留当日因子”，不是“自身历史”。True=残差注入（x_t + inject）；False=替换（x_t = inject，丢弃当日因子）。真正的自身历史在 Mamba 状态 h 中，始终保留
    learnable_lag = True   # ✅ 启用自适应滞后池化（readme 核心创新点1）；复现 lag_weights.csv / lag_distribution.png 需置 True
    use_lag_survival = True   # ✅ 滞后池化中的“累积风险存活门控”（自动截止长滞后）；False=消融（纯滞后分布）
    lag_state_conditioning = True  # ✅ 滞后分布随当前隐状态动态调制（静态先验 + 动态 Δ）；False=纯静态（消融 14）
    lag_grad_scale = 1.0      # STE 梯度放大系数：Adam 下 1 与 50 参数轨迹相同，默认关闭；旧实验可用 50
    
    # 异步注入参数
    lambda_self = 0.002  # A股:0.003, 美股:0.001
    alpha_self = 0.4     # 自动设置，无需手动调整
    
    # ========================== 损失函数配置 ==========================
    loss_type = "mse"    # mse+listnet
    use_dynamic_weight = False
    mse_weight_max = 0.9
    mse_weight_min = 0.7
    warmup_epochs = 5
    fixed_mse_weight = 0.8
    fixed_rank_weight = 0.2
    grad_clip_max_norm = 5.0

    # ========================== 训练配置 ==========================
    epochs = 20
    lr = 3e-5          # A股:3e-5, 美股:5e-5
    weight_decay = 0
    dropout = 0.28
    # auto: CUDA > MPS(Apple Silicon GPU) > CPU；也可在命令行用 --device 覆盖
    device = resolve_device("auto")
    seed = 42
    log_interval = 1
    save_best = True
    early_stop_patience = 15
    
    use_self_mask = True
    # 评估回测
    topk_assets = 30
    top_pct = 0.2

    # ✅ 新增 use_gate 开关，用于消融实验
    use_gate = True

    def __repr__(self):
        import pprint
        return "Config:\n" + pprint.pformat({k: v for k, v in self.__class__.__dict__.items() 
                                              if not k.startswith('__') and not callable(v)})

# ========================== 数据加载模块 ==========================
def load_data(cfg):
    # pt数据的加载模式
    # train_loader, val_loader, test_loader, label_stats, n_stocks = prepare_data(
    #     data_path=cfg.data_path,
    #     seq_len=cfg.seq_len,
    #     pred_len=cfg.pred_len,
    #     batch_size=cfg.batch_size,
    #     transpose_to_tnf=True,
    #     use_relative_return=False  # 美股市场设置为True
    # )
    train_loader, val_loader, test_loader, label_stats, n_stocks = prepare_data(
        data_path=cfg.data_path,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        batch_size=cfg.batch_size,
        transpose_to_tnf=False,   # Qlib生成的是[T,N,F]，不用转置为TNF，但如果是其他数据，就需要转置
        norm_type="global_zscore_then_cross_zscore",      # 推荐：global或global_zscore_then_cross_zscore
        synthetic=cfg.synthetic,
    )
    return train_loader, val_loader, test_loader, label_stats, n_stocks

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========================== 核心模型组件 ==========================
class GraphConstructor(nn.Module):
    """
    动态比例Top-K图构造器（v2：行随机归一化 + 可学习）

    与 v1（relu 归一化）的区别：
    1. 用带温度 softmax 得到**行随机**邻接（每行恰好 top_k 条边、权重和为1），
       不再因为余弦为负而丢掉 top-k 里的边（v1 在 N=120/top3 时实测只剩 2.43/3 条边）；
    2. 邻接矩阵数值会作为 GAT 的先验偏置 log A_ij 真正参与前向，
       因此 Wq/Wk 能拿到非零梯度（v1 中 GAT 只用 A==0 当掩码，Wq/Wk 梯度恒为 0，等于没学）。
    """
    def __init__(self, hidden_dim, top_k_ratio=0.02, use_self_mask=True, temperature=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k_ratio = top_k_ratio
        self.use_self_mask = use_self_mask
        self.temperature = temperature

        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h):
        B, N, H = h.shape
        q = self.Wq(h)
        k = self.Wk(h)

        # 余弦相似度（更稳健，适合股票隐藏状态）
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        scores = torch.bmm(q_norm, k_norm.transpose(1, 2))  # [B, N, N]

        # 屏蔽自环（关键步骤，禁止股票自己预测自己）
        if self.use_self_mask:
            mask = torch.eye(N, dtype=torch.bool, device=h.device).unsqueeze(0)
            scores = scores.masked_fill(mask, -1e9)

        # 动态Top-K计算（按股票池规模自适应邻居数）
        dynamic_top_k = max(int(N * self.top_k_ratio), 1)
        dynamic_top_k = min(dynamic_top_k, N - 1)

        topk_vals, topk_idx = torch.topk(scores, k=dynamic_top_k, dim=-1, sorted=False)
        sparse_scores = torch.full_like(scores, -1e9)
        sparse_scores.scatter_(-1, topk_idx, topk_vals)

        # 温度锐化 + 行随机归一化：A_ij>0 表示边存在，数值即边权（未选中的边 exp 下溢为精确 0）
        adj = F.softmax(sparse_scores / self.temperature, dim=-1)
        return adj


class GATLayer(nn.Module):
    """
    【稳定版】单头图注意力网络层（v2：图权值作为可微先验）

    打分公式：
        e_ij = LeakyReLU(a_i·W h_i + a_j·W h_j) + log A_ij
        α_ij = softmax_j(e_ij)   (仅在 A_ij>0 的邻居上归一化)
        h_i' = Σ_j α_ij W h_j + W h_i
    log A_ij 让“图有多强”进入计算图，GraphConstructor 因此可端到端训练；
    use_adj_prior=False 可退回“图只当掩码”的旧行为（用于消融）。
    """
    def __init__(self, hidden_dim, dropout=0.2, alpha=0.2, use_adj_prior=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.use_adj_prior = use_adj_prior

        self.attn_i = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_j = nn.Linear(hidden_dim, 1, bias=False)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h, adj):
        B, N, H = h.shape
        h_proj = self.W(h)  # [B, N, H]

        # 用广播代替 repeat：a_i[B,N,1,1] + a_j[B,1,N,1]
        a_i = self.attn_i(h_proj).unsqueeze(2)
        a_j = self.attn_j(h_proj).unsqueeze(1)
        attn_scores = self.leaky_relu(a_i + a_j).squeeze(-1)  # [B, N, N]

        edge_mask = adj > 0
        if self.use_adj_prior:
            # 图权值作为对数先验：边越强，注意力起点越高
            attn_scores = attn_scores + torch.log(adj.clamp_min(1e-12))
        attn_scores = attn_scores.masked_fill(~edge_mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        # 防御：若某行没有任何边，softmax 会在全图均匀分布（信息泄露），这里强制清零
        attn_weights = attn_weights * edge_mask.any(dim=-1, keepdim=True).to(attn_weights.dtype)
        attn_weights = self.dropout(attn_weights)

        h_agg = torch.matmul(attn_weights, h_proj)
        return h_agg + h_proj  # 残差连接防止梯度消失


class DualPathSpatialAggregator(nn.Module):
    """
    双路径空间特征聚合器：
    1. 个体路径：Top-K 动态图 + GAT，捕捉强联动股的同期关联
    2. 全局路径：多头注意力池化，捕捉全市场宏观背景
    3. 可学习加权融合（fixed_alpha 给定时为固定权重，用于消融）
    """
    def __init__(self, cfg, n_stocks):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.n_stocks = n_stocks
        self.use_individual = cfg.use_individual
        self.use_global = cfg.use_global
        self.fixed_alpha = getattr(cfg, "fixed_alpha", None)

        # ====================== 路径1：个股间联动路径 ======================
        if self.use_individual:
            self.graph_constructor = GraphConstructor(
                hidden_dim=cfg.hidden_dim,
                top_k_ratio=cfg.top_k_ratio,  # CSI500: 500*0.02=10个邻居
                use_self_mask=cfg.use_self_mask,
                temperature=getattr(cfg, "graph_temperature", 0.2)
            )
            self.gat = GATLayer(
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
                use_adj_prior=getattr(cfg, "use_adj_prior", True)
            )

        if self.use_global:
            # 多头全局注意力池化：学习不同股票对宏观市场的贡献
            self.global_pool = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.Tanh(),
                nn.Linear(cfg.hidden_dim, 4),  # 4头注意力
                nn.Softmax(dim=1)
            )

        # ✅ 自动设置alpha初始值（根据股票数量）；fixed_alpha 用于固定融合权重消融
        if self.fixed_alpha is not None:
            a0 = float(min(max(self.fixed_alpha, 1e-4), 1 - 1e-4))
            init_logit = float(np.log(a0 / (1 - a0)))
            self.alpha = nn.Parameter(torch.tensor(init_logit), requires_grad=False)
        else:
            if n_stocks >= 200:
                # CSI300(215只)：个股效应强，初始alpha=0.7
                initial_alpha = 0.7
            elif n_stocks >= 80:
                # NAS100(92只)：行业效应强，初始alpha=0.4
                initial_alpha = 0.4
            else:
                # DJIA(30只)：宏观效应强，初始alpha=0.2
                initial_alpha = 0.2
            self.alpha = nn.Parameter(torch.tensor(initial_alpha))

    def forward(self, h):
        # h: [B, N, H] 当前时刻的隐藏状态
        B, N, H = h.shape

        # 路径1输出：个股间联动特征
        if self.use_individual:
            adj = self.graph_constructor(h)   # [B, N, N] 动态稀疏邻接矩阵
            s_individual = self.gat(h, adj)   # [B, N, H] 聚合后的个体特征
        else:
            s_individual = torch.zeros_like(h)
            adj = torch.zeros(B, N, N, device=h.device)   # 保证不是 None

        # 路径2输出：全局宏观特征
        if self.use_global:
            attn_weights = self.global_pool(h)      # [B, N, 4]
            # 多头池化：4 个头分别池化后取均值
            s_global = torch.einsum('bnh, bnk -> bkh', h, attn_weights)  # [B, 4, H]
            s_global = s_global.mean(dim=1, keepdim=True)  # [B, 1, H]
            s_global = s_global.expand(-1, self.n_stocks, -1)      # [B, N, H]
        else:
            s_global = torch.zeros_like(h)

        # ✅ 单路径特判：只保留一条路径时直接返回，避免被可学习 alpha 缩放到 [0.1,0.9]（保证 E9/E10 消融公平）
        if self.use_individual and not self.use_global:
            return s_individual, adj
        if self.use_global and not self.use_individual:
            return s_global, adj

        # 双路径融合：可学习加权和
        alpha = torch.sigmoid(self.alpha)
        alpha = torch.clamp(alpha, 0.1, 0.9)
        s_fused = alpha * s_individual + (1 - alpha) * s_global
        return s_fused, adj


class LearnableLagPooling(nn.Module):
    """
    可学习多步延迟池化（baseline）：
    对过去 max_lag 步的空间特征做加权求和，权重可学习
    """
    def __init__(self, n_stocks, hidden_dim, max_lag=5):
        super().__init__()
        self.max_lag = max_lag
        # 每只股票独立的延迟权重
        self.lag_weights = nn.Parameter(torch.randn(n_stocks, max_lag))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, spatial_history):
        # spatial_history: [B, N, H, max_lag] 最近max_lag步的空间特征
        w = self.lag_weights
        w = w - torch.max(w, dim=1, keepdim=True)[0]
        w = self.softmax(w)   # [N, max_lag]
        out = torch.einsum('bnhl,nl->bnh', spatial_history, w)   # [B, N, H]
        return out


class AdaptiveLagPooling(nn.Module):
    """
    自适应滞后池化（v2：滞后偏好分布 × 单调存活门控）

    对第 n 只股票、滞后 l ∈ {1..L}（l=1 为最近一天）：
        w_n(l)  = softmax_l( (lag_weight[n] + Δ_n(l)) / temperature )       # 滞后偏好（静态先验 + 状态调制）
        h_n(j)  = σ( lag_hazard[n, j] )                                    # 第 j 阶“失效风险”
        S_n(l)  = Π_{j<=l} (1 - h_n(j))                                    # 存活/截止门控（单调递减）
        p_n(l)  = w_n(l) S_n(l) / Σ_l' w_n(l') S_n(l')                     # 实际使用的滞后分布
    s_lagged = Σ_l p_n(l) · s_stack[..., l]

    三层结构（每层语义不同、互不冗余）：
        ① 个股静态先验 lag_weight[N, L]  —— “这只股票通常几天后才反应”
        ② 状态调制 Δ = MLP(h_{t-1})     —— “今天这种状态下应该看几天前”
        ③ 存活门控 S(l)                 —— “传导链条能活到第几阶”，单调递减、可真正截止长滞后
    ② 的末层近零初始化（std=1e-3），训练起点≈纯静态先验，不会破坏已有收敛路径；
    `use_state_conditioning=False` 可退回纯静态版本（消融 14）。

    为什么这样改（相对 v1 的 softmax × 自由门控）：
    1. v1 的 p ∝ softmax(w/τ)·σ(g) 恒等于 softmax(w/τ + log σ(g))，门控与滞后权重完全冗余，
       “自动过滤无效历史”从未真正发生；这里门控由累积风险构造，天然单调，可把长滞后压到 1e-10 量级，
       语义 = “传导链条在第 j 阶断掉的概率”。
    2. 全程可导，不需要硬阈值，也不需要 STE 梯度放大（Adam 下 ×50 与 ×1 的参数轨迹完全相同，已实测）。
    3. 主体参数仍是 2×N×L（lag_weight + lag_hazard），另加一个 H→L 的小 MLP，容量与创新点强度不降反升。

    额外产出（可视化/论文指标，forward 时缓存，均为 batch 平均后的 [N]/[N,L]）：
        lag_probs [N, L]        实际滞后分布（含状态调制与门控）
        lag_probs_static [N, L] 静态先验分布（仅 lag_weight，用于对照“动态调制学了多少”）
        expected_lag [N]        期望滞后 Σ l·p(l)
        lag_entropy [N]         滞后分布熵（越小越“确定”）
        effective_max_lag [N]   累计概率质量首次 ≥ cutoff_threshold 的滞后阶数（有效传导窗口）
    """
    def __init__(self, n_stocks, hidden_dim, max_possible_lag=10,
                 temperature=0.5, cutoff_threshold=0.5,
                 grad_scale=1.0, hazard_init=-6.0, use_survival=True,
                 use_state_conditioning=True, state_hidden=32):
        super().__init__()
        self.n_stocks = n_stocks
        self.hidden_dim = hidden_dim
        self.max_possible_lag = max_possible_lag
        self.temperature = temperature
        self.cutoff_threshold = cutoff_threshold
        self.grad_scale = grad_scale
        self.use_survival = use_survival
        self.use_state_conditioning = use_state_conditioning

        # 滞后偏好：零初始化 → 初始为均匀分布
        self.lag_weight = nn.Parameter(torch.zeros(n_stocks, max_possible_lag))
        # 累积风险：σ(-6)≈0.0025 → 初始存活率≈[.9975, .995, .9925] 几乎全开，不破坏初始均匀性
        self.lag_hazard = nn.Parameter(
            torch.full((n_stocks, max_possible_lag), float(hazard_init))
        )
        # 状态调制头：末层近零初始化 → 训练起点≈纯静态先验，但 W1 立刻有梯度（避免冷启动）
        if self.use_state_conditioning:
            self.state_to_lag = nn.Sequential(
                nn.Linear(hidden_dim, state_hidden),
                nn.Tanh(),
                nn.Linear(state_hidden, max_possible_lag),
            )
            nn.init.normal_(self.state_to_lag[2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.state_to_lag[2].bias)
        else:
            self.state_to_lag = None

        self.register_buffer(
            "lag_index",
            # s_stack 最后一维索引 0 = 最早一天(t-L)、索引 L-1 = 最近一天(t-1)；
            # 故“滞后阶”应为 [L, L-1, ..., 1]，而非 [1, 2, ..., L]
            torch.arange(max_possible_lag, 0, -1, dtype=torch.float32),
            persistent=False
        )
        self.lag_probs = None
        self.lag_probs_static = None
        self.expected_lag = None
        self.lag_entropy = None
        self.effective_max_lag = None

    def _survival(self):
        """存活/截止门控 S(l)=Π_{j<=l}(1-h_j)：[N, L]，随滞后阶 l 单调递减；每个 hazard 都参与计算。
        注意 s_stack 索引 0 是最早一天(t-L)、索引 L-1 是最近一天(t-1)，
        故“越久远存活越低”等价于 S 沿 dim=-1 单调递增（最早一天最低、最近一天最高）；
        用 flip-cumprod-flip 实现，替代原先沿 dim=-1 递减（语义反向）的写法。"""
        if not self.use_survival:
            return None
        hazard = torch.sigmoid(self.lag_hazard)          # [N, L]
        survival = 1.0 - hazard
        return torch.flip(torch.cumprod(torch.flip(survival, dims=[-1]), dim=-1), dims=[-1])

    def _static_probs(self):
        """静态滞后先验 softmax(lag_weight/τ)：[N, L]，用于对照/可视化"""
        return F.softmax(self.lag_weight / self.temperature, dim=-1)

    def _distributions(self, state=None):
        """返回 (p, survival)：p 为 [B,N,L]（有状态调制）或 [N,L]（静态）；survival 为 [N,L] 或 None"""
        logits = self.lag_weight / self.temperature      # [N, L]
        if self.use_state_conditioning and state is not None:
            delta = self.state_to_lag(state) / self.temperature   # [B, N, L]
            w = F.softmax(logits.unsqueeze(0) + delta, dim=-1)    # [B, N, L]
        else:
            w = F.softmax(logits, dim=-1)                         # [N, L]

        survival = self._survival()
        if survival is None:
            return w, None
        p = w * survival
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
        return p, survival

    def forward(self, s_stack, state=None):
        """
        输入: s_stack [B, N, H, L] 最近L天的空间特征栈（最后一位为最近一天）
              state  [B, N, H]    当前隐藏状态 h_{t-1}（None 时退化为用最近一天的空间特征）
        输出: s_lagged [B, N, H]
        """
        B, N, H, L = s_stack.shape
        assert L == self.max_possible_lag, \
            f"输入特征栈长度{L}与最大可能滞后{self.max_possible_lag}不匹配"

        if self.use_state_conditioning and state is None:
            state = s_stack[..., -1]        # 退化：用最新已知的空间特征做调制

        p, _ = self._distributions(state)
        if p.dim() == 2:
            p = p.unsqueeze(0).expand(B, -1, -1)
        # 可选：STE 梯度放大（默认 1.0，即关闭；保留开关只为复现旧实验/做消融）
        if self.grad_scale != 1.0:
            p = p * self.grad_scale - p.detach() * (self.grad_scale - 1)

        s_lagged = torch.einsum('bnhl,bnl->bnh', s_stack, p)

        # 缓存解释性指标（detach，避免额外显存）
        with torch.no_grad():
            p_det = p.detach()                                     # [B, N, L]
            self.lag_probs = p_det.mean(dim=0)                     # [N, L]
            self.lag_probs_static = self._static_probs().detach()  # [N, L]
            self.expected_lag = (p_det * self.lag_index).sum(-1).mean(dim=0)
            self.lag_entropy = -(p_det.clamp_min(1e-12).log() * p_det).sum(-1).mean(dim=0)
            mass = torch.cumsum(p_det, dim=-1)
            eff = ((mass < self.cutoff_threshold).sum(dim=-1) + 1).clamp(1, L)
            self.effective_max_lag = eff.float().mean(dim=0).round().long()

        return s_lagged

    def get_lag_probs(self):
        """返回实际使用的滞后分布 [N, L]（供 plot.extract_lag_weights 使用）"""
        if self.lag_probs is not None:
            return self.lag_probs
        # 未前向时没有 batch，无法做状态调制 → 退回静态先验
        survival = self._survival()
        p = self._static_probs()
        if survival is not None:
            p = p * survival
            p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
        return p.detach()

class AsyncMambaBlock(nn.Module):
    """
    【最终修正版】异步传导Mamba核心模块
    严格实现：时间步推进 → 空间特征计算 → 异步注入 → 下一个时间步 的交替循环
    使用正确的SimpleMambaCell实现单步时间推进
    """
    def __init__(self, cfg, n_stocks):
        super().__init__()
        self.cfg = cfg
        self.n_stocks = n_stocks  # 股票池总数量，例如215只
        self.hidden_dim = cfg.hidden_dim  # 隐藏维度，例如128
        self.use_async = cfg.use_async  # 是否启用异步注入
        self.use_self_history = cfg.use_self_history  # 是否保留当日因子（残差 vs 替换）；自身历史始终由 Mamba 状态 h 承载
        self.learnable_lag = cfg.learnable_lag  # 是否启用可学习多步延迟
        # self.l_max = cfg.l_max  # 最大搜索滞后步数，例如5天
        self.l_max = cfg.max_possible_lag  # 最大可能滞后步数，例如10天
        self.max_possible_lag = cfg.max_possible_lag

        # ✅ 保存 use_gate 开关，避免 AttributeError
        self.use_gate = cfg.use_gate

        # 多层Mamba堆叠，例如num_mamba_layers=2就是2层Mamba
        self.mamba_layers = nn.ModuleList([
            SimpleMambaCell(
                d_model=cfg.hidden_dim,
                d_state=cfg.d_state
            ) for _ in range(cfg.num_mamba_layers)
        ])

        # 输入投影：将原始459维Alpha特征映射到128维隐藏空间
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        
        # 双路径空间聚合器：输入隐藏状态，输出融合后的空间特征
        self.spatial_agg = DualPathSpatialAggregator(cfg, n_stocks)
        
        # ✅ 异步注入门控根据 cfg.use_gate 开关创建
        if self.use_gate:
            self.causal_gate = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.Sigmoid()  # 输出范围[0,1]，0表示完全不注入，1表示完全注入
            )
        else:
            self.causal_gate = None

        # 全局异步强度系数：控制整体异步注入的强度
        # 初始值0.1，训练过程中自动学习最优值
        self.lambda_param = nn.Parameter(torch.tensor(cfg.lambda_self))
        
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        if self.learnable_lag:
            # self.lag_pool = LearnableLagPooling(n_stocks, cfg.hidden_dim, max_lag=cfg.l_max)
            self.lag_pool = AdaptiveLagPooling(
                n_stocks=n_stocks,
                hidden_dim=cfg.hidden_dim,
                max_possible_lag=cfg.max_possible_lag,
                temperature=0.5,
                cutoff_threshold=0.5,
                grad_scale=getattr(cfg, "lag_grad_scale", 1.0),
                use_survival=getattr(cfg, "use_lag_survival", True),
                use_state_conditioning=getattr(cfg, "lag_state_conditioning", True),
            )
        else:
            self.lag_pool = None

    def forward(self, x):
        B, T, N, F = map(int, x.shape)
        x_emb = self.input_proj(x)
        x_emb = self.dropout(x_emb)

        for layer in self.mamba_layers:
            layer.reset_cache(B * N, x.device)   # 防止不同 batch 之间的信息泄露
        h = torch.zeros(B, N, self.hidden_dim, device=x.device)
        s_history = []
        adj_seq = []
        h_seq = []
        for t in range(T):
            x_t = x_emb[:, t, :, :]

            if self.use_async and t > 0:
                if self.learnable_lag and self.n_stocks >= 50:
                    # ==============================================
                    # ✅ 修改这里：构建长度为max_possible_lag的特征栈
                    # ==============================================
                    recent_s = s_history[-self.max_possible_lag:]
                    if len(recent_s) < self.max_possible_lag:
                        pad_length = self.max_possible_lag - len(recent_s)
                        recent_s = [recent_s[0]] * pad_length + recent_s
                    s_stack = torch.stack(recent_s, dim=-1)  # [B, N, H, max_possible_lag]
                    
                    s_lagged = self.lag_pool(s_stack, h)
                else:
                    s_lagged = s_history[-1]

                # ✅ 异步注入逻辑适配 use_gate
                lambda_val = torch.sigmoid(self.lambda_param)
                if self.n_stocks >= 500:
                    lambda_max = 0.5
                elif self.n_stocks >= 100:
                    lambda_max = 0.2
                else:
                    lambda_max = 0.1
                lambda_val = lambda_val * lambda_max

                if self.use_gate and self.causal_gate is not None:
                    gate = self.causal_gate(s_lagged)
                    inject = lambda_val * gate * s_lagged
                else:
                    inject = lambda_val * s_lagged

                if self.use_self_history:
                    x_t = x_t + inject   # 残差：当日因子 + 滞后空间注入
                else:
                    x_t = inject         # 替换：丢弃当日因子，只用滞后空间注入（自身历史仍在 h 中）

            for layer in self.mamba_layers:
                h = layer(x_t, h)

            s_t, adj_t = self.spatial_agg(h)
            s_history.append(s_t)
            adj_seq.append(adj_t)
            h_seq.append(h)

        h_last = h_seq[-1]
        out = self.output_proj(h_last)
        out = self.dropout(out)
        adj = adj_seq[-1] if adj_seq else torch.zeros(B, N, N, device=h.device)
        
        return out, adj

class AsyncMambaModel(nn.Module):
    def __init__(self, cfg, n_stocks):
        super().__init__()
        self.encoder = AsyncMambaBlock(cfg, n_stocks)
        self.pred_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_dim, 1)
        )

    def forward(self, x):
        h, adj = self.encoder(x)
        pred = self.pred_head(h).squeeze(-1)
        return pred, adj

# ========================== 主程序 ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='ablation', choices=['train', 'ablation'])
    parser.add_argument('--device', type=str, default=None,
                        help="计算设备，可选 auto/cpu/cuda/mps；默认沿用 Config.device（本机 auto→mps）")
    parser.add_argument('--synthetic', action='store_true',
                        help="无真实数据时强制使用虚拟合成数据跑通全流程（仅流程自检，指标无金融含义）")
    args = parser.parse_args()
    # 创建带数据集标识的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{Config.market_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    # ========================== 1. 启动全局计时器 ==========================
    timer = start_timer(experiment_name=f"AsyncMamba_{Config.data_name}", mode=args.mode)
    cfg = Config()
    # 命令行 --device 优先于 Config.device：方便在 mac 上 `--device mps` 跑 GPU、
    # 或 `--device cpu` 做数值对照（注意 cfg.__repr__ 只打印类属性，故单独打印设备行）。
    if args.device:
        cfg.device = resolve_device(args.device)
    if args.synthetic:
        cfg.synthetic = True
    print("\n" + "="*80, flush=True)
    print("【本次实验完整配置】", flush=True)
    print("="*80, flush=True)
    print(cfg, flush=True)
    print(f"\n【计算设备】{cfg.device}", flush=True)
    if str(cfg.device).startswith("mps"):
        print("            (Apple Silicon GPU；如某算子未实现会自动回落到 CPU，"
              "设 PYTORCH_ENABLE_MPS_FALLBACK=0 可改为强制报错)", flush=True)
    print("="*80 + "\n", flush=True)
    set_seed(cfg.seed)
    # ========================== 2. 数据加载阶段计时 ==========================
    timer.start_stage("数据加载与预处理")
    train_loader, val_loader, test_loader, label_stats, n_stocks = load_data(cfg)
    print(f"Number of stocks: {n_stocks}", flush=True)
    timer.end_stage("数据加载与预处理")

    if args.mode == 'ablation':
        # ========================== 消融实验阶段计时 ==========================
        timer.start_stage("消融实验")
        results = run_ablation(cfg, train_loader, val_loader, test_loader, n_stocks, label_stats)
        print("\n============ Ablation Results (sorted by ICIR) ============", flush=True)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['RankICIR'], reverse=True)
        for name, metrics in sorted_results:
            print(f"{name:35s} | IC={metrics['RankIC_Mean']:.4f} | ICIR={metrics['RankICIR']:.4f} | LS_Sharpe={metrics['LS_Sharpe']:.4f}", flush=True)
        timer.end_stage("消融实验")
        # 结束全局计时
        timer.end()
        return
    device = torch.device(cfg.device)
    model = AsyncMambaModel(cfg, n_stocks).to(device)
    # print_model_architecture(model)
    # ========== 第二阶段：放开所有参数，正常训练 ==========
    print("\n" + "="*60)
    print("🚀 第二阶段：全参数训练")
    print("="*60)
    # 放开所有参数
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    # ========================== 新增：初始化损失权重调度器 ==========================
    loss_scheduler = None
    if cfg.use_dynamic_weight and cfg.loss_type != "mse":
        loss_scheduler = DynamicLossWeightScheduler(
            total_epochs=cfg.epochs,
            mse_max=cfg.mse_weight_max,
            mse_min=cfg.mse_weight_min,
            warmup_epochs=cfg.warmup_epochs)
        print(f"\n✅ 已开启动态损失权重调度:")
        print(f"   预热轮数: {cfg.warmup_epochs} (纯MSE)")
        print(f"   MSE权重范围: {cfg.mse_weight_min:.2f} ~ {cfg.mse_weight_max:.2f}")
    best_val_ir = -float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    train_losses = []
    val_metrics_history = []
    no_improve_count = 0

    # 在主训练循环中添加
    running_rank_std = None
    timer.start_stage("模型训练")   # ✅ 修复：与 line timer.end_stage("模型训练") 配对，避免“没有正在进行的阶段计时”告警
    for epoch in range(1, cfg.epochs+1):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_loss, train_metrics, avg_mse_loss, avg_rank_loss, running_rank_std = train_epoch(
            cfg, model, train_loader, optimizer, device, epoch, n_stocks, 
            loss_scheduler=loss_scheduler, running_rank_std=running_rank_std
        )
        # 在每个epoch结束后加
        alpha = torch.sigmoid(model.encoder.spatial_agg.alpha).item()
        lambda_val = torch.sigmoid(model.encoder.lambda_param).item()
        print(f"双路径融合权重alpha: {alpha:.4f}")
        print(f"异步强度lambda: {lambda_val:.4f}")
        # ✅ 安全打印滞后参数，防止 lag_pool 为 None 报错
        if model.encoder.lag_pool is not None:
            lag_pool = model.encoder.lag_pool
            print(f"滞后权重均值: {lag_pool.lag_weight.mean().item():.4f}")
            # v2：存活门控（累积风险）与滞后分布的可解释性指标
            if getattr(lag_pool, "lag_hazard", None) is not None:
                survival = torch.cumprod(1.0 - torch.sigmoid(lag_pool.lag_hazard), dim=-1)
                print(f"存活门控均值: {survival.mean().item():.4f} "
                      f"(逐阶: {[round(v, 4) for v in survival.mean(0).tolist()]})")
            if getattr(lag_pool, "expected_lag", None) is not None:
                print(f"期望滞后均值: {lag_pool.expected_lag.mean().item():.4f} 天")
                print(f"滞后分布熵均值: {lag_pool.lag_entropy.mean().item():.4f} nats")
        val_metrics, long_only_rets, long_short_rets = evaluate(cfg, model, val_loader, device, mode="val")
        scheduler.step()
        
        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)
        # 修复：使用RankICIR
        current_ir = val_metrics["RankICIR"]
        current_ic = val_metrics["RankIC_Mean"]
        
        # ✅ 修正后的代码
        if not np.isnan(current_ir) and current_ir > best_val_ir:
            best_val_ir = current_ir
            best_model_state = copy.deepcopy(model.state_dict())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 带数据集名称的唯一文件名
            save_path = f"best_model_{cfg.data_name}_icir_{best_val_ir:.4f}_{timestamp}.pth"
            # 带数据集名称的通用文件名，不同数据集不会互相覆盖
            best_model_path = f"best_model_{cfg.data_name}.pth"
            
            torch.save(best_model_state, save_path)
            torch.save(best_model_state, best_model_path)
            no_improve_count = 0
            print(f"[{current_time}]  -> ✅ 保存最佳模型: {save_path} (Val ICIR: {best_val_ir:.4f}, IC: {current_ic:.4f})", flush=True)
        else:
            no_improve_count += 1
        if no_improve_count >= cfg.early_stop_patience:
            print(f"[{current_time}] Early stopping triggered after {cfg.early_stop_patience} epochs of no improvement.")
            break
            # ========================== 修改：日志输出加入损失分量 ==========================
        if epoch % cfg.log_interval == 0:
            print(f"[{current_time}] Epoch {epoch:3d} | "
                  f"Total Loss: {train_loss:.4f} | MSE: {avg_mse_loss:.4f} | Rank: {avg_rank_loss:.4f} | "
                  f"Val RankIC: {current_ic:.4f} | Val RankICIR: {current_ir:.4f} | Val MSE: {val_metrics['MSE']:.6f}",flush=True)
    timer.end_stage("模型训练")

    # ✅ 修正后的代码
    print("\n========== 开始测试 ==========")
    # 1. 股票代码：优先取数据文件旁的 stocks.npy（合成兜底数据写在那里），缺失/数量不符就用占位代码兜底
    _sf = os.path.join(os.path.dirname(os.path.abspath(cfg.data_path)), "stocks.npy")
    if not os.path.exists(_sf):
        _sf = stocks_path(cfg.data_name, check=False)
    try:
        stock_names = np.load(_sf, allow_pickle=True)
        if len(stock_names) != n_stocks:
            raise ValueError(f"len={len(stock_names)} != N={n_stocks}")
    except Exception as _err:
        stock_names = np.array([f"S{i:05d}" for i in range(n_stocks)], dtype=object)
        print(f"⚠️  stocks.npy 不可用({_err})，使用虚拟股票代码（共 {n_stocks} 个）。")
    
    best_model_path = f"best_model_{cfg.data_name}.pth"
    # map_location：MPS/CUDA 上存的权重也能在 CPU 上加载（反之亦然）
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_metrics, long_only_rets, long_short_rets = evaluate(
        cfg, model, test_loader, device, mode="test", 
        label_stats=label_stats, stock_names=stock_names
    )
    print("\n========== 最终测试结果 (完整展示) ==========")
    print_test_results(test_metrics, cfg.topk_assets)
    plot_results(train_losses, val_metrics_history,save_path=f"{output_dir}/training_curves.png")
    plot_full_results(train_losses, val_metrics_history, long_only_rets, long_short_rets, save_prefix=f"{output_dir}/result")

    # ========== 测试完成后添加这段代码 ==========
    print("\n========== 生成传导延迟可视化 ==========")

    # ✅ 防御：只有启用自适应滞后池化(Config.learnable_lag=True)时，模型里才存在 lag_pool，
    # 才能提取滞后权重、生成 lag_weights.csv / lag_distribution.png 等产物。
    # learnable_lag=False 时直接跳过，避免 AttributeError 让整个流程失败。
    if getattr(model.encoder, "lag_pool", None) is None:
        print("⚠️  当前 Config.learnable_lag=False，模型未启用自适应滞后池化，")
        print("    跳过「传导延迟可视化」环节（不会生成 lag_weights.csv/lag_distribution.png）。")
        print("    如需该产物，请在 Config 中设置 learnable_lag = True 后重新训练。")
    else:
        # 2. 提取延迟权重
        lag_df = extract_lag_weights(model, stock_names)
        if lag_df is None:
            print("⚠️  滞后权重提取为空，跳过可视化。")
        else:
            # 3. 打印延迟统计信息
            print_lag_statistics(lag_df)
            # 4. 保存延迟权重到CSV（方便后续分析）
            lag_df.to_csv(f"{output_dir}/lag_weights.csv", encoding="utf-8-sig")
            print(f"\n✅ 延迟权重已保存到: {output_dir}/lag_weights.csv")
            # 5. 生成可视化图片
            plot_lag_distribution(lag_df, save_path=f"{output_dir}/lag_distribution.png")
            plot_lag_weight_heatmap(lag_df, top_n=20, save_path=f"{output_dir}/lag_weight_heatmap.png")
            # ✅ 可选：新增一张有效最大滞后的分布图
            # plot_lag_distribution(lag_df, column="effective_max_lag", title="max_lag_distribution",save_path=f"{output_dir}/effective_lag_distribution.png", show_kde=False)  # 有效最大滞后是离散截止值，不需要kde曲线
    # ========================== 6. 结束全局计时 ==========================
    timer.end()

if __name__ == '__main__':
    main()