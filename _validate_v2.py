"""临时脚本：验证 main.py v2 架构改动（删除 _old + 图先验 + 存活门控 + 残差注入）。"""
import numpy as np
import torch
import torch.nn.functional as F

import main as M
from plot import extract_lag_weights, print_lag_statistics

torch.manual_seed(0)
cfg = M.Config()
cfg.hidden_dim = 32
cfg.d_state = 8
cfg.num_mamba_layers = 1
cfg.dropout = 0.0
cfg.use_async = True
cfg.use_global = True
cfg.use_individual = True
cfg.use_self_history = True
cfg.use_gate = True
cfg.learnable_lag = True
cfg.use_lag_survival = True
cfg.max_possible_lag = 3
cfg.top_k_ratio = 0.03
cfg.lambda_self = 0.002

N = 120
model = M.AsyncMambaModel(cfg, N)
x = torch.randn(2, 6, N, cfg.input_dim)
y = torch.randn(2, N)

pred, adj = model(x)
loss = F.mse_loss(pred, y)
loss.backward()

print("=" * 78)
print("[A] 组件存在性：旧类应已删除")
for name in ["GraphConstructor_old", "GATLayer_old", "AdaptiveLagPooling_old"]:
    print(f"   hasattr(main, {name}) =", hasattr(M, name))

print("=" * 78)
print("[B] 图是否可学（关键修复）")
sd = dict(model.named_parameters())
gn = sum(sd[k].grad.norm().item() ** 2 for k in sd
         if "graph_constructor" in k) ** 0.5
print("   graph_constructor grad_norm =", gn, "（修复前恒为 0）")
print("   邻接行和 =", adj.sum(-1).mean().item(),
      "| 非零边/行 =", (adj > 0).sum(-1).float().mean().item(),
      "| 全零行 =", (adj.sum(-1) == 0).float().mean().item())
print("   邻接权值 min/max =", adj[adj > 0].min().item(), adj.max().item())

if model.encoder.lag_pool.state_to_lag is not None:
    head = model.encoder.lag_pool.state_to_lag
    print("   state_to_lag 末层 grad_norm =", head[2].weight.grad.norm().item(),
          "| 首层 grad_norm =", head[0].weight.grad.norm().item(),
          "（末层近零初始化，两层都应有梯度）")

print("=" * 78)
print("[C] 滞后分布 + 存活门控")
lp = model.encoder.lag_pool
p = lp.get_lag_probs()
print("   p 行和 min/max =", p.sum(-1).min().item(), p.sum(-1).max().item())
print("   初始 lag_probs[0] =", p[0].tolist())
print("   expected_lag  min/mean/max =", lp.expected_lag.min().item(),
      lp.expected_lag.mean().item(), lp.expected_lag.max().item())
print("   lag_entropy mean =", lp.lag_entropy.mean().item())
print("   effective_max_lag 分布 =",
      torch.bincount(lp.effective_max_lag.long()).tolist())
print("   lag_weight.grad abs mean =", lp.lag_weight.grad.abs().mean().item(),
      "| lag_hazard.grad abs mean =", lp.lag_hazard.grad.abs().mean().item())

# 手工把某只股票的 hazard 拉高，验证门控能真正“截止”
with torch.no_grad():
    lp.lag_hazard[0] = torch.tensor([-4.0, 8.0, 8.0])
    s_stack = torch.randn(1, N, cfg.hidden_dim, 3)
    out = lp(s_stack)
    p2 = lp.get_lag_probs()
print("   拉高 hazard 后 p[0] =", p2[0].tolist(),
      "-> effective_max_lag[0] =", lp.effective_max_lag[0].item())

print("=" * 78)
print("[D] 残差注入（修复前 t>=1 的当日特征被完全丢弃）")
model.eval()
with torch.no_grad():
    base, _ = model(x)
    x_zero = x.clone(); x_zero[:, 1:] = 0
    p_zero, _ = model(x_zero)
print("   置零 t>=1 输入 -> max|Δpred| =", (p_zero - base).abs().max().item(),
      "（应显著 > 0）")

print("=" * 78)
print("[E] 滞后权重导出 & 消融变体构建")
stock_names = [f"S{i:04d}" for i in range(N)]
df = extract_lag_weights(model, stock_names)
print_lag_statistics(df)
print(df.head(3).to_string())

from ablation import get_ablation_configs
names = [n for n, _ in get_ablation_configs(cfg)]
print("\n   消融变体数 =", len(names))
print("  ", names)

# 每个变体都能前向
for name, c in get_ablation_configs(cfg):
    c.epochs = 1
    m = M.AsyncMambaModel(c, N)
    o, a = m(x[:1, :3])
    assert torch.isfinite(o).all(), name
print("   ✅ 全部消融变体前向通过（含 12_No_GraphPrior / 13_No_Lag_Survival）")

print("=" * 78)
print("[F] 可识别性：目标只来自 lag=3 的特征，门控能否打开到 3 阶")
lp2 = M.AdaptiveLagPooling(n_stocks=8, hidden_dim=16, max_possible_lag=3,
                           temperature=0.5, cutoff_threshold=0.5)
opt = torch.optim.Adam(lp2.parameters(), lr=0.05)
s = torch.randn(8, 8, 16, 3)
target = s[..., 2].detach()          # 只有 lag=3 含信息
for _ in range(300):
    opt.zero_grad()
    out = lp2(s)
    loss = F.mse_loss(out, target)
    loss.backward()
    opt.step()
p3 = lp2.get_lag_probs()
print("   p[0] =", [round(v, 4) for v in p3[0].tolist()],
      "| loss =", round(loss.item(), 6))
print("   expected_lag mean =", round(lp2.expected_lag.mean().item(), 3),
      "| effective_max_lag mean =", round(lp2.effective_max_lag.float().mean().item(), 3))

print("=" * 78)
print("[G] 真实优化下 GraphConstructor 是否在更新（10 步 Adam）")
model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
wq_before = model.encoder.spatial_agg.graph_constructor.Wq.weight.detach().clone()
haz_before = model.encoder.lag_pool.lag_hazard.detach().clone()
for _ in range(10):
    opt.zero_grad()
    pred, _ = model(x)
    F.mse_loss(pred, y).backward()
    opt.step()
dwq = (model.encoder.spatial_agg.graph_constructor.Wq.weight - wq_before).norm().item()
dhaz = (model.encoder.lag_pool.lag_hazard - haz_before).norm().item()
print("   ||ΔWq|| =", dwq, "| ||Δlag_hazard|| =", dhaz, "（两者均应 > 0）")

print("=" * 78)
print("[H] 状态依赖调制的有效性（同一股票、不同状态 → 不同滞后分布）")
lp3 = M.AdaptiveLagPooling(n_stocks=4, hidden_dim=16, max_possible_lag=3)
# 模拟训练后的调制头：让 lag-3 的 logit 依赖状态
with torch.no_grad():
    lp3.state_to_lag[2].weight[2] = torch.randn(
        lp3.state_to_lag[2].weight.shape[1]) * 0.5
    lp3.state_to_lag[2].bias.zero_()
s1 = torch.randn(2, 4, 16)
s2 = torch.randn(2, 4, 16)
p1, _ = lp3._distributions(s1)
p2, _ = lp3._distributions(s2)
print("   max|p(state1) - p(state2)| =", (p1 - p2).abs().max().item(), "（应 > 0）")
print("   p 形状 =", tuple(p1.shape), "| 行和 =", p1.sum(-1).mean().item())
out = lp3(torch.randn(2, 4, 16, 3), s1)
out.sum().backward()
print("   state_to_lag 首层 grad_norm =",
      lp3.state_to_lag[0].weight.grad.norm().item(),
      "| 末层 grad_norm =", lp3.state_to_lag[2].weight.grad.norm().item())
print("   lag_probs_static 行和 =",
      lp3.get_lag_probs().sum(-1).mean().item(),
      "| lag_probs_static 形状 =", tuple(lp3.lag_probs_static.shape))
