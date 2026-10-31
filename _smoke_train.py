"""冒烟测试：用 metric.train_epoch 跑两轮真实训练循环（synthetic 数据）。"""
import torch
from metric import train_epoch
import main as M

torch.manual_seed(1)
cfg = M.Config()
cfg.hidden_dim, cfg.d_state = 32, 8
cfg.max_possible_lag = 3
cfg.loss_type = "mse"
N = 120
model = M.AsyncMambaModel(cfg, N)
opt = torch.optim.Adam(model.parameters(), lr=3e-5)
batches = [(torch.randn(2, 6, N, cfg.input_dim), torch.randn(2, N)) for _ in range(3)]

for ep in (1, 2):
    total, m, mse, rk, std = train_epoch(cfg, model, batches, opt,
                                         torch.device("cpu"), ep, N)
    print(f"epoch {ep}: total={total:.4f} mse={mse:.4f} rank={rk:.4f}")

pred, adj = model(batches[0][0])
print("pred", tuple(pred.shape), "adj", tuple(adj.shape))
print("lag_probs[0]", [round(v, 4) for v in model.encoder.lag_pool.get_lag_probs()[0].tolist()])
print("expected_lag mean", round(model.encoder.lag_pool.expected_lag.mean().item(), 4))
print("effective_max_lag mean", round(model.encoder.lag_pool.effective_max_lag.float().mean().item(), 3))
print("OK")
