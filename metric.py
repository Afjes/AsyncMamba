import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from plot import (
    plot_conduction_heatmap, plot_top_conduction_heatmap, 
    plot_conduction_network
)

def print_test_results(test_metrics, topk_assets=30):
    """
    统一打印完整的测试结果
    Args:
        test_metrics: evaluate函数返回的指标字典
        topk_assets: 回测使用的Top K股票数量
    """
    print("\n========== 最终测试结果 (完整展示) ==========")
    print(f"【1. 预测能力】")
    print(f"  IC Mean:  {test_metrics['RankIC_Mean']:.4f}")
    print(f"  IC Std:   {test_metrics['RankIC_Std']:.4f}")
    print(f"  ICIR:     {test_metrics['RankICIR']:.4f}")

    print(f"\n【2. 回测结果 - Long-Only (纯多头, Top {topk_assets} 只)】")
    print(f"  ARR:      {test_metrics['LO_ARR']:.4f} ({test_metrics['LO_ARR']*100:.2f}%)")
    print(f"  AVol:     {test_metrics['LO_AVol']:.4f}")
    print(f"  MDD:      {test_metrics['LO_MDD']:.4f} ({test_metrics['LO_MDD']*100:.2f}%)")
    print(f"  Sharpe:   {test_metrics['LO_Sharpe']:.4f}")

    print(f"\n【3. 回测结果 - Long-Short (多空对冲, Top {topk_assets} 只)】")
    print(f"  ARR:      {test_metrics['LS_ARR']:.4f} ({test_metrics['LS_ARR']*100:.2f}%)")
    print(f"  AVol:     {test_metrics['LS_AVol']:.4f}")
    print(f"  MDD:      {test_metrics['LS_MDD']:.4f} ({test_metrics['LS_MDD']*100:.2f}%)")
    print(f"  Sharpe:   {test_metrics['LS_Sharpe']:.4f}")


def listmle_loss(y_pred, y_true, eps=1e-8):
    """
    ListMLE排序损失（可选，单数据集调优用）
    优化排列的极大似然，部分数据集效果更好但稳定性稍差
    """
    # 按真实收益率降序排序
    sorted_indices = torch.argsort(y_true, dim=1, descending=True)
    y_pred_sorted = torch.gather(y_pred, 1, sorted_indices)
    
    # 数值稳定
    y_pred_max = torch.max(y_pred_sorted, dim=1, keepdim=True)[0]
    y_pred_sorted = y_pred_sorted - y_pred_max
    
    exp_pred = torch.exp(y_pred_sorted)
    cum_exp = torch.cumsum(exp_pred.flip(1), dim=1).flip(1)
    
    log_cum_exp = torch.log(cum_exp + eps)
    loss = torch.mean(torch.sum(log_cum_exp - y_pred_sorted, dim=1))
    return loss

class DynamicLossWeightScheduler:
    """
    ✅ 修正版：损失量级匹配的余弦退火调度器
    解决排序损失初始值远大于MSE的问题;
    余弦退火动态损失权重调度器
    策略：
    - 训练前期（warmup阶段）：MSE权重高，学习基础收益率模式
    - 训练后期：排序损失权重高，优化选股排名
    公式：
        mse_weight = mse_min + (mse_max - mse_min) * 0.5 * (1 + cos(epoch / total_epochs * π))
        rank_weight = 1 - mse_weight
    """
    def __init__(self, total_epochs, mse_max=0.9, mse_min=0.7, warmup_epochs=5):
        self.total_epochs = total_epochs
        self.mse_max = mse_max
        self.mse_min = mse_min  # ✅ 从0.5提高到0.7，限制排序损失最大权重为0.3
        self.warmup_epochs = warmup_epochs

    def get_weights(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段：纯MSE
            return 1.0, 0.0
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            mse_weight = self.mse_min + (self.mse_max - self.mse_min) * 0.5 * (1 + np.cos(progress * np.pi))
            rank_weight = 1 - mse_weight
            return mse_weight, rank_weight

def calculate_quant_finance_metrics(pred, label, top_pct=0.1, top_k=None,
                                     annual_days=252, risk_free_rate=0.02, 
                                     mode="test", market_type="csi500"):
    """
    量化金融指标计算（纯净版）
    核心原则：指标计算函数只做计算，不负责数据清洗
    修改：
    1. 移除所有极端值/空值过滤（应在预处理完成）
    2. 保留基础防御性检查（防崩溃）
    3. 添加market_type参数适配不同市场
    4. 修复交易成本计算（核心！）
    """
    metrics = {}
    # 交易成本配置（按市场类型调整）
    cost_config = {
        "csi500": 0.001,    # 0.1% (A股高换手)
        "csi300": 0.0008,   # 0.08% 
        "sp500": 0.0005,    # 0.05% (美股低摩擦)
        "nasdaq100": 0.0006 # 0.06%
    }
    cost_rate = cost_config.get(market_type.lower(), 0.001)  # 默认CSI500

    # ========== 基础类型转换（必要防御）==========
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.detach().cpu().numpy()
    pred = np.asarray(pred)
    label = np.asarray(label)
    
    # 基础防御：空输入检查（防崩溃，不修改数据分布）
    if pred.size == 0 or label.size == 0:
        empty_metrics = {"MSE": 0.0, "MAE": 0.0, "RankIC_Mean": 0.0, 
                        "RankIC_Std": 0.0, "RankICIR": 0.0}
        return empty_metrics, [], [] if mode != "train" else empty_metrics

    # ========== 核心指标计算（不做数据修改）==========
    # 1. 回归指标（直接计算，NaN会自然传播）
    metrics["MSE"] = np.nanmean((pred - label) ** 2)
    metrics["MAE"] = np.nanmean(np.abs(pred - label))
    
    if mode == "train":
        return metrics, [], []

    # 2. RankIC计算（基础防御，不清洗数据）
    ic_list = []
    T = pred.shape[0]
    
    for i in range(T):
        p = pred[i].ravel()
        l = label[i].ravel()
        
        # 最小样本检查（防崩溃，不修改数据）
        if len(p) < 5 or len(l) < 5:
            continue
            
        # 标准差检查（防除零，不修改数据）
        if np.nanstd(l) < 1e-8 or np.nanstd(p) < 1e-8:
            continue
            
        try:
            ic, _ = spearmanr(p, l, nan_policy='omit')  # 仅跳过NaN
            if np.isfinite(ic):
                ic_list.append(ic)
        except:
            continue
    
    # 重命名IC→RankIC（避免歧义）
    metrics["RankIC_Mean"] = np.nanmean(ic_list) if ic_list else 0.0
    metrics["RankIC_Std"] = np.nanstd(ic_list) if ic_list else 0.0
    metrics["RankICIR"] = metrics["RankIC_Mean"] / (metrics["RankIC_Std"] + 1e-8) if ic_list else 0.0
    
    if mode == "val":
        return metrics, [], []

    # 3. 回测指标（核心：修复交易成本）
    N = pred.shape[1]
    if top_k is None:
        top_k = max(1, int(N * top_pct))
    top_k = min(top_k, N // 2)  # 保证多空对称
    
    long_only_rets = []
    long_short_rets = []
    prev_long_idx = None  # 跟踪持仓
    
    for t in range(T):
        p_t = pred[t].ravel()
        l_t = label[t].ravel()
        
        # 基础索引处理（不清洗数据）
        sorted_idx = np.argsort(p_t)
        long_idx = sorted_idx[-top_k:] if len(sorted_idx) >= top_k else sorted_idx
        short_idx = sorted_idx[:top_k] if len(sorted_idx) >= top_k else sorted_idx
        
        # 换手率计算（真实跟踪）
        turnover = 1.0  # 默认
        if prev_long_idx is not None and len(prev_long_idx) > 0:
            common = np.intersect1d(prev_long_idx, long_idx)
            turnover = 1 - len(common) / len(prev_long_idx)
        
        # 关键修复：真实成本计算
        cost = turnover * cost_rate  # 动态成本
        
        # 收益计算（不做数据修改）
        lo_ret = np.nanmean(l_t[long_idx]) - cost
        long_only_rets.append(lo_ret)
        
        ls_ret_long = np.nanmean(l_t[long_idx])
        ls_ret_short = np.nanmean(l_t[short_idx])
        ls_ret = (ls_ret_long - ls_ret_short) - 2 * cost  # 双边成本
        long_short_rets.append(ls_ret)
        
        prev_long_idx = long_idx.copy()

    # 4. 回测统计（移除极端值过滤）
    def _calc_backtest(daily_returns, annual_days=252, risk_free_rate=0.02):
        rets = np.array(daily_returns)
        # 仅做基础防御
        rets = rets[np.isfinite(rets)]
        T_valid = len(rets)
        
        if T_valid < 5 or np.nanstd(rets) < 1e-8:
            return {"ARR": 0.0, "AVol": 0.0, "MDD": 0.0, "Sharpe": 0.0}
        
        # 原始净值计算（无clip！）
        nav = np.nancumprod(1 + rets)
        total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0
        
        annual_factor = annual_days / max(T_valid, 1)
        arr = (1 + total_ret) ** annual_factor - 1 if total_ret > -1 else -1.0
        
        daily_std = np.nanstd(rets)
        avol = daily_std * np.sqrt(annual_days) if daily_std > 0 else 0.0
        
        # 基础回撤计算
        if len(nav) > 0:
            max_nav = np.maximum.accumulate(np.nan_to_num(nav, nan=1.0))
            drawdown = (max_nav - nav) / np.maximum(max_nav, 1e-8)
            mdd = np.nanmax(drawdown)
        else:
            mdd = 0.0
        
        sharpe = (arr - risk_free_rate) / avol if avol > 1e-8 else 0.0
        
        return {"ARR": arr, "AVol": avol, "MDD": mdd, "Sharpe": sharpe}
    
    lo_metrics = _calc_backtest(long_only_rets, annual_days, risk_free_rate)
    ls_metrics = _calc_backtest(long_short_rets, annual_days, risk_free_rate)
    
    # 修复：命名统一，避免LO/LS混淆
    metrics["LO_ARR"] = lo_metrics["ARR"]
    metrics["LO_AVol"] = lo_metrics["AVol"]
    metrics["LO_MDD"] = lo_metrics["MDD"]
    metrics["LO_Sharpe"] = lo_metrics["Sharpe"]
    
    metrics["LS_ARR"] = ls_metrics["ARR"]
    metrics["LS_AVol"] = ls_metrics["AVol"]
    metrics["LS_MDD"] = ls_metrics["MDD"]
    metrics["LS_Sharpe"] = ls_metrics["Sharpe"]
    
    return metrics, long_only_rets, long_short_rets

def train_epoch(cfg, model, loader, optimizer, device, epoch, n_stocks, loss_scheduler=None, running_rank_std=None):
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_rank_loss = 0.0
    rank_losses = []
    all_preds, all_targets = [], []

    # 获取当前epoch的损失权重
    if loss_scheduler is not None and cfg.use_dynamic_weight:
        mse_weight, rank_weight = loss_scheduler.get_weights(epoch)
    else:
        mse_weight = cfg.fixed_mse_weight
        rank_weight = cfg.fixed_rank_weight

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred, adj = model(x)

        # 打印信息
        if batch_idx == 0 and epoch == 1:
            alpha = torch.sigmoid(model.encoder.spatial_agg.alpha).item()
            print(f"双路径融合权重alpha: {alpha:.4f}")
            lambda_val = torch.sigmoid(model.encoder.lambda_param).item()
            print(f"异步强度lambda: {lambda_val:.4f}")
            non_zero_count = torch.count_nonzero(adj > 1e-6).item()
            total_count = adj.numel()
            non_zero_ratio = non_zero_count / total_count
            print(f"邻接矩阵非零元素比例: {non_zero_ratio:.4f} (预期≈{cfg.top_k_ratio:.4f})")
            if cfg.loss_type != "mse":
                print(f"   损失权重: MSE={mse_weight:.4f}, Rank={rank_weight:.4f}")
        
        has_nan = torch.isnan(pred).any().item()
        has_inf = torch.isinf(pred).any().item()
        if has_nan or has_inf:
            print(f"⚠️  Warning: Epoch {epoch}, Batch {batch_idx}: Model output has NaN={has_nan}, Inf={has_inf}. Skipping.")
            optimizer.zero_grad()
            continue

        # ========================== 损失计算 ==========================
        mse_loss = F.mse_loss(pred, y)
        
        if cfg.loss_type == "mse":
            loss = mse_loss
            rank_loss = torch.tensor(0.0, device=device)

        elif cfg.loss_type == "mse+listnet":
            # ✅ 【正确】传入 running_std
            rank_loss = listnet_loss(pred, y, running_std=running_rank_std)
            loss = mse_weight * mse_loss + rank_weight * rank_loss

        elif cfg.loss_type == "mse+listmle":
            rank_loss = listmle_loss(pred, y)
            loss = mse_weight * mse_loss + rank_weight * rank_loss

        else:
            raise ValueError(f"不支持的损失类型: {cfg.loss_type}")
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_max_norm)
        optimizer.step()

        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_rank_loss += rank_loss.item()
        rank_losses.append(rank_loss.item())  # ✅ 记录

        all_preds.append(pred.detach().cpu())
        all_targets.append(y.detach().cpu())
        
    # 平均损失
    avg_loss = total_loss / len(loader)
    avg_mse_loss = total_mse_loss / len(loader)
    avg_rank_loss = total_rank_loss / len(loader)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_quant_finance_metrics(all_preds, all_targets, mode="train")[0]

    # ✅ 更新滑动标准差
    new_running_std = None
    if rank_losses:
        current_std = np.mean(rank_losses)  # 用 mean 最稳定
        if running_rank_std is None:
            new_running_std = current_std
        else:
            new_running_std = 0.9 * running_rank_std + 0.1 * current_std

    return avg_loss, metrics, avg_mse_loss, avg_rank_loss, new_running_std

def evaluate(cfg, model, loader, device, mode="val", label_stats=None, stock_names=None):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # 修复：确保label_stats包含均值和标准差
    if label_stats is not None and mode == "test":
        if isinstance(label_stats, tuple) and len(label_stats) == 2:
            y_mean, y_std = label_stats
            all_preds = all_preds * y_std + y_mean
            all_targets = all_targets * y_std + y_mean
            print("反标准化后预测值示例:", all_preds[0, :5].detach().cpu().numpy())
            print("反标准化后真实值示例:", all_targets[0, :5].detach().cpu().numpy())
            print("预测值均值:", all_preds.mean().item(), "标准差:", all_preds.std().item())
        else:
            print("Warning: label_stats format unexpected, skipping denormalization")

    # 修复：传入market_type参数
    metrics, long_only_rets, long_short_rets = calculate_quant_finance_metrics(
        all_preds, all_targets,top_k=cfg.topk_assets,mode=mode,market_type=cfg.market_type)

    if mode == "test":
        metrics['long_only_rets'] = long_only_rets
        metrics['long_short_rets'] = long_short_rets
        # 只在测试阶段画这个热力图
        with torch.no_grad():
            for x, y in loader:  # 使用传入的loader，确保一致
                x = x.to(device)
                _, adj = model(x)
                # plot_conduction_heatmap(adj)
                 # 1. 热力图（降低维度）
                plot_conduction_heatmap(
                    adj, 
                    stock_names=stock_names,
                    save_path=f"{cfg.market_type}_heatmap_top30.png",
                    top_n=30
                )
                # 2. 精简热力图（只显示最强连接）
                plot_top_conduction_heatmap(
                    adj,
                    stock_names=stock_names,
                    save_path=f"{cfg.market_type}_heatmap_top20_top5.png",
                    top_n=20,
                    top_k_per_stock=5
                )
                # 3. 网络图（最直观）
                plot_conduction_network(adj,
                    stock_names=stock_names,
                    save_path=f"{cfg.market_type}_network_top30.png",
                    top_n=30,
                    min_weight=0.02
                )
                break  # 只画第一个batch
    return metrics, long_only_rets, long_short_rets

def listnet_loss(y_pred, y_true, eps=1e-8, temperature=0.1, running_std=None):
    """
    ✅ 股票预测专用ListNet损失（带温度参数）
    Args:
        temperature: 温度参数，放大收益率差异，股票预测最优值=0.1
    """
    # 数值稳定
    y_pred = y_pred - y_pred.max(dim=1, keepdim=True)[0]
    y_true = y_true - y_true.max(dim=1, keepdim=True)[0]
    # ✅ 关键：用温度参数放大收益率差异
    exp_pred = torch.exp(y_pred / temperature)
    exp_true = torch.exp(y_true / temperature)
    
    pred_probs = exp_pred / (exp_pred.sum(dim=1, keepdim=True) + eps)
    true_probs = exp_true / (exp_true.sum(dim=1, keepdim=True) + eps)

    loss = -torch.sum(true_probs * torch.log(pred_probs + eps), dim=1).mean()
    # 标准化到和MSE相同量级
    if running_std is not None and running_std > 0:
        loss = loss / running_std
    return loss


def check_gradients(model, epoch, batch_idx):
    """检查关键参数的梯度"""
    grad_stats = {}
    
    # 检查滞后参数梯度（v2：滞后偏好 + 存活门控风险）
    if hasattr(model.encoder, 'lag_pool'):
        lag_pool = model.encoder.lag_pool
        for pname in ('lag_weight', 'lag_hazard'):
            p = getattr(lag_pool, pname, None)
            if p is not None and p.grad is not None:
                grad_norm = p.grad.norm().item()
                grad_mean = p.grad.abs().mean().item()
                grad_stats[pname] = (grad_norm, grad_mean)
    
    # 检查图构造梯度
    if hasattr(model.encoder.spatial_agg, 'graph_constructor'):
        gc = model.encoder.spatial_agg.graph_constructor
        if hasattr(gc, 'Wq') and gc.Wq.weight.grad is not None:
            grad_norm = gc.Wq.weight.grad.norm().item()
            grad_stats['graph_constructor'] = grad_norm
    
    # 检查alpha和lambda梯度
    if hasattr(model.encoder.spatial_agg, 'alpha') and model.encoder.spatial_agg.alpha.grad is not None:
        grad_stats['alpha'] = model.encoder.spatial_agg.alpha.grad.item()
    
    if hasattr(model.encoder, 'lambda_param') and model.encoder.lambda_param.grad is not None:
        grad_stats['lambda'] = model.encoder.lambda_param.grad.item()
    
    # 打印梯度统计
    if grad_stats:
        print(f"Epoch {epoch}, Batch {batch_idx} 梯度统计:")
        for name, value in grad_stats.items():
            if isinstance(value, tuple):
                print(f"  {name}: norm={value[0]:.6f}, mean_abs={value[1]:.6f}")
            else:
                print(f"  {name}: {value:.6f}")

def monitor_param_updates(model, optimizer, epoch):
    """监控参数更新情况"""
    update_stats = {}
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                param_norm = p.norm().item()
                grad_norm = p.grad.norm().item()
                update_ratio = grad_norm / (param_norm + 1e-8)
                
                if update_ratio > 1e-6:  # 只记录有更新的参数
                    update_stats[f"param_{id(p)}"] = {
                        'param_norm': param_norm,
                        'grad_norm': grad_norm,
                        'update_ratio': update_ratio
                    }
    
    # 汇总统计
    if update_stats:
        avg_update_ratio = np.mean([stats['update_ratio'] for stats in update_stats.values()])
        print(f"Epoch {epoch} 参数更新统计:")
        print(f"  有效更新参数比例: {len(update_stats)}/{sum(1 for p in model.parameters() if p.requires_grad)}")
        print(f"  平均更新比率: {avg_update_ratio:.6f}")
        print(f"  典型梯度范数: {np.mean([stats['grad_norm'] for stats in update_stats.values()]):.6f}")