import copy
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from metric import train_epoch, evaluate, DynamicLossWeightScheduler

def get_ablation_configs(base_cfg):
    variants = []

    # 1. 纯Mamba
    # cfg1 = copy.deepcopy(base_cfg)
    # cfg1.use_async = False
    # cfg1.use_individual = False
    # cfg1.use_global = False
    # cfg1.learnable_lag = False
    # variants.append(("1_Baseline_Mamba", cfg1))

    # 2. 异步1天 + 微观（个股）联动特征
    # cfg2 = copy.deepcopy(base_cfg)
    # cfg2.use_async = True
    # cfg2.use_individual = True
    # cfg2.use_global = False
    # cfg2.learnable_lag = False
    # cfg2.l_max = 1     # 这里的异步一天是怎么回事？？
    # variants.append(("2_Async_1Day_Individual", cfg2))

    # 3. 异步1天 + 宏观特征
    # cfg3 = copy.deepcopy(base_cfg)
    # cfg3.use_async = True
    # cfg3.use_individual = False
    # cfg3.use_global = True
    # cfg3.learnable_lag = False
    # cfg3.l_max = 1
    # variants.append(("3_Async_1Day_Global", cfg3))

    # 4. 异步1天 + 双路径
    # cfg4 = copy.deepcopy(base_cfg)
    # cfg4.use_async = True
    # cfg4.use_individual = True
    # cfg4.use_global = True
    # cfg4.learnable_lag = False
    # cfg4.l_max = 1
    # variants.append(("4_Async_1Day_DualPath", cfg4))

    # 5. 同步双路径（无异步）
    # cfg5 = copy.deepcopy(base_cfg)
    # cfg5.use_async = False
    # cfg5.use_individual = True
    # cfg5.use_global = True
    # cfg5.learnable_lag = False
    # variants.append(("5_Sync_DualPath_NoAsync", cfg5))

    # 6. 异步多步 + 双路径（完整模型）
    cfg6 = copy.deepcopy(base_cfg)
    cfg6.use_async = True
    cfg6.use_individual = True
    cfg6.use_global = True
    cfg6.learnable_lag = True
    # variants.append(("6_Async_MultiLag_DualPath", cfg6))

    # 7. 无自身历史
    # cfg7 = copy.deepcopy(cfg6)
    # cfg7.use_self_history = False
    # variants.append(("7_No_SelfHistory", cfg7))

    # 8. 无门控机制
    cfg8 = copy.deepcopy(cfg6)
    cfg8.use_gate = False
    variants.append(("8_No_Gating", cfg8))

    # 建议补充的消融变体
    # 9. 仅全局路径
    cfg9 = copy.deepcopy(base_cfg)
    cfg9.use_async = True
    cfg9.use_individual = False  # 关闭个体路径
    cfg9.use_global = True
    cfg9.learnable_lag = True
    variants.append(("9_Global_Only", cfg9))

    # 10. 仅个体路径  
    cfg10 = copy.deepcopy(base_cfg)
    cfg10.use_async = True
    cfg10.use_individual = True
    cfg10.use_global = False  # 关闭全局路径
    cfg10.learnable_lag = True
    variants.append(("10_Individual_Only", cfg10))

    # 11. 固定权重双路径
    cfg11 = copy.deepcopy(base_cfg)
    cfg11.use_async = True
    cfg11.use_individual = True
    cfg11.use_global = True
    cfg11.learnable_lag = True
    cfg11.fixed_alpha = 0.5  # 添加固定权重配置
    variants.append(("11_Fixed_Weight_DualPath", cfg11))

    # 12. 关闭图先验：邻接只当 0/1 掩码（等价于 v1 行为，用于验证“图可学”的价值）
    cfg12 = copy.deepcopy(cfg6)
    cfg12.use_adj_prior = False
    variants.append(("12_No_GraphPrior", cfg12))

    # 13. 关闭滞后存活门控：只保留滞后分布（用于验证“自动截止”的价值）
    cfg13 = copy.deepcopy(cfg6)
    cfg13.use_lag_survival = False
    variants.append(("13_No_Lag_Survival", cfg13))

    # 14. 关闭滞后状态调制：纯静态个股滞后先验（用于验证“动态调制”的价值）
    cfg14 = copy.deepcopy(cfg6)
    cfg14.lag_state_conditioning = False
    variants.append(("14_Static_Lag_Prior", cfg14))

    return variants

def run_ablation(cfg, train_loader, val_loader, test_loader, n_stocks, label_stats):
    from main import AsyncMambaModel
    device = torch.device(cfg.device)
    print(f"\n[ablation] device = {device}（{len(get_ablation_configs(cfg))} 个变体）", flush=True)
    results = {}

    for name, config in get_ablation_configs(cfg):
        print(f"\n{'='*20} Running ablation: {name} {'='*20}")
        model = AsyncMambaModel(config, n_stocks).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

        loss_scheduler = None
        if config.use_dynamic_weight and config.loss_type != "mse":
            loss_scheduler = DynamicLossWeightScheduler(
                total_epochs=config.epochs,
                mse_max=config.mse_weight_max,
                mse_min=config.mse_weight_min,
                warmup_epochs=config.warmup_epochs
            )

        best_val_ir = -float('inf')
        best_model_state = copy.deepcopy(model.state_dict())
        no_improve_count = 0
        running_rank_std = None

        for epoch in range(1, config.epochs + 1):
            train_loss, train_metrics, avg_mse_loss, avg_rank_loss, running_rank_std = train_epoch(
                config, model, train_loader, optimizer, device, epoch, n_stocks,
                loss_scheduler=loss_scheduler, running_rank_std=running_rank_std
            )
            val_metrics, _, _ = evaluate(config, model, val_loader, device, mode="val")
            scheduler.step()

            current_ir = val_metrics.get("RankICIR", 0.0)
            if not np.isnan(current_ir) and current_ir > best_val_ir:
                best_val_ir = current_ir
                best_model_state = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % config.log_interval == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val RankIC: {val_metrics.get('RankIC_Mean',0):.4f} | "
                      f"Val RankICIR: {val_metrics.get('RankICIR',0):.4f}")

        model.load_state_dict(best_model_state)
        test_metrics, _, _ = evaluate(
            config, model, test_loader, device, mode="test", label_stats=label_stats
        )
        print(f"Test Metrics: RankIC={test_metrics.get('RankIC_Mean',0):.4f}, "
              f"RankICIR={test_metrics.get('RankICIR',0):.4f}, "
              f"LS_Sharpe={test_metrics.get('LS_Sharpe',0):.4f}")
        results[name] = test_metrics

    return results