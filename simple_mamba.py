import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Union
from einops import rearrange, repeat, einsum
import torch.nn.functional as F

# ---------------------- 基础组件 ----------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

@dataclass
class _ModelArgs:
    d_model: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

# ---------------------- 单步SimpleMambaCell（终极修复版） ----------------------
class SimpleMambaCell(nn.Module):
    """
    单步Mamba单元，修复了梯度爆炸问题
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.args = _ModelArgs(d_model=d_model, d_state=d_state)
        self.d_model = d_model
        self.d_inner = self.args.d_inner
        self.d_state = self.args.d_state
        
        # 所有参数与批量版完全一致
        self.in_proj = nn.Linear(self.args.d_model, self.args.d_inner * 2, bias=self.args.bias)
        self.conv1d = nn.Conv1d(
            self.args.d_inner, self.args.d_inner, self.args.d_conv, 
            groups=self.args.d_inner, padding=self.args.d_conv-1, bias=self.args.conv_bias
        )
        self.x_proj = nn.Linear(self.args.d_inner, self.args.dt_rank + self.args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.args.dt_rank, self.d_inner, bias=True)
        
        # ✅ 官方Mamba的dt_proj.bias初始化（关键修复）
        dt_min = 0.001
        dt_max = 0.1
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        A = repeat(torch.arange(1, self.args.d_state+1), 'n -> d n', d=self.args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.args.d_inner))
        self.out_proj = nn.Linear(self.args.d_inner, self.args.d_model, bias=self.args.bias)
        
        # ✅ 添加RMSNorm
        self.norm = RMSNorm(d_model)
        
        # 缓存
        self.conv_cache = None
        self.ssm_state = None

    def reset_cache(self, batch_size, device):
        batch_size = int(batch_size)
        self.conv_cache = torch.zeros(
            batch_size, self.d_inner, self.args.d_conv - 1, 
            device=device
        )
        self.ssm_state = torch.zeros(
            batch_size, self.d_inner, self.d_state, 
            device=device
        )

    def forward(self, x_t, h_prev):
        # ✅ 输入前添加RMSNorm
        x_t = self.norm(x_t)
        
        # 将前两个维度合并
        x_flat = x_t.flatten(0, 1)
        
        # 1. 输入投影
        x_and_res = self.in_proj(x_flat)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # 2. 因果卷积
        x_conv_in = torch.cat([self.conv_cache, x.unsqueeze(-1)], dim=-1)
        self.conv_cache = x_conv_in[:, :, 1:]
        x_conv = torch.sum(
            x_conv_in * rearrange(self.conv1d.weight, 'd 1 w -> 1 d w'), 
            dim=-1
        )
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x = F.silu(x_conv)
        
        # 3. 计算SSM参数
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.args.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        # 4. 单步SSM状态更新
        deltaA = torch.exp(einsum(delta, A, 'b d, d n -> b d n'))
        deltaB_u = einsum(delta, B, x, 'b d, b n, b d -> b d n')
        
        self.ssm_state = deltaA * self.ssm_state + deltaB_u
        y = einsum(self.ssm_state, C, 'b d n, b n -> b d')
        y = y + x * D
        
        # 5. 门控和输出投影
        y = y * F.silu(res)
        h_flat = self.out_proj(y)
        
        # 恢复维度
        h_t = h_flat.unflatten(0, x_t.shape[:2])
        
        # ✅ 添加残差连接
        h_t = h_t + x_t
        
        return h_t

# ---------------------- 批量版SimpleMamba（同步修复） ----------------------
class _MambaBlock(nn.Module):
    def __init__(self, args: _ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(args.d_inner, args.d_inner, args.d_conv, 
                                 groups=args.d_inner, padding=args.d_conv-1, bias=args.conv_bias)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state+1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        # ✅ 同样修复shape类型问题
        b, l, d = map(int, x.shape)
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split([self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        
        (d_in, n) = map(int, self.A_log.shape)
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split([self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, x, 'b l d, b l n, b l d -> b l d n')
        x_state = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x_state = deltaA[:, i] * x_state + deltaB_u[:, i]
            y = einsum(x_state, C[:, i, :], 'b d n, b n -> b d')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + x * D
        
        y = y * F.silu(res)
        return self.out_proj(y)

class SimpleMamba(nn.Module):
    def __init__(self, d_model=256, d_state=16, n_layer=1):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            args = _ModelArgs(d_model=d_model, d_state=d_state)
            self.layers.append(nn.ModuleDict({
                "mixer": _MambaBlock(args),
                "norm": RMSNorm(d_model)
            }))

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mixer"](x)
            x = x + residual
        return x