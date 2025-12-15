# state_transition_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class StateTransitionNet(nn.Module):
    """
    StateTransitionNet (Dense Version)
    ----------------------------------
    输入支持 Batch 处理形式。
    
    Dimensions:
      B: Batch Size (DataLoader定义的训练批次大小，例如 32)
      N: Num Agents (单次模拟中的用户数量，对应 Dataset 中的 16)
      D_u: Agent Feature Dimension (例如 768)
      D_c: Text Embedding Dimension (例如 768)
    """

    def __init__(
        self,
        agent_feat_dim: int,
        text_emb_dim: int,
        hidden_dim: int = 512,
        use_layernorm: bool = False,
    ):
        super().__init__()

        self.agent_feat_dim = agent_feat_dim
        self.text_emb_dim = text_emb_dim
        
        # 输入维度 = Agent特征 + 上一时刻分布(3) + 环境文本嵌入
        input_dim = agent_feat_dim + 3 + text_emb_dim

        # 保持2层的MLP + GELU，如果训练困难考虑加入Residual
        layers = [
            nn.Linear(input_dim, 512), 
            nn.LayerNorm(512),         
            nn.GELU(),                 
            nn.Dropout(0.1),           
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 3)
        ]

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        mu_prev: torch.Tensor,
        text_emb: torch.Tensor,
        agent_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
          mu_prev:     (B, 3)       -> 上一时刻的分布 (上一行数据)
          text_emb:    (B, d_c)     -> 当前时刻mf的 Embedding
          agent_feats: (B, N, d_u)  -> 当前时刻参与的 N 个用户的画像 (从 Dataset 出来是 (B, 16, 768))

        返回:
          mu_pred: (B, 3)           -> 预测的当前时刻分布 (聚合后)
          probs:   (B, N, 3)        -> 每个 Agent 的具体概率 (未聚合)
        """
        # 1. 获取维度信息
        B, N, d_u = agent_feats.shape
        
        # 2. 扩展维度以匹配 Agent 数量
        # mu_prev: (B, 3) -> (B, 1, 3) -> (B, N, 3)
        mu_prev_exp = mu_prev.unsqueeze(1).expand(B, N, -1)
        
        # text_emb: (B, d_c) -> (B, 1, d_c) -> (B, N, d_c)
        text_emb_exp = text_emb.unsqueeze(1).expand(B, N, -1)

        # 3. 拼接输入特征
        # 结果 shape: (B, N, d_u + 3 + d_c)
        h_in = torch.cat([agent_feats, mu_prev_exp, text_emb_exp], dim=-1)

        # 4. 这里的 MLP 通常处理最后一维，所以不需要 flatten 也可以直接传
        # Linear 层会对 (B, N, input_dim) 的 input_dim 进行操作
        logits = self.mlp(h_in)  # (B, N, 3)

        # 5. 计算概率
        probs = F.softmax(logits, dim=-1)  # (B, N, 3)

        # 6. 聚合得到平均场分布 (Mean Field)
        # 在 N (dim=1) 维度上取平均，得到该时刻整体的分布
        mu_pred = probs.mean(dim=1)  # (B, 3)

        return mu_pred, probs