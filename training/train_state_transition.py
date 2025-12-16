# train_state_transition.py
# 路径: MFSim/training/train_state_transition.py

import datetime
import os
import sys
import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer  # 如果用 BERT

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 假设 encoders 和 dataset 都在正确位置
from model.state_transition.encoders import build_text_encoder
from model.state_transition.state_transition_net import StateTransitionNet
from datasets import StateTransitionDataset  # 引用你修改后的 Dataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    # --- 目录配置 ---
    event_data_dir: str = "/root/Mean-Field-LLM/mf_llm/data/rumdect/Weibo/test"
    mf_dir: str = "/root/ICML/data/test_mf"
    state_trajectory_dir: str = "/root/ICML/data/test_state_distribution"
    
    # --- 全局共享文件 ---
    profile_path: str = "/root/ICML/data/profile/cluster_core_user_profile.jsonl"
    uid_mapping_path: str = "/root/ICML/data/profile/user_clusters_map.csv"
    cluster_info_path: str = "/root/ICML/data/profile/cluster_details.json"

    # --- 模型与训练参数 ---
    encoder_type: str = "bert"
    model_name: str = "bert-base-chinese"
    text_emb_dim: int = 768
    agent_feat_dim: int = 768
    hidden_dim: int = 256
    use_layernorm: bool = True
    
    train_batch_size: int = 32
    max_event: int = 100
    num_agents: int = 16 
    num_epochs: int = 20
    lr: float = 2e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    save_dir: str = os.path.join(ROOT_DIR, "checkpoints")
    save_name: str = "state_transition_best.pt"

import glob
from torch.utils.data import ConcatDataset

def build_dataloader(cfg: TrainConfig, tokenizer) -> DataLoader:
    # 构建包含所有文件的巨型 Dataset
    full_dataset = build_full_dataset(cfg)
    
    loader = DataLoader(
        full_dataset,
        batch_size=cfg.train_batch_size, # 这里是 32
        shuffle=True, # 这一步很关键！它会打乱不同事件里的样本
        num_workers=4, # 此时可以开启多进程加速读取
        pin_memory=True
    )
    
    return loader

def build_full_dataset(cfg: TrainConfig):
    """
    逻辑变更：
    1. 扫描 state_trajectory_dir 下所有的 *_trajectory.csv 文件 (作为锚点)
    2. 提取 ID
    3. 反向查找对应的 .json (raw data) 和 _mf.csv (environment context)
    4. 如果齐全，创建 Dataset
    5. 合并
    """
    
    # 1. 以 Trajectory (状态分布 GT) 文件为锚点进行扫描
    # 注意：这里扫描的是 state_trajectory_dir
    traj_pattern = os.path.join(cfg.state_trajectory_dir, "*_trajectory.csv")
    traj_files = glob.glob(traj_pattern)
    
    if not traj_files:
        raise ValueError(f"未在 {cfg.state_trajectory_dir} 下找到任何 *_trajectory.csv 文件")
    
    traj_files = sorted(traj_files)
    if cfg.max_event is not None and cfg.max_event > 0:
        original_len = len(traj_files)
        traj_files = traj_files[:cfg.max_event]
        print(f"选取{cfg.max_event}测试文件……")

    dataset_list = []
    
    # 准备配置
    file_config = {
        'cluser_user_profile': cfg.profile_path,
        'uid_mapping_path': cfg.uid_mapping_path,
        'cluster_info_path': cfg.cluster_info_path
    }
    
    encoder_config = {
        "type": cfg.encoder_type,
        "model_name": cfg.model_name
    }

    print(f"🔍 开始扫描 Trajectory 目录: {cfg.state_trajectory_dir} ...")
    print(f"   (共发现 {len(traj_files)} 个分布文件)")

    for traj_path in traj_files:
        # traj_path = ".../4264473811_trajectory.csv"
        filename = os.path.basename(traj_path)  # "4264473811_trajectory.csv"
        
        # 2. 提取 ID (去除后缀 _trajectory.csv)
        event_id = filename.replace("_trajectory.csv", "") # "4264473811"
        
        # 排除非数据文件
        if "cluster" in event_id or "profile" in event_id:
            continue

        # 根据 ID 去找 json
        json_path = os.path.join(cfg.event_data_dir, f"{event_id}.json")
        # 根据 ID 去找 mf.csv
        mf_path = os.path.join(cfg.mf_dir, f"{event_id}_mf.csv")
        
        # 4. 检查原材料是否存在
        if not os.path.exists(json_path):
            print(f"⚠️ 跳过 {event_id}: 有 Trajectory 但缺少原始 JSON 数据 -> {json_path}")
            continue
        if not os.path.exists(mf_path):
            print(f"⚠️ 跳过 {event_id}: 有 Trajectory 但缺少 MF 环境数据 -> {mf_path}")
            continue
            
        # 5. 实例化单个 Dataset
        try:
            ds = StateTransitionDataset(
                trajectory_path=traj_path,  # 锚点文件
                mf_path=mf_path,
                test_data_path=json_path,
                profile_path=cfg.profile_path,
                encoder_config=encoder_config,
                file_config=file_config,
                batch_size=cfg.num_agents
            )
            dataset_list.append(ds)
        except Exception as e:
            print(f"❌ 加载 {event_id} 失败: {e}")

    if not dataset_list:
        raise RuntimeError("没有成功加载任何有效的数据集！")

    print(f"✅ 成功加载 {len(dataset_list)} 个事件的数据集")
    
    # 6. 合并
    full_dataset = ConcatDataset(dataset_list)
    return full_dataset


def build_models(cfg: TrainConfig):
    # 1. 文本编码器 (用于处理环境文本 mf_text)
    encoder_config = {
        "type": cfg.encoder_type,
        "model_name": cfg.model_name,
        "output_dim": cfg.text_emb_dim,
        "freeze": False # 训练时是否微调 BERT
    }
    text_encoder = build_text_encoder(encoder_config)

    # 2. 状态转移网络
    state_net = StateTransitionNet(
        agent_feat_dim=cfg.agent_feat_dim,
        text_emb_dim=cfg.text_emb_dim,
        hidden_dim=cfg.hidden_dim,
        use_layernorm=cfg.use_layernorm,
    )

    return text_encoder, state_net

def train_one_epoch(
    epoch: int,
    cfg: TrainConfig,
    text_encoder: torch.nn.Module,
    state_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    writer: SummaryWriter
):
    text_encoder.train()
    state_net.train()
    
    total_loss = 0.0
    total_steps = 0

    for batch_idx, batch_data in enumerate(train_loader):
        # 1. 解包数据并送入设备
        mu_prev = batch_data["mu_prev"].to(cfg.device)        # (B, 3)
        target_dist = batch_data["target_dist"].to(cfg.device) # (B, 3)
        agent_feats = batch_data["profile_vecs"].to(cfg.device) # (B, N, D_u)
        mf_texts = batch_data["mf_text"] # List[str]
        
        # 2. 处理环境文本 -> Embedding
        tokenizer = text_encoder.tokenizer 
        tokenized_inputs = tokenizer(
            mf_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(cfg.device)
        
        if cfg.encoder_type == 'bert':
            text_emb = text_encoder(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
            if isinstance(text_emb, tuple): text_emb = text_emb[0]
        else:
             text_emb = text_encoder(tokenized_inputs['input_ids'])

        # 3. 状态转移网络前向传播
        mu_pred, _ = state_net(mu_prev, text_emb, agent_feats)

        # 4. 计算 Loss (KL Divergence)
        log_mu_pred = torch.log(mu_pred + 1e-8)
        loss = F.kl_div(log_mu_pred, target_dist, reduction='batchmean')

        # ==========================================================
        # [新增] 计算辅助指标 (不参与梯度回传，使用 no_grad)
        # ==========================================================
        with torch.no_grad():
            # A. MAE (平均绝对误差): 直观理解概率偏离了多少
            mae = F.l1_loss(mu_pred, target_dist).item()

            # B. Trend Accuracy (趋势准确率): 预测的主要方向(Pos/Neu/Neg)是否对齐
            pred_label = torch.argmax(mu_pred, dim=1)
            target_label = torch.argmax(target_dist, dim=1)
            acc = (pred_label == target_label).float().mean().item()

        # 5. 反向传播
        optimizer.zero_grad()
        loss.backward()

        # ==========================================================
        # [新增] 计算梯度范数 (用于监控梯度爆炸/消失)
        # ==========================================================
        grad_norm = 0.0
        for p in list(text_encoder.parameters()) + list(state_net.parameters()):
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(text_encoder.parameters()) + list(state_net.parameters()),
                cfg.grad_clip
            )

        optimizer.step()

        # ==========================================================
        # [新增] 写入 TensorBoard 多条曲线
        # ==========================================================
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        
        # 1. 训练损失 (最重要)
        writer.add_scalar('Loss/train_kl', loss.item(), global_step)
        
        # 2. 业务指标 (给人看)
        writer.add_scalar('Metric/MAE', mae, global_step)
        writer.add_scalar('Metric/Accuracy', acc, global_step)
        
        # 3. 调试指标 (给开发者看)
        writer.add_scalar('Debug/Grad_Norm', grad_norm, global_step)
        # 监控学习率变化 (如果是动态学习率的话很有用)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Debug/LR', current_lr, global_step)

        total_loss += loss.item()
        total_steps += 1

        if (batch_idx + 1) % cfg.log_interval == 0:
            logger.info(
                f"[Epoch {epoch}] Step {batch_idx+1}/{len(train_loader)} "
                f"Loss: {loss.item():.6f} | MAE: {mae:.4f} | Acc: {acc:.2%} | Grad: {grad_norm:.2f}"
            )

    avg_loss = total_loss / max(1, total_steps)
    logger.info(f"[Epoch {epoch}] Finished. Avg Loss: {avg_loss:.6f}")
    return avg_loss


def save_checkpoint(cfg, text_encoder, state_net, optimizer, epoch, loss, is_best=False):
    os.makedirs(cfg.save_dir, exist_ok=True)

    ckpt = {
        'epoch': epoch,
        'model_state_dict': state_net.state_dict(),
        'encoder_state_dict': text_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # [关键] 保存优化器状态
        'loss': loss,
        'config': str(cfg)
    }

    last_path = os.path.join(cfg.save_dir, "checkpoint_last.pt")
    torch.save(ckpt, last_path)

    if is_best:
        best_path = os.path.join(cfg.save_dir, cfg.save_name)
        torch.save(ckpt, best_path)
        logger.info(f"🌟 Best model saved: {best_path} (Loss: {loss:.6f})")
    
    logger.info(f"💾 Checkpoint saved: {last_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # 初始化配置
    cfg = TrainConfig()
    cfg.num_epochs = args.epochs
    cfg.train_batch_size = args.batch_size

    log_dir = f"checkpoints/runs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    
    logger.info(f"Device: {cfg.device}")
    
    # 1. 构建模型
    text_encoder, state_net = build_models(cfg)
    text_encoder.to(cfg.device)
    state_net.to(cfg.device)
    
    # 2. 构建 DataLoader
    # 确保传入正确的函数调用
    train_loader = build_dataloader(cfg, getattr(text_encoder, 'tokenizer', None))

    # 3. 优化器
    optimizer = torch.optim.AdamW(
        list(text_encoder.parameters()) + list(state_net.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # 4. 训练循环
    start_epoch = 1
    best_loss = float('inf')
    
    if args.resume:
        ckpt_path = os.path.join(cfg.save_dir, "checkpoint_last.pt")
        if os.path.exists(ckpt_path):
            print(f"🔄 正在从 {ckpt_path} 恢复训练...")
            checkpoint = torch.load(ckpt_path, map_location=cfg.device)
            
            # 恢复模型权重
            state_net.load_state_dict(checkpoint['model_state_dict'])
            text_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
            # 恢复优化器 (这就保证了学习率和动量是接着上次的)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复 Epoch (从下一轮开始)
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('loss', float('inf')) # 尝试获取上次的 loss
            
            print(f"✅ 恢复成功！将从 Epoch {start_epoch} 开始继续训练。")
        else:
            print(f"⚠️ 未找到 {ckpt_path}，将从头开始训练。")

    # 4. 训练循环
    #range 从 start_epoch 开始
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        # 记得把 writer 传进去
        loss = train_one_epoch(epoch, cfg, text_encoder, state_net, optimizer, train_loader, writer)
        
        # 判断是否是最佳
        is_best = loss < best_loss
        if is_best:
            best_loss = loss
            
        # 保存 (注意参数变了，传入了 optimizer 和 is_best)
        save_checkpoint(cfg, text_encoder, state_net, optimizer, epoch, loss, is_best)

    writer.close()

if __name__ == "__main__":
    main()