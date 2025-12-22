import sys
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from torch.utils.data import DataLoader

# 确保能找到项目根目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_state_transition import TrainConfig, build_models
from datasets.state_datasets import StateTransitionDataset

def get_event_datasets(cfg):
    """扫描目录获取所有独立事件数据集"""
    traj_pattern = os.path.join(cfg.state_trajectory_dir, "*_trajectory.csv")
    traj_files = sorted(glob.glob(traj_pattern))
    
    if cfg.max_event > 0:
        traj_files = traj_files[:cfg.max_event]
        
    datasets = []
    for traj_path in traj_files:
        event_id = os.path.basename(traj_path).replace("_trajectory.csv", "")
        json_path = os.path.join(cfg.event_data_dir, f"{event_id}.json")
        mf_path = os.path.join(cfg.mf_dir, f"{event_id}_mf.csv")
        
        if not (os.path.exists(json_path) and os.path.exists(mf_path)):
            continue
            
        ds = StateTransitionDataset(
            trajectory_path=traj_path,
            mf_path=mf_path,
            test_data_path=json_path,
            profile_path=cfg.profile_path,
            encoder_config={"type": cfg.encoder_type, "model_name": cfg.model_name},
            file_config={
                'cluser_user_profile': cfg.profile_path,
                'uid_mapping_path': cfg.uid_mapping_path,
                'cluster_info_path': cfg.cluster_info_path
            },
            batch_size=cfg.num_agents
        )
        datasets.append((event_id, ds))
    return datasets

def get_cached_features(event_id, dataset, text_encoder, cfg):
    """缓存机制：加速 BERT 提取过程"""
    cache_dir = os.path.join(cfg.save_dir, "eval_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{event_id}_features.pt")

    if os.path.exists(cache_path):
        return torch.load(cache_path)

    print(f"📦 正在提取并缓存事件 {event_id} 的特征向量...")
    all_text_embs, all_profile_vecs = [], []
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            mf_texts = batch["mf_text"]
            tokens = text_encoder.tokenizer(mf_texts, padding=True, truncation=True, return_tensors="pt").to(cfg.device)
            emb = text_encoder(tokens['input_ids'], tokens['attention_mask'])
            if isinstance(emb, tuple): emb = emb[0]
            all_text_embs.append(emb.cpu())
            all_profile_vecs.append(batch["profile_vecs"].cpu())

    features = {"text_embs": torch.cat(all_text_embs, dim=0), "profile_vecs": torch.cat(all_profile_vecs, dim=0)}
    torch.save(features, cache_path)
    return features

def evaluate_event(event_id, dataset, state_net, features, cfg):
    """执行时间步循环预测"""
    preds, targets = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            mu_prev = item["mu_prev"].unsqueeze(0).to(cfg.device)
            target = item["target_dist_sum"].numpy()
            text_emb = features["text_embs"][i].unsqueeze(0).to(cfg.device)
            profile_vec = features["profile_vecs"][i].unsqueeze(0).to(cfg.device)
            mu_pred, _ = state_net(mu_prev, text_emb, profile_vec)
            preds.append(mu_pred.cpu().numpy()[0])
            targets.append(target)
    return np.array(preds), np.array(targets)

def plot_single_event_style(event_id, preds, targets, save_dir):
    """
    按照用户要求的风格绘制对比图：
    - 实线代表真实值 (Target/Real)
    - 虚线代表模型预测值 (Pred)
    - 颜色区分：正向(绿)、中立(灰)、负向(红)
    """
    plt.figure(figsize=(12, 6))
    x = range(len(preds))
    
    # --- 颜色定义 ---
    colors = {
        'pos': '#006400', # 深绿
        'neu': '#4c4c4c', # 深灰
        'neg': '#8B0000'  # 深红
    }

    # --- 1. 绘制真实值 (Target/Real) - 使用粗实线 ---
    plt.plot(x, targets[:, 0], color=colors['pos'], linestyle='-', linewidth=3, label='Target Pos (+1)')
    plt.plot(x, targets[:, 1], color=colors['neu'], linestyle='-', linewidth=3, label='Target Neu (0)')
    plt.plot(x, targets[:, 2], color=colors['neg'], linestyle='-', linewidth=3, label='Target Neg (-1)')
    
    # --- 2. 绘制预测值 (Pred) - 使用细虚线 ---
    plt.plot(x, preds[:, 0], color=colors['pos'], linestyle='--', linewidth=1.5, alpha=0.8, label='Pred Pos (+1)')
    plt.plot(x, preds[:, 1], color=colors['neu'], linestyle='--', linewidth=1.5, alpha=0.8, label='Pred Neu (0)')
    plt.plot(x, preds[:, 2], color=colors['neg'], linestyle='--', linewidth=1.5, alpha=0.8, label='Pred Neg (-1)')
    
    # --- 装饰 ---
    plt.title(f'State Prediction Comparison: {event_id}', fontsize=16, fontweight='bold')
    plt.xlabel('Time Step (Batch)', fontsize=12)
    plt.ylabel('Proportion (占比)', fontsize=12)
    plt.ylim(-0.05, 1.05) 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 图例放在右侧外部
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f"{event_id}_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 图表已保存: {save_path}")

def main():
    cfg = TrainConfig()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.event_data_dir = "/root/Mean-Field-LLM/enhance/data/append_data"
    cfg.mf_dir = "/root/Mean-Field-LLM/enhance/result/test_mf/tmp"
    cfg.state_trajectory_dir = "/root/Mean-Field-LLM/enhance/result/code_trajectory/tmp"
    
    res_dir = os.path.join(cfg.save_dir, "evaluation_results")
    os.makedirs(res_dir, exist_ok=True)

    text_encoder, state_net = build_models(cfg)
    ckpt_path = os.path.join(cfg.save_dir, cfg.save_name)
    checkpoint = torch.load(ckpt_path, map_location=cfg.device)
    state_net.load_state_dict(checkpoint['model_state_dict'])
    text_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    state_net.to(cfg.device).eval()
    text_encoder.to(cfg.device).eval()

    event_list = get_event_datasets(cfg)
    summary_data = []

    for event_id, ds in event_list:
        features = get_cached_features(event_id, ds, text_encoder, cfg)
        preds, targets = evaluate_event(event_id, ds, state_net, features, cfg)
        
        mae = np.abs(preds - targets).mean(axis=0)
        summary_data.append([event_id, mae[0], mae[1], mae[2], mae.mean()])
        
        # 调用新风格的绘图函数
        plot_single_event_style(event_id, preds, targets, res_dir)

    summary_df = pd.DataFrame(summary_data, columns=['event_id', 'mae_pos', 'mae_neu', 'mae_neg', 'mae_avg'])
    summary_df.to_csv(os.path.join(res_dir, "all_events_summary.csv"), index=False)
    print(f"📊 汇总结果已保存至: {os.path.join(res_dir, 'all_events_summary.csv')}")

if __name__ == "__main__":
    main()