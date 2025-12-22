import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.state_transition.encoders import build_text_encoder
from model.state_transition.state_transition_net import StateTransitionNet
# 引入上面的 dataset
from datasets.state_datasets import TrajectoryDistDataset, trajectory_dist_collate

def train_distribution_matching():
    # === 配置 ===
    config = {
        "batch_size_batches": 4, # 这里指的是一次训练取多少个 "Batch任务" (每个任务含16人)
        "lr": 1e-4,
        "epochs": 20,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "encoder": {
            "type": "bert",
            "model_name": "bert-base-chinese",
            "output_dim": 768,
            "freeze": True
        }
    }
    
    print(f"Using Device: {config['device']}")
    
    # === 1. 数据准备 ===
    dataset = TrajectoryDistDataset(
        trajectory_path="10031994215_trajectory.csv",
        mf_path="4264473811_mf.csv", 
        profile_path="cluster_core_user_profile.jsonl",
        encoder_config=config['encoder']
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=config['batch_size_batches'], 
        shuffle=True, 
        collate_fn=trajectory_dist_collate
    )
    
    # === 2. 模型构建 ===
    mf_encoder = build_text_encoder(config['encoder']).to(config['device'])
    
    model = StateTransitionNet(
        agent_feat_dim=768, 
        text_emb_dim=768,
        hidden_dim=256
    ).to(config['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # 核心修改：使用 MSE Loss 来对齐分布
    # 也可以用 KLDivLoss，但 MSE 在分布匹配上通常更稳定
    criterion = nn.MSELoss() 
    
    # === 3. 训练 ===
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        
        for batch_idx, batch in enumerate(loader):
            # batch['profile_vecs'] shape: (B_task, 16, 768)
            # mu_prev shape: (B_task, 3)
            
            mu_prev = batch['mu_prev'].to(config['device'])
            target_dist = batch['target_dist'].to(config['device'])
            profile_vecs_batch = batch['profile_vecs'].to(config['device'])
            mf_texts = batch['mf_texts']
            
            # A. 编码 MF 文本
            if hasattr(mf_encoder, 'tokenizer'):
                inputs = mf_encoder.tokenizer(
                    mf_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
                ).to(config['device'])
                mf_emb = mf_encoder(inputs['input_ids'], inputs['attention_mask']) # (B_task, 768)
            else:
                mf_emb = torch.randn(len(mf_texts), 768).to(config['device'])
            
            # B. Forward (需要模型支持 forward_dense)
            # 调用我们在上一步讨论的 forward_dense 接口
            # 输入: mu_prev(B,3), mf_emb(B,768), profile_vecs(B,16,768)
            # 输出: mu_pred(B,3), probs(B,16,3)
            
            # 如果你的 state_transition_net.py 里实现了 forward_dense，直接调用:
            mu_pred_mean, probs_individual = model.forward_dense(mu_prev, mf_emb, profile_vecs_batch)
            
            # mu_pred_mean 就是这一组 16 个人的平均预测分布
            
            # C. 计算 Loss
            # 我们希望 预测的平均分布 == 真实的 Batch 分布
            loss = criterion(mu_pred_mean, target_dist)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.6f}")
        
        # 打印一个样本看看效果
        if (epoch+1) % 5 == 0:
            print(f"  Target: {target_dist[0].detach().cpu().numpy()}")
            print(f"  Pred:   {mu_pred_mean[0].detach().cpu().numpy()}")

if __name__ == "__main__":
    train_distribution_matching()