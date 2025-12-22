import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

class PolicyMLPModel(nn.Module):
    def __init__(self, base_model_name, k_steps=3, state_dim=3, lora_rank=8, gamma=0.9):
        super().__init__()
        self.k_steps = k_steps
        self.gamma = gamma # 衰减因子
        
        # 1. 加载基座模型 Qwen2-1.5B
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 2. 配置并应用 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=lora_rank, 
            lora_alpha=2*lora_rank, 
            lora_dropout=0.1,
            # 应用的地方
            target_modules=["q_proj","v_proj",]
        )
        self.llm = get_peft_model(self.base_model, peft_config)
        
        # 获取 LLM 的 hidden size (Qwen2-1.5B 通常是 1536)
        hidden_size = self.base_model.config.hidden_size
        
        # 3. 定义 K 个预测头 (Residual Heads)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + state_dim, 512),    # hidden部分已经经过llm处理了，state dim部分是三分类的概率
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, state_dim) # 预测 Delta_k
            ) for _ in range(k_steps)
        ])
        
        # 确保 MLP 的权重是 float32 (为了数值稳定性)，即使 LLM 是 bfloat16
        # 显卡支持吗
        for head in self.prediction_heads:
            head.to(torch.float32)

    def forward(self, input_ids, attention_mask, labels=None, current_state=None, future_states=None):
        # --- Part A & B 保持不变 (LLM 前向与特征提取) ---
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        text_loss = outputs.loss
        last_hidden_state = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        feature_vector = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
        feature_vector = feature_vector.to(torch.float32)

        # --- Part C: 修改为 Batch-Average KL Loss ---
        pred_loss = 0.0
        predicted_trajectories = []
        
        if current_state is not None:
            combined_input = torch.cat([feature_vector, current_state], dim=-1) 
            
            for i, head in enumerate(self.prediction_heads):
                # 1. 预测与归一化
                delta = head(combined_input)
                pred_dist = torch.softmax(current_state + delta, dim=-1) 
                predicted_trajectories.append(pred_dist)
                
                if future_states is not None:
                    # 2. 计算当前步的 KL Loss
                    target_dist = future_states[:, i, :]
                    avg_pred = pred_dist.mean(dim=0)
                    avg_target = target_dist.mean(dim=0)
                    
                    step_kl_loss = nn.functional.kl_div(
                        avg_pred.log(), 
                        avg_target, 
                        reduction='sum'
                    )
                    
                    # 3. 应用加权：weight = gamma ^ i
                    weight = self.gamma ** i
                    pred_loss += weight * step_kl_loss

        return {
            "loss": text_loss,
            "pred_loss": pred_loss,
            "logits": outputs.logits,
            "predicted_trajectories": torch.stack(predicted_trajectories, dim=1)
        }
    # 用于保存模型时，只保存 adapter 和 mlp
    def save_pretrained(self, path):
        self.llm.save_pretrained(path)
        torch.save(self.prediction_heads.state_dict(), f"{path}/prediction_heads.pt")