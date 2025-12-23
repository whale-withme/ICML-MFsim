import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

class PolicyMLPModel(nn.Module):
    def __init__(self, base_model_name, k_steps=10, state_dim=3, lora_rank=16, gamma=0.9):
        super().__init__()
        self.k_steps = k_steps
        self.gamma = gamma
        
        # 1. 加载基座模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 2. 配置并应用 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=lora_rank, 
            lora_alpha=2*lora_rank, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.llm = get_peft_model(self.base_model, peft_config)
        
        # 3. 定义 K 个预测头
        hidden_size = self.base_model.config.hidden_size
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + state_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, state_dim)
            ) for _ in range(k_steps)
        ])
        
        # 确保 MLP 权重是 float32 且开启梯度
        for head in self.prediction_heads:
            head.to(torch.float32)
            for param in head.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None, current_state=None, future_states=None, **kwargs):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        text_loss = outputs.loss
        
        # 提取特征向量
        last_hidden_state = outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        feature_vector = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
        feature_vector = feature_vector.to(torch.float32)

        pred_loss = 0.0
        predicted_trajectories = []
        
        if current_state is not None:
            combined_input = torch.cat([feature_vector, current_state], dim=-1) 
            
            for i, head in enumerate(self.prediction_heads):
                delta = head(combined_input)
                # 状态演变预测
                pred_dist = torch.softmax(current_state + delta, dim=-1) 
                predicted_trajectories.append(pred_dist)
                
                if future_states is not None:
                    target_dist = future_states[:, i, :]
                    # 使用 batchmean 计算 KL 散度
                    step_kl_loss = nn.functional.kl_div(
                        pred_dist.log(), 
                        target_dist, 
                        reduction='batchmean'
                    )
                    pred_loss += (self.gamma ** i) * step_kl_loss

        return {
            "loss": text_loss,
            "pred_loss": pred_loss,
            "logits": outputs.logits,
            "predicted_trajectories": torch.stack(predicted_trajectories, dim=1) if predicted_trajectories else None
        }

    def save_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        # 保存 LoRA (adapter_model.bin)
        self.llm.save_pretrained(path)
        # 保存所有 MLP 头 (prediction_heads.pt)
        torch.save(self.prediction_heads.state_dict(), os.path.join(path, "prediction_heads.pt"))
        print(f"[SUCCESS] Model and MLP heads saved to {path}")