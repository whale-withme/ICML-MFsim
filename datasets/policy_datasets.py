import torch
from torch.utils.data import Dataset

class PolicyModelDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, k_steps=10):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_steps = k_steps

    def __len__(self):
        return len(self.data)

    """
    item读取需要的, 假设K=3
    [
        {
            "user_profile": "位置: 北京, 性别: 男, 爱好: 科技...", 
            "user_profile_vector": .........,
            "topic": "某科技公司发布新一代芯片",
            "mf_text": "当前舆论普遍表示兴奋，但也有人担心价格过高。",
            "mf_text_vector": .....,
            "real_comments": "这绝对是技术突破！只要价格不超过5000我肯定买。",
            "prev_state": [0.7, 0.2, 0.1]
            "user_state": [0.8, 0.1, 0.1], 
            "future_states_ground_truth": [
            [0.85, 0.05, 0.1],
            [0.90, 0.05, 0.05],
            [0.88, 0.07, 0.05]
            ]
        }
        ...
    ]
    """
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 构建 Prompt (Input)
        # 格式：[Profile] + [Context] + [MeanField Summary] + 
        prompt = (
            f"<|im_start|>system\n"
            f"你是一个社交媒体用户。以下是你的个人画像：\n{item['user_profile']}\n"
            f"你的当前状态/立场：{item['user_state']}\n"  # user_state这里要试一下直接的数字还是文本label好
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"当前讨论的话题是：{item['topic']}\n"
            f"当前已有的网友评论情况（舆论风向）：{item['mf_text']}\n"
            f"请根据其他网友的评论情况，推测你可能的情绪、观点和立场，模拟发布一条评论或转发。\n"
            f"要求：符合你的人设，直接输出模拟的内容，不要包含任何分析过程。<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # 2. 构建 Target Text (Output)
        target_text = item['real_comments'] + "<|im_end|>"
        
        # 3. Tokenize
        # 注意：这里我们将 input 和 target 拼在一起训练，利用 mask 区分 loss
        full_text = prompt + target_text
        encodings = self.tokenizer(
            full_text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        
        # 4. 构造 Text Labels (只计算回答部分的 Loss，Prompt 部分设为 -100)
        labels = input_ids.clone()
        # 找到 "assistant\n" 的位置，这之前都 mask 掉
        # 简单处理：你可以计算 prompt 的长度来 mask
        prompt_len = len(self.tokenizer(prompt)['input_ids'])
        labels[:prompt_len] = -100 
        
        # 5. 状态数据 (数值部分)
        # current_state: [3] -> [Pos, Neg, Neu]
        current_state = torch.tensor(item['user_state'], dtype=torch.float32) 
        
        # future_states: [K, 3] -> 未来 K 步的真实分布
        future_states = torch.tensor(item['future_states_ground_truth'], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_state": current_state,
            "future_states": future_states
        }