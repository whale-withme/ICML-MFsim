import os
import sys
import torch
import json
import pandas as pd
import argparse
from torch.utils.data import random_split
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- 导入你提供的自定义模块 ---
# 确保这三个文件在同一目录下，或者在PYTHONPATH中
from model.policyLLM.predict_trainer import PolicyTrainer, SaveMLPCallback
from model.policyLLM.lora_base_model import PolicyMLPModel
from datasets.policy_datasets import PolicyModelDataset

def load_raw_data(data_dir_or_file):
    """
    支持传入单个csv/json文件或目录（目录下全是csv）。
    处理CSV数据并将其转换为PolicyModelDataset期望的格式。
    """
    if data_dir_or_file and os.path.exists(data_dir_or_file):
        if os.path.isdir(data_dir_or_file):
            # 目录模式，合并所有csv
            all_records = []
            for fname in sorted(os.listdir(data_dir_or_file)):
                if fname.endswith(".csv"):
                    fpath = os.path.join(data_dir_or_file, fname)
                    df = pd.read_csv(fpath)
                    processed_records = _process_csv_data(df)
                    all_records.extend(processed_records)
            print(f"Loaded {len(all_records)} samples from {data_dir_or_file}")
            return all_records
        elif data_dir_or_file.endswith(".csv"):
            df = pd.read_csv(data_dir_or_file)
            processed_records = _process_csv_data(df)
            print(f"Loaded {len(processed_records)} samples from {data_dir_or_file}")
            return processed_records
        else:
            with open(data_dir_or_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    else:
        print("No data path provided or file not found. Generating MOCK data for testing...")
        return _generate_mock_data()

import pandas as pd

def _process_csv_data(df):
    """
    处理CSV数据，转换为PolicyModelDataset期望的格式。

    映射逻辑：
    - user_profile: profile_text
    - user_state: [pre_pos, pre_neg, pre_neu] (Pos, Neg, Neu 顺序)
    - topic: topic
    - mf_text: batch_mf
    - real_comments: topic (CSV中topic列包含了具体文本)
    - future_states_ground_truth: 提取 dist_t0 到 dist_t10 的状态序列
    """
    processed_records = []

    for _, row in df.iterrows():
        # 1. 提取当前用户状态 [Pos, Neg, Neu]
        # 注意 CSV 列名是 pre_pos, pre_neu, pre_neg，需要按顺序重排
        user_state = [
            float(row['pre_pos']),
            float(row['pre_neg']),
            float(row['pre_neu'])
        ]

        # 2. 提取未来 K 步的状态序列 (从 t0 到 t10)
        future_states = []
        for k in range(11):
            pos_col = f'dist_t{k}_pos'
            neg_col = f'dist_t{k}_neg'
            neu_col = f'dist_t{k}_neu'
            
            # 检查列是否存在
            if pos_col in row:
                future_states.append([
                    float(row[pos_col]),
                    float(row[neg_col]),
                    float(row[neu_col])
                ])

        # 3. 构建 Dataset 所需的记录
        record = {
            "user_profile": str(row['profile_text']),
            "user_state": user_state,
            "topic": str(row['topic']),
            "mf_text": str(row['batch_mf']) if pd.notna(row['batch_mf']) else "",
            "real_comments": str(row['real_comments']), 
            "future_states_ground_truth": future_states
        }

        processed_records.append(record)

    return processed_records

# 使用示例
# df = pd.read_csv('10031994215.csv')
# data = _process_csv_data(df)
# dataset = PolicyModelDataset(data, tokenizer)
def _generate_mock_data():
    """
    生成用于测试的模拟数据。
    """
    import random

    mock_data = []
    topics = ["科技产品发布", "体育赛事", "政治新闻", "娱乐八卦", "经济政策"]

    for i in range(10):  # 生成10条模拟数据
        # 随机生成状态分布
        pos = random.uniform(0.1, 0.8)
        neg = random.uniform(0.1, 0.8)
        neu = random.uniform(0.1, 0.8)
        total = pos + neg + neu

        user_state = [pos/total, neg/total, neu/total]

        # 生成未来状态
        future_states = []
        for _ in range(5):
            import copy
            future_state = copy.deepcopy(user_state)
            # 添加小幅变化
            for j in range(3):
                future_state[j] = max(0.01, future_state[j] + random.uniform(-0.1, 0.1))
            # 重新归一化
            f_total = sum(future_state)
            future_states.append([s/f_total for s in future_state])

        record = {
            "user_profile": f"模拟用户{i+1}，关注科技和时事",
            "user_state": user_state,
            "topic": random.choice(topics),
            "mf_text": "当前网友对此话题讨论热烈，观点多样",
            "real_comments": f"这是模拟用户{i+1}的评论内容",
            "future_states_ground_truth": future_states
        }
        mock_data.append(record)

    return mock_data

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/root/qwen2-1.5B")
    parser.add_argument("--data_path", type=str, default="/root/ICML/data/pre_policy")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_policy/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_coeff", type=float, default=0.4)
    parser.add_argument("--k_steps", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # 1. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备数据并划分验证集
    raw_data = load_raw_data(args.data_path) # 假设该函数已定义
    full_dataset = PolicyModelDataset(raw_data, tokenizer, max_length=1024, k_steps=args.k_steps)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 3. 初始化模型
    model = PolicyMLPModel(args.model_name, k_steps=args.k_steps, lora_rank=16)

    # --- 断点续训逻辑：手动恢复 MLP 头 ---
    if args.resume:
        checkpoints = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), 
                             key=lambda x: int(x.split('-')[-1]))
        if checkpoints:
            last_ckpt = checkpoints[-1]
            mlp_path = os.path.join(last_ckpt, "prediction_heads.pt")
            if os.path.exists(mlp_path):
                print(f"[INFO] Resuming MLP heads from {mlp_path}")
                model.prediction_heads.load_state_dict(torch.load(mlp_path, map_location='cpu'))

    # 4. 配置训练参数 (每100步保存/评估，保留最优)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps", 
        eval_steps=100,
        # --------------------------------------------------------
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        bf16=True,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    # 5. 初始化 Trainer
    trainer = PolicyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        lambda_coeff=args.lambda_coeff,
        callbacks=[SaveMLPCallback()] # 注入回调以保存 MLP
    )

    # 6. 开始训练
    print("[INFO] Starting training...")
    # 如果开启了 resume，Trainer 会自动寻找最新的 checkpoint-XXX
    trainer.train(resume_from_checkpoint=args.resume)

    # 7. 最终保存 (保存的是 load_best_model_at_end 加载后的最优版本)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[INFO] Training and saving finished.")

if __name__ == "__main__":
    main()