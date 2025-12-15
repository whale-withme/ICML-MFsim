import json
import logging
import pandas as pd
import torch
import random
import os
from torch.utils.data import Dataset
from typing import Dict, List

os.environ["TOKENIZERS_PARALLELISM"] = "false" #关闭tokenizer内部多线程

logging.basicConfig(
    # level=logging.INFO,  # 或 DEBUG/ERROR
    # format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# 假设 encoders 在 models 目录下
from model.state_transition.encoders import build_text_encoder

class StateTransitionDataset(Dataset):
    def __init__(
        self,
        trajectory_path: str,       # 10031994215_trajectory.csv (Ground Truth 分布)
        mf_path: str,               # 4264473811_mf.csv (舆论环境)
        test_data_path: str,        # 4264473811.json (真实用户流, 包含 uid)
        profile_path: str,          # cluster_core_user_profile.jsonl (核心用户池)
        encoder_config: dict,
        file_config: dict,
        batch_size: int = 16
    ):
        """
        self.cluser_user_profile: 核心用户的画像
        self.uid_map: id -> cluster id 
        self.cluster_info: 保存聚类中的用户信息
        """
        self.batch_size = batch_size
        self.cluser_user_profile = file_config['cluser_user_profile']
        self.cluster_info = file_config['cluster_info_path']
        self.text_encoder = build_text_encoder(encoder_config)

        # 1. 加载grouond truth状态分布
        self.traj_df = pd.read_csv(trajectory_path)
        
        # 2. 加载 MF Context
        mf_df = pd.read_csv(mf_path)
        self.mf_texts = mf_df['mean_field'].tolist()
        
        # 3. 加载测试集用户流 (Test Users)
        # 这里我们只需要 uid，按顺序排好
        with open(test_data_path, 'r', encoding='utf-8') as f:
            try:
                self.test_users = json.load(f) # 假设是 List[Dict]
            except json.JSONDecodeError:
                # 兼容 JSONL 格式
                f.seek(0)
                self.test_users = [json.loads(line) for line in f]
        
        print(f"✅ 加载测试用户: {len(self.test_users)} 人")

        # 4. 加载并向量化核心用户画像 (构建聚类池)
        # TODO: 已经做了聚类，这里应该是映射
        print("正在构建核心用户聚类池...")
        df = pd.read_csv(file_config['uid_mapping_path'])
        df.drop_duplicates(subset=['uid'])
        self.uid_map = dict(zip(df['uid'], df['cluster_id']))
        

    def _uid2profile(self, uid):
        """
        uid -> cluster id -> random cluster user profile，也避免了某些类中的用户信息不全的问题
        1. 根据uid查找其cluster_id。
        2. 从cluster_info文件中获取该类所有用户uid。
        3. 随机打乱这些uid，依次在画像文件中查找，找到第一个有画像的用户就返回。
        """
        profile_path = self.cluser_user_profile
        cluster_info_path = self.cluster_info
        cluster_id = self.uid_map.get(uid)
        logger.info(f"[uid2profile] 输入uid: {uid}, 查到cluster_id: {cluster_id}")
        if cluster_id is None:
            logger.error(f"[uid2profile] uid {uid} 没有找到对应的 cluster_id")
            cluster_id = random.randint(0, 19)
        
        # 1. 获取该类所有用户uid
        cluster_users = []
        if cluster_info_path and os.path.exists(cluster_info_path):
            with open(cluster_info_path, 'r', encoding='utf-8') as f:
                cluster_list = json.load(f)
                for c in cluster_list:
                    if str(c.get('cluster_id')) == str(int(cluster_id)):
                        cluster_users = [str(u['uid']) for u in c.get('typical_users', [])]
                        logger.info(f"[uid2profile] cluster_id: {cluster_id} typical_users: {cluster_users}")
                        break
        if not cluster_users:
            logger.error(f"[uid2profile] cluster_id {cluster_id} 没有 typical_users")
            return None

        # 2. 读取画像文件，建立uid->profile映射
        uid2profile = {}
        if profile_path and os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    uid_profile = str(obj.get('user_id'))
                    uid2profile[uid_profile] = obj
                    line_count += 1
                # logger.info(f"[uid2profile] 画像文件共加载 {line_count} 条, uid2profile keys样例: {list(uid2profile.keys())[:5]}")

        # 3. 随机遍历用户，找到第一个有画像的
        random.shuffle(cluster_users)
        logger.info(f"[uid2profile] cluster_users 随机顺序: {cluster_users}")
        for u in cluster_users:
            if u in uid2profile:
                logger.info(f"[uid2profile] 命中: {u}, 返回profile")
                return uid2profile[u]

        logger.error(f"[uid2profile] cluster_id {cluster_id} 所有 typical_users 都没有画像")
        return None
    
    def _text2vector(self, input):
        """文本转换向量"""
        inputs = self.text_encoder.tokenizer(
            input, return_tensors='pt', padding='max_length', truncation=True, max_length=256
        )
        with torch.no_grad():
            vec = self.text_encoder(
                inputs['input_ids'], inputs['attention_mask']
            )[0]

        return vec

    def __len__(self):
        return len(self.traj_df)

    def __getitem__(self, idx):
        """
        datasets item:
        {
            "mu_prev": Tensor(3,),
            "mf_text": str,
            "profile_vecs": Tensor(batch_size, hidden_dim),
            "target_dist": Tensor(3,)
        }
        """
        # 1. 获取当前 Batch 对应的测试用户片段
        # 逻辑：Trajectory 的每一行代表一个 Time Step，包含 batch_size 个新用户行为
        start_idx = idx * self.batch_size
        
        batch_profiles = []
        
        for i in range(self.batch_size):
            curr_idx = (start_idx + i) % len(self.test_users)
            user_item = self.test_users[curr_idx]
            uid = str(user_item.get('uid', ''))

            text_profile = self._uid2profile(uid)
            print(uid)
            if text_profile is not None:
                vector_profile = self._text2vector(
                    text_profile['stance_nuance']
                    + '，'.join(text_profile['expression_style'])
                    + text_profile['persona_description']
                )
                batch_profiles.append(vector_profile)
            else:
                logger.error("user profile is None")
        # 堆叠画像向量 (16, Hidden_Dim)
        profile_vecs_tensor = torch.stack(batch_profiles)

        # 2. 处理状态分布输入
        row = self.traj_df.iloc[idx]
        target_dist = torch.tensor([
            row['batch_ratio_pos'], row['batch_ratio_neu'], row['batch_ratio_neg']
        ], dtype=torch.float32)

        # 上一时刻状态
        if idx == 0:
            mu_prev = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        else:
            prev_row = self.traj_df.iloc[idx - 1]
            mu_prev = torch.tensor([
                prev_row['cum_ratio_pos'], prev_row['cum_ratio_neu'], prev_row['cum_ratio_neg']
            ], dtype=torch.float32)
            
        mf_text = str(self.mf_texts[(idx * self.batch_size + 1) % len(self.mf_texts)])

        return {
            "mu_prev": mu_prev,
            "mf_text": mf_text,
            "profile_vecs": profile_vecs_tensor, # (16, 768)
            "target_dist": target_dist
        }