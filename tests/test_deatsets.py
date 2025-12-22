import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from datasets.state_datasets import StateTransitionDataset


def test_dataset_basic():
    print("=== [Test 1] Dataset 基本功能测试 ===")

    encoder_config = {
        "type": "bert",
        "model_name": "bert-base-chinese",
        "output_dim": 768,
        "freeze": True
    }

    file_config = {
        "uid_mapping_path": "/root/ICML/data/profile/user_clusters_map.csv",
        "cluser_user_profile": "/root/ICML/data/profile/cluster_core_user_profile.jsonl",
        "cluster_info_path": "/root/ICML/data/profile/cluster_details.json"
    }

    dataset = StateTransitionDataset(
        trajectory_path="/root/ICML/data/test_state_distribution/10031994215_trajectory.csv",
        mf_path="/root/ICML/data/test_mf/10031994215_mf.csv",
        test_data_path="/root/Mean-Field-LLM/mf_llm/data/rumdect/Weibo/test/10031994215.json",
        profile_path=file_config["cluser_user_profile"],
        encoder_config=encoder_config,
        file_config=file_config,
        batch_size=16
    )

    print(f"✅ Dataset 初始化成功，总长度: {len(dataset)}")

    print("\n--- 取第 0 个样本 ---")
    item = dataset[0]

    # ====== Key 检查 ======
    expected_keys = {"mu_prev", "mf_text", "profile_vecs", "target_dist"}
    print("Keys:", item.keys())
    assert set(item.keys()) == expected_keys, "❌ 返回字段不匹配"

    # ====== 形状检查 ======
    print("mu_prev:", item["mu_prev"].shape)
    print("target_dist:", item["target_dist"].shape)
    print("profile_vecs:", item["profile_vecs"].shape)

    assert item["mu_prev"].shape == (3,)
    assert item["target_dist"].shape == (3,)
    assert item["profile_vecs"].ndim == 2
    assert item["profile_vecs"].shape[1] == 768

    # ====== NaN / 全 0 检查 ======
    if torch.isnan(item["profile_vecs"]).any():
        print("❌ profile_vecs 含 NaN")
    elif torch.sum(item["profile_vecs"]) == 0:
        print("⚠️ profile_vecs 全 0，说明 uid → profile 映射失败")
    else:
        print("✅ profile_vecs 数值正常")

    print("MF Text Preview:", item["mf_text"][:50])
    print("✅ 单样本测试通过")


def test_dataloader():
    print("\n=== [Test 2] DataLoader 测试 ===")

    encoder_config = {
        "type": "bert",
        "model_name": "bert-base-chinese",
        "output_dim": 768,
        "freeze": True
    }

    file_config = {
        "uid_mapping_path": "/root/ICML/data/profile/user_clusters_map.csv",
        "cluser_user_profile": "/root/ICML/data/profile/cluster_core_user_profile.jsonl",
        "cluster_info_path": "/root/ICML/data/profile/cluster_details.json"
    }

    dataset = StateTransitionDataset(
        trajectory_path="/root/ICML/data/test_state_distribution/10031994215_trajectory.csv",
        mf_path="/root/ICML/data/test_mf/10031994215_mf.csv",
        test_data_path="/root/Mean-Field-LLM/mf_llm/data/rumdect/Weibo/test/10031994215.json",
        profile_path=file_config["cluser_user_profile"],
        encoder_config=encoder_config,
        file_config=file_config,
        batch_size=16
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    batch = next(iter(loader))

    print("Batch keys:", batch.keys())
    print("mu_prev:", batch["mu_prev"].shape)         # (4, 3)
    print("target_dist:", batch["target_dist"].shape) # (4, 3)
    print("profile_vecs:", batch["profile_vecs"].shape)
    # 期望: (4, 16, 768)

    assert batch["profile_vecs"].shape[1:] == (16, 768)
    print("✅ DataLoader 测试通过")


if __name__ == "__main__":
    test_dataset_basic()
    test_dataloader()
