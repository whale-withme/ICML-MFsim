import os
import sys
import glob
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# =====================================================
# 运行模式设置（只改这里）
# =====================================================
RUN_ALL_EVENTS = True       # True: 跑全部事件；False: 只跑一个
TEST_EVENT_INDEX = 1        # RUN_ALL_EVENTS=False 时生效

CHECKPOINT_PATH = "/root/ICML/checkpoints/state_transition_best.pt"
SAVE_DIR = "/root/ICML/data/pre_state_distribution"

# =====================================================
# Path setup
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(current_dir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# =====================================================
# Imports
# =====================================================
from model.state_transition.encoders import build_text_encoder
from model.state_transition.state_transition_net import StateTransitionNet
# from model.state_transition import StateTransitionDataset
from datasets.state_datasets import StateTransitionDataset
from training.train_state_transition import TrainConfig

IDX2STATE = {0: "Positive", 1: "Neutral", 2: "Negative"}
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# Load model
# =====================================================
def load_model(device):
    cfg = TrainConfig()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    encoder = build_text_encoder({
        "type": cfg.encoder_type,
        "model_name": cfg.model_name,
        "output_dim": cfg.text_emb_dim,
        "freeze": True
    })

    model = StateTransitionNet(
        agent_feat_dim=cfg.agent_feat_dim,
        text_emb_dim=cfg.text_emb_dim,
        hidden_dim=cfg.hidden_dim,
        use_layernorm=cfg.use_layernorm
    )

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])

    encoder.to(device).eval()
    model.to(device).eval()

    return encoder, model, cfg

# =====================================================
# Build dataset for ONE event
# =====================================================
def build_event_dataset(cfg, traj_path):
    event_id = os.path.basename(traj_path).replace("_trajectory.csv", "")

    json_path = os.path.join(cfg.event_data_dir, f"{event_id}.json")
    mf_path = os.path.join(cfg.mf_dir, f"{event_id}_mf.csv")

    if not (os.path.exists(json_path) and os.path.exists(mf_path)):
        return None, None

    dataset = StateTransitionDataset(
        trajectory_path=traj_path,
        mf_path=mf_path,
        test_data_path=json_path,
        profile_path=cfg.profile_path,
        encoder_config={
            "type": cfg.encoder_type,
            "model_name": cfg.model_name
        },
        file_config={
            "cluser_user_profile": cfg.profile_path,
            "uid_mapping_path": cfg.uid_mapping_path,
            "cluster_info_path": cfg.cluster_info_path
        },
        batch_size=cfg.num_agents
    )

    return dataset, event_id

# =====================================================
# Predict ONE event
# =====================================================
def predict_event(dataset, event_id, encoder, model, cfg, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    batches = []
    total_mae = 0.0
    total_acc = 0.0

    start_time = time.time()

    for batch_id, batch in tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Event {event_id}",
        leave=False
    ):
        mu_prev = batch["mu_prev"].to(device)
        gt = batch["target_dist"].to(device)
        agent_feats = batch["profile_vecs"].to(device)
        texts = batch["mf_text"]

        tokens = encoder.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        text_emb = encoder(tokens["input_ids"], tokens.get("attention_mask"))
        if isinstance(text_emb, tuple):
            text_emb = text_emb[0]

        mu_pred, probs = model(mu_prev, text_emb, agent_feats)

        mae = F.l1_loss(mu_pred, gt).item()
        acc = (
            torch.argmax(mu_pred, dim=1)
            == torch.argmax(gt, dim=1)
        ).float().mean().item()

        total_mae += mae
        total_acc += acc

        prev_dist = mu_prev[0].cpu().tolist()
        pred_dist = mu_pred[0].cpu().tolist()
        gt_dist   = gt[0].cpu().tolist()

        batch_record = {
            "batch_id": batch_id,
            "global": {
                "processed_count": probs.shape[1],
                "prev_dist": prev_dist,
                "pred_dist": pred_dist,
                "gt_dist": gt_dist,
                "prev_state": IDX2STATE[int(np.argmax(prev_dist))],
                "pred_state": IDX2STATE[int(np.argmax(pred_dist))],
                "gt_state":   IDX2STATE[int(np.argmax(gt_dist))]
            },
            "agents": []
        }

        for agent_id, p in enumerate(probs[0].detach().cpu().numpy()):
            batch_record["agents"].append({
                "agent_id": agent_id,
                "dist": p.tolist(),
                "state": IDX2STATE[int(np.argmax(p))]
            })

        batches.append(batch_record)

    elapsed = time.time() - start_time

    output = {
        "num_batches": len(batches),
        "num_agents": cfg.num_agents,
        "batches": batches,
        "metrics": {
            "avg_mae": total_mae / max(1, len(batches)),
            "avg_acc": total_acc / max(1, len(batches)),
            "elapsed_sec": elapsed
        }
    }

    save_path = os.path.join(SAVE_DIR, f"{event_id}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    return elapsed

# =====================================================
# Main
# =====================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, model, cfg = load_model(device)

    traj_files = sorted(
        glob.glob(os.path.join(cfg.state_trajectory_dir, "*_trajectory.csv"))
    )

    if not RUN_ALL_EVENTS:
        traj_files = [traj_files[TEST_EVENT_INDEX]]

    total_start = time.time()

    for traj_path in tqdm(traj_files, desc="All Events"):
        dataset, event_id = build_event_dataset(cfg, traj_path)
        if dataset is None:
            continue

        predict_event(dataset, event_id, encoder, model, cfg, device)

    print(f"\n✅ All done. Total time: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()