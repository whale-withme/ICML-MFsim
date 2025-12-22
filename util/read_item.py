import os
import json
import glob
import random
from numpy import real
import pandas as pd
from tqdm import tqdm

def load_uid_map(uid_mapping_path):
    df = pd.read_csv(uid_mapping_path, dtype={"uid": str}, low_memory=False).drop_duplicates(subset=["uid"])
    # 处理 cluster_id 字段，强制转 int，无法转的填-1
    def safe_int(x):
        try:
            return int(float(x))
        except Exception:
            return -1
    cluster_ids = df["cluster_id"].apply(safe_int)
    return dict(zip(df["uid"].astype(str), cluster_ids))

def load_cluster_users(cluster_info_path):
    with open(cluster_info_path, "r", encoding="utf-8") as f:
        cluster_list = json.load(f)
    cluster2uids = {}
    for c in cluster_list:
        cid = str(int(c["cluster_id"]))
        cluster2uids[cid] = [str(u["uid"]) for u in c.get("typical_users", [])]
    return cluster2uids

def load_profiles(profile_path):
    uid2profile = {}
    with open(profile_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            uid2profile[str(obj["user_id"])] = obj
    return uid2profile

def uid2profile_text(uid, uid_map, cluster2uids, uid2profile):
    cid = uid_map.get(uid)
    if cid is None:
        # print(f"[DEBUG] uid {uid} 未在uid_map中找到，随机分配cluster")
        cid = random.randint(0, len(cluster2uids) - 1)
        is_random = True
    else:
        is_random = False
    cid = str(int(cid))
    candidates = cluster2uids.get(cid, [])
    # print(f"[DEBUG] uid: {uid} -> cluster_id: {cid} (随机分配: {is_random})，typical_users: {candidates if candidates else '无'}")
    random.shuffle(candidates)

    for u in candidates:
        if u in uid2profile:
            p = uid2profile[u]
            # print(f"[DEBUG] 命中typical_user: {u}，返回profile。")
            return (
                p["stance_nuance"]
                + "，".join(p["expression_style"])
                + p["persona_description"]
            )
    # print(f"[DEBUG] cluster {cid} 的typical_users都未找到profile，uid: {uid}")
    return ""

def build_event_csv(
    event_id,
    traj_path,
    mf_path,
    json_path,
    output_csv,
    uid_map,
    cluster2uids,
    uid2profile,
    batch_size=16,
    k_future=3,
):
    traj_df = pd.read_csv(traj_path)
    mf_df = pd.read_csv(mf_path)
    mf_texts = mf_df["mean_field"].astype(str).tolist()
    topic = mf_df["topic"].sample(1).iloc[0]

    with open(json_path, "r", encoding="utf-8") as f:
        test_users = json.load(f)

    rows = []

    for t in tqdm(range(len(traj_df)), desc=f"Event {event_id}"):
        start_idx = t * batch_size

        for i in range(batch_size):
            user = test_users[(start_idx + i) % len(test_users)]
            uid = str(user.get("uid", ""))
            real_comments = str(user.get("text") or user.get("original_text") or "")


            profile_text = uid2profile_text(
                uid, uid_map, cluster2uids, uid2profile
            )

            # ---------- pre_dist ----------
            if t == 0:
                pre = [0.0, 1.0, 0.0]
            else:
                prev = traj_df.iloc[t - 1]
                pre = [
                    prev["cum_ratio_pos"],
                    prev["cum_ratio_neu"],
                    prev["cum_ratio_neg"],
                ]

            # ---------- future dists ----------
            dist_fields = {}
            for k in range(k_future + 1):
                idx = min(t + k, len(traj_df) - 1)
                row_k = traj_df.iloc[idx]
                dist_fields[f"dist_t{k}_pos"] = row_k["batch_ratio_pos"]
                dist_fields[f"dist_t{k}_neu"] = row_k["batch_ratio_neu"]
                dist_fields[f"dist_t{k}_neg"] = row_k["batch_ratio_neg"]

            if t == 0:
                mf_text = ""
            else:
                mf_text = mf_texts[((t-1) * batch_size + 1) % len(mf_texts)]

            rows.append({
                "time_step": t,
                "topic": topic,
                "uid": uid,
                "real_comments": real_comments,
                "profile_text": profile_text,
                "batch_mf": mf_text,
                "pre_pos": pre[0],
                "pre_neu": pre[1],
                "pre_neg": pre[2],
                **dist_fields,
            })

    pd.DataFrame(rows).to_csv(output_csv, index=False)


# =====================

def build_all_events(
    trajectory_dir,
    mf_dir,
    json_dir,
    output_dir,
    profile_path,
    uid_mapping_path,
    cluster_info_path,
    batch_size=16,
    k_future=3,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 输出目录已创建: {output_dir}")

    print("[INFO] 加载uid映射...")
    uid_map = load_uid_map(uid_mapping_path)
    print(f"[INFO] 加载cluster信息...")
    cluster2uids = load_cluster_users(cluster_info_path)
    print(f"[INFO] 加载用户profile...")
    uid2profile = load_profiles(profile_path)

    traj_files = glob.glob(os.path.join(trajectory_dir, "*_trajectory.csv"))
    print(f"[INFO] 共发现{len(traj_files)}个trajectory文件，开始处理...")

    for idx, traj_path in enumerate(sorted(traj_files)):
        event_id = os.path.basename(traj_path).replace("_trajectory.csv", "")
        mf_path = os.path.join(mf_dir, f"{event_id}_mf.csv")
        json_path = os.path.join(json_dir, f"{event_id}.json")

        if not (os.path.exists(mf_path) and os.path.exists(json_path)):
            print(f"[WARN] 缺少mf或json，跳过: {event_id}")
            continue

        out_csv = os.path.join(output_dir, f"{event_id}.csv")
        print(f"[INFO] ({idx+1}/{len(traj_files)}) 处理事件: {event_id}")

        build_event_csv(
            event_id,
            traj_path,
            mf_path,
            json_path,
            out_csv,
            uid_map,
            cluster2uids,
            uid2profile,
            batch_size,
            k_future,
        )
        print(f"[INFO] 已保存: {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_dir', type=str, default="/root/ICML/data/test_state_distribution", help="trajectory目录")
    parser.add_argument('--mf_dir', type=str, default="/root/ICML/data/test_mf", help="mean field目录")
    parser.add_argument('--json_dir', type=str, default="/root/Mean-Field-LLM/mf_llm/data/rumdect/Weibo/test", help="原始json目录")
    parser.add_argument('--output_dir', type=str, default="/root/ICML/data/pre_policy", help="输出csv目录")
    parser.add_argument('--profile_path', type=str, default="/root/ICML/data/profile/cluster_core_user_profile.jsonl", help="用户profile路径")
    parser.add_argument('--uid_mapping_path', type=str, default="/root/ICML/data/profile/user_clusters_map.csv", help="uid映射路径")
    parser.add_argument('--cluster_info_path', type=str, default="/root/ICML/data/profile/cluster_details.json", help="cluster info路径")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--k_future', type=int, default=10)
    args = parser.parse_args()

    print("[INFO] 参数解析完毕，开始批量处理事件数据...")
    build_all_events(
        trajectory_dir=args.trajectory_dir,
        mf_dir=args.mf_dir,
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        profile_path=args.profile_path,
        uid_mapping_path=args.uid_mapping_path,
        cluster_info_path=args.cluster_info_path,
        batch_size=args.batch_size,
        k_future=args.k_future,
    )
    print("[INFO] 全部处理完成！")
    