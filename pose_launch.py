import os
import numpy as np
import subprocess
from datetime import datetime

# ====================================== 【唯一需要改的配置区域】 ======================================
# 1. 模式选择
USE_KEYFRAMES = True  # True=关键帧模式, False=全帧模式

# 2. 路径配置
PRED_POSE_TXT = "/home/haoyang22/project/VGGT-Long/exps/dataset_KF_7scenes_chess01/2026-05-04-11-56-40/camera_poses.txt"
GT_POSE_DIR = "/home/haoyang22/project/VGGT-Long/dataset/7scenes/chess/seq-01"
KEYFRAME_INDICES_TXT = "/home/haoyang22/project/VGGT-Long/eval_results/pose/keyframe_index/KF_7scenes_chess01.txt"  # 仅关键帧模式用

# 3. 输出配置
OUTPUT_ROOT = "/home/haoyang22/project/VGGT-Long/eval_results/pose"
SCENE_ID = "chess_seq-01"
# ====================================================================================================

def rotation_matrix_to_quaternion(matrix):
    R = np.array(matrix, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def save_tum_trajectory(poses, output_path, timestamps=None):
    with open(output_path, "w") as f:
        for i, p in enumerate(poses):
            t = p[:3, 3]
            q = rotation_matrix_to_quaternion(p[:3, :3])
            ts = timestamps[i] if timestamps is not None else i
            f.write(f"{ts} {t[0]} {t[1]} {t[2]} {q[1]} {q[2]} {q[3]} {q[0]}\n")

def load_keyframe_indices(txt_path):
    indices = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.append(int(line))
    return indices

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "keyframe" if USE_KEYFRAMES else "full"
    output_dir = os.path.join(OUTPUT_ROOT, f"{SCENE_ID}_{mode_str}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "result.txt")

    print(f"\n{'='*60}")
    print(f"VGGT-Long 位姿评估 (ATE) - 最终优化版")
    print(f"{'='*60}")
    print(f"  模式: {'关键帧模式' if USE_KEYFRAMES else '全帧模式'}")
    print(f"  场景: {SCENE_ID}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}\n")

    # 加载预测位姿
    print(f"[1/5] 加载预测位姿...")
    pred_poses = []
    with open(PRED_POSE_TXT, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mat = np.array(list(map(float, line.split()))).reshape(4, 4)
            pred_poses.append(mat)
    print(f"      完成: {len(pred_poses)} 帧")

    # 加载GT位姿
    print(f"[2/5] 加载GT位姿...")
    gt_poses = []
    gt_frame_indices = []
    pose_files = sorted([f for f in os.listdir(GT_POSE_DIR) if f.endswith(".pose.txt")])
    for f in pose_files:
        try:
            idx = int(f.split("-")[1].split(".")[0])
        except:
            continue
        mat = np.loadtxt(os.path.join(GT_POSE_DIR, f))
        gt_frame_indices.append(idx)
        gt_poses.append(mat)
    print(f"      完成: {len(gt_poses)} 帧")

    # 关键帧处理
    kf_indices = None
    if USE_KEYFRAMES:
        print(f"[3/5] 加载并筛选关键帧...")
        if not os.path.exists(KEYFRAME_INDICES_TXT):
            raise FileNotFoundError(f"[ERROR] 关键帧列表不存在")
        kf_indices = load_keyframe_indices(KEYFRAME_INDICES_TXT)
        
        gt_kf_mask = np.isin(gt_frame_indices, kf_indices)
        gt_poses = [gt_poses[i] for i in range(len(gt_poses)) if gt_kf_mask[i]]
        gt_frame_indices = [gt_frame_indices[i] for i in range(len(gt_frame_indices)) if gt_kf_mask[i]]
        print(f"      完成: 关键帧 {len(kf_indices)} 个")
    else:
        print(f"[3/5] 全帧模式")

    # 转换TUM格式
    print(f"[4/5] 转换标准格式...")
    pred_tum_path = os.path.join(output_dir, "pred.tum")
    gt_tum_path = os.path.join(output_dir, "gt.tum")
    if USE_KEYFRAMES:
        save_tum_trajectory(pred_poses, pred_tum_path, timestamps=kf_indices)
        save_tum_trajectory(gt_poses, gt_tum_path, timestamps=gt_frame_indices)
    else:
        save_tum_trajectory(pred_poses, pred_tum_path)
        save_tum_trajectory(gt_poses, gt_tum_path, timestamps=gt_frame_indices)

    # 运行evo，捕获输出，不生成ZIP
    print(f"[5/5] 计算官方ATE...")
    evo_cmd = ["evo_ape", "tum", "gt.tum", "pred.tum", "-v", "--align"]
    result = subprocess.run(evo_cmd, cwd=output_dir, capture_output=True, text=True)

    # 打印并保存结果
    print("\n" + result.stdout)
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    print(f"{'='*60}")
    print(f"✅ 评估完成！")
    print(f"📄 结果已保存至: result.txt")
    print(f"📂 目录: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)