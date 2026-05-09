import os
import numpy as np

# ====================================== 配置 ======================================
PRED_POSE_TXT = "/home/haoyang22/project/VGGT-Long/exps/dataset_7scenes_chess_seq01_only_images/2026-05-04-14-38-07/camera_poses.txt"
GT_POSE_DIR = "/home/haoyang22/project/VGGT-Long/dataset/7scenes/chess/seq-01"
OUTPUT_DIR = "/home/haoyang22/project/VGGT-Long/eval_results/pose/tum_format"
# =================================================================================

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

def save_tum_trajectory(poses, output_path):
    """保存为TUM轨迹格式：timestamp tx ty tz qx qy qz qw"""
    with open(output_path, "w") as f:
        for i, p in enumerate(poses):
            t = p[:3, 3]
            q = rotation_matrix_to_quaternion(p[:3, :3])
            # TUM格式：timestamp tx ty tz qx qy qz qw
            f.write(f"{i} {t[0]} {t[1]} {t[2]} {q[1]} {q[2]} {q[3]} {q[0]}\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载预测位姿
    print(f"[INFO] 加载预测位姿...")
    pred_poses = []
    with open(PRED_POSE_TXT, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mat = np.array(list(map(float, line.split()))).reshape(4, 4)
            pred_poses.append(mat)
    
    # 加载GT位姿
    print(f"[INFO] 加载GT位姿...")
    gt_poses = []
    pose_files = sorted([f for f in os.listdir(GT_POSE_DIR) if f.endswith(".pose.txt")])
    for f in pose_files:
        mat = np.loadtxt(os.path.join(GT_POSE_DIR, f))
        gt_poses.append(mat)
    
    # 保存为TUM格式
    print(f"[INFO] 保存TUM格式轨迹...")
    save_tum_trajectory(pred_poses, os.path.join(OUTPUT_DIR, "pred.tum"))
    save_tum_trajectory(gt_poses, os.path.join(OUTPUT_DIR, "gt.tum"))
    
    print(f"[INFO] 完成！TUM文件保存在: {OUTPUT_DIR}")
    print(f"\n接下来运行 evo 官方命令：")
    print(f"cd {OUTPUT_DIR}")
    print(f"evo_ape tum gt.tum pred.tum -v --align --save_results results.zip")

if __name__ == "__main__":
    main()