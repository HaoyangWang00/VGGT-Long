# merge_run.py
from loop_utils.sim3utils import merge_ply_files

input_dir = "/home/haoyang22/project/VGGT-Long/exps/kitti_after_vins_images/2026-04-23-16-56-55/pcd"
output_path = input_dir + "/combined_2pcd.ply"

merge_ply_files(input_dir, output_path)