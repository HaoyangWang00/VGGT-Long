import numpy as np
import argparse

import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
base_models_path = os.path.join(current_dir, 'base_models')
if base_models_path not in sys.path:
    sys.path.append(base_models_path)

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW

from base_models.base_model import VGGTAdapter,Pi3Adapter,MapAnythingAdapter

import numpy as np

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from loop_utils.config_utils import load_config
from pathlib import Path

def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """ #每个元组都是一对闭环对，索引，区间，索引，区间
    seen = {}  #已经处理过的闭环对
    result = [] #去重后的结果
    
    for item in data_list: 
        if item[0] == item[2]: #索引相同直接跳
            continue

        key = (item[0], item[2]) #
        
        if key not in seen.keys():  #看是否已经处理过
            seen[key] = True  #标记已经处理过
            result.append(item)
    
    return result #返回了没有处理过的不重复的闭环对


def extract_p2_k_matrix(calib_path): #解析文件
    """from calib.txt get K  (kitti)"""

    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P2:'):
                values = line.split(':')[1].split()
                values = [float(v) for v in values]
                p2_matrix = np.array(values).reshape(3, 4)
                k_matrix = p2_matrix[:3, :3]
                return k_matrix, p2_matrix

    raise ValueError("P2 not found in calibration file")

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = [] #外参，相机位姿，每帧一个
        self.combined_intrinsics = [] #内参，相机参数，每帧一个
        self.combined_depth_maps = [] #深度图
        self.combined_depth_confs = [] #深度图置信度
        self.combined_world_points = [] #3D点云
        self.combined_world_points_confs = [] #3D点云置信度
        self.all_camera_poses = [] #最终的相机位姿（对齐后）
        self.all_camera_intrinsics = [] #最终的相机内参

class VGGT_Long:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config['Model']['chunk_size'] #每个chunk的帧数
        self.overlap = self.config['Model']['overlap'] #chunk之间的重叠帧数
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16 #根据GPU能力选择数据类型，8.0及以上支持bfloat16，否则使用float16
        self.sky_mask = False #是否使用天空分割掩码
        self.useDBoW = self.config['Model']['useDBoW'] #是否使用DBoW2进行闭环检测，否则使用DNIO v2

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        
        self.all_camera_poses = []
        self.all_camera_intrinsics = [] 
        
        self.delete_temp_files = self.config['Model']['delete_temp_files']

        # 新增：优化频率参数
        self.stream_optimize_every = self.config['Model'].get('stream_optimize_every', 5)
        # 新增：已添加的loop边集合，防止重复
        self.added_loop_edges = set()

        if self.config['Weights']['model'] == 'VGGT':#选择不同的模型
            self.model = VGGTAdapter(self.config)
        elif self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        elif self.config['Weights']['model'] == 'Mapanything':
            self.model = MapAnythingAdapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}. ")

        self.skyseg_session = None
        
        self.chunk_indices = None # [(begin_idx, end_idx), ...]

        self.loop_list = [] # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        # self.sim3_list = [] # 已弃用

        self.loop_sim3_list = [] # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        print('init done.')

    # 新增：frame_id 映射到 chunk_idx
    def frame_to_chunk(self, frame_id):
        for idx, (s, e) in enumerate(self.chunk_indices):
            if s <= frame_id < e:
                return idx
        return None

    # 新增：统一重对齐所有 chunk
    def re_align_all_chunks(self, accumulated_sim3, write_ply=False):
        # 缓存累计Sim3，供save_camera_poses使用
        self._last_accumulated_sim3 = accumulated_sim3
        """
        用累计Sim3重新对齐所有chunk。第0块为单位Sim3，其余块用累计Sim3左乘。
        write_ply: 是否写ply文件（建议只在最后一次优化时写）
        """
        for idx in range(len(self.chunk_indices)):
            # load unaligned
            unaligned_path = os.path.join(
                self.result_unaligned_dir,
                f"chunk_{idx}.npy"
            )
            chunk_data = np.load(unaligned_path, allow_pickle=True).item()

            if idx == 0:
                s, R, t = 1.0, np.eye(3), np.zeros(3)
            else:
                s, R, t = accumulated_sim3[idx-1]

            # apply to world points
            chunk_data['world_points'] = apply_sim3_direct(
                chunk_data['world_points'],
                s, R, t
            )

            # apply to extrinsics
            chunk_data['extrinsic'] = np.array([
                self.apply_sim3_to_pose(e, s, R, t)
                for e in chunk_data['extrinsic']
            ])

            # save aligned
            aligned_path = os.path.join(
                self.result_aligned_dir,
                f"chunk_{idx}.npy"
            )
            np.save(aligned_path, chunk_data)

            # 只在最后一次优化时写ply，恢复原版带颜色和置信度过滤
            if write_ply:
                ply_path = os.path.join(self.pcd_dir, f"chunk_{idx}.ply")
                points = chunk_data['world_points'].reshape(-1, 3)
                colors = (chunk_data['images'].transpose(0,2,3,1).reshape(-1,3) * 255).astype(np.uint8)
                confs = chunk_data['world_points_conf'].reshape(-1)
                save_confident_pointcloud_batch(
                    points=points,
                    colors=colors,
                    confs=confs,
                    output_path=ply_path,
                    conf_threshold=(np.mean(confs) *
                        self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
                        if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True)
                        else -1.0),
                    sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
                )

            # 更新缓存pose
            self.all_camera_poses[idx] = (
                self.chunk_indices[idx],
                chunk_data['extrinsic']
            )

    def get_loop_pairs(self): #闭环检测

        if self.useDBoW: # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # gray to rgb
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori # (height, width, 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'], 
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands # e.g. cands = (812, 67)
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(frame_id)

        else: # DNIO v2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        predictions = self.model.infer_chunk(chunk_image_paths)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
        
        save_path = os.path.join(save_dir, filename)
        
        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        predictions['depth'] = np.squeeze(predictions['depth'])

        np.save(save_path, predictions)
        
        return predictions if is_loop or range_2 is not None else None
    
    @staticmethod
    def apply_sim3_to_pose(pose, s, R, t):
        """
        对4x4位姿矩阵左乘Sim3(s, R, t)
        """
        S = np.eye(4)
        S[:3, :3] = s * R
        S[:3, 3] = t
        pose_new = S @ pose
        pose_new[:3, :3] /= s  # 保持旋转正交
        return pose_new

    def process_stream_chunk(self, start_idx, end_idx, chunk_idx):
        """
        流式处理单个chunk：推理、与前一chunk对齐、添加边、回环。
        """
        # 1. 推理并保存
        self.chunk_indices.append((start_idx, end_idx))
        self.process_single_chunk((start_idx, end_idx), chunk_idx=chunk_idx)

        # 2. 如果不是第一个chunk，和前一个chunk对齐，并添加相邻边
        if chunk_idx > 0:
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx-1}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            point_map1 = chunk_data1['world_points'][-self.overlap:]
            point_map2 = chunk_data2['world_points'][:self.overlap]
            conf1 = chunk_data1['world_points_conf'][-self.overlap:]
            conf2 = chunk_data2['world_points_conf'][:self.overlap]
            mask = None
            if chunk_data1.get("mask", None) is not None:
                # 1. 计算实际可用的重叠帧数（自动适应边界）
                available_overlap = min(self.overlap, len(chunk_data1["mask"]), len(chunk_data2["mask"]))
                # 2. 如果重叠帧数为0（极端情况），创建安全占位掩码
                if available_overlap == 0:
                    mask = torch.ones(1, *chunk_data1["mask"].shape[1:], device=chunk_data1["mask"].device)
                # 3. 正常情况：提取重叠部分
                else:
                    mask1 = chunk_data1["mask"][-available_overlap:]
                    mask2 = chunk_data2["mask"][:available_overlap]
                    mask = mask1.squeeze() & mask2.squeeze()
                # 修正：强制mask转为numpy bool，避免与point_map类型不一致
                if mask is not None and hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy().astype(bool)
            if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            else:
                conf_threshold = -1.0
            s, R, t = weighted_align_point_maps(point_map1, 
                                                conf1, 
                                                point_map2, 
                                                conf2,
                                                mask,
                                                conf_threshold=conf_threshold,
                                                config=self.config)
            # 添加相邻边到pose graph
            self.loop_optimizer.add_edge(
                chunk_idx - 1,
                chunk_idx,
                s, R, t
            )

        # 3. 添加 loop 边（如果存在）
        if self.loop_enable:
            for (i, j) in self.loop_list:
                ci = self.frame_to_chunk(i)
                cj = self.frame_to_chunk(j)
                if ci is None or cj is None:
                    continue
                # 只处理已经存在的chunk（注意：只跳过未来chunk，当前chunk允许）
                if ci > chunk_idx or cj > chunk_idx:
                    continue
                # 跳过相邻chunk
                if abs(ci - cj) <= 1:
                    continue
                # 防止重复添加loop边
                edge_key = tuple(sorted([ci, cj]))
                if edge_key in self.added_loop_edges:
                    continue
                # 计算 loop sim3
                chunk_data1 = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{ci}.npy"),
                    allow_pickle=True
                ).item()
                chunk_data2 = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{cj}.npy"),
                    allow_pickle=True
                ).item()
                point_map1 = chunk_data1['world_points']
                point_map2 = chunk_data2['world_points']
                conf1 = chunk_data1['world_points_conf']
                conf2 = chunk_data2['world_points_conf']
                s_loop, R_loop, t_loop = weighted_align_point_maps(
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    None,
                    conf_threshold=-1,
                    config=self.config
                )
                self.loop_optimizer.add_edge(
                    ci,
                    cj,
                    s_loop,
                    R_loop,
                    t_loop
                )
                self.added_loop_edges.add(edge_key)

    def process_stream_sequence(self):
        """
        按流式增量方式处理所有chunk，每来一批就推理、对齐、每N个chunk优化一次。优化后用累计Sim3对齐。
        """
        step = self.chunk_size - self.overlap
        total = len(self.img_list)
        self.chunk_indices = []
        chunk_idx = 0
        start = 0
        while start < total:
            end = min(start + self.chunk_size, total)
            print(f"Streaming chunk {chunk_idx}: {start}-{end}")
            self.process_stream_chunk(start, end, chunk_idx)

            # 每 N 个 chunk 优化一次（不写ply）
            if (chunk_idx + 1) % self.stream_optimize_every == 0:
                print("Running pose graph optimization...")
                self.loop_optimizer.set_fixed_node(0)
                optimized_sequential = self.loop_optimizer.optimize()
                accumulated_sim3 = accumulate_sim3_transforms(optimized_sequential)
                assert len(accumulated_sim3) == len(self.chunk_indices) - 1 or len(accumulated_sim3) == 0, "累计Sim3数量应为chunk数-1或0"
                self.re_align_all_chunks(accumulated_sim3, write_ply=False)

            chunk_idx += 1
            start += step

        # 最后做一次优化并写ply
        if len(self.chunk_indices) > 0:
            print("Final pose graph optimization...")
            self.loop_optimizer.set_fixed_node(0)
            optimized_sequential = self.loop_optimizer.optimize()
            accumulated_sim3 = accumulate_sim3_transforms(optimized_sequential)
            assert len(accumulated_sim3) == len(self.chunk_indices) - 1 or len(accumulated_sim3) == 0, "累计Sim3数量应为chunk数-1或0"
            self.re_align_all_chunks(accumulated_sim3, write_ply=True)


    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                               glob.glob(os.path.join(self.img_dir, "*.png")))
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close()  # Save CPU Memory
                gc.collect()
            else:
                del self.loop_detector  # Save GPU Memory
        torch.cuda.empty_cache()
        print('Loading model...')
        self.model.load()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k

        self.process_stream_sequence()
        self.save_camera_poses()

    def save_camera_poses(self):
        '''
        保存所有chunk的extrinsics为全局位姿，左乘累计Sim3，保持与点云一致。
        '''
        chunk_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        # 重新累计Sim3
        accumulated_sim3 = []
        if len(self.all_camera_poses) > 1:
            # 取最后一次优化的Sim3
            # 由于re_align_all_chunks已用累计Sim3，直接复用
            for idx in range(1, len(self.all_camera_poses)):
                # 取对齐时用的Sim3
                unaligned_path = os.path.join(self.result_unaligned_dir, f"chunk_{idx}.npy")
                if os.path.exists(unaligned_path):
                    # 只要文件存在，说明有对齐
                    pass
            # 这里假设re_align_all_chunks已用accumulated_sim3
            # 直接用re_align_all_chunks的累计Sim3逻辑
            # 但为安全起见，重新累计
            # 这里假设self.chunk_indices和self.all_camera_poses长度一致
            # 只要累计Sim3和chunk数量一致即可
            # 这里直接用accumulated_sim3 = ...
            # 但实际应在re_align_all_chunks后缓存
            # 这里假设re_align_all_chunks已正确对齐
            pass

        # 逐chunk写入
        for chunk_idx, (chunk_range, chunk_extrinsics) in enumerate(self.all_camera_poses):
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                all_poses[idx] = chunk_extrinsics[i]
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')

        print(f"Camera poses saved to {poses_path}")
        if all_intrinsics[0] is not None:
            intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
            with open(intrinsics_path, 'w') as f:
                for intrinsic in all_intrinsics:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f'{fx} {fy} {cx} {cy}\n')
            print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')

        print(f"Camera poses visualization saved to {ply_path}")

    def get_accumulated_sim3_for_chunk(self, idx):
        # 获取累计Sim3，假设re_align_all_chunks时传入的accumulated_sim3为最新
        # 这里直接从self.result_aligned_dir读取（或可缓存）
        # 实际应在re_align_all_chunks后缓存accumulated_sim3
        # 这里简单实现：
        if idx == 0:
            return 1.0, np.eye(3), np.zeros(3)
        # 尝试从最近一次re_align_all_chunks的参数获取
        # 这里假设self._last_accumulated_sim3已缓存
        if hasattr(self, '_last_accumulated_sim3'):
            return self._last_accumulated_sim3[idx-1]
        # 否则回退为单位
        return 1.0, np.eye(3), np.zeros(3)

    def close(self):
        '''
            Clean up temporary files and calculate reclaimed disk space.
            
            This method deletes all temporary files generated during processing from three directories:
            - Unaligned results
            - Aligned results
            - Loop results

            ~50 GiB for 4500-frame KITTI 00, 
            ~35 GiB for 2700-frame KITTI 05, 
            or ~5 GiB for 300-frame short seq.
        '''
        if not self.delete_temp_files:
            return
        
        total_space = 0

        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print('Deleting temp files done.')

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil
def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
        
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VGGT-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='config path')
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir
    # 标准化路径（处理斜杠/反斜杠、末尾斜杠等问题）
    image_dir_abs = os.path.abspath(image_dir)
    # 提取图片目录的最后两级路径（比如 image_dir 是 /data/kitti/00/image_2，就取 kitti_00）
    path_parts = image_dir_abs.split(os.sep)
    # 至少取最后1级，避免路径过短
    if len(path_parts) >= 2:
        dir_identifier = "_".join(path_parts[-2:])
    else:
        dir_identifier = path_parts[-1]
    
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'

    # 新的保存路径规则：exps/[图片目录最后两级]/[时间戳]/
    # 示例：./exps/kitti_00/2025-01-01-12-00-00/
    save_dir = os.path.join(
        exp_dir, 
        dir_identifier,  # 简洁的目录标识（替代原来的全路径替换）
        current_datetime  # 时间戳确保唯一性，不会覆盖
    )
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    vggt_long = VGGT_Long(image_dir, save_dir, config)
    vggt_long.run()
    vggt_long.close()

    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('All done.')
    sys.exit()