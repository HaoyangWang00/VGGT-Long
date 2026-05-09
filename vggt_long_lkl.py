#截止17:04,2026-3-22，本文件能实现带有冗余回环检测的半流式结构
#在这份代码里，我基于半流式结构做出的改动是：配合了 LoopModel 文件，替它储存，为他分工
#争取实现真正的流式

#四月初又进行了修改适配检测和优化代码

#4.5 五点，改回没有计时模块的代码

#5.9 整套流式代码为正常的，接下来要修改对齐逻辑尝试用ORBSLAM3的位姿进行辅助
import numpy as np
import argparse
import time

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


# ========== 新增：改进版SimpleTimer类 ==========
class SimpleTimer:
    def __init__(self):
        self.start_times = {}
        self.total_times = {}
        self.current_chunk_stats = {}
        self.chunk_summaries = {}
        self.current_chunk_id = None
    
    def set_current_chunk(self, chunk_id):
        """设置当前处理的chunk ID"""
        self.current_chunk_id = chunk_id
        self.current_chunk_stats = {}
    
    def start(self, name):
        """开始计时"""
        self.start_times[name] = time.time()
    
    def end(self, name, extra_info=None):
        """结束计时，记录到当前chunk统计中"""
        if name not in self.start_times:
            return 0.0
        elapsed = time.time() - self.start_times[name]
        del self.start_times[name]
        
        # 累计总时间
        if name not in self.total_times:
            self.total_times[name] = 0.0
        self.total_times[name] += elapsed
        
        # 记录到当前chunk统计
        if self.current_chunk_id is not None:
            if self.current_chunk_id not in self.chunk_summaries:
                self.chunk_summaries[self.current_chunk_id] = {}
            
            record = {
                'time': elapsed,
                'extra_info': extra_info
            }
            self.chunk_summaries[self.current_chunk_id][name] = record
        
        return elapsed
    
    def print_chunk_summary(self, chunk_id):
        """打印指定chunk的详细统计信息"""
        if chunk_id not in self.chunk_summaries:
            return
        
        stats = self.chunk_summaries[chunk_id]
        
        print(f"\n{'='*60}")
        print(f"Chunk {chunk_id} Performance Summary")
        print(f"{'='*60}")
        
        # 按重要性和逻辑顺序排序
        order = ['inference', 'sequential_align', 'loop_detection', 'optimization', 'loop_inference']
        printed = set()
        
        for name in order:
            if name in stats:
                record = stats[name]
                if name == 'inference':
                    print(f"├─ Inference: {record['time']:.3f}s")
                    if record['extra_info']:
                        print(f"│  └─ Info: {record['extra_info']}")
                elif name == 'sequential_align':
                    print(f"├─ Sequential alignment: {record['time']:.3f}s")
                    if record['extra_info']:
                        print(f"│  └─ Info: {record['extra_info']}")
                elif name == 'loop_detection':
                    print(f"├─ Loop detection: {record['time']:.3f}s")
                    if record['extra_info']:
                        print(f"│  └─ Info: {record['extra_info']}")
                elif name == 'optimization':
                    print(f"├─ Optimization: {record['time']:.3f}s")
                    if record['extra_info']:
                        print(f"│  └─ Info: {record['extra_info']}")
                elif name == 'loop_inference':
                    print(f"├─ Loop inference: {record['time']:.3f}s")
                    if record['extra_info']:
                        print(f"│  └─ Info: {record['extra_info']}")
                printed.add(name)
        
        # 打印其他未排序的项
        for name, record in stats.items():
            if name not in printed:
                print(f"├─ {name.replace('_', ' ').title()}: {record['time']:.3f}s")
                if record['extra_info']:
                    print(f"│  └─ Info: {record['extra_info']}")
        
        # 计算chunk总时间（取所有阶段时间之和）
        chunk_total = sum(stats[name]['time'] for name in stats)
        print(f"└─ Chunk total time: {chunk_total:.3f}s")
        print(f"{'='*60}\n")
    
    def print_final_summary(self):
        """打印最终的全局统计信息"""
        print(f"\n{'='*65}")
        print(f"Final Performance Summary - All Chunks")
        print(f"{'='*65}")
        
        # 按重要性排序
        order = ['initialization', 'inference', 'sequential_align', 'loop_detection', 'optimization', 'final_optimization', 'total_runtime']
        printed = set()
        
        for name in order:
            if name in self.total_times:
                total_time = self.total_times[name]
                if name == 'initialization':
                    print(f"├─ Initialization: {total_time:.3f}s")
                elif name == 'inference':
                    print(f"├─ Frame inference (all chunks): {total_time:.3f}s")
                elif name == 'sequential_align':
                    print(f"├─ Sequential alignment (all chunks): {total_time:.3f}s")
                elif name == 'loop_detection':
                    print(f"├─ Loop detection (all chunks): {total_time:.3f}s")
                elif name == 'optimization':
                    print(f"├─ On-the-fly optimization: {total_time:.3f}s")
                elif name == 'final_optimization':
                    print(f"├─ Final optimization: {total_time:.3f}s")
                elif name == 'total_runtime':
                    print(f"└─ Total pipeline runtime: {total_time:.3f}s")
                printed.add(name)
        
        # 打印其他未排序的项
        for name, total_time in self.total_times.items():
            if name not in printed and name != 'total_runtime':
                print(f"├─ {name.replace('_', ' ').title()}: {total_time:.3f}s")
        
        # 计算并打印效率指标
        if 'total_runtime' in self.total_times and 'inference' in self.total_times:
            inference_time = self.total_times['inference']
            total_time = self.total_times['total_runtime']
            efficiency = inference_time / total_time * 100 if total_time > 0 else 0
            print(f"\nPerformance Metrics:")
            print(f"├─ Inference efficiency: {efficiency:.1f}%")
            print(f"└─ Average time per chunk: {total_time/len(self.chunk_summaries):.3f}s" if self.chunk_summaries else "└─ No chunks processed")
        
        print(f"{'='*65}\n")
# ========== 新增结束 ==========


def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    #有相同回环区间的回环只保留一个，避免重复添加边
    #同一个chunk内部的回环无意义，直接去除
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])  # ✅ 原版：只用 chunk_idx 去重
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result


def extract_p2_k_matrix(calib_path):
    """from calib.txt get K  (kitti)"""
    #读取相机内参
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
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []
        self.all_camera_intrinsics = []

class VGGT_Long:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']

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

        self.stream_optimize_every = self.config['Model'].get('stream_optimize_every', 5)
        self.added_loop_edges = set()
        self._need_optimize = False

        # ========== 修复ID体系：废除缓存计数器，使用全局ID ==========
        self.descriptor_cache = {}  # {frame_id: descriptor_numpy_array}
        # 删除 _cached_frame_count，不再需要
        # ===========================================================

        if self.config['Weights']['model'] == 'VGGT':
            self.model = VGGTAdapter(self.config)
        elif self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        elif self.config['Weights']['model'] == 'Mapanything':
            self.model = MapAnythingAdapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}. ")

        self.skyseg_session = None
        
        self.chunk_indices = []

        self.loop_list = []

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = []

        self.loop_sim3_list = []

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        # 显式初始化 LoopDetector 状态（与修复后的 LoopModel 兼容）
        self.loop_state = {
            'faiss_index': None,
            'frame_ids': [],
            'loop_closures': []
        }
        self._dbow_processed_count = 0
        self.loop_output_path = os.path.join(save_dir, "loop_closures.txt")

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

        # ========== 新增：初始化计时器 ==========
        self.timer = SimpleTimer()
        # ========== 新增结束 ==========
        
        print('init done.')

    def frame_to_chunk(self, frame_id):
        for idx, (s, e) in enumerate(self.chunk_indices):
            if s <= frame_id < e:
                return idx
        return None

    def re_align_all_chunks(self, accumulated_sim3, write_ply=False):
        self._last_accumulated_sim3 = accumulated_sim3
        """
        用累计 Sim3 重新对齐所有 chunk。第 0 块为单位 Sim3，其余块用累计 Sim3 左乘。
        write_ply: 是否写 ply 文件（建议只在最后一次优化时写）
        """
        # 修复问题5：确保 all_camera_poses 有足够空间
        while len(self.all_camera_poses) < len(self.chunk_indices):
            self.all_camera_poses.append(None)
            self.all_camera_intrinsics.append(None)
        
        for idx in range(len(self.chunk_indices)):
            unaligned_path = os.path.join(
                self.result_unaligned_dir,
                f"chunk_{idx}.npy"
            )
            chunk_data = np.load(unaligned_path, allow_pickle=True).item()

            if idx == 0:
                s, R, t = 1.0, np.eye(3), np.zeros(3)
            else:
                s, R, t = accumulated_sim3[idx-1]

            chunk_data['world_points'] = apply_sim3_direct(
                chunk_data['world_points'],
                s, R, t
            )

            chunk_data['extrinsic'] = np.array([
                self.apply_sim3_to_pose(e, s, R, t)
                for e in chunk_data['extrinsic']
            ])

            aligned_path = os.path.join(
                self.result_aligned_dir,
                f"chunk_{idx}.npy"
            )
            np.save(aligned_path, chunk_data)

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

            # 修复问题5：使用索引赋值前确保列表有足够空间
            self.all_camera_poses[idx] = (
                self.chunk_indices[idx],
                chunk_data['extrinsic']
            )

    def _save_loop_closures_to_file(self):
        """保存回环结果到文件（覆盖写入，保证文件是最新的）"""
        if self.loop_state is None:
            return
        
        loop_closures = self.loop_state['loop_closures']
        
        with open(self.loop_output_path, 'w') as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.loop_detector.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.loop_detector.nms_threshold}\n")
            f.write("\n# Loop pairs:\n")
            for i, j, sim in loop_closures:
                f.write(f"{i}, {j}, {sim:.4f}\n")
        
        print(f"[VGGT_Long] Saved {len(loop_closures)} loop pairs to {self.loop_output_path}")

    def get_loop_pairs(self, start_idx, end_idx, new_image_paths=None):
        """
        获取回环对（使用全局ID）
        """
        if self.useDBoW:
            if hasattr(self, '_dbow_processed_count'):
                base_frame_id = self._dbow_processed_count
            else:
                base_frame_id = 0
            if new_image_paths is None:
                image_paths = self.img_list[start_idx:end_idx]
                base_frame_id = start_idx
            else:
                image_paths = new_image_paths

            for frame_id, img_path in tqdm(enumerate(image_paths)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = cv2.resize(image_ori, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                global_frame_id = base_frame_id + frame_id

                self.retrieval(frame, global_frame_id)

                cands = self.retrieval.detect_loop(
                    thresh=self.config['Loop']['DBoW']['thresh'],
                    num_repeat=self.config['Loop']['DBoW']['num_repeat']
                )

                if cands is not None:
                    (i, j) = cands
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(global_frame_id)

            self._dbow_processed_count = base_frame_id + len(image_paths)

        else:  # ===== DNIO v2（修正版）=====
            if new_image_paths is None:
                print("[VGGT_Long] DNIO v2 offline mode")
                new_image_paths = self.img_list[start_idx:end_idx]

            # ========= 核心：区分 new vs overlap =========
            new_paths = []
            new_frame_ids = []

            frame_ids_to_process = list(range(start_idx, end_idx))

            for i, img_path in enumerate(new_image_paths):
                global_frame_id = frame_ids_to_process[i]

                # ✅ 如果已经存在 → overlap → 跳过（不参与检测）
                if global_frame_id in self.descriptor_cache:
                    continue
                else:
                    new_paths.append(img_path)
                    new_frame_ids.append(global_frame_id)

            # ========= 只处理真正的新帧 =========
            if not new_paths:
                print("[VGGT_Long] No new frames (all overlap), skip loop detection")
                return []

            print(f"[VGGT_Long] Extracting descriptors for {len(new_paths)} new frames")

            new_descriptors = self.loop_detector.extract_descriptors(new_paths)
            new_descriptors_np = (
                new_descriptors.numpy()
                if isinstance(new_descriptors, torch.Tensor)
                else new_descriptors
            )

            # ========= 写入缓存 =========
            for i, frame_id in enumerate(new_frame_ids):
                self.descriptor_cache[frame_id] = new_descriptors_np[i]

            # ========= 只用 new frames 做检测 =========
            input_descriptors = torch.from_numpy(new_descriptors_np)

            new_loops, self.loop_state = self.loop_detector.detect_loops_with_descriptors(
                input_descriptors,
                new_frame_ids,
                self.loop_state
            )

            # ========= 更新 loop_list =========
            self.loop_list = [
                (idx1, idx2)
                for idx1, idx2, _ in self.loop_state['loop_closures']
            ]

            # ========= 保存 =========
            self._save_loop_closures_to_file()

            print(f"[VGGT_Long] Total loops: {len(self.loop_list)}, New loops: {len(new_loops)}")

            return new_loops

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # ========== 新增：推理计时 ==========
        timer_name = "loop_inference" if is_loop else "inference"
        extra_info = f"loop region {range_1[0]}-{range_1[1]} & {range_2[0]}-{range_2[1]}" if is_loop else None
        self.timer.start(timer_name)
        # ========== 新增结束 ==========
        
        predictions = self.model.infer_chunk(chunk_image_paths)
        
        # ========== 新增：结束推理计时 ==========
        elapsed = self.timer.end(timer_name, extra_info)
        # ========== 新增结束 ==========
        
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
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
        对 4x4 位姿矩阵左乘 Sim3(s, R, t)
        """
        S = np.eye(4)
        S[:3, :3] = s * R
        S[:3, 3] = t
        pose_new = S @ pose
        pose_new[:3, :3] /= s
        return pose_new

    def process_stream_chunk(self, start_idx, end_idx, chunk_idx):
        """
        流式处理单个 chunk：推理、与前一 chunk 对齐、添加边、回环。
        """
        # ========== 新增：设置当前chunk ID ==========
        self.timer.set_current_chunk(chunk_idx)
        # ========== 新增结束 ==========
        
        print(f"\n{'='*60}")
        print(f"Processing Chunk {chunk_idx}: frames {start_idx}-{end_idx}")
        print(f"{'='*60}")
        
        # 1. 推理并保存
        self.chunk_indices.append((start_idx, end_idx))
        self.process_single_chunk((start_idx, end_idx), chunk_idx=chunk_idx)

        # 2. 如果不是第一个 chunk，和前一个 chunk 对齐，并添加相邻边
        if chunk_idx > 0:
            # ========== 新增：顺序对齐计时 ==========
            self.timer.start("sequential_align")
            # ========== 新增结束 ==========
            
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx-1}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            overlap = min(
                self.overlap,
                len(chunk_data1['world_points']),
                len(chunk_data2['world_points'])
            )
            point_map1 = chunk_data1['world_points'][-overlap:]
            point_map2 = chunk_data2['world_points'][:overlap]
            conf1 = chunk_data1['world_points_conf'][-overlap:]
            conf2 = chunk_data2['world_points_conf'][:overlap]
            mask = None
            if chunk_data1.get("mask", None) is not None:
                available_overlap = min(self.overlap, len(chunk_data1["mask"]), len(chunk_data2["mask"]))
                if available_overlap == 0:
                    mask = torch.ones(1, *chunk_data1["mask"].shape[1:], device=chunk_data1["mask"].device)
                else:
                    mask1 = chunk_data1["mask"][-available_overlap:]
                    mask2 = chunk_data2["mask"][:available_overlap]
                    mask = mask1.squeeze() & mask2.squeeze()
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
            self.loop_optimizer.add_sequential(s, R, t)
            self.sim3_list.append((s, R, t))  # 保留
            
            # ========== 新增：结束顺序对齐计时 ==========
            self.timer.end("sequential_align", f"chunk {chunk_idx-1} → {chunk_idx}")
            # ========== 新增结束 ==========

        # 3. 每个 chunk 后做一次回环检测（流式模式）
        if self.loop_enable:
            # ========== 新增：回环检测计时 ==========
            self.timer.start("loop_detection")
            loop_count = 0
            # ========== 新增结束 ==========
            
            chunk_image_paths = self.img_list[start_idx:end_idx]
            # 关键：传入全局ID范围
            new_loops = self.get_loop_pairs(start_idx, end_idx, new_image_paths=chunk_image_paths)
            if len(new_loops) > 0:
                print(f"[VGGT_Long] Processing {len(new_loops)} new loop(s)...")
                loop_half_window = int(self.config['Model'].get('loop_chunk_size', 20) / 2)
                # ========== 修复：将 new_loops 转换为二元组格式 ==========
                # new_loops 可能是 (idx1, idx2, similarity) 三元组
                # process_loop_list 期望 (idx1, idx2) 二元组
                loop_pairs = [(idx1, idx2) for idx1, idx2, *_ in new_loops]
                # ========================================================
                loop_results = process_loop_list(
                    self.chunk_indices,
                    loop_pairs,  # ✅ 使用转换后的二元组
                    half_window=loop_half_window
                )
                loop_results = remove_duplicates(loop_results)
                print(f"[VGGT_Long] Loop results after dedup: {loop_results}")

                for item in loop_results:
                    chunk_idx_a, range_a, chunk_idx_b, range_b = item
                    # ========== 细化去重 key ==========
                    edge_key = (chunk_idx_a, chunk_idx_b, tuple(range_a), tuple(range_b))
                    if edge_key in self.added_loop_edges:
                        continue
                    print(f"[VGGT_Long] Processing loop region: chunk {chunk_idx_a} ({range_a[0]}-{range_a[1]}) "
                          f"<-> chunk {chunk_idx_b} ({range_b[0]}-{range_b[1]})")
                    loop_predictions = self.process_single_chunk(
                        range_a,
                        range_2=range_b,
                        is_loop=True
                    )
                    # --- Step 1: Chunk A 与 loop 区域对齐 ---
                    range_a_start, range_a_end = range_a
                    chunk_a_data = np.load(
                        os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                        allow_pickle=True
                    ).item()
                    chunk_a_begin = self.chunk_indices[chunk_idx_a][0]
                    rela_a_start = range_a_start - chunk_a_begin
                    rela_a_end = range_a_end - chunk_a_begin
                    point_map_a = chunk_a_data['world_points'][rela_a_start:rela_a_end]
                    conf_a = chunk_a_data['world_points_conf'][rela_a_start:rela_a_end]
                    loop_len_a = range_a_end - range_a_start
                    point_map_loop_a = loop_predictions['world_points'][:loop_len_a]
                    conf_loop_a = loop_predictions['world_points_conf'][:loop_len_a]
                    if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                        conf_threshold_a = min(np.median(conf_a), np.median(conf_loop_a)) * 0.1
                    else:
                        conf_threshold_a = -1.0
                    mask_a = None
                    if chunk_a_data.get("mask") is not None and loop_predictions.get("mask") is not None:
                        mask_loop_a = loop_predictions["mask"][:loop_len_a]
                        mask_chunk_a = chunk_a_data["mask"][rela_a_start:rela_a_end]
                        mask_a = mask_loop_a.squeeze() & mask_chunk_a.squeeze()
                        if hasattr(mask_a, 'cpu'):
                            mask_a = mask_a.cpu().numpy().astype(bool)
                    s_a, R_a, t_a = weighted_align_point_maps(
                        point_map_a, conf_a, point_map_loop_a, conf_loop_a,
                        mask_a, conf_threshold=conf_threshold_a, config=self.config
                    )
                    # --- Step 2: Chunk B 与 loop 区域对齐 ---
                    range_b_start, range_b_end = range_b
                    chunk_b_data = np.load(
                        os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                        allow_pickle=True
                    ).item()
                    chunk_b_begin = self.chunk_indices[chunk_idx_b][0]
                    rela_b_start = range_b_start - chunk_b_begin
                    rela_b_end = range_b_end - chunk_b_begin
                    point_map_b = chunk_b_data['world_points'][rela_b_start:rela_b_end]
                    conf_b = chunk_b_data['world_points_conf'][rela_b_start:rela_b_end]
                    loop_len_b = range_b_end - range_b_start
                    point_map_loop_b = loop_predictions['world_points'][-loop_len_b:]
                    conf_loop_b = loop_predictions['world_points_conf'][-loop_len_b:]
                    if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                        conf_threshold_b = min(np.median(conf_b), np.median(conf_loop_b)) * 0.1
                    else:
                        conf_threshold_b = -1.0
                    mask_b = None
                    if chunk_b_data.get("mask") is not None and loop_predictions.get("mask") is not None:
                        mask_loop_b = loop_predictions["mask"][-loop_len_b:]
                        mask_chunk_b = chunk_b_data["mask"][rela_b_start:rela_b_end]
                        mask_b = mask_loop_b.squeeze() & mask_chunk_b.squeeze()
                        if hasattr(mask_b, 'cpu'):
                            mask_b = mask_b.cpu().numpy().astype(bool)
                    s_b, R_b, t_b = weighted_align_point_maps(
                        point_map_b, conf_b, point_map_loop_b, conf_loop_b,
                        mask_b, conf_threshold=conf_threshold_b, config=self.config
                    )
                    # --- Step 3: 计算 a → b 的 Sim3 ---
                    s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                    self.loop_optimizer.add_loop(chunk_idx_a, chunk_idx_b, s_ab, R_ab, t_ab)
                    self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, s_ab, R_ab, t_ab))
                    self.added_loop_edges.add(edge_key)
                    self._need_optimize = True
                    print(f"[VGGT_Long] Added loop edge: chunk {chunk_idx_a} <-> chunk {chunk_idx_b} "
                          f"(frames {range_a_start}-{range_a_end} ↔ {range_b_start}-{range_b_end})")
                    print(f"[VGGT_Long] Sim3: scale={s_ab:.4f}, trans={t_ab}")
                    loop_count += 1
            
            # ========== 新增：结束回环检测计时 ==========
            self.timer.end("loop_detection", f"{loop_count} loops found")
            # ========== 新增结束 ==========
        
        # 新增：如果有新 loop，触发优化
        if self._need_optimize:
            print(f"[VGGT_Long] Triggering optimization with {len(self.loop_sim3_list)} loop constraints...")
            
            # ========== 新增：优化计时 ==========
            self.timer.start("optimization")
            # ========== 新增结束 ==========
            
            optimized = self.loop_optimizer.optimize()
            accumulated = accumulate_sim3_transforms(optimized)
            self.re_align_all_chunks(accumulated, write_ply=False)
            
            # ========== 新增：结束优化计时 ==========
            self.timer.end("optimization", f"{len(self.loop_sim3_list)} edges")
            # ========== 新增结束 ==========
            
            self._need_optimize = False
            print("[VGGT_Long] Optimization completed.")
        
        # ========== 新增：打印当前chunk统计 ==========
        self.timer.print_chunk_summary(chunk_idx)
        # ========== 新增结束 ==========

    def process_stream_sequence(self):
        """
        按流式增量方式处理所有 chunk，每来一批就推理、对齐、每 N 个 chunk 优化一次。优化后用累计 Sim3 对齐。
        """
        step = self.chunk_size - self.overlap
        self.chunk_indices = []
        chunk_idx = 0
        start = 0
        idle_count = 0

        print("Streaming mode started...")

        last_total = 0
        while True:
            current_img_list = sorted(
                glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                glob.glob(os.path.join(self.img_dir, "*.png"))
            )

            total = len(current_img_list)
            new_images = total - last_total
            print(f"Current frames: {total}, New images: {new_images}")

            if total < start + self.chunk_size:
                idle_count += 1
                if idle_count > 200:
                    if total > start:
                        end = total
                        print(f"Final chunk {chunk_idx}: {start}-{end}")
                        self.img_list = current_img_list
                        self.process_stream_chunk(start, end, chunk_idx)
                        chunk_idx += 1
                    print("No new frames. Final optimization.")
                    break
                time.sleep(0.5)
                continue

            idle_count = 0

            end = start + self.chunk_size

            print(f"Streaming chunk {chunk_idx}: {start}-{end}")

            self.img_list = current_img_list

            self.process_stream_chunk(start, end, chunk_idx)

            last_total = total

            chunk_idx += 1
            start += step

        # 最后做一次优化并写 ply
        if len(self.chunk_indices) > 0:
            print("Final pose graph optimization...")
            
            # ========== 新增：最终优化计时 ==========
            self.timer.start("final_optimization")
            # ========== 新增结束 ==========
            
            if self.loop_enable and len(self.loop_sim3_list) > 0:
                print(f"[VGGT_Long] Optimizing with {len(self.loop_sim3_list)} loop constraints...")
                optimized = self.loop_optimizer.optimize()
            else:
                optimized = self.loop_optimizer.optimize()
            accumulated = accumulate_sim3_transforms(optimized)
            self.re_align_all_chunks(accumulated, write_ply=True)
            
            # ========== 新增：结束最终优化计时 ==========
            self.timer.end("final_optimization")
            # ========== 新增结束 ==========


    def run(self):
        # ========== 新增：开始总时间计时 ==========
        self.timer.start("total_runtime")
        # ========== 新增结束 ==========
        
        print(f"Loading images from {self.img_dir}...")
        MIN_INIT_FRAMES = 50
        
        # ========== 新增：开始初始化计时 ==========
        self.timer.start("initialization")
        # ========== 新增结束 ==========
        
        while True:
            self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                                   glob.glob(os.path.join(self.img_dir, "*.png")))
            if len(self.img_list) >= MIN_INIT_FRAMES:
                print(f"Found {len(self.img_list)} images, start initialization.")
                break
            print(f"Waiting for enough images... current: {len(self.img_list)} / {MIN_INIT_FRAMES}")
            time.sleep(2)
        print(f"Found {len(self.img_list)} image files")

        if self.loop_enable:  
            if not self.config['Model'].get('streaming_mode', True):
                print("[VGGT_Long] Offline mode, running loop detection...")
                # 离线模式：传入全局ID范围 0 到总帧数
                self.get_loop_pairs(0, len(self.img_list))
            
        torch.cuda.empty_cache()
        print('Loading model...')
        self.model.load()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k
        
        # ========== 新增：结束初始化计时 ==========
        self.timer.end("initialization")
        # ========== 新增结束 ==========
        
        self.process_stream_sequence()
        self.save_camera_poses()
        
        # ========== 新增：结束总时间计时 ==========
        self.timer.end("total_runtime")
        # ========== 新增结束 ==========
        
        # ========== 新增：打印最终总结 ==========
        self.timer.print_final_summary()
        # ========== 新增结束 ==========

    def save_camera_poses(self):
        '''
        保存所有 chunk 的 extrinsics 为全局位姿。
        注意：all_camera_poses 已在 re_align_all_chunks 中应用 Sim3 变换，此处直接使用。
        '''
        # 修复问题6：更新注释，说明 extrinsics 已对齐
        chunk_colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
            [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

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
        if idx == 0:
            return 1.0, np.eye(3), np.zeros(3)
        if hasattr(self, '_last_accumulated_sim3'):
            return self._last_accumulated_sim3[idx-1]
        return 1.0, np.eye(3), np.zeros(3)

    def close(self):
        '''
        Clean up temporary files and calculate reclaimed disk space.
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

        # ========== 清理 descriptor 缓存 ==========
        if hasattr(self, 'descriptor_cache'):
            self.descriptor_cache.clear()
            print("[VGGT_Long] Descriptor cache cleared.")
        # ========================================

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
    image_dir_abs = os.path.abspath(image_dir)
    path_parts = image_dir_abs.split(os.sep)
    if len(path_parts) >= 2:
        dir_identifier = "_".join(path_parts[-2:])
    else:
        dir_identifier = path_parts[-1]
    
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'

    save_dir = os.path.join(
        exp_dir, 
        dir_identifier,
        current_datetime
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