#截止16:49, 2026-3-22,这个项目未做修改
#第一次修改，目的是将loopmodel变成流式结构
# 这份代码遵循的思想是，把本代码作为一个离线计算元件，相当于是流水线的一部分，
# 只负责完成主代码安排的分工，至于具体怎么实现实时功能由主代码定夺

# 4月初，第二次修改：修复skipping pair的问题
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path
import numpy as np
import faiss

from LoopModels.vpr_model import VPRModel


class LoopDetector:
    """Loop detector class for detecting loop closures in image sequences
    
    设计原则：无状态工具类，状态由主代码 (VGGT_Long) 维护
    """
    
    def __init__(self, image_dir=None, 
                 output="loop_closures.txt",
                 config=None):
        """Initialize the loop detector
        
        Args:
            image_dir: Directory path containing images (可选，流式模式下由主代码传入图片路径)
            output: Output file path
            config: Configuration dictionary
        """
        self.config = config
        self.image_dir = image_dir
        self.output = output
        
        # 从 config 读取参数
        if config:
            self.ckpt_path = self.config['Weights']['SALAD']
            self.image_size = self.config['Loop']['SALAD']['image_size']
            self.batch_size = self.config['Loop']['SALAD']['batch_size']
            self.similarity_threshold = self.config['Loop']['SALAD']['similarity_threshold']
            self.top_k = self.config['Loop']['SALAD']['top_k']
            self.use_nms = self.config['Loop']['SALAD']['use_nms']
            self.nms_threshold = self.config['Loop']['SALAD']['nms_threshold']
        else:
            # 默认值（用于 standalone 模式）
            self.ckpt_path = None
            self.image_size = [336, 336]
            self.batch_size = 32
            self.similarity_threshold = 0.7
            self.top_k = 5
            self.use_nms = True
            self.nms_threshold = 25
        
        # 保留的变量（配置/资源，可复用）
        self.model = None
        self.device = None
        self.descriptors = None  # 用于 find_loop_closures
        self.state = {'frame_ids': [], 'faiss_index': None}  # 废除 processed_frames
    
    def _input_transform(self, image_size=None):
        """Create image transformation function"""
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        if image_size:
            return T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
    
    def load_model(self):
        """Load model (只加载一次，后续调用可复用)"""
        if self.model is not None:
            return
            
        model = VPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
            vggt_long_config=self.config
        )

        # 加载权重
        model.load_state_dict(torch.load(self.ckpt_path, map_location='cpu', weights_only=True))
        model = model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model loaded: {self.ckpt_path}")
        
        self.model = model
        self.device = device
    
    def get_image_paths(self, image_dir=None):
        """Get paths of all image files in directory
        
        Args:
            image_dir: 可选，覆盖 self.image_dir
        """
        if image_dir is None:
            image_dir = self.image_dir
            
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(Path(image_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(image_dir).glob(f"*{ext.upper()}")))
        
        image_paths = sorted(image_paths)
        return image_paths
    
    def extract_descriptors(self, image_paths):
        """Extract image feature descriptors
        
        Args:
            image_paths: 图片路径列表
        
        Returns:
            descriptors: 特征描述子 tensor [N, D]
        """
        if self.model is None:
            self.load_model()
            
        transform = self._input_transform(self.image_size)
        descriptors = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_imgs = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    h, w = self.image_size if self.image_size else (224, 224)
                    img = torch.zeros(3, h, w)
                    batch_imgs.append(img)
            
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                    batch_descriptors = self.model(batch_tensor).cpu()
            
            descriptors.append(batch_descriptors)
        
        descriptors = torch.cat(descriptors)
        return descriptors
    
    def _apply_nms_filter(self, loop_closures, nms_threshold):
        """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()
        
        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)
        
        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                continue
            
            filtered_loops.append((idx1, idx2, sim))
            
            # 抑制附近的回环
            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2) 
            suppress_range = set(range(start1, end1))
            
            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))
            
            suppressed.update(suppress_range)
        
        return filtered_loops
    
    def _ensure_descending_order(self, tuples_list):
        """确保回环对中 idx1 > idx2（新帧在前，旧帧在后）"""
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]
    
    def detect_loops(self, new_image_paths, new_frame_ids, state=None):
        """
        主入口函数：检测新帧中的回环
        
        Args:
            new_image_paths: 新帧的图片路径列表
            new_frame_ids: 新帧的全局ID列表，必须与图片一一对应
            state: 历史状态字典，第一次调用为 None
        
        Returns:
            new_loops: 本次检测到的新回环列表 [(idx1, idx2, sim), ...]
            state: 更新后的状态字典
        """
        if not new_image_paths or not new_frame_ids:
            if state is None:
                state = {
                    'faiss_index': None,
                    'frame_ids': [],
                    'loop_closures': []
                }
            return [], state
        
        # 提取新帧特征
        new_descriptors = self.extract_descriptors(new_image_paths)
        
        # 调用核心检测函数
        return self.detect_loops_with_descriptors(new_descriptors, new_frame_ids, state)
    
    def detect_loops_with_descriptors(self, new_descriptors, new_frame_ids, state=None):
        """
        核心检测函数：使用已提取的特征进行回环检测
        
        Args:
            new_descriptors: 新帧的特征描述子
            new_frame_ids: 新帧的全局ID列表
            state: 历史状态字典
        
        Returns:
            new_loops: 本次检测到的新回环列表 [(idx1, idx2, sim), ...]
            state: 更新后的状态字典
        """
        if not new_frame_ids or len(new_frame_ids) == 0:
            if state is None:
                state = {
                    'faiss_index': None,
                    'frame_ids': [],
                    'loop_closures': []
                }
            return [], state

        # 类型校验：确保 new_frame_ids 是整数列表
        if not all(isinstance(fid, int) for fid in new_frame_ids):
            raise TypeError("new_frame_ids must be list of integers")
        
        # 转换特征为 numpy 数组
        if isinstance(new_descriptors, torch.Tensor):
            new_descriptors_np = new_descriptors.cpu().numpy()
        else:
            new_descriptors_np = new_descriptors
            
        new_count = len(new_descriptors_np)
        if new_count == 0:
            if state is None:
                state = {
                    'faiss_index': None,
                    'frame_ids': [],
                    'loop_closures': []
                }
            return [], state
            
        embed_size = new_descriptors_np.shape[1]
        
        # ========== 初始化模式：第一次调用 ==========
        if state is None:
            print(f"[LoopDetector] Initialize mode: {new_count} frames")
            
            # 创建 FAISS 索引
            faiss_index = faiss.IndexFlatIP(embed_size)
            
            # 归一化向量（FAISS IndexFlatIP 需要归一化）
            faiss.normalize_L2(new_descriptors_np)
            faiss_index.add(new_descriptors_np)
            
            # 创建初始状态
            state = {
                'faiss_index': faiss_index,
                'frame_ids': list(new_frame_ids),  # 保存全局ID
                'loop_closures': []
            }
            
            return [], state
        
        # ========== 检测模式：后续调用 ==========
        faiss_index = state['faiss_index']
        
        # 检查 FAISS 索引是否需要初始化
        if faiss_index is None:
            print(f"[LoopDetector] Initialize FAISS index: {new_count} frames")
            faiss_index = faiss.IndexFlatIP(embed_size)
            faiss.normalize_L2(new_descriptors_np)
            faiss_index.add(new_descriptors_np)
            state['faiss_index'] = faiss_index
            state['frame_ids'] = list(new_frame_ids)
            return [], state
        
        print(f"[LoopDetector] Detect mode: {new_count} new frames vs {len(state['frame_ids'])} history frames")
        
        # 归一化新描述子
        new_descriptors_np_norm = new_descriptors_np.copy()
        faiss.normalize_L2(new_descriptors_np_norm)
        
        # 搜索最近邻
        similarities, indices = faiss_index.search(new_descriptors_np_norm, self.top_k)
        
        # 检测回环
        new_loops = []
        for i in range(new_count):
            global_new_idx = new_frame_ids[i]  # 使用传入的全局ID
            
            for j in range(self.top_k):
                neighbor_local_idx = indices[i, j]
                similarity = similarities[i, j]
                
                # 检查索引有效性
                if neighbor_local_idx < 0 or neighbor_local_idx >= len(state['frame_ids']):
                    continue
                
                # 获取历史帧的全局 ID
                global_old_idx = state['frame_ids'][neighbor_local_idx]
                
                # 过滤条件：相似度阈值 + 时间排除窗口
                if (similarity > self.similarity_threshold and 
                    abs(global_new_idx - global_old_idx) > 10 and
                    global_new_idx > global_old_idx):  # 确保新帧在后
                    
                    new_loops.append((global_new_idx, global_old_idx, similarity))
        
        # 去重
        new_loops = list(set(new_loops))
        new_loops.sort(key=lambda x: x[2], reverse=True)
        
        # NMS 过滤
        if self.use_nms and self.nms_threshold > 0:
            new_loops = self._apply_nms_filter(new_loops, self.nms_threshold)
        
        # 确保顺序（新帧在前，旧帧在后）
        new_loops = self._ensure_descending_order(new_loops)
        
        # ========== 更新状态 ===========
        # 1. 添加新帧到 FAISS 索引
        faiss.normalize_L2(new_descriptors_np)  # 确保归一化
        faiss_index.add(new_descriptors_np)
        
        # 2. 更新帧 ID 列表
        state['frame_ids'].extend(new_frame_ids)
        
        # 3. 更新回环列表
        state['loop_closures'].extend(new_loops)
        
        print(f"[LoopDetector] Found {len(new_loops)} new loop(s)")
        
        return new_loops, state
    
    def find_loop_closures(self, descriptors=None):
        """离线模式：查找回环（保留用于向后兼容）"""
        if descriptors is None:
            if not hasattr(self, 'descriptors') or self.descriptors is None:
                image_paths = self.get_image_paths()
                self.descriptors = self.extract_descriptors(image_paths)
            descriptors = self.descriptors
        
        # 使用流式接口实现离线模式
        all_frame_ids = list(range(len(descriptors)))
        new_loops, state = self.detect_loops_with_descriptors(descriptors, all_frame_ids, state=None)
        
        return state['loop_closures']
    
    def save_results(self, loop_closures, image_paths=None, output_path=None):
        """保存回环检测结果到文件"""
        if output_path is None:
            output_path = self.output
        
        if image_paths is None and self.image_dir:
            image_paths = self.get_image_paths()
            
        with open(output_path, 'w') as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            f.write("\n# Loop pairs:\n")
            for i, j, sim in loop_closures:
                f.write(f"{i}, {j}, {sim:.4f}\n")
            
            if image_paths:
                f.write("\n# Image path list:\n")
                for i, path in enumerate(image_paths):
                    f.write(f"# {i}: {path}\n")
        
        print(f"Found {len(loop_closures)} loop pairs, results saved to {output_path}")
        
        if loop_closures:
            print("\nTop 10 loop pairs:")
            for i, (idx1, idx2, sim) in enumerate(loop_closures[:10]):
                print(f"{idx1}, {idx2}, similarity: {sim:.4f}")
                if i >= 9:
                    break
    
    def get_loop_list(self, loop_closures):
        """从回环三元组中提取帧 ID 对列表"""
        return [(idx1, idx2) for idx1, idx2, _ in loop_closures]
    
    def run(self):
        """Run complete loop detection pipeline (离线模式，向后兼容)"""
        print('Loading model...')
        self.load_model()
        
        image_paths = self.get_image_paths()
        if not image_paths:
            print(f"No image files found in {self.image_dir}")
            return []
        
        print(f"Found {len(image_paths)} image files")
        
        # 使用 detect_loops 离线模式
        all_frame_ids = list(range(len(image_paths)))
        new_loops, state = self.detect_loops(image_paths, all_frame_ids, state=None)
        
        # 保存结果
        self.save_results(state['loop_closures'], image_paths)
        
        return state['loop_closures']


def main():
    parser = argparse.ArgumentParser(description="Loop detection using SALAD model")
    parser.add_argument("--image_dir", type=str, default="/media/deng/Data/KITTIdataset/data_odometry_color/dataset/sequences/00/image_2", help="Directory path containing images")
    parser.add_argument("--ckpt_path", type=str, default="./weights/dino_salad.ckpt", help="Model checkpoint path")
    parser.add_argument("--image_size", nargs=2, type=int, default=[336, 336], help="Image resize dimensions [height width]")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, help="Similarity threshold for loop closure")
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors to check for each image")
    parser.add_argument("--output", type=str, default="loop_closures.txt", help="Output file path")
    parser.add_argument("--use_nms", action="store_true", default=True, help="Whether to use Non-Maximum Suppression (NMS) filtering")
    parser.add_argument("--nms_threshold", type=int, default=25, help="NMS threshold for minimum frame difference between loop pairs")
    
    args = parser.parse_args()
    
    # 构建 config 字典
    config = {
        'Weights': {'SALAD': args.ckpt_path},
        'Loop': {
            'SALAD': {
                'image_size': args.image_size,
                'batch_size': args.batch_size,
                'similarity_threshold': args.similarity_threshold,
                'top_k': args.top_k,
                'use_nms': args.use_nms,
                'nms_threshold': args.nms_threshold
            }
        }
    }
    
    detector = LoopDetector(
        image_dir=args.image_dir,
        output=args.output,
        config=config
    )
    
    detector.run()


if __name__ == "__main__":
    main()