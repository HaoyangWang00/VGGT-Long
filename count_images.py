import os
import argparse
from pathlib import Path

def count_images(folder_path):
    """统计指定文件夹内所有图片文件的数量"""
    # 定义常见图片扩展名（不区分大小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        raise ValueError(f"错误：文件夹 '{folder_path}' 不存在或不是文件夹")
    
    # 遍历文件夹中的所有文件
    image_count = 0
    for file in os.listdir(folder_path):
        # 获取文件扩展名并转换为小写
        ext = Path(file).suffix.lower()
        if ext in image_extensions:
            image_count += 1
    
    return image_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统计文件夹中的图片数量')
    parser.add_argument('folder', type=str, help='要统计图片的文件夹路径')
    args = parser.parse_args()
    
    try:
        count = count_images(args.folder)
        print(f"文件夹 '{args.folder}' 中包含 {count} 张图片")
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)