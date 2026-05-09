import os
import sys
import glob  # <--- 补上了这一行
from datetime import datetime

def delete_recent_images(folder_path, num_to_delete=5):
    """
    从指定文件夹中删除最近的num_to_delete张图像文件
    """
    # 获取所有图片文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    
    if not image_files:
        print(f"❌ 文件夹 {folder_path} 中没有找到任何图像文件")
        return
    
    # 按修改时间排序（从新到旧）
    # reverse=True 表示降序，最新的在前面
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 确定要删除的文件（最新的num_to_delete个）
    files_to_delete = image_files[:num_to_delete]
    
    if not files_to_delete:
        print(f"⚠️ 没有找到需要删除的文件（需要删除 {num_to_delete} 张，但实际只有 {len(image_files)} 张）")
        return
    
    # 显示将要删除的文件
    print(f"🔍 将要删除以下 {len(files_to_delete)} 个最新图像文件:")
    for f in files_to_delete:
        mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - {os.path.basename(f)} (时间: {mtime})")
    
    # 确认
    confirm = input("\n是否确认删除？(输入 y 确认，其他键取消): ").strip().lower()
    if confirm != 'y':
        print("❌ 操作已取消")
        return
    
    # 执行删除
    deleted_count = 0
    for f in files_to_delete:
        try:
            os.remove(f)
            deleted_count += 1
            print(f"✅ 已删除: {os.path.basename(f)}")
        except Exception as e:
            print(f"❌ 删除失败 {os.path.basename(f)}: {str(e)}")
    
    print(f"\n🎉 已成功删除 {deleted_count}/{len(files_to_delete)} 张图像文件")

if __name__ == "__main__":
    # ================= 配置区域 =================
    FOLDER_PATH = "./http_server/received_images"  # 你的图像文件夹路径
    NUM_TO_DELETE = 1500                              # 要删除的图像数量
    # ===========================================

    if not os.path.exists(FOLDER_PATH):
        print(f"❌ 错误：文件夹不存在 -> {FOLDER_PATH}")
        sys.exit(1)

    print(f"🚀 开始删除文件夹 {FOLDER_PATH} 中的最新 {NUM_TO_DELETE} 张图像...")
    delete_recent_images(FOLDER_PATH, NUM_TO_DELETE)
    print("\n✅ 操作完成！")