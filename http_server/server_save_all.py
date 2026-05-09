import flask
from flask import request
import os
import cv2
import threading

app = flask.Flask(__name__)

SAVE_FOLDER = '/home/haoyang22/project/VGGT-Long/http_server/received_images'
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ====== 内存计数器（重启即归零） ======
image_counter = 0
counter_lock = threading.Lock()

def get_next_index():
    global image_counter
    with counter_lock:
        current = image_counter
        image_counter += 1
        return current


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image", 400

    file = request.files['image']
    
    # ====== 获取序号命名（6位补零） ======
    index = get_next_index()
    filename = f"{index:06d}.png"  # 000000.png, 000001.png, 000002.png...

    tmp_path = os.path.join(SAVE_FOLDER, filename + ".tmp")
    final_path = os.path.join(SAVE_FOLDER, filename)

    # 先写临时文件
    file.save(tmp_path)

    # ====== 完整性检查 ======
    img = cv2.imread(tmp_path)

    if img is None:
        print("Bad image received:", filename)
        os.remove(tmp_path)
        return "Bad image", 400

    # ====== 原子重命名 ======
    os.rename(tmp_path, final_path)

    print(f"Saved: {filename} (index={index})")

    return "OK", 200


@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    """手动重置计数器（可选）"""
    global image_counter
    with counter_lock:
        image_counter = 0
    return "Counter reset to 0"


if __name__ == '__main__':
    print("=== Image Server Started ===")
    print("Counter will start from 0")
    print("============================")
    app.run(host='0.0.0.0', port=5001, threaded=True)