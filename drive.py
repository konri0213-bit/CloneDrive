# ====== 作用：加载你训练好的模型，连接Udacity模拟器，实时接收摄像头图像，预测方向盘转角并发送控制指令 ======

import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

# ====== 初始化 SocketIO 服务器和 Flask 应用 ======
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


# ====== 定义图像预处理函数 ======
# 这个预处理必须和训练时的预处理完全一致！
def preprocess(image):
    """
    将图像从模拟器格式 (160, 320, 3) 转换为模型输入格式。
    这里做了裁剪（去掉天空和车头）和归一化。
    """
    from PIL import Image
    import numpy as np
    import cv2

    # 模拟器传来的图像是 RGB 格式
    image = np.array(image)
    # 裁剪：去掉顶部70行（天空）和底部25行（车头），保留道路区域
    image = image[70:-25, :, :]
    # 归一化到 [-0.5, 0.5]
    image = image / 255.0 - 0.5
    return image


# ====== SocketIO 事件：模拟器连接成功 ======
@sio.on('connect')
def connect(sid, environ):
    print("Connected to simulator!")
    # 发送一条“准备就绪”的消息，告诉模拟器可以开始发送数据了
    sio.emit('ready', data={'status': 'ready'}, room=sid)


# ====== SocketIO 事件：收到新的图像帧 ======
@sio.on('telemetry')
def telemetry(sid, data):
    """
    每收到一帧模拟器传来的数据，就执行一次推理。
    data 包含：
        - "image": base64编码的图像
        - "steering_angle": 当前方向盘转角（由人类驾驶员产生）
        - "throttle": 当前油门
        - "speed": 当前车速
    """
    global model

    # 1. 检查是否有图像数据
    if data:
        # 2. 解码base64图像为PIL Image
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        try:
            # 3. 预处理图像
            image_array = preprocess(image)
            # 4. 增加一个批次维度 (1, height, width, channels)
            image_array = np.expand_dims(image_array, axis=0)

            # 5. 模型推理：预测方向盘转角
            steering_angle = float(model.predict(image_array, batch_size=1))

            # 6. 设置油门（这里使用简单策略：车速低于10时给全油门，否则给0.2）
            speed = float(data['speed'])
            throttle = 1.0 if speed < 10 else 0.2

            # 7. 打印调试信息
            print(f"Speed: {speed:.2f} | Steering: {steering_angle:.3f} | Throttle: {throttle:.2f}")

            # 8. 发送控制指令回模拟器
            send_control(steering_angle, throttle)

        except Exception as e:
            print(f"Error in telemetry: {e}")

        # 9. 可选：保存当前帧为图像（用于制作视频）
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # 没有数据时，什么也不做
        sio.emit('manual', data={}, skip_sid=True)


def send_control(steering_angle, throttle):
    """
    向模拟器发送方向盘转角和油门指令
    """
    sio.emit(
        'steer',
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


# ====== 主函数入口 ======
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # 检查Keras版本是否兼容
    print(f'Keras version: {keras_version}')

    # 加载模型
    model = load_model(args.model)

    # 如果指定了图像保存路径，则创建文件夹
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # 启动SocketIO服务器，监听4567端口（模拟器默认连接此端口）
    # wrap with Flask application
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)