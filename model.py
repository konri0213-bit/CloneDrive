#   行为克隆自动驾驶 ============================
# 1. 导入必要的库
# ============================
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D

# ============================
# 2. 加载数据路径
# ============================
# 假设你的 driving_log.csv 和 IMG 文件夹都在 './data/' 目录下
lines = []
with open('驾驶数据的csv文件') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# 将数据集分为训练集和验证集（80% 训练，20% 验证）
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# ============================
# 3. 数据生成器（动态加载，节省内存）
# ============================
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:  # 无限循环，供 Keras 训练时调用
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # 获取图像路径（注意：CSV里存储的是绝对路径或相对路径，这里只取文件名）
                # 如果你CSV里是 "data/IMG/xxx.jpg"，直接 split 取最后一部分
                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                current_path = './data/IMG/' + filename

                # 读取图像
                image = cv2.imread(current_path)
                # 获取方向盘转角（CSV第4列）
                angle = float(batch_sample[3])

                # 原始图像加入训练集
                images.append(image)
                angles.append(angle)

                # 数据增强：水平翻转（图像翻转，角度取反），有效应对赛道转弯不平衡
                images.append(cv2.flip(image, 1))
                angles.append(-angle)

            # 转换为 numpy 数组并归一化（归一化也可以在模型里做，这里先保留原始像素）
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# 创建训练和验证生成器
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# ============================
# 4. 定义 NVIDIA 模型架构
# ============================
def nvidia_model():
    model = Sequential()
    # 归一化：将像素值从 0-255 缩放到 -0.5 到 0.5
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # 裁剪：去掉顶部 70 像素（天空）和底部 25 像素（车头）
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # 卷积层（提取特征）
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # 全连接层（映射到转向角度）
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # 输出一个连续的转向角度值

    return model


model = nvidia_model()

# ============================
# 5. 编译并训练模型
# ============================
model.compile(loss='mse', optimizer='adam')

print("开始训练...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_samples) * 2 // 32,  # *2 是因为做了翻转增强
    validation_data=validation_generator,
    validation_steps=len(validation_samples) * 2 // 32,
    epochs=10,  # 训练轮数，可以适当增加
    verbose=1
)

# ============================
# 6. 保存模型为 model.h5
# ============================
model.save('model.h5')
print("✅ 模型已成功保存为 model.h5")
