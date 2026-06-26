# 🚗 行为克隆 - 端到端自动驾驶

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 项目简介

本项目实现了一个基于**行为克隆**的自动驾驶系统。简单来说，就是让一个深度神经网络观看人类驾驶员在模拟器中的操作，学习如何根据摄像头画面来决定方向盘该打多少角度，最终实现车辆在赛道上的自主行驶。

**核心原理**：模型接收车载摄像头拍摄的道路图像作为输入，输出一个连续的方向盘转角值（回归任务）。整个过程是端到端的，即"图像像素 → 转向角度"，不需要人工提取任何特征。

---

## 🎯 核心特性

- **端到端学习**：从原始像素直接预测转向角度，无需人工设计特征
- **NVIDIA 架构**：采用英伟达提出的经典卷积神经网络结构，专为自动驾驶设计
- **数据增强**：训练时实时进行图像翻转、亮度调整等操作，提升模型泛化能力
- **仿真测试**：连接 Udacity 模拟器，在虚拟环境中验证模型效果

---

## 📁 项目结构
behavioral-cloning/
│
├── model.py # 模型定义与训练脚本（核心）
├── drive.py # 驱动脚本（连接模型与模拟器）
├── model.h5 # 训练好的模型权重（运行后生成）
├── README.md # 本文件
├── requirements.txt # Python 依赖清单
│
└── data/ # 训练数据（需自己录制）
├── driving_log.csv # 驾驶日志
└── IMG/ # 图像文件夹
├── center_xxx.jpg
└── ...

text

---

## 🛠️ 环境准备

### 1. 安装 Python 依赖

打开终端（CMD 或 PyCharm 的 Terminal），进入项目文件夹，执行：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
如果提示 NumPy 版本冲突，请执行：

bash
pip install "numpy<2.0" --force-reinstall
2. 下载 Udacity 模拟器
访问官方发布页下载对应你操作系统的模拟器：
👉 https://github.com/udacity/self-driving-car-sim/releases

Windows：下载 windows.zip

macOS：下载 mac.zip

Linux：下载 linux.zip

下载后解压到任意位置，双击 term1_sim.exe（Windows）或对应文件（Mac）即可运行。

⚠️ 注意：如果模拟器启动后闪退，请尝试右键点击 → "以管理员身份运行"，或者检查显卡驱动是否为最新版本。

📊 录制训练数据（最关键的一步）
模型最终表现如何，80% 取决于你录制的数据质量，请认真对待。

操作步骤
启动模拟器，点击 Training Mode（训练模式）。

点击 Record 按钮，在弹出的路径选择框中，选择项目目录下的 data 文件夹（如果不存在请手动创建）。

开始驾驶：

↑ 键：油门加速

↓ 键：刹车减速

← / → 键：左转 / 右转

驾驶技巧（非常重要！）：

尽量保持在车道正中央行驶

方向盘操作要平滑，避免忽左忽右

速度控制在 15 ~ 25 mph 之间

至少跑 2 整圈（建议正反方向各一圈，平衡左右转弯数据）

收集"救车"数据（极其关键！）：

故意把车开到车道边缘（接近白线，但不要冲出路肩）

然后缓慢平稳地打方向盘，让车回到中心

重复 5 ~ 8 次，这能教会模型如何在冲出赛道时自救

驾驶完成后，点击 Stop 停止录制。

检查数据
录制完成后，你的 data 文件夹应该包含：

driving_log.csv —— 记录每一帧的图像路径和驾驶指令

IMG/ 文件夹 —— 所有驾驶截图（可能有几千到上万张）

🧠 训练模型
确保终端当前路径在项目根目录下，执行：

bash
python model.py
训练过程说明
屏幕上会滚动显示训练进度和损失值（loss 和 val_loss）

loss 数值逐渐下降，说明模型正在学习

训练时间取决于电脑配置（有无独立显卡），大约 5 ~ 30 分钟

训练完成后，项目根目录会生成 model.h5 文件，这就是训练好的模型权重

🚀 测试自动驾驶
重新启动模拟器，这次选择 Autonomous Mode（自动驾驶模式）

在终端执行：

bash
python drive.py model.h5
观察模拟器画面：车辆应该开始自动打方向盘，沿赛道行驶！

保存行驶录像（可选）：

bash
python drive.py model.h5 ./run1
每一帧图像会保存到 run1/ 文件夹中，可后期合成视频。

❓ 常见问题排查
问题	可能原因	解决方法
ModuleNotFoundError: No module named 'xxx'	依赖包未安装	执行 pip install -r requirements.txt
NumPy 报错 _ARRAY_API not found	NumPy 版本太新	执行 pip install "numpy<2.0" --force-reinstall
FileNotFoundError: data/driving_log.csv	数据路径不对	确认 data 文件夹在项目根目录下，且 CSV 文件名正确
模拟器启动后闪退	权限不够或缺少运行库	右键"以管理员身份运行"；更新显卡驱动
训练时 loss 显示 nan	学习率太高或图像读取失败	降低学习率；检查图像路径是否正确
车辆开出赛道	训练数据不足或质量差	补充更多"救车"数据和弯道数据，重新训练
车辆原地打转	油门逻辑问题	调整 drive.py 中的油门阈值（speed < 10 中的数值）
🔧 参数调优建议
调整训练参数（model.py 中）
python
epochs = 10          # 训练轮数，可增加到 15~20
batch_size = 32      # 批次大小，显存不够可调小为 16 或 8
learning_rate = 1e-4 # 学习率，在 model.compile 中修改
调整油门逻辑（drive.py 中）
python
throttle = 1.0 if speed < 10 else 0.2
车太慢：把 10 改大（如 15）

车太快容易冲出：把 0.2 改小（如 0.15）

📄 许可证
本项目采用 MIT 许可证，可自由使用和修改。

🙏 致谢
Udacity 提供的模拟器和课程资料

NVIDIA 提供的端到端驾驶架构论文

祝训练顺利！ 🚗💨
