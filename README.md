# HybridDeeplabV3+

[![Paper](https://img.shields.io/badge/Paper-10.1016%2Fj.catena.2026.110343-blue)](https://doi.org/10.1016/j.catena.2026.110343)

Official PyTorch implementation of **"A framework for gully extraction based on thalweg pre-learning and an improved Hybrid-DeepLabv3+ model"**.

本项目是基于 PyTorch 深度学习框架对侵蚀沟提取任务进行高度优化的开源实现[cite: 2]。

## 🌟 模型改进与特性 (Model Highlights)

考虑到侵蚀沟目标复杂的地形特征与狭长形态，本项目在传统网络的基础上进行了以下针对性深度优化：

*   **强化特征提取**：采用大参数量的主干网络 (Backbone)，并专门融入了**轴向注意力机制 (Axial Attention)**[cite: 2]，能够更精准地捕捉侵蚀沟的全局上下文和局部狭长空间特征。
*   **优化的 ASPP 模块**：引入了深度可分离卷积与可变形卷积以适应目标特征。同时，在配置中为 ASPP 模块设计了**手动控制开关**，使研究人员在训练配置时可以自由控制该模块的开启与关闭，提升了实验的灵活性。
*   **复合损失函数优化**：在模型训练阶段，采用 **Cross Entropy (CE) 与 Dice Loss 的组合**进行联合优化，显著提升了模型对不规则边缘的分割精度。
*   **清晰的代码解耦**：训练与推理逻辑分离设计，其中 `deeplab.py` 脚本被专门用于执行高效的模型预测 (Prediction/Inference)[cite: 2]，便于实际部署。

## 💻 运行环境 (Environment)

建议在以下或更高配置的环境中运行本代码：

| 硬件/软件 (Component) | 规格/版本 (Specification) |
| :--- | :--- |
| **操作系统 (OS)** | Windows 11 Home China |
| **显卡 (GPU)** | Nvidia GeForce RTX 5070Ti Laptop GPU 12GB |
| **处理器 (CPU)** | Intel(R) Core(TM) i9-13900HX |
| **内存 (RAM)** | 32GB |
| **编程语言** | Python 3.12.7 |
| **IDE** | Pycharm 2024.3 |
| **深度学习框架** | PyTorch 2.8.0 |
| **CUDA / cuDNN** | CUDA 12.9 / cuDNN 9.9 |

## 📊 数据集 (Dataset)

本模型所使用的数据集由作者以 **GF1 卫星高分辨率影像**为数据源，并使用 Labelme 工具进行精细标注构建而成。
*   **数据集构建与训练详情**：请参阅我们的正式发表论文 (DOI: [10.1016/j.catena.2026.110343](https://doi.org/10.1016/j.catena.2026.110343))。
*   **数据集获取**：如需获取该侵蚀沟数据集用于学术研究，请通过邮件联系作者：`804638568@qq.com`。

## 🖼️ 实验效果 (Results)

模型网络结构示意图：
<img width="620" height="702" alt="Hybrid-Deeplabv+" src="https://github.com/user-attachments/assets/afd47e48-f788-4cd5-b138-d744377027f8" />

侵蚀沟实际提取效果展示：
<img width="829" height="254" alt="image" src="https://github.com/user-attachments/assets/9e99c658-631c-4e74-a179-884e78c3433a" />

<img width="780" height="571" alt="e49afa39-6941-4480-bdfb-817985a235aa" src="https://github.com/user-attachments/assets/239403f5-0ad9-4e87-942f-2a4720e18843" />

## 🙏 致谢 (Acknowledgements)

*   本项目在早期开发与学习阶段，受益于 B 站 UP 主 **“东北Abner说AI”** 提供的 U-Net 模型课程，在此表示诚挚的感谢。
*   部分基础架构代码借鉴并改进自开源项目：[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)。
