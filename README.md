# HybridDeeplabV3-
Official PyTorch implementation of "A framework for gully extraction based on thalweg pre-learning and an improved Hybrid-DeepLabv3+ model".
该模型环境配置：操作系统 (System)：Windows 11 Home China  显卡 (GPU)：Nvidia GeForce RTX 5070Ti Laptop GPU 12GB  处理器 (CPU)：Intel(R) Core(TM) i9-13900HX  内存 (RAM)：32GB  编程语言 (Language)：Python 3.12.7  集成开发环境 (IDE)：Pycharm 2024.3  深度学习框架 (Deep learning framework)：Pytorch 2.8.0  CUDA 版本：12.9  cuDNN 版本：9.9  
该模型是作者学习B站up主“东北Abner说AI”学习Unet模型课程后，根据up主提供的优化后的代码针对侵蚀沟提取任务改进的模型，源模型链接:"https://github.com/milesial/Pytorch-UNet"
作者使用的数据集是自己以GF1卫星影像为数据源，使用Labelme标注的侵蚀沟数据集，具体数据集构建方法和训练 效果见论文：“10.1016/j.catena.2026.110343”。数据集获取可联系：“804638568@qq.com”
作者考虑到侵蚀沟目标的特征，使用大参数的主干网络，并根据目标特征修改了ASPP模块，添加了深度可分离和可变形卷积
最后再次感谢up主“东北Abner说AI”。
