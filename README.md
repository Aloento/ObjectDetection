# Simplified Custom Object Detection

```prompt
我正在为 Mihály Kolodko's Mini Statues 制作一个基于 CNN 的，接受不同图片大小输入的物体检测框架
我不能使用基础模型和预训练模型，我需要自己从头实现一个简单网络

有 17 种不同的雕像，每一种我有都一张图片，其中只有雕像本身，背景透明
它们都有一个统一的标签 "MiniStatue"，而它们的文件名是它们的类别名

我还从 GoogleImageCrawler 中随机下载了 50 张大小不同的背景图片

我定义了 ObjectDetectionDataset，它会随机的把雕像合成到背景图片中
生成的图片全部为 600\*600，我还引入了 albumentations 数据增强
有 getitem ：return img, bbox (YOLO 格式) 均为 Tensor

我还有使用 ObjectDetectionDataset 的 train_loader, val_loader, test_loader

接下来请帮助我设计一个简单 CNN 网络，接受不同大小的图片输入，输出物体检测的 bbox
需要引入 CBAM 注意力，使用 AdamW，使用 YOLO 格式的 bbox，使用 YOLO Loss，使用 Adaptive Pooling，使用 PReLU

首先我们来设计网络结构，不需要代码，尽可能详细
```
