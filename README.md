# Simplified Custom Object Detection

```prompt
我正在为 Mihály Kolodko's Mini Statues 制作一个基于 CNN 的物体检测 TorchVision 框架

有 17 种不同的雕像，每一种有都一张图片，其中只有雕像本身，背景透明
它们的文件名是它们的类别名，模型需要能够区分这 17 种雕像

我还从 BingImageCrawler 中随机下载了 50 张大小不同的背景图片

我定义了 ObjectDetectionDataset，它会随机的把雕像合成到背景图片中
使用了 albumentations 数据增强，会把所有图片自动缩放到 640*640

bbox = [
   (rand_x + statue_width / 2) / bg_width,
   (rand_y + statue_height / 2) / bg_height,
   statue_width / bg_width,
   statue_height / bg_height,
   statue_id
]

def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[float]]:
    return img, bbox

还有使用 ObjectDetectionDataset 的 train_loader, val_loader, test_loader
```

# Network Structure

```prompt
使用 CBAM 注意力，使用 AdamW，
使用 YOLO bbox，使用 YOLO Loss，
使用 DCN v2，使用 PReLU，使用 DSC

首先我们来设计网络结构
```

> AdamW: `lr=1e-4, weight_decay=1e-3`

## 1. 特征提取层

### 1.1

1. Conv2d

   ```python
   in_channels: 3
   out_channels: 16
   kernel_size: 3
   stride: 1
   padding: 1
   ```

2. BatchNorm2d
3. PReLU
4. MaxPool2d

   ```python
   kernel_size: 2
   stride: 2
   ```

### 1.2

1. DSC

   ```python
   in_channels: 16
   out_channels: 32
   kernel_size: 3
   stride: 1
   padding: 1
   ```

2. BatchNorm2d
3. PReLU
4. MaxPool2d

   ```python
   kernel_size: 2
   stride: 2
   ```

Depthwise Separable Convolution 用于减少参数量，提高计算效率，
然后通过一个点卷积合并深度卷积的输出

### 1.3

0. ResidualBlock = input

1. Dilated Conv2d

   ```python
   in_channels: 32
   out_channels: 64
   kernel_size: 3
   stride: 1
   padding: 2
   dilation: 2
   ```

2. BatchNorm2d
3. PReLU
4. MaxPool2d

   ```python
   kernel_size: 2
   stride: 2
   ```

5. CBAM 块

### 1.4

1. DSC

   ```python
   in_channels: 64
   out_channels: 128
   kernel_size: 3
   stride: 1
   padding: 1
   ```

2. BatchNorm2d
3. PReLU
4. DropBlock
5. output += 卷积(ResidualBlock)

## 2. 增强层

1. DCN v2

   Dynamic Convolution Network v2  
   来增加网络的适应性，让模型更好地适应雕像的形状变化和尺寸变化

   ```python
   in_channels: 128
   out_channels: 128
   kernel_size: 3
   stride: 1
   padding: 1
   ```

2. BatchNorm2d
3. PReLU
4. CBAM

## 3. 输出层

1. Conv2d

   ```python
   in_channels: 128
   out_channels: 22
   kernel_size: 1
   stride: 1
   padding: 0
   ```

   [x Center, y Center, Width, Height, Confidence, Class]  
   其中 Class 有 17 种，则通道数为 $5 + 17 = 22$

## 4. 后处理

1. NMS

   torchvision.ops  
   非极大值抑制，去除重叠的检测框

2. Loss (Focal + L1)

   计算预测框和真实框之间的损失

3. mAP

   为每个类别计算 AP，然后对这些 AP 值取平均

4. Precision & Recall

   计算 Precision 和 Recall

5. F1 Score

   精确度和召回率的调和平均，是一个综合考虑查准率和查全率的评价指标

6. TensorBoard

   记录训练过程中的损失和评价指标

# Training

```prompt
class Model 和 class ComputeLoss 已经实现
```
