# Simplified Custom Object Detection

```prompt
我正在为 Mihály Kolodko's Mini Statues 制作一个基于 CNN 的，接受不同图片大小输入的物体检测框架

有 17 种不同的雕像，每一种有都一张图片，其中只有雕像本身，背景透明
它们的文件名是它们的类别名，模型需要能够区分这 17 种雕像

我还从 BingImageCrawler 中随机下载了 50 张大小不同的背景图片

我定义了 ObjectDetectionDataset，它会随机的把雕像合成到背景图片中
生成的图片全部为 640*640，使用了 albumentations 数据增强

bbox = [
   statue_name,
   (rand_x + statue_width / 2) / bg_width,
   (rand_y + statue_height / 2) / bg_height,
   statue_width / bg_width,
   statue_height / bg_height
]
getitem ：return img, torch.tensor(bbox, dtype=torch.float)

还有使用 ObjectDetectionDataset 的 train_loader, val_loader, test_loader
```

# Network Structure

```prompt
使用 CBAM 注意力，使用 AdamW，
使用 YOLO bbox，使用 YOLO Loss，
使用 DCN v2，使用 PReLU，使用 DSC

首先我们来设计网络结构，无需代码，尽可能详细
```

## 1. 输入层

AdaptiveAvgPool2d

```python
output_size: 640
```

统一特征图的大小，使得网络能够接受不同大小的输入图片

## 2. 特征提取层

### 2.1

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
4. 最大池化层

   ```python
   kernel_size: 2
   stride: 2
   ```

### 2.2

1. DSC v2

   ```python
   in_channels: 16
   out_channels: 32
   kernel_size: 3
   stride: 1
   padding: 1
   ```

2. 点卷积
3. BatchNorm2d
4. PReLU
5. MaxPool2d

   ```python
   kernel_size: 2
   stride: 2
   ```

Depthwise Separable Convolution 用于减少参数量，提高计算效率，
然后通过一个点卷积合并深度卷积的输出

> ResidualBlock: 记录

### 2.3

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

### 2.4

1. DSC v2

   ```python
   in_channels: 64
   out_channels: 128
   kernel_size: 3
   stride: 1
   padding: 1
   ```

2. 点卷积
3. BatchNorm2d
4. PReLU
5. DropBlock

> ResidualBlock: 写入

## 3. 增强层

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

4. CBAM 块

   Channel Attention  
   Spatial Attention

## 4. 输出层

使用卷积层将特征图映射到目标检测框的表示上。
最后一个卷积层的输出通道数应等于检测框参数的数量

[x Center, y Center, Width, Height, Confidence, Class]  
其中 Class 有 17 种，则通道数为 $5 + 17 = 22$

对于前五个参数使用 sigmoid 函数，将输出限制在 0 到 1 之间  
而 Class 使用 softmax 函数，将所有输出的和限制为 1
