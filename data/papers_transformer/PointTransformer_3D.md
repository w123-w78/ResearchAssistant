# Point Transformer：三维点云分析

## Abstract
本文提出 Point Transformer，将自注意力引入三维点云处理，
设计适合无序点集的局部注意力机制。
在 S3DIS 语义分割上 mIoU 达到 70.4%，在 ModelNet40 分类上达到 93.7%。

## Method
点云被表示为无序坐标集合，无法直接套用图像的位置编码。
本文提出向量自注意力：注意力权重由位置编码向量调制，
位置编码为两点坐标差经 MLP 映射得到的向量。

网络为 U-Net 式编码器-解码器结构，5 个 stage，
每 stage 内做最远点采样下采样，邻域大小 k=16。
注意力采用减法关系：gamma = MLP(p_i - p_j)，
相比标量点积注意力在点云上 mIoU 提升 2.1 个点。

训练使用 AdamW，学习率 5e-4，batch size 16，训练 100 个 epoch。

## Experiments
数据集：S3DIS（6 个区域，13 类）、ModelNet40（12311 个 CAD 模型）、
ShapeNetPart（部件分割）。
S3DIS Area5 mIoU：Point Transformer 为 70.4%，
PointNet++ 为 53.5%，KPConv 为 67.1%。
ModelNet40 总体准确率 93.7%。
消融显示，邻域 k 从 16 降到 8 时 mIoU 下降 1.3；
去掉位置编码调制后 mIoU 暴跌至 61.2%。

## Conclusion
向量形式的相对位置注意力是点云 Transformer 的关键，
局部邻域注意力比全局注意力更适合三维数据。
