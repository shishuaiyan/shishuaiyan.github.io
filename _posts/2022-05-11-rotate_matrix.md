---
layout:     post
title:      "旋转矩阵"
data: 2022年5月11日17:30:44
permalink:  rotate_matrix.html
categories: control
tags: control
excerpt: 控制中的旋转矩阵相关知识整理
---
* content
  {:toc}

# 1 基础知识

## 1.1 左右手坐标系

# 2 矩阵

矩阵的意义：

- 描述一个坐标系
- 描述一个运动

符号约定：

- $^AP_a$: 点$a$在坐标系$A$中的位姿（矩阵）
- $^A_BR$: 坐标系$B$到坐标系$A$的旋转矩阵
  - $^A_BR=[v_x,v_y,v_z]$ 其中列向量$v_x,v_y,v_z$分别为坐标系$B$的xyz轴单位向量在坐标系$A$中的坐标

# 3 三维旋转矩阵

**旋转方向**

- 右手定则
  - 右手大拇指指向旋转轴正方向，四指合上的方向为正

- 顺着旋转轴（人眼在旋转轴0点向旋转轴正方向看）时，顺时针旋转为正，逆时针旋转为负
- 逆着旋转轴（人眼在旋转轴正方向上向旋转轴0点看）时，顺时针旋转为负，逆时针旋转为正

举例：Rz = 30°：逆着$\overrightarrow{v}=[0, 0, 1]$的方向看（人眼在z轴正方向上向z轴0点看），表示逆时针旋转30 degree

## 内旋，外旋

世界坐标系{O}

### 内旋

内旋(intrinsic rotations)、旋转轴(rotated axis)

旋转顺序`zyx`：

1. {O}绕{O}中的`z`轴旋转得到{A}

2. {A}绕{A}中的`y`轴旋转得到{B}

3. {B}绕{B}中的`x`轴旋转得到{C}，即最终坐标系

旋转矩阵可表示为：
$$
R=R_zR_yR_x
$$
对应**矩阵右乘**

### 外旋

外旋(extrinsic rotations)、固定轴(static/fixed axis)

旋转顺序`xyz`：

1. {O}绕{O}中的`x`轴旋转得到{A}

2. {A}绕{O}中的`y`轴旋转得到{B}

3. {B}绕{O}中的`z`轴旋转得到{C}，即最终坐标系

旋转矩阵可表示为：
$$
R=R_zR_yR_x
$$
对应**矩阵左乘**

### tips

- 绕一个轴旋转+180°等效于绕该轴旋转-180°
- 内旋的ZYX等效于外旋的XYZ（顺序相反），其他旋转方式同理（最终的旋转矩阵相同）



```python
from lnodes.libs.motion_control.math_utils import MathUtils
import numpy as np
np.set_printoptions(suppress=True, precision=6)

# 内旋：矩阵右乘
# 外旋：矩阵左乘
# R = Rz@Ry@Rx 外旋旋转顺序：XYZ  内旋旋转顺序：ZYX

vec = np.array([0, 0, 1], dtype=np.float64)
# 旋转矩阵：{A} -> {B}; 内旋：ZYX  外旋：XYZ
# rotate_mat = MathUtils.eulerAnglesToRotationMatrix([-np.pi / 2, np.pi / 2, -np.pi])
rotate_mat = MathUtils.eulerAnglesToRotationMatrix([np.pi / 2, np.pi / 2, 0])  # 与上述旋转等效
# 绕一个轴旋转 +pi 等效于绕该轴旋转 -pi
print(rotate_mat)
#[[ 0.  1.  0.]
# [ 0.  0. -1.]
# [-1.  0.  0.]]
angles = MathUtils.rotationMatrixToEulerAngles(rotate_mat)
print(angles)
# [1.570796 1.570796 0.      ]

# 向量在坐标系{B}下的坐标（向量本身未运动）
vec.dot(rotate_mat)
# array([-1., -0.,  0.])

# 向量旋转后在原始坐标系{A}下的坐标
rotate_mat.dot(vec)
# array([ 0., -1.,  0.])

r1 = MathUtils.eulerAnglesToRotationMatrix([np.pi / 2, 0, 0])
r2 = MathUtils.eulerAnglesToRotationMatrix([0, np.pi / 2, 0])
# print(r1.dot(r2))
print(r2.dot(r1)) # *
#[[ 0.  1.  0.]
# [ 0.  0. -1.]
# [-1.  0.  0.]]
```

# 4 四维齐次矩阵

3×3变换矩阵表示的是线性变换，不包含平移。因为矩阵乘法的性质，零向量总是变换成零向量，因此，任何能用矩阵乘法表达的变换都不包含平移。这很不幸，因为矩阵乘法和它的逆是种非常方便的工具，不仅可以用来将复杂的变换组合成简单的单一变换，还可以操纵嵌入式坐标系间的关系。如果能找到一种方法将3×3变换矩阵进行扩展，使它能处理平移，这将是一件多么美妙的事情啊，4×4矩阵恰好提供了一种数学上的“技巧”，使我们能够做到这一点。 

4D向量有4个分量，前3个是标准的x,y和z分量，第四个是w，有时称作**齐次坐标**。

为了理解标准3D坐标是怎样扩展到4D坐标的，让我们先看一下2D中的齐次坐标，它的形式为(x，y，w)。想象在3D中w=1处的标准2D平面，实际的2D点(x，y)用齐次坐标表示为(x，y，1)，对于那些不在w=1平面上的点，则将它们投影到w=1平面上。所以齐次坐标(x，y，w)映射的实际2D点为(x/w，y/w)。

因此，给定一个2D点(x，y)，齐次空间中有无数多个点与之对应。所有点的形式都为(kx，ky，k)，k≠0。这些点构成一条穿过齐次原点的直线。

当w=0时，除法未定义，因此不存在实际的2D点。然而，可以将2D齐次点(x，y，0)解释为“位于无穷远的点”，它描述了一个方向而不是一个位置。

4D坐标的基本思想相同。实际的3D点能被认为是在4D中w=1“平面”上。4D点的形式为(x，y，z，w)，将4D点投影到这个"平面"上得到相应的实际3D点(x/w，y/w，z/w)。w=0时4D点表示"无限远点",它描述了一个方向而不是一个位置。

齐次坐标和通过除以w来投影是很有趣的，那我们为什么要使用4D坐标呢？有两个基本原因使得我们要使用4D向量和4×4矩阵。第一个原因实际上就是因为它是一种方便的记法。

## SE(3)

将旋转矩阵和平移向量写在同一个矩阵中，形成的4x4矩阵，称为special euclidean group, 即SE(3)
$$
T=\begin{bmatrix}R&p\\0&1\end{bmatrix}
=\begin{bmatrix}1&0&0&p_1\\0&1&0&p_2\\0&0&1&p_3\\0&0&0&1\end{bmatrix}
 \begin{bmatrix}r_{11}&r_{12}&r_{13}&0\\r_{21}&r_{22}&r_{23}&0\\r_{31}&r_{32}&r_{33}&0\\0&0&0&1\end{bmatrix}
=\begin{bmatrix}r_{11}&r_{12}&r_{13}&p_1\\r_{21}&r_{22}&r_{23}&p_2\\r_{31}&r_{32}&r_{33}&p_3\\0&0&0&1\end{bmatrix}
$$
**先移动，后旋转**

容易验证，齐次变换矩阵满足群（group）所具有的的性质，即封闭性，结合律，幺元，逆
$$
T^{-1}=\begin{bmatrix}R&p\\0&1\end{bmatrix}^{-1}=\begin{bmatrix}R^T&-R^Tp\\0&1\end{bmatrix}
$$

$$
(T_1T_2)T_3=T_1(T_2T_3)
$$

此外**齐次变换矩阵还能保持变换前后的距离和角度不变**

## 旋转平移



## 齐次坐标系的左乘、右乘

世界坐标系{s}

齐次坐标系$T_b=\begin{bmatrix}R_b&p_b\\0&1\end{bmatrix}=\begin{bmatrix}0&0&1&0\\0&-1&0&-2\\1&0&0&0\\0&0&0&1\end{bmatrix}$ （平面运动[0, -2, 0]；ZYX内旋欧拉角[0°, -90°, 180°]）

齐次坐标系$T=\begin{bmatrix}R&p\\0&1\end{bmatrix}=\begin{bmatrix}0&-1&0&0\\1&0&0&2\\0&0&1&0\\0&0&0&1\end{bmatrix}$ （平面运动[0, 2, 0]；ZYX内旋欧拉角[0°, 0°, 90°]）

左乘（fixed frame）：
$$
T_{b^\prime}=TT_b
=\begin{bmatrix}R&p\\0&1\end{bmatrix}\begin{bmatrix}R_b&p_b\\0&1\end{bmatrix}
=\begin{bmatrix}RR_b&Rp_b+p\\0&1\end{bmatrix}
=\begin{bmatrix}0&1&0&2\\0&0&1&2\\1&0&0&0\\0&0&0&1\end{bmatrix}
$$
右乘(body frame)：
$$
T_{b^\prime}=T_bT
=\begin{bmatrix}R_b&p_b\\0&1\end{bmatrix}\begin{bmatrix}R&p\\0&1\end{bmatrix}
=\begin{bmatrix}R_bR&Rp+p_b\\0&1\end{bmatrix}
=\begin{bmatrix}0&0&1&0\\-1&0&0&-4\\0&-1&0&0\\0&0&0&1\end{bmatrix}
$$
![img](E:\code\github\shishuaiyan.github.io\img\hom_mat_multi.png)

# 坐标变换

这里的坐标变换指的是**将一个坐标系中的向量在其他坐标系通进行变换（描述），向量本身并没有变换，只不过对它的描述变换了**

已知向量$^AP=\begin{bmatrix}p_x\\p_y\\p_z\end{bmatrix}$

## 旋转
