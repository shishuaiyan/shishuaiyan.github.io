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


# 机械臂中的坐标变换

# 向量

## 向量的定义

- 定义
  - 向量（也称为矢量），指具有大小和方向的量。可以形象化地表示为带箭头的线段。
    - 箭头指向代表向量的方向
    - 线段长度代表向量的大小

## 向量的运算

### 向量加法、减法

![](static/GF6mbexR3o52l3xjP9UcUpWmnOg.PNG)

![](static/FVVEbvEVTouiXlxuV86cgvMHnGc.PNG)

### 向量点乘

![](static/IzZUb9Q83oRxI9xLfaEcWXfVnid.PNG)

已知两个向量：

$$
\overrightarrow{a}=(x_1, y_1)\\
\overrightarrow{b}=(x_2, y_2)
$$

- 点乘的数学定义：$\overrightarrow{a}\cdot\overrightarrow{b}=x_1x_2+y_1y_2$
- 点乘的几何含义：

  - $$
    \overrightarrow{a}\cdot\overrightarrow{b}=|\overrightarrow{a}|\cdot|\overrightarrow{b}|\cdot cos(\theta)
    $$
  - 向量的点乘可以用来计算两个向量之间的夹角，进一步判断这两个向量是否正交（垂直）等方向关系。同时，还可以用来计算一个向量在另一个向量方向上的投影长度。
- 点乘的结果是一个标量
- np.dot(a, b)

### 向量叉乘

![](static/AEa0brOu9oyEH7xtibIcIa0xnme.PNG)

对于向量 $\overrightarrow{a}=(x_1, y_1)$和 $\overrightarrow{b}=(x_2, y_2)$

- 叉乘的数学定义：$\overrightarrow{a}\times\overrightarrow{b}=x_1y_2-y_1x_2$
- 叉乘的几何含义：

  - $$
    \overrightarrow{a}\times\overrightarrow{b}=|\overrightarrow{a}|\cdot|\overrightarrow{b}|\cdot sin(\theta)
    $$
- 叉乘的结果是一个向量而不是标量，上述结果是它的模，结果向量 $\overrightarrow{c}$ 的方向与 $\overrightarrow{a}, \overrightarrow{b}$所在的平面垂直，方向用“右手法则”判断

  - 右手手掌张开，四指并拢，大拇指垂直于四指指向的方向
  - 伸出右手，四指弯曲，四指与 A 旋转到 B 方向一致，那么大拇指指向为 C 向量的方向

![](static/H3c3bN4oio2nFlxhHB4ccgO3n4d.png)

- np.cross(a, b)

# 刚体运动状态的描述

- 刚体 rigid body
- DOF degree of freedom 自由度

## 平面

- 移动：2 DOFs
- 转动：1 DOF

## 三维

- 移动：3 DOFs
- 转动：3 DOFs

# 旋转矩阵

$$
^A_BR=
\begin{bmatrix}
|&|&|\\
^A\hat{X}_B&^A\hat{Y}_B&^A\hat{Z}_B\\
|&|&|\end{bmatrix}
=\begin{bmatrix}
\hat{X}_B\cdot\hat{X}_A&
\hat{Y}_B\cdot\hat{X}_A&
\hat{Z}_B\cdot\hat{X}_A\\
\hat{X}_B\cdot\hat{Y}_A&
\hat{Y}_B\cdot\hat{Y}_A&
\hat{Z}_B\cdot\hat{Y}_A\\
\hat{X}_B\cdot\hat{Z}_A&
\hat{Y}_B\cdot\hat{Z}_A&
\hat{Z}_B\cdot\hat{Z}_A\end{bmatrix}
$$

- B 相对于（relative to）A 的旋转矩阵
- R 的三个列向量为 frame{B} 的基坐标： $\hat{X}_B,\hat{Y}_B,\hat{Z}_B$（由{A}看）

  - $^A\hat{X}_B$为 frame{B}中 X 轴的基在 frame{A}中的投影，$^A\hat{Y}_B,^A\hat{Z}_B$同理
- 向量内积 :  $\begin{bmatrix}a_1&a_2&a_3\end{bmatrix}\cdot\begin{bmatrix}b_1&b_2&b_3\end{bmatrix}=a_1b_1+a_2b_2+a_3b_3$

  - 几何意义：两向量的长度相乘，再乘以两向量的夹角的 cos

由于点乘具有交换律： $\hat{X}_B\cdot\hat{X}_A=\hat{X}_A\cdot\hat{X}_B$，将矩阵中所有点乘交互顺序：

$$
=\begin{bmatrix}
\hat{X}_A\cdot\hat{X}_B&
\hat{X}_A\cdot\hat{Y}_B&
\hat{X}_A\cdot\hat{Z}_B\\
\hat{Y}_A\cdot\hat{X}_B&
\hat{Y}_A\cdot\hat{Y}_B&
\hat{Y}_A\cdot\hat{Z}_B\\
\hat{Z}_A\cdot\hat{X}_B&
\hat{Z}_A\cdot\hat{Y}_B&
\hat{Z}_A\cdot\hat{Z}_B\end{bmatrix}
=
\begin{bmatrix}
-&{^B\hat{X}_A}^T&-\\
-&{^B\hat{Y}_A}^T&-\\
-&{^B\hat{Z}_A}^T&-\end{bmatrix}
$$

- R 的三个行向量分别为 frame{A}的基在 frame{B}中的投影

$$
=\begin{bmatrix}
|&|&|\\
^B\hat{X}_A&^B\hat{Y}_A&^B\hat{Z}_A\\
|&|&|\end{bmatrix}^T=^B_AR^T
$$

特性如下（正交矩阵）：

- $$
  ^A_BR=^B_AR^T
  $$
- $$
  ^A_BR^T=^B_AR^{-1}
  $$

## 单次旋转（以 Z 轴旋转为例）

![](static/XqrpboaO1oLsBNxmmiRc3X25nnf.png)

- 旋转方向
  - 从 z 轴正方向向下看，逆时针旋转为正
  - 顺着右手定则方向为正，反之为负

$R_{\hat{Z}_A}(\theta)=\begin{bmatrix}
c\theta&-s\theta&0\\
s\theta&c\theta&0\\
0&0&1\end{bmatrix}=^A_BR$

- $\hat{Z}_A$为旋转轴
- $\theta$为旋转角度

## 旋转矩阵的三种用法

### 描述坐标系

1. 描述一个 frame 相对于另一个 frame 的姿态

![](static/AeHwbfYuxo04VaxH9mocfo5Bnrh.png)

$^A_BR=
\begin{bmatrix}
|&|&|\\
^A\hat{X}_B&^A\hat{Y}_B&^A\hat{Z}_B\\
|&|&|\end{bmatrix}$

### 描述坐标变化

1. 将 point 由某一个 frame 的表达转换到另一个和此 frame 仅有相对转动的 frame 来表达
   1. 向量不运动，坐标系旋转（mapping）
      ![](static/ESPDby7FVoqtylxj07VcIj4Wnke.png)

$^AP=^A_BR^BP$

### 描述一个运动

1. 将 point（vector）在同一个 frame 中进行转动
   1. 向量旋转，坐标系不动（operate）

![](static/MIKobbBxgo7arUxxSJlces80nme.png)

$^AP'=R(\theta)^AP$

## 将旋转矩阵拆解成三次旋转连乘

- 注意事项：

  - 旋转顺序，不同的旋转顺序会导致最终结果不同
  - 旋转轴，是对固定不动的轴旋转，还是对转动 frame 下的轴旋转
- 两种拆解方式：

  - Fixed angles: 对固定不动的转轴旋转
  - Euler angles：对旋转 frame 当下所在的转轴方向旋转

### Fixed Angles

![](static/FmpmbIgCUoCTXFxYtutcJdEpnUb.png)

$^A_BR_{XYZ}(\gamma,\beta,\alpha)=R_Z(\alpha)R_Y(\beta)R_X(\gamma)$

固定轴：

- 参考上面的向量 operate: $^AP'=R(\theta)^AP$
- 初始状态：frame{B}的三个基向量 $\hat{X}_{B},\hat{Y}_{B},\hat{Z}_{B}$与 frame{A}的三个基向量$\hat{X}_{A},\hat{Y}_{A},\hat{Z}_{A}$完全重合（相等）

  - $\hat{X}_{B},\hat{Y}_{B},\hat{Z}_{B}$在固定 frame{A}中运动（旋转）至$\hat{X}_{B'},\hat{Y}_{B'},\hat{Z}_{B'}$, 旋转矩阵为 $R_1$
  - $\hat{X}_{B'},\hat{Y}_{B'},\hat{Z}_{B'}$在固定 frame{A}中运动（旋转）至$\hat{X}_{B''},\hat{Y}_{B''},\hat{Z}_{B''}$, 旋转矩阵为 $R_2$
  - $\hat{X}_{B''},\hat{Y}_{B''},\hat{Z}_{B''}$在固定 frame{A}中运动（旋转）至$\hat{X}_{B'''},\hat{Y}_{B'''},\hat{Z}_{B'''}$, 旋转矩阵为 $R_3$
- $v'=^A_BRv=R_3R_2R_1v$

  - 按照对向量 $v$操作（旋转）的先后顺序依次左乘，先操作的贴着$v$，后操作的远离$v$

### Euler Angles

![](static/BVEEbLdopo3UARxZUO4cv0vpnUd.png)

$^A_BR_{Z'Y'X'}(\alpha,\beta,\gamma)
=^A_{B'}R^{B'}_{B''}R^{B''}_{B'''}R
=R_Z(\alpha)R_Y(\beta)R_X(\gamma)$

非固定轴：

- 参考上面的向量 mapping：$^AP=^A_BR^BP$
- 对某一个向量 $^BP$，从最后一个 frame 逐渐转换（旋转）来回到第一个 frame $^AP$
- $^AP=^A_BR^BP=R_1R_2R_3^BP$

## 矩阵运算

### 矩阵乘向量

已知 $v=\begin{bmatrix}a&b&c\end{bmatrix}^T,
^A_BR=
\begin{bmatrix}
r_{11}&r_{12}&r_{13}\\
r_{21}&r_{22}&r_{23}\\
r_{31}&r_{32}&r_{33}\end{bmatrix}
=\begin{bmatrix}
|&|&|\\
^A\hat{X}_B&^A\hat{Y}_B&^A\hat{Z}_B\\
|&|&|\end{bmatrix}
=\begin{bmatrix}
-&{^B\hat{X}_A}^T&-\\
-&{^B\hat{Y}_A}^T&-\\
-&{^B\hat{Z}_A}^T&-\end{bmatrix}$

![](static/XdKSbnhrao0W8Zx6D9Kcp2oDnTb.png)

```python
A_B_R = trot2(pi/3)                # 逆时针旋转60度
v = np.array([np.sqrt(3),1,1])     # 齐次表示
A_B_R @ v
# [0, 2, 1]
```

$^A_BRv=v'$

- 几何意义

  - 点（向量）$v$如果是 frame{A}中的一点（上图中的 A_null_v），则表示点的运动 $v\rightarrow v'$
  - 点（向量）$v$如果是 frame{B}中的一点（上图中的 B_null_v），$v'$表示点$v$在 frame{A}中的坐标（点未运动）
- 从数值上看

  - $v'=\begin{bmatrix}
    a\cdot{^B\hat{X}_A}^T&
    b\cdot{^B\hat{Y}_A}^T&
    c\cdot{^B\hat{Z}_A}^T\end{bmatrix}^T$

![](static/AVtCbUu4toIylcxbiQ7cWdTunAb.png)

```python
A_B_R = trot2(pi/3)                # 逆时针旋转60度
v = np.array([np.sqrt(3),1,1])     # 齐次表示
v @ A_B_R
# [1.732, -1, 1]
```

${v^T}^A_BR=v''$

- 几何意义

  - frame{A}中的点 $v$（A_null_v）在旋转后的 frame{B}中的坐标（B_null_v）
- 从数值上看

  - $v''=\begin{bmatrix}
    a\cdot^A\hat{X}_B&
    b\cdot^A\hat{Y}_B&
    c\cdot^A\hat{Z}_B\end{bmatrix}$

### 矩阵乘矩阵

$^A_BR ^B_CR ^C_DR=^A_CR ^C_DR=^A_DR$

- $^A_BR$ 的三个列向量为 frame{B} 的基：$\hat{X}_B,\hat{Y}_B,\hat{Z}_B$在 frame{A}中的坐标
- $^B_CR$ 的三个列向量为 frame{C} 的基：$\hat{X}_C,\hat{Y}_C,\hat{Z}_C$在 frame{B}中的坐标
- ...
- 矩阵的连乘是 Euler Angles 的形式，在当前坐标系下继续旋转

# 齐次矩阵

## 基础

将移动和转动整合在一起描述

$$
^A_BT=
\begin{bmatrix}
&^A_BR_{3\times3}&&^AP_{B_{org}}\\
0&0&0&1\end{bmatrix}
$$

- Mapping
  ![](static/VeRFbeAEmof3qaxOn5AchLt3nrg.png)

  - $^AP_{3\times 1}=^A_BR ^BP+^AP_{B_{org}}$
  - $\begin{bmatrix}^AP\\1\end{bmatrix}=
    \begin{bmatrix}
    &^A_BR_{3\times3}&&^AP_{B_{org}}\\
    0&0&0&1\end{bmatrix}
    \begin{bmatrix}^BP\\1\end{bmatrix}=
    \begin{bmatrix}^A_BR ^BP+^AP_{B_{org}}\\1\end{bmatrix}$
- Operate
  ![](static/TQA8bUIRqons0IxjkRvciLLmneb.png)

  - $$
    ^AP_{2}=R(\theta) ^AP_1+^AQ
    $$
  - $\begin{bmatrix}^AP_2\\1\end{bmatrix}=
    \begin{bmatrix}
    &R(\theta)&&^AQ\\
    0&0&0&1\end{bmatrix}
    \begin{bmatrix}^AP_1\\1\end{bmatrix}=
    \begin{bmatrix}
    R(\theta)^AP_1 +^AQ\\1\end{bmatrix}$
    ![](static/R1o8bK0xsosJ6GxbT37cftQZnth.png)
    ![](static/ZH7EbkhBBo74uOxLwpwcKAhMnNh.png)

```python
R = trot2(pi/3) @ transl2(1.5, 1)
 v = np.array([1, 0])
 R @ v
 # array([0.383975, 2.665064, 1.      ])
```

## 运算

### 连乘

![](static/Tt5UbJrmKo6D2Hx0q4ccd2Qyncb.png)

### 反矩阵

![](static/AeX8bGFjPo9yiOxRmNgcRPohnWd.png)

## 实例

已知机械臂末端

$$
\begin{aligned}
^A_CT^{-1} ^A_BT &=
\begin{bmatrix}
^A_CR&^AP_{AC}\\
0&1\end{bmatrix}^{-1}\begin{bmatrix}
^A_BR&^AP_{AB}\\
0&1\end{bmatrix} \\&=
\begin{bmatrix}
^C_AR&-^C_AR^AP_{AC}\\
0&1\end{bmatrix}\begin{bmatrix}
^A_BR&^AP_{AB}\\
0&1\end{bmatrix} \\&=
\begin{bmatrix}
^C_AR^A_BR&^C_AR^AP_{AB}-^C_AR^AP_{AC}\\
0&1\end{bmatrix} \\&=
\begin{bmatrix}
^C_BR & ^CP_{AB}-^CP_{AC}\\
0&1\end{bmatrix} \\&=
\begin{bmatrix}
^C_BR & ^CP_{CB}\\
0&1\end{bmatrix}
\end{aligned}
$$

![](static/H0LpbQPaqo8y2oxi4ancCqrxnUg.png)

```python
A_B_T = transl2(2, 1) @ trot2(0.3)
A_C_T = transl2(3, 3) @ trot2(0.2)

trplot2( np.identity(3), frame='A', width=2, color='black' )
trplot2( A_B_T, frame='B', width=2, color='blue' )
trplot2( A_C_T, frame='C', width=2, color='g', dims=[-1, 5, -1, 5] )
plot_line((0, 0), (2, 1), linestyle=':', color='blue')
plot_line((0, 0), (3, 3), linestyle=':', color='g')
plot_line((2, 1), (3, 3), linestyle='-', color='orange')
plt.grid(True)

print(A_B_T)
#[[ 0.955336 -0.29552   2.      ]
# [ 0.29552   0.955336  1.      ]
# [ 0.        0.        1.      ]]
print(A_C_T)
#[[ 0.980067 -0.198669  3.      ]
# [ 0.198669  0.980067  3.      ]
# [ 0.        0.        1.      ]]
print(np.linalg.inv(A_C_T) @ A_B_T)
#[[ 0.995004 -0.099833 -1.377405]
# [ 0.099833  0.995004 -1.761464]
# [ 0.        0.        1.      ]]
```
