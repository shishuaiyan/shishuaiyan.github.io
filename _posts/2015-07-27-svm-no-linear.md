---
layout:     post
title:      "非线性支持向量机学习算法"
data: 2015-07-27 21:11:50
permalink:  svm-no-linear.html
categories: 机器学习
tags: svm
excerpt: 非线性支持向量机的学习算法
mathjax: true
---

* content
{:toc}

## 算法
$\qquad$ 输入：线性可分训练集 $T=\\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\\}$，其中 $x_i\in \Bbb R^n,y_i\in\\{-1,+1\\}$

$\qquad$ 输出：分类决策函数

$\qquad$ (1) 选择适当的核函数 $\color{blue}{K(x,z)}$ 和惩罚参数 $\color{red}{C\gt 0}$，构造并求解凸二次规划问题

$$\begin{align}\min_\alpha \quad&\frac 1 2 \sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\color{blue}{K(x_i,x_j)}-\sum_{i=1}^N \alpha_i\\
\text {s.t.}\quad &\sum_{i=1}^N\alpha_iy_i=0\\
&\color{red}{0 \le\alpha_i\le C },\quad i=1,2,\cdots,N\end{align}$$

求得最优解 $\alpha^\*=(\alpha_1^\*,\alpha_2^\*,\cdots,\alpha_N^\*)^T$ .

$\qquad$ (2) 选择 $\alpha^\*$ 的一个分量 $\color{red}{0\lt\alpha_j^\* \lt C}$，计算

$$b^*=y_j-\sum_{i=1}^N\alpha_i^*y_i\color{blue}{K(x_i,x_j)}$$

$\qquad$ (3) 构造分类决策函数：

$$f(x)=\text {sign}\left(\sum_{i=1}^N\alpha_i^*y_i\color{blue}{K(x,x_i)}+b^*\right)$$

## 常用核函数
### 多项式核函数

$$K(x,z)=(x\cdot z+1)^p$$

### 高斯核函数

$$K(x,z)=\exp\left(-\frac{\|x-z\|^2}{2\sigma^2}\right)$$

## 一些说明
$\quad$当训练样本容量很大时，一般的凸二次规划最优化算法效率比较低，可以使用序列最小最优化 (sequential minimal optimization, SMO) 算法。
