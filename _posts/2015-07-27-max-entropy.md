---
layout:     post
title:      "最大熵模型"
data: 2015-07-27 15:47:56
permalink:  maxentropy.html
categories: 机器学习
tags: 最大熵
excerpt: 最大熵模型
mathjax: true
---

* content
{:toc}


## 模型

$$P_w(y|x)=\frac 1 {Z_w(x)}\exp\left(\sum_{i=1}^nw_if_i(x,y)\right)$$

其中，

$$Z_w(x)=\sum_y \exp\left(\sum_{i=1}^nw_if_i(x,y)\right)$$

## 模型的导出
最大熵模型的学习等价于约束最优化问题：

$$\min_{P\in \text C} \quad -H(P)=\sum_{x,y} \widetilde P(x)P(y|x)\log P(y|x)$$

$$\text {s.t.}\quad E_P(f_i)-E_{\widetilde P}(f_i)=0,\quad i=1,2,\cdots,n$$

$$\sum_yP(y|x)=1$$

约束最优化问题转化为无约束对偶问题，定义拉格朗日函数 $L(P,w):$

$$L(P,w)\equiv -H(P)+w_0\left(1-\sum_yP(y|x)\right)+\sum_{i=1}^nw_i\left(E_{\widetilde P}(f_i)-E_P(f_i)\right)$$

最优化的原始问题为

$$\min_{P \in\text C} \max_wL(P,w)$$

对偶问题是

$$\max_w\min_{P \in\text C} L(P,w)$$

求 $L(P,w)$ 对 $P(y\|x)$ 的偏导数并令为 $0$，得

$$P_w(y|x)=\frac 1 {Z_w(x)}\exp\left(\sum_{i=1}^nw_if_i(x,y)\right)$$

其中，

$$Z_w(x)=\sum_y \exp\left(\sum_{i=1}^nw_if_i(x,y)\right)$$

## 算法
之后，求解对偶问题外部的极大化问题的出 $w^*$ . 
简单问题可以令导数为 $0$，复杂的可以参见改进的迭代尺度法(improved iterative scaling，IIS)或者拟牛顿法(如BFGS算法) .
## 补充说明
$f(x,y)$ 为特征函数，定义为

$$
f(x,y) =\begin{cases}
1,  & \text{$x$ 与 $y$ 满足某一事实} \\
0, & \text{否则}
\end{cases}
$$
