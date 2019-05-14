---
layout:     post
title:      "感知机(perceptron)学习算法的对偶形式"
data: 2015-07-24 11:40:13
permalink:  perceptron-dual.html
categories: 机器学习
tags: 感知机
excerpt: 感知机的对偶形式
mathjax: true
---

* content
{:toc}

## 算法
$\qquad$ 输入：训练数据集$T=\\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\\}$，
其中$x_i\in \Bbb R^n$,$y_i \in \\{-1,1\\}，i=1,2,\ldots,N$ ；学习率 $\eta(0\lt\eta\le1)$;

$\qquad$ 输出：$\alpha,b$；感知机模型$f(x)=sign(\sum_{j=1}^N \alpha_jy_jx_j\cdot x+b)$.
其中$\alpha=(\alpha_1,\alpha_2,\ldots,\alpha_N)^T$.

$\qquad$ (1) $\alpha\leftarrow 0 ,b\leftarrow 0$

$\qquad$ (2) 在训练集中选取数据$(x_i,y_i)$

$\qquad$ (3) 如果$y_i(\sum_{j=1}^N \alpha_jy_jx_j\cdot x_i+b)\le 0$ 

$$\alpha_i\leftarrow \alpha_i+\eta $$

$$b\leftarrow b+\eta y_i$$

$\qquad$ (4) 转至(2),直至训练集中没有误分类点。


## 模型

$$f(x)=sign(\sum_{j=1}^N \alpha_jy_jx_j\cdot x+b)$$

## 策略
误分类点到超平面 $S$  的总距离

## 推理思路
原始形式：

$$w\leftarrow w+\eta y_ix_i$$

$$b\leftarrow b+\eta y_i$$

最终修改完成后$w,b$关于$(x_i,y_i)$的增量分别为$\alpha_iy_ix_i$和$\alpha_iy_i$

$$w=\sum_{i=1}^N \alpha_iy_ix_i$$

$$b=\sum_{i=1}^N \alpha_iy_i$$
