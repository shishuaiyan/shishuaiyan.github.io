---
layout:     post
title:      "感知机(perceptron)学习算法的原始形式"
data: 2015-07-24 11:01:37
permalink:  perceptron.html
categories: 机器学习
tags: 感知机
excerpt: 感知机的原始形式
mathjax: true
---

* content
{:toc}

## 算法
$\qquad$ 输入：训练数据集$T=\\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\\}$,其中$x_i\in \Bbb R^n$,$y_i \in \\{-1,1\\}，i=1,2,\ldots,N$ ；学习率 $\eta(0\lt\eta\le1)$;

$\qquad$ 输出：$w,b$ ; 感知机模型$f(x)=sign(w\cdot x+b)$.

$\qquad$ (1) 选择初值$w_0,b_0$

$\qquad$ (2) 在训练集中选取数据$(x_i,y_i)$

$\qquad$ (3) 如果$y_i(w\cdot x_i+b)\le 0$ 

$$w\leftarrow w+\eta y_ix_i$$ 

$$b\leftarrow b+\eta y_i$$

$\qquad$ (4) 转至(2),直至训练集中没有误分类点。


## 模型

$$f(x)=sign(w\cdot x+b)$$

## 策略
误分类点到超平面 $S$  的总距离

## 推理思路
任意一点$x_0$到超平面 $S$ 的距离为

$$\frac 1 {\lVert w \rVert} \lvert w\cdot x_0 +b\rvert$$

误分类点到超平面 $S$ 的距离为

$$-\frac 1 {\lVert w \rVert} y_i (w\cdot x_i + b)$$

所有误分类点到$S$的总距离为

$$-\frac 1 {\lVert w\rVert} \sum_{x_i\in M} y_i(w\cdot x_i+b)$$

不考虑$\lVert w\rVert$，损失函数定义为

$$L(w,b)=-\sum_{x_i\in M} y_i(w\cdot x_i+b)$$

其中$M$为误分类点的集合。

最优化方法为随机梯度下降法

$$\nabla_w L(w,b)=-\sum_{x_i \in M} y_ix_i$$

$$\nabla_b L(w,b)=-\sum_{x_i\in M} y_i$$

随机选择一个误分类点$(x_i,y_i)$，对$w,b$进行更新：

$$w\leftarrow w+\eta y_ix_i$$

$$b\leftarrow b+\eta y_i$$
