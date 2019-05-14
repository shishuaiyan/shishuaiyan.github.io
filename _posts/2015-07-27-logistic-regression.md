---
layout:     post
title:      "逻辑斯谛（logistic regression）回归"
data: 2015-07-27 10:42:10
permalink:  lr.html
categories: 机器学习
tags: 逻辑斯谛
excerpt: 逻辑斯谛回归
mathjax: true
---

* content
{:toc}


## 模型

$$P(Y=1|x)=\frac{\exp(w\cdot x)}{1+\exp(w\cdot x)}$$

$$P(Y=0|x)=\frac 1 {1+\exp(w\cdot x)}$$

这里权值向量和输入向量都为扩充后的表示 .
## 参数估计
用极大似然估计法估计模型参数
设：$P(Y=1|x)=\pi(x),\quad P(Y=0|x)=1-\pi(x)$
似然函数为

$$\prod_{i=1}^N[\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}$$

对数似然函数为

$$
\begin{aligned}
L(w)&=\sum_{i=1}^N[y_i\log\pi(x_i)+(1-y_i)\log(1-\pi(x_i))] \\
&=\sum_{i=1}^N\left[y_i\log\frac{\pi(x_i)}{1-\pi(x_i)}+\log(1-\pi(x_i))\right] \\
&=\sum_{i=1}^N[y_i(w\cdot x_i)-\log(1+\exp(w\cdot x_i))]
\end{aligned}
$$

对 $L(w)$ 求极大值，得到 $w$ 的估计值，通常采用的方法是梯度下降法及拟牛顿法 .
