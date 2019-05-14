---
layout:     post
title:      "朴素贝叶斯算法"
data: 2015-07-24 17:43:47
permalink:  naive-bayes.html
categories: 机器学习
tags: 贝叶斯
excerpt: 朴素贝叶斯算法
mathjax: true
---

* content
{:toc}

## 算法
$\qquad$ 输入：训练数据 $T=\\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\\}$，其中 $x_i=(x_i^{(1)},x_i^{(2)},\ldots,x_i^{(n)})^T$, $x_i^{(j)}$ 是第 $i$ 个样本的第 $j$ 个特征， $x_i^{(j)}\in\\{a_{j1},a_{j2},\ldots,a_{js_j}\\}$，$a_{jl}$ 是第 $j$ 个特征可能取的第 $l$ 个值， $j=1,2,\ldots,n$， $l=1,2,\dots,S_j$ ，$y_i\in\\{c_1,c_2,\ldots,c_K\\}$；实例 $x$ ；

$\qquad$ 输出：实例 $x$ 的分类。

$\qquad$ (1) 计算先验概率及条件概率

$$P(Y=c_k)=\frac {\sum_{i=1}^N I(y_i=c_k)} N,\quad k=1,2,\ldots,K$$

$$P(X^{(j)}=a_{jl}|Y=c_k)=\frac {\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)} {\sum_{i=1}^N I(y_i=c_k)}$$

$$j=1,2,\ldots,n;\quad l=1,2,\ldots ,S_j;\quad k=1,2,\ldots,K$$

$\qquad$ (2) 对于给定的实例 $x=(x^{(1)},x^{(2)},\ldots,x^{(n)})^T$，计算

$$P(Y=c_k)\cdot\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k),\quad k=1,2,\ldots,K$$

$\qquad$ (3) 确定实例 $x$ 的类

$$y=arg\max_{c_k}P(Y=c_k)\cdot\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)$$

## 模型
$P(X,Y)$，条件独立性假设
## 策略
极大后验概率估计
## 推理思路
$\quad$ 通过训练数据集学习 $P(X,Y)$ ，具体地，学习先验概率分布 $P(Y=c_k)$ 和条件概率分布 $P(X=x|Y=c_k)$

朴素 = 条件独立性假设 = 特征在类确定的情况下是条件独立的

$$
\begin{aligned}
P(X=x|Y=c_k)&=P(X^{(1)}=x^{(1)},\ldots,X^{(n)}=x^{(n)}|Y=c_k) \\
& =\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)
\end{aligned}
$$

将后验概率$P(Y=c_k\|X=x)$ 最大的类作为 $x$ 的类输出

$$
\begin{aligned}
P(Y=c_k|X=x) &=\frac {P(X=x|Y=c_k)P(Y=c_k)} {\sum_k P(X=x|Y=c_k)P(Y=c_k)} \\
&=\frac{P(Y=c_k)\prod_j P(X^{(j)}=x^{(j)}|Y=c_k)} {\sum_k P(Y=c_k)\prod_j P(X^{(j)}=x^{(j)}|Y=c_k)}
\end{aligned}
$$

分母对所有 $c_k$ 都是相同的，所以

$$y=arg\max_{c_k}P(Y=c_k)\prod_{j=1}^n P(X^{(j)}=x^{(j)}|Y=c_k)$$

## 算法缺陷及改进
可能出现估计的概率为0，采用贝叶斯估计代替极大似然估计

$$P_\color{red}\lambda(Y=c_k)=\frac {\sum_{i=1}^N I(y_i=c_k)+\color{red}\lambda} {N+\color{red}{K\lambda}},\quad k=1,2,\ldots,K$$

$$P_\color{red}\lambda(X^{(j)}=a_{jl}|Y=c_k)=\frac {\sum_{i=1}^N I(x_i^{(j)}=a_{jl},y_i=c_k)+\color{red}\lambda} {\sum_{i=1}^N I(y_i=c_k)+\color{red}{S_j\lambda}}$$

$\lambda \ge 0$，当 $\lambda=0$ 即为极大似然估计。常取 $\lambda=1$ 此时称为拉普拉斯平滑。
