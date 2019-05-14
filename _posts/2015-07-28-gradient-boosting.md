---
layout:     post
title:      "梯度提升算法"
data: 2015-07-28 11:21:12
permalink:  gradient-boosting.html
categories: 机器学习
tags: boost
excerpt: 梯度提升（gradient boosting）算法
mathjax: true
---

* content
{:toc}

## 算法
$\qquad$ 输入：训练数据集 $T=\\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\\},x_i\in\Bbb R^n,y_i\in\Bbb R\text{；损失函数} L(y,f(x));$

$\qquad$ 输出：回归树 $\hat f(x)$ .

$\qquad$ (1) 初始化

$$f_0(x)=\arg\min_c\sum_{i=1}^NL(y_i,c)$$

$\qquad$ (2) 对 $m=1,2,\cdots,M$

$\qquad\quad$ (a) 对 $i=1,2,\cdots,N$，计算

$$r_{mi}=-\left[\frac{\partial L(y_i,f(x_i))}{\partial f(x_i)}\right]_{f(x)=f_{m-1}(x)}$$

$\qquad\quad$ (b) 拟合残差 $r_{mi}$ 学习一个回归树，得到第 $m$ 棵树的叶结点区域 $R_{mj},\quad j=1,2,\cdots,J$

$\qquad\quad$ (c) 对 $j=1,2,\cdots,J$，计算

$$c_{mj}=\arg\min_c\sum_{x_i\in R_{mj}}L(y_i,f_{m-1}(x_i)+c)$$

$\qquad\quad$ (d) 更新

$$f_m(x)=f_{m-1}(x)+\sum_{j=1}^Jc_{mj}I(x\in R_{mj})$$

$\qquad$ (3) 得到回归树

$$\hat f(x)=f_M(x)=\sum_{m=1}^M\sum_{j=1}^Jc_{mj}I(x\in R_{mj})$$

## 思路说明
$\quad$ 算法第 1 步初始化，估计使损失函数极小化的常数值，它是只有一个根结点的树。第 2(a) 步计算损失函数的负梯度在当前模型的值，将它作为残差的估计。对于平方损失函数，他就是通常所说的残差；对于一般损失函数，它就是残差的近似值。第 2(b) 步估计回归树叶结点区域，以拟合残差的近似值。第 2(c) 步利用线性搜索估计叶结点区域的值，是损失函数极小化。第 2(d) 步更新回归树。第 (3) 步得到输出的最终模型 $\hat f(x)$ .   
