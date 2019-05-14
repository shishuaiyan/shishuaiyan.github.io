---
layout:     post
title:      "回归问题的提升树（boosting tree）算法"
data: 2015-07-28 10:35:35
permalink:  boosting-tree.html
categories: 机器学习
tags: boost
excerpt: 回归问题的提升树（boosting tree）算法
mathjax: true
---

* content
{:toc}

## 算法
$\qquad$ 输入：训练数据集 $T=\\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\\},x_i\in\Bbb R^n,y_i\in\Bbb R;$

$\qquad$ 输出：提升树 $f_M(x)$ .

$\qquad$ (1) 初始化 $f_0(x)=0$

$\qquad$ (2) 对 $m=1,2,\cdots,M$

$\qquad\quad$ (a) 计算残差

$$r_{mi}=y_i-f_{m-1}(x_i),\quad i=1,2,\cdots,N$$

$\qquad\quad$ (b) 拟合残差 $r_{mi}$ 学习一个回归树，得到 $T(x;\Theta_m)$

$\qquad\quad$ (c) 更新

$$f_m(x)=f_{m-1}(x)+T(x;\Theta_m)$$

$\qquad$ (3) 得到回归问题提升数

$$f_M(x)=\sum_{m=1}^MT(x;\Theta_m)$$

## 思路说明
回归问题采用平方误差损失函数

$$L(y,f(x))=(y-f(x))^2$$

按照[前向分步算法](../forward-stagewise.html)极小化损失函数，则损失为

$$
\begin{align}L(y,f_{m-1}(x)+T(x;\Theta_m))&=[y-f_{m-1}(x)-T(x;\Theta_m)]^2\\
&=[r-T(x;\Theta_m)]^2\end{align}$$

这里 $r=y-f_{m-1}(x)$ .

所以回归问题的提升树算法需要计算残差并拟合残差 。
