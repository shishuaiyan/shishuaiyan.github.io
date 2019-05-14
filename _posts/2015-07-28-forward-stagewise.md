---
layout:     post
title:      "前向分布算法"
data: 2015-07-28 09:55:09
permalink:  forward-stagewise.html
categories: 机器学习
tags: boost
excerpt: 前向分布算法（forward stagewise algorithm）
mathjax: true
---

* content
{:toc}

## 算法
$\qquad$ 输入：训练数据集 $T=\\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\\}$；损失函数 $L(y,f(x))$ ；基函数集 $\\{b(x;\gamma)\\}$；

$\qquad$ 输出：加法模型 $f(x)$ .

$\qquad$ (1) 初始化 $f_0(x)=0$

$\qquad$ (2) 对 $m=1,2,\cdots,M$

$\qquad\quad$ (a) 极小化损失函数

$$(\beta_m,\gamma_m)=\arg\min_{\beta,\gamma}\sum_{i=1}^NL\left(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma)\right)$$

得到参数 $\beta_m,\gamma_m$

$\qquad\quad$ (b) 更新

$$f_m(x)=f_{m-1}(x)+\beta_mb(x;\gamma_m)$$

$\qquad$ (3) 得到加法模型

$$f(x)=f_M(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)$$

## 一点说明
$\quad$[AdaBoost](../adaboost.html) 算法可以认为是模型为加法模型、损失函数为指数函数、学习算法为前向分步算法的二类分类学习方法。


