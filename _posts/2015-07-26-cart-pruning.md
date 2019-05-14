---
layout:     post
title:      "CART剪枝算法"
data: 2015-07-26 13:13:27
permalink:  cartpruning.html
categories: 机器学习
tags: 决策树
excerpt: CART树的剪枝算法
mathjax: true
---

* content
{:toc}

## 算法
$\qquad$ 输入：CART 算法生成的决策树 $T_0$；

$\qquad$ 输出：最优决策树 $T_\alpha$ .

$\qquad$ (1) 设 $k=0,\quad T=T_0$ .

$\qquad$ (2) 设 $\alpha=+\infty$ .

$\qquad$ (3) 自下而上地对各内部结点 $t$ 计算 $C(T_t)，\|T_t\|$以及

$$g(t)=\frac{C(t)-C(T_t)}{|T_t|-1}$$

$$\alpha=\min(\alpha,g(t))$$

这里 $T_t$ 表示以 $t$ 为结点的子树，$C(T_t)$ 是对训练数据的误差估计，$\|T_t\|$ 是 $T_t$ 的叶结点个数。

$\qquad$ (4) 自上而下地访问内部结点 $t$，如果有 $g(t)=\alpha$，进行剪枝，并对叶结点 $t$ 以多数表决法决定其类，得到树 $T$ .

$\qquad$ (5) 设 $k=k+1,\quad \alpha_k=\alpha,\quad T_k=T$ .

$\qquad$ (6) 如果 $T$ 不是由根结点单独构成的树，则返回步骤(4) .

$\qquad$ (7) 采用交叉验证法在子树序列 $T_0,T_1,\cdots,T_n$ 中选取最优子树 $T_\alpha$ .
