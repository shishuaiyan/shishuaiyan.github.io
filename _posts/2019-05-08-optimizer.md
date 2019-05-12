---
layout:     post
title:      "深度学习优化器"
data: 2019-05-08 12:34:56
permalink:  optimizer.html
categories: 深度学习
tags: optimizer
excerpt: 常用优化器整理
mathjax: true
---

* content
{:toc}


## BGD

$$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta)$$

缺点：再一次更新中对整个数据集计算剃度，速度慢

## SGD

$$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta; x^{(i)}; y^{(i)})$$

缺点：虽然更新快，但是容易震荡

## MBGD

$$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta; x^{(i:i+n)}; y^{(i:i+n)})$$

缺点：
* 学习率太小，收敛很慢，太大，loss会震荡甚至不收敛
* 对所有参数使用了相同的学习率
* 容易困在鞍点
  
## Momentum

$$
\begin{aligned}
v_t &= \gamma\cdot v_{t-1} + \eta \cdot \nabla_{\theta}J(\theta) \\
\theta &= \theta - v_t
\end{aligned}
$$

特点：使得剃度方向不变的维度速度变快，剃度方向改变的维度速度变慢，就可以达到加速收敛和减小震荡的作用

超参数：$\gamma$一般选择0.9

缺点：这种情况相当于小球从山上滚下来时是在盲目地沿着坡滚，如果它能具备一些先知，例如快要上坡时，就知道需要减速了的话，适应性会更好

## Nesterov(NAG)

$$
\begin{aligned}
v_t &= \gamma\cdot v_{t-1} + \eta \cdot \nabla_{\theta}J(\theta-\gamma\cdot v_{t-1}) \\
\theta &= \theta - v_t
\end{aligned}
$$

特点：用$\theta-\gamma\cdot v_{t-1}$来近似当做参数下一步会变成的值，则在计算梯度时，不是在当前位置，而是未来的位置上

超参数：$\gamma$一般选择0.9

## AdaGrad

$$
\theta = \theta - \frac{\eta}{\sqrt{G+\epsilon}} \cdot g
$$

说明：其中$g$为梯度，$g = \nabla_{\theta}J(\theta)$，$G$为历史梯度的平方和，因此可以做到对低频的参数做较大的更新，对高频的参数做较小的更新，且减小了学习率的手动调节

超参数：$\eta$一般选择0.01

缺点：分母会不断积累，导致学习率变得非常小

## Adadelta
这个算法是对AdaGrad的改进，和AdaGrad相比，将分母的G换成了过去梯度的衰减平均值

$$
\Delta\theta = -\frac{\eta}{\sqrt{E[g^2]+\epsilon}}\cdot g
$$

其中$E$的公式如下，$t$时刻依赖于前一时刻的平均值和当前值：

$$
E[g^2]_t = \gamma\cdot E[g^2]_{t-1} + (1-\gamma)\cdot g_t^2
$$

分母相当于梯度的均方根root mean square(RMS)，用RMS简写则为

$$
\Delta\theta = -\frac{\eta}{RMS[g]}\cdot g
$$ 

此外，将学习率换成$RMS[\Delta\theta]$，这样就不需要设置学习率

$$
\theta_{t+1} = \theta_t - \frac{RMS[\Delta\theta]_{t-1}}{RMS[g]_t}\cdot g_t
$$ 

超参数：$\gamma$一般选择0.9

## RMSprop
RMSprop和Adadelta第一种形式相同：

$$
\theta = \theta -\frac{\eta}{\sqrt{E[g^2]+\epsilon}}\cdot g
$$

超参数：$\gamma$一般选择0.9，$\eta$一般设置为0.001

## Adam
既像Adadelta和RMSprop一样存储了过去梯度的平方的指数衰减平均值，也像Momentum一样保存了过去梯度的指数衰减平均值

$$
\begin{aligned}
m_t &= \beta_1m_{t-1} + (1-\beta_1)g_t \\
v_t &= \beta_2v_{t-1} + (1-\beta_2)g_t^2
\end{aligned}
$$

如果$m_t$和$v_t$被初始化0向量，那么就会像0偏置，所以做了偏差校正：

$$
\begin{aligned}
\hat m_t &= \frac{m_t}{1-\beta_1^t} \\
\hat v_t &= \frac{v_t}{1-\beta_2^t}
\end{aligned}
$$

最终，梯度更新规则为：

$$
\theta_{t+1} = \theta_t -\frac{\eta}{\sqrt{\hat v_t}+\epsilon}\cdot \hat m_t
$$

超参数设置：建议$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10e-8$

## AdaMax
将Adam中$v_t$的二范数改为无穷范数

## Nadam
adam可以看做Momentum+RMSprop，将Momentum改为NAG即为Nadam