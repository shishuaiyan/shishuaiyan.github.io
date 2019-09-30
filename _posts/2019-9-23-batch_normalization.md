---
layout:     post
title:      "BN层原理及TF中调用方法"
data: 2019年9月23日20:59:05
permalink:  batch_normalization.html
categories: 深度学习
tags: batch_normalization
excerpt: BN层的优缺点及TF调用BN层的方法
mathjax: true
---

* content
{:toc}

## batch normalization
BN层基本原理([参考至知乎言有三](https://www.zhihu.com/question/38102762/answer/607815171))：
> 现在一般采用批梯度下降方法对深度学习进行优化，这种方法把数据分为若干组，按组来更新参数，一组中的数据共同决定了本次梯度的方向，下降时减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也下降了很多。  
> Batch Normalization(简称BN)中的batch就是批量数据，即每一次优化时的样本数目，通常BN网络层用在卷积层后，用于重新调整数据分布。假设神经网络某层一个batch的输入为X=[x1,x2,...,xn]，其中xi代表一个样本，n为batch size。  
> 1. 计算mini-batch中元素均值  
> 
> $$\mu_B=\frac 1 n\sum^n_{i=1}x_i$$  
> 
> 2. 求mini-batch的方差  
> 
> $$\sigma_B^2=\frac 1 n\sum^n_{i=1}(x_i-\mu_B)^2$$  
> 
> 3. 对每个元素进行归一化  
> 
> $$x_i^\prime=\frac {x_i-\mu_B} {\sqrt{\sigma^2_B+\varepsilon}}$$  
> 
> 4. 最后进行尺度缩放和偏移操作，这样可以变换回原始的分布，实现恒等变换，这样的目的是为了补偿网络的非线性表达能力，因为经过标准化之后，偏移量丢失。具体的表达如下，yi就是网络的最终输出  
> 
> $$y_i=\gamma_i\cdot x_i^\prime+\beta$$  

假如gamma等于方差，beta等于均值，就实现了恒等变换。    
从某种意义上来说，gamma和beta代表的其实是输入数据分布的方差和偏移。对于没有BN的网络，这两个值与前一层网络带来的非线性性质有关，而经过变换后，就跟前面一层无关，变成了当前层的一个学习参数，这更加有利于优化并且不会降低网络的能力。  
对于CNN，BN的操作是在各个特征维度之间单独进行，也就是说各个通道是分别进行Batch Normalization操作的。  

BN层带来的好处：  
> - 减轻了对参数初始化的依赖，利于调参
> - 训练更快，可以使用更高的学习率
> - 在一定程度上增加了泛华能力

BN层的缺陷：

> batch normalization依赖于batch的大小，当batch值很小时，计算的均值和方差不稳定。研究表明对于ResNet类模型在ImageNet数据集上，batch从16降低到8时开始有非常明显的性能下降，在训练过程中计算的均值和方差不准确，而在测试的时候使用的就是训练过程中保持下来的均值和方差。
> 这一个特性，导致batch normalization不适合以下的几种场景。
> - batch非常小，比如训练资源有限无法应用较大的batch，也比如在线学习等使用单例进行模型参数更新的场景。
> - RNN，因为它是一个动态的网络结构，同一个batch中训练实例有长有短，导致每一个时间步长必须维持各自的统计量，这使得BN并不能正确的使用。在rnn中，对bn进行改进也非常的困难。不过，困难并不意味着没人做，事实上现在仍然可以使用的，不过这超出了咱们初识境的学习范围。

BN层为何有效的观点：
> - 主流观点，Batch Normalization调整了数据的分布，不考虑激活函数，它让每一层的输出归一化到了均值为0方差为1的分布，这保证了梯度的有效性，目前大部分资料都这样解释，比如BN的原始论文认为的缓解了Internal Covariate Shift(ICS)问题。
> - 可以使用更大的学习率，文[2]指出BN有效是因为用上BN层之后可以使用更大的学习率，从而跳出不好的局部极值，增强泛化能力，在它们的研究中做了大量的实验来验证。  
> - 损失平面平滑。文[3]的研究提出，BN有效的根本原因不在于调整了分布，因为即使是在BN层后模拟ICS，也仍然可以取得好的结果。它们指出，BN有效的根本原因是平滑了损失平面。之前我们说过，Z-score标准化(零均值归一化)对于包括孤立点的分布可以进行更平滑的调整。


BN层在使用过程中的`training`参数非常重要（参考至[BN原理与实战](https://zhuanlan.zhihu.com/p/34879333)）：    
> BN层在训练的过程中每一层计算的期望$\mu$与方差$\sigma^2$都是基于当前batch中的训练数据，之后更新$Z^{[l]}$；但在测试阶段有可能只需要预测一个样本或很少的样本，此时计算的期望与方差一定是有偏估计，因此，我们需要使用整个样本的统计量来对测试数据进行归一化，即使用均值与方差的无偏估计([无偏估计样本方差为什么分母是m-1而不是m?](https://www.matongxue.com/madocs/607.html))：
> $$\mu_{test}=\Epsilon(\mu_{batch})$$
> $$\sigma_{batch}^{2}=\frac{m}{m-1}\Epsilon(\sigma_{batch}^2)$$
> 得到每个特征的均值与方差的无偏估计后，对test数据才用同样的normalization方法：
> $$BN(X_{test})=\gamma \cdot \frac{X_{test}-\mu_{test}}{\sqrt{\sigma_{test}^2 + \epsilon}} + \beta$$
> 此外，同样可使用train阶段每个batch计算的mean\variance的加权平均数来得到test阶段mean\variance的估计

tf中使用BN层有如下几种方法：

1. tf.layers.batch_normalization()   
tf.layers提供高层的神经网络，主要和卷积相关，是对tf.nn的进一步封装    
示例如下：
```python
tf.layers.batch_normalization(inputs, training=True)
```
2. tf.nn.batch_normalization()    
相对于tf.layers, tf.nn更底层，属于低阶API，提供神经网络相关操作的支持，且tf.nn.batch_normalization()中无`training`参数（需要手动传入期望与方差），因此需要在其基础上继续封装，下面是一个示例：
```python
from tensorflow.python.training import moving_averages
def create_variable(name, shape, initializer, dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable)
# batchnorm layer
def batchnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):
    inputs_shape = inputs.get_shape().as_list()
    params_shape = inputs_shape[-1:]
    axis = list(range(len(inputs_shape) - 1))

    with tf.variable_scope(scope):
        beta = create_variable("beta", params_shape,
                               initializer=tf.zeros_initializer())
        gamma = create_variable("gamma", params_shape,
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_variable("moving_mean", params_shape,
                            initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = create_variable("moving_variance", params_shape,
                            initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(inputs, axes=axis)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=momentum)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
```
3. tf.keras.layers.BatchNormalization()  
   使用keras高阶API
```python
x = tf.keras.layers.BatchNormalization()(x)
```



