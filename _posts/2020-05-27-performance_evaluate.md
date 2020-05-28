---
layout:     post
title:      "C++/Python性能评估"
data: 2020年5月27日11:00:26
permalink:  performance_evaluate.html
categories: performacne
tags: performance
excerpt: C/C++/Python高性能编程的第一步：性能评估(profile)总结
---

* content
{:toc}

# Python
> reference: [Python优化第一步: 性能分析实践](https://zhuanlan.zhihu.com/p/24495603)

# C/C++
> reference: [linux下的profile工具GNU gprof和Valgrind](https://blog.csdn.net/clarstyle/article/details/41747817)

## Valgrind
> reference: [Valgrind安装和简单使用介绍](https://segmentfault.com/a/1190000017465212)

valgrind官网：[http://valgrind.org/](http://valgrind.org/)

### introduction
> reference: [linux下的profile工具GNU gprof和Valgrind](https://blog.csdn.net/clarstyle/article/details/41747817)

Valgrind是用于内存泄露和threading bugs检测以及程序性能分析的一款开源工具。作者曾经获得Google开源Best Tool Maker奖项。Valgrind含有很多工具：

(1) Memcheck  
&emsp;&emsp;用来检测程序中出现的内存问题，所有对内存的读写都会被检测到，一切对malloc/new/free/delete的调用都会被捕获，来检测问题。 

(2) Cachegrind  
&emsp;&emsp;Cache分析工具，它模拟CPU中的一级缓存I1，Dl和二级缓存，能够精确地指出程序中cache的丢失和命中。如果需要，它还能够为我们提供cache丢失次数，内存引用次数，以及每行代码，每个函数，每个模块，整个程序产生的指令数。

(3) Callgrind  
&emsp;&emsp;是Cachegrind的扩充版，它在提供Cachegrind工具所有信息的基础上，还提供函数调用图。Callgrind是和gprof类似的分析工具，但它对程序的运行观察更是入微，能给我们提供更多的信息。和gprof不同，它不需要在编译源代码时附加特殊选项，但加上调试选项是推荐的。Callgrind收集程序运行时的一些数据，建立函数调用关系图，还可信以有选择地进行cache模拟。在运行结束时，它会把分析数据写入一个文件。也是我们在profile时主要使用的工具

(4) Massif  
&emsp;&emsp;Massif是一个堆栈分析工具。它能测量程序在堆栈中使用了多少内存，告诉我们堆块，堆管理块和栈的大小。Massif能帮助我们减少内存的使用，在带有虚拟内存的现代系统中，它还能够加速我们程序的运行。

(5) Helgrind  
&emsp;&emsp;Helgrind主要用来检查多线程程序中出现的竞争问题。Helgrind寻找内存中被多个线程访问，而又没有一贯加锁的区域，这些区域往往是线程之间失去同步的地方，而且会导致难以发掘的错误。Helgrind仍然处于实验阶段。

### 安装
```bash
# download
curl -O https://sourceware.org/pub/valgrind/valgrind-3.15.0.tar.bz2
# unzip
tar xvf valgrind-3.15.0.tar.bz2
cd valgrind-3.15
# install
./autogen.sh
./configure
make
make install
# check
valgrind --version
```

### 使用
```bash
valgrind --tool=callgrind <run_exe_command>     # valgrind --tool=callgrind ./a.out
```
> 注意：编译可执行文件时要添加`-g`选项，生成的profile结果才有行号信息。

上述命令会在当前目录下生成`callgrind.out.x`文件，`x`为进程的PID编号。该文件为文本文件，可直接打开，但不易观看，建议使用`kcachegrind`将结果可视化。

### 查看
linux平台下可直接`apt get kcachegring`下载使用，注意：该工具基于QT，需要图形界面。
windows平台可下载[QCacheGrind](https://sourceforge.net/projects/qcachegrindwin/files/latest/download)解压使用。

## 

# profile可视化
## Graphviz
> [官网](http://graphviz.org/)


`Graphviz`是一个很好用的绘图工具，linux/windows下都可快速使用，通过简单的语句编写`.dot`文本并生成图片，这里是为了生成性能分析的流程图。
linux下可直接通过`apt`命令安装：
```bash
apt install graphviz
# check
dot -V
# use
dot -Tpng -o result.png <path_to_dot_file>
```
上述命令会根据传入的dot文件在当前目录下生成名为`result.png`的图片。

## gprof2dot
> [git项目地址](https://github.com/jrfonseca/gprof2dot)

对于valgrind, cProfiler等工具生成的结果文件，还需要使用`gprof2dot`工具将log转换为dot文件，之后才能使用`graphviz`工具生成图片。
```bash
# gprof2dot是一个Python库，支持py2.7以及python3
conda activate
# install
pip install gprof2dot
# check
gprof2dot -h
# use 注意，对于不同工具生成的log文件，需要制定不同的format参数
gprof2dot -f pstats <log_file> | dot -Tpng -o result.png    # python cProfile/profile
gprof2dot -f callgrind <log_file> | dot -Tpng -o result.png    # c++ valgrind
```

# C++优化方法
## 并行计算
1. 并行库
2. SIMD指令集，例如：128位的寄存器可以同时存放4个32位的浮点数，因此可同时处理4个数据

## 内存优化
1. 内存块频繁的申请与释放耗时比较大，必要情况下可以自己实现内存管理。对于内存大小比较相近，频繁申请时可以自己缓冲内存列表，类似 Look aside list
2. 内存对齐
3. 空间换时间：能用到的中间数据存储起来；for循环展开等

## 数值优化
1. 考虑使用些基本代数库
2. 少用乘法/除法，多用加法
3. 计算n的10次方以下时不要使用std::pow()
4. 使用广播机制将一个数乘/加到矩阵的所有元素上时将这个数提前计算出来或者放到矩阵的前面
