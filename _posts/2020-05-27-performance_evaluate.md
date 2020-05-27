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

# C/C++

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