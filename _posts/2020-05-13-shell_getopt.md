---
layout:     post
title:      "Shell命令行选项与参数详解(getopt/getopts)"
data: 2020年5月13日13:45:34
permalink:  shell_getopt.html
categories: linux
tags: linux
excerpt: linux shell命令行参数详解
---

* content
{:toc}


# Shell命令行选项与参数详解(getopt/getopts)
> 参考1：[linux shell命令行选项与参数用法详解--getopts、getopt](https://www.jianshu.com/p/6393259f0a13)  
> 参考2：[Shell 参数(2) --解析命令行参数工具：getopts/getopt](https://www.cnblogs.com/yxzfscg/p/5338775.html)  
> 参考3：[Shell 脚本传参方法总结](https://www.jianshu.com/p/d3cd36c97abc)

在bash中，可以用以下三种方式来处理命令行参数：

- 直接处理：使用`$1,$2,...,$n`进行解析
- getopts：单个字符选项的情况（如：`-n 10 -f file.txt`等选项）
- getopt：可以处理单个字符选项，也可以处理长选项long-option（如：--prefix=/home等）  
总结：小脚本直接处理即可，getopts能处理绝大多数的情况，getopt较复杂、功能也更强大。

# 直接处理
基于如下几个命令直接处理输入参数：
```bash
#!/bin/bash
echo $0    # 当前脚本的文件名（间接运行时还包括绝对路径）。
echo $n    # 传递给脚本或函数的参数。n 是一个数字，表示第几个参数。例如，第一个参数是 $1 。
echo $#    # 传递给脚本或函数的参数个数。
echo $*    # 传递给脚本或函数的所有参数。
echo $@    # 传递给脚本或函数的所有参数。被双引号 (" ") 包含时，与 $* 不同，下面将会讲到。
echo $?    # 上个命令的退出状态，或函数的返回值。
echo $$    # 当前 Shell 进程 ID。对于 Shell 脚本，就是这些脚本所在的进程 ID。
echo $_    # 上一个命令的最后一个参数
echo $!    # 后台运行的最后一个进程的 ID 号
shift      # 用于对参数的移动(左移)，每次运行shift(不带参数的),销毁一个参数，后面的参数前移
shift n    # 一次销毁n个参数
```

# getopts
- `getopts`是bash的内部命令
- `getopts`有两个参数，第一个参数是一个字符串，包括字符和`":"`
- 每一个字符都是一个有效的选项（option），如果字符后面带有`":"`，表示这个选项有自己的argument，argument保存在内置变量`OPTARG`中
- `${OPTIND}`总是存储原始`$*`中下一个要处理的元素位置


例如`getopts.sh`：
```bash
#!/bin/bash

echo original parameters=[$*]
echo original OPTIND=[$OPTIND]
# ":a:bc"中第一个":"表示忽略错误
# "a:bc"，表示可接受的选项为-a -b -c，
# 其中-a选项后接参数，-b -c选项后不接参数
while getopts ":a:bc" opt
do
    case $opt in
        a)
            echo "this is -a option. OPTARG=[$OPTARG] OPTIND=[$OPTIND]"
            ;;
        b)
            echo "this is -b option. OPTARG=[$OPTARG] OPTIND=[$OPTIND]"
            ;;
        c)
            echo "this is -c option. OPTARG=[$OPTARG] OPTIND=[$OPTIND]"
            ;;
        ?)      # 未知选项处理
            echo "there is unrecognized parameter."
            exit 1
            ;;
    esac
done
#通过shift $(($OPTIND - 1))的处理，$*中就只保留了除去选项内容的参数，
#可以在后面的shell程序中进行处理
shift $(($OPTIND - 1))

echo remaining parameters=[$*]
echo \$1=[$1]
echo \$2=[$2]
```

测试代码如下：
```bash
# ./getopts.sh -a 12 -b -c file1 file2
original parameters=[-a 12 -b -c file1 file2]
original OPTIND=[1]
this is -a option. OPTARG=[12] OPTIND=[3]
this is -b option. OPTARG=[] OPTIND=[4]
this is -c option. OPTARG=[] OPTIND=[5]
remaining parameters=[file1 file2]
$1=[file1]
$2=[file2]
```

# getopt
- `getopt`是一个外部命令，不是bash内置命令，Linux发行版通常会自带
- `getopt`支持短选项和长选项
- `getopt`命令解析选项后会添加`"--"`作为分隔符
- 老版本的getopt问题较多，增强版getopt比较好用，执行命令`getopt -T; echo $?`，如果输出4，则代表是增强版的
- 如果短选项带argument且参数可选时，argument必须紧贴选项，如`-carg` 而不能是`-c arg`
- 如果长选项带argument且参数可选时，argument和选项之间用“=”，如`--clong=arg`而不能是`--clong arg`

例如`getopt.sh`：
```bash
#!/bin/bash

echo original parameters=[$@]

# -o或--options选项后面是可接受的短选项，如ab:c::，表示可接受的短选项为-a -b -c，
# 其中-a选项不接参数，-b选项后必须接参数，-c选项的参数为可选的
# -l或--long选项后面是可接受的长选项，用逗号分开，冒号的意义同短选项。
# -n选项后接选项解析错误时提示的脚本名字
# getopt是外部命令，需要使用$()或``实现命令替换
ARGS=$(getopt -o ab:c:: --long along,blong:,clong:: -n "$0" -- "$@")
if [ $? != 0 ]; then
    echo "Terminating..."
    exit 1
fi

echo ARGS=[$ARGS]
# eval set "${ARGS}" 将变量"ARGS"中的值最为当前shell脚本的输入分配至位置参数（$1,$2,...)
# 但对于"-"开头的参数会被当做选项来解析，需要加"--"
# 举一个例子比较好理解：
# 我们要创建一个名字为 "-f"的目录你会怎么办？
# mkdir -f #不成功，因为-f会被mkdir当作选项来解析，这时就可以使用
# mkdir -- -f 这样-f就不会被作为选项。
eval set -- "${ARGS}"
echo formatted parameters=[$@]

while true
do
    case "$1" in
        -a|--along) 
            echo "Option a";
            shift
            ;;
        -b|--blong)
            echo "Option b, argument $2";
            shift 2
            ;;
        -c|--clong)
            case "$2" in
                "")
                    echo "Option c, no argument";
                    shift 2  
                    ;;
                *)
                    echo "Option c, argument $2";
                    shift 2
                    ;;
            esac
            ;;
        --)
            shift
            break       # 这里跳出while循环
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

#处理剩余的参数
echo remaining parameters=[$@]
echo \$1=[$1]
echo \$2=[$2]
```

测试：
```bash
#短选项
# ./getopt.sh -a -b1 -c2 file1 file2
original parameters=[-a -b1 -c2 file1 file2]
ARGS=[ -a -b '1' -c '2' -- 'file1' 'file2']
formatted parameters=[-a -b 1 -c 2 -- file1 file2]
Option a
Option b, argument 1
Option c, argument 2
remaining parameters=[file1 file2]
$1=[file1]
$2=[file2]

#长选项
./getopt.sh --along --blong=1 --clong=2 file1 file2
original parameters=[--along --blong=1 --clong=2 file1 file2]
ARGS=[ --along --blong '1' --clong '2' -- 'file1' 'file2']
formatted parameters=[--along --blong 1 --clong 2 -- file1 file2]
Option a
Option b, argument 1
Option c, argument 2
remaining parameters=[file1 file2]
$1=[file1]
$2=[file2]

#长短混合
# ./getopt.sh -a -b1 --clong=2 file1 file2
original parameters=[-a -b1 --clong=2 file1 file2]
ARGS=[ -a -b '1' --clong '2' -- 'file1' 'file2']
formatted parameters=[-a -b 1 --clong 2 -- file1 file2]
Option a
Option b, argument 1
Option c, argument 2
remaining parameters=[file1 file2]
$1=[file1]
$2=[file2]
```

对于可选参数出错的情况：
```bash
#短选项和所带argument中间含有空格
# ./getopt.sh -a -b 1 -c 2 file1 file2
original parameters=[-a -b 1 -c 2 file1 file2]
ARGS=[ -a -b '1' -c '' -- '2' 'file1' 'file2']
formatted parameters=[-a -b 1 -c -- 2 file1 file2]
Option a
Option b, argument 1
Option c, no argument
remaining parameters=[2 file1 file2]
$1=[2]
$2=[file1]

#长选项和所带argument中间含有空格
# ./getopt.sh --along --blong 1 --clong 2 file1 file2
original parameters=[--along --blong 1 --clong 2 file1 file2]
ARGS=[ --along --blong '1' --clong '' -- '2' 'file1' 'file2']
formatted parameters=[--along --blong 1 --clong -- 2 file1 file2]
Option a
Option b, argument 1
Option c, no argument
remaining parameters=[2 file1 file2]
$1=[2]
$2=[file1]
```

# 使用示例
指定目录，遍历目录下的`cif`文件，并依次执行`test_exe`.
```bash
#!/bin/bash

COLOR_ERROR="\e[38;5;198mError:"
COLOR_NONE="\e[0m"
COLOR_WARN="\e[1;33;198mWaning:"
COLOR_SUCC="\e[92mSuccess:"

COLOR_GREEN='\e[1;32m' #绿
COLOR_RED='\E[1;31m'  #红
COLOR_YELLOW='\E[1;33m' #黄
COLOR_BLUE='\E[1;34m'  #蓝

echo original parameters=[$@]

# -o或--options选项后面是可接受的短选项，如ab:c::，表示可接受的短选项为-a -b -c，
# 其中-a选项不接参数，-b选项后必须接参数，-c选项的参数为可选的
# -l或--long选项后面是可接受的长选项，用逗号分开，冒号的意义同短选项。
# -n选项后接选项解析错误时提示的脚本名字
# getopt是外部命令，需要使用$()或``实现命令替换
ARGS=$(getopt -o d:f:e: --long files_dir:,single_file_path:,extension:,func:,exe_path: -n "$0" -- "$@")
if [ $? != 0 ]; then
    echo "Terminating..."
    exit 1
fi

echo ARGS=[$ARGS]
# eval set "${ARGS}" 将变量"ARGS"中的值最为当前shell脚本的输入分配至位置参数（$1,$2,...)
# 但对于"-"开头的参数会被当做选项来解析，需要加"--"
# 举一个例子比较好理解：
# 我们要创建一个名字为 "-f"的目录你会怎么办？
# mkdir -f #不成功，因为-f会被mkdir当作选项来解析，这时就可以使用
# mkdir -- -f 这样-f就不会被作为选项。
eval set -- "${ARGS}"
echo formatted parameters=[$@]

while true
do
    case "$1" in
        -d|--files_dir) 
            echo "Option -d/--files_dir, argument $2";
            files_dir=$2
            if [ ! ${files_dir: -1} == '/' ];then       # 获取最后一个字符并判断
                files_dir="${files_dir}/"
            fi
            shift 2
            ;;
        -f|--single_file_path)
            echo "Option -f/--single_file_path, argument $2";
            single_file_path=$2
            shift 2
            ;;
        --func)
            echo "Option --func, argument $2";
            func=$2
            shift 2
            ;;
        --exe_path)
            echo "Option --exe_path, argument $2";
            exe_path=$2
            shift 2
            ;;
        -e|--extension)
            echo "Option -e/--extension, argument $2";
            extension=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

#处理剩余的参数
echo remaining parameters=[$@]
echo \$1=[$1]
echo \$2=[$2]
echo

if [ -z "$extension" ];then extension="cif";fi

if [ -z "$files_dir" ] && [ -z $single_file_path ];then 
    echo -e "${COLOR_ERROR} Please assign the test file or dir by --single_file_path/--files_dir ${COLOR_NONE}"
    exit 1
fi

if $exe_path;then 
    if [ -f "./test_exe" ];then
        exe_path="./test_exe"
    else
        echo -e "${COLOR_ERROR} Please assign the path to test_exe ${COLOR_NONE}"
        exit 1
    fi
    else
        if [ ! -f ${exe_path} ];then
            echo -e "${COLOR_ERROR} ${exe_path} not exit! ${COLOR_NONE}"
            exit 1
        fi
fi

if [ -z "$func" ];then      # 判断字符串"$func"是否为空，是为真
    echo -e "${COLOR_ERROR} Please assign the func to test_exe ${COLOR_NONE}"
    exit 1
fi

function run_test_exe() {
    local file_name=$1
    local cur_extension=${file_name: 0-3: 3}
    # 1. []中condition两边需要有空格
    # 2. =等效于==，且两边需要有空格
    # 3. "$cur_extension"x最后的x，这是特意安排的，因为当$cur_extension为空的时候，上面的表达式就变成了x = "${extension}"x, 显然是不相等的。而如果没有这个x就会报错
    if [ "${cur_extension}"x == "${extension}"x ];then
        # echo ">>>>>> run test_exe -func ${func} -im "${files_dir}${file_name}""
        echo -e "${COLOR_GREEN}>>>${COLOR_NONE} ${func} ${COLOR_GREEN}${file_name}${COLOR_NONE} "
        ${exe_path} -func ${func} -im "${files_dir}/${file_name}"
        echo
    fi
}

if [ -n "${files_dir}" ] && [ -d "${files_dir}" ];then
    file_names=$(ls "$files_dir")
    for file_name in ${file_names}
    do
        run_test_exe ${file_name}
    done
fi

if [ -f $single_file_path ];then
    run_test_exe ${single_file_path}
fi

```
