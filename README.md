# CudaEzDemo
Use Cuda to realize Pytorch model.

01 ops/src 是cuda/c++代码
02 setup.py 是编译算子的配置文件
03 ops/ops_py 是用PyTorch包装的算子函数
04 test_ops.py 是调用算子的测试文件

对于一个算子实现，
需要用到.cu编写核函数
.cpp(C++)编写包装函数并调用PYBIND11_MODULE对算子进行封装

注意:CUDA文件和CPP文件不能同名


// two_sum.cpp
在C++文件中实现算子的封装。
宏定义函数是为了保证传入的向量在cuda上
(CHECK_CUDA),传入的向量中元素地址连续(CHECK_CONTIGUOUS)
two_sum_launcher是对cuda文件中的声明
two_sum_gpu是与python的接口，传入的参数是pytorch的tensor。在这一部分需要对tensor
做CHECK检验(可选).并通过.data_ptr得到Tensor的变量指针。
对于tensor在C++中的使用可查阅

最后PYBIND11_MODULE的作用是对整个算子进行封装，
能够通过python调用C++函数
对于自定义的其他算子，只用改动m.def()中的三个参数


"forward": 算子的方法名，假如算子的整个模块命名为sum_double，则在python中通过
'sum_double.forward'调用该算子
&two_sum_gpu：进行绑定的函数，这里根据自己实现的不同函数进行更改
"sum two arrays (CUDA)":算子注释，在python端调用help(sum_double.forward)时会出现帮助函数。


Q：为什么要把算子和模块分开。
A:假如整个sum_double有许多不同的功能，我就可以在一个模块中绑定多个算子，具体只用在PYBIND11_MODULE中写入多个m.def()，再通过sum_double.SUANZI_NAME调用不同的算子

// setup.py 
编译配置
利用setuptools对算子打包

name:包名
version:版本号
author:作者名称
ext_modules:编译C/C++扩展，list类型，每个元素为一个模块的相关信息

CUDAExtension
在ext_modules中采用CUDAExtension指明Cuda/C++的文件路径，其中第一个参数为对应模块的名字，第二个参数为包含所有文件路径的列表。

这里的模块名和Cuda/C++中m.def()定义的算子名共同决定了调用算子的方式。
例如两数数组相加的模块名是sum_double、算子方法名是forward,
所以在python中调用该算子的方式为sum_double.forward()

值得一提的是packages的值为list[str]，表示本地需要打包的package,
这里find_packages()是找出本地所有的package。
因为这里打爆只考虑ops/src/中的文件，所以packages=['ops/src']也能正常编译
这里采用find_packages().


Pytorch包装
为了让自定义算子能够正向传播、反向传播，
我们需要集成torch.autograd.Function进行算子包装。
这里以sum_double为例进行介绍。

# ops/ops_py/sum.py
import sum_double就是导入的setup.py中定义的模块名。
自定义的torch.autograd.Function类型要实现forward、backward函数，并声明为静态成员函数


forward
前向传播的前半部分就是正常传入Tensor进入接口，如果传入向量在之前的代码里是所引出来的很可能非连续，所以建议在传入算子的时候使其连续。
如果算子不需要考虑反向传播，可以用ctx.mark_non_differentiable(ans)将函数的输出标记不需要微分

backward
backward的输入对应的是forward的输出，输出对应的是forward的输入，
这里backward的输入g_out对应forward输出ans，backward的输出g_in1,g_in2对应forward输入array1,array2.

如果算子不需要考虑反向传播，则直接return None, None。 
否则就按照输入变量的梯度进行计算。

问题一1：如果反向传播需要用到forward的信息，可以用ctx进行记录存取。
例如对一个数组求和，则反向传播的梯度为原数组长度的向量。就可以在forward中用
ctx.shape = array.shape[0]记录输入数组长度，并在backward中通过n = ctx.shape进行读取。

如果存取的是Tensor则建议使用save_for_backward(x, y, z, ...)存储向量，
并用x,y,z, ... = ctx.saved_tensors 取向量，而不是直接用ctx

注：save_for_backward()只能存向量，标量用ctx直接存取。
最后用sum_double_op = SumDouble.apply 获取最终的函数形式。


__init__.py

