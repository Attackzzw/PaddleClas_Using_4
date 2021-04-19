# PaddleClas_Using_4

AiStudio:[PaddleClas 预测部署（四）](https://aistudio.baidu.com/aistudio/projectdetail/1589564)
# 基于上篇文章[部署三](https://aistudio.baidu.com/aistudio/projectdetail/1184186)，将详细介绍Python 和 C# 调用Dll传参 实现PaddleClas的预测部署

#### **上篇文章主要介绍了：如何生成 exe可执行文件进行预测，如何生成dll动态链接库Python/C#调用预测（图片路径是再生成dll前规定好的，并不能实现从外界传参调用）**
#### **接下来主要介绍传参进来实现更便捷的dll调用**


## 准备：
### **往下进行前，需要已经完成文章[部署三](https://aistudio.baidu.com/aistudio/projectdetail/1184186)的cmake编译工作**

#### **注意生成dll前，需要把deploy/cpp_infer/CMakeLists.txt文件内的**
`add_executable(${DEMO_NAME} ${SRCS})`

**改为**
`ADD_library(${DEMO_NAME} SHARED ${SRCS})`

**否则会生成exe文件**

<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/f307bc33f09e46e6aceaab65140a3f2155fccefc7a614b509b9fa0a6ff86d4b3" width = "800" height = "400" align=center />

&nbsp; 

#### **能够按照文章[部署三](https://aistudio.baidu.com/aistudio/projectdetail/1184186)1.4章节成功编译，并打开：**

<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/9e2385cbb6244858a093b7c82ec8edf841b6caa48d304818a2defb3d3fdb6497" width = "800" height = "400" align=center />

<br/>

‘
# 一，Python调用DLL 进行预测（对DLL 传输字符串）
## 1.1更改main.cpp
#### **传输前，需要在编译生成的C++项目中更改main.cpp，(如果忘记如何更改请查看文章[部署三](https://aistudio.baidu.com/aistudio/projectdetail/1184186))：**

#### 主要是将预测分为了 加载模型  和  图片预测  两个函数用于分别调用，这样也可以缩短我们的预测时间

#### **main.cpp代码具体如下：**
```
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/core/utils/filesystem.hpp>
#include <ostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <numeric>

#include <include/cls.h>
#include <include/cls_config.h>

using namespace std;
using namespace cv;
using namespace PaddleClas;

static PaddleClas::Classifier *clas;  //定义全局对象

extern "C" __declspec(dllexport) int LoadModel(char* cfig); //表示python可以调用该dll   
extern "C" __declspec(dllexport) int Infer(char* jpg); //表示python可以调用该dll   

int LoadModel(char* cfig) {    //加载模型

    ClsConfig config(cfig);
    config.PrintConfigInfo();
    clas = new PaddleClas::Classifier(config.cls_model_path, config.cls_params_path,
        config.use_gpu, config.gpu_id, config.gpu_mem,
        config.cpu_math_library_num_threads, config.use_mkldnn,
        config.use_tensorrt, config.use_fp16,
        config.resize_short_size, config.crop_size);

    return 0;
}

int Infer(char* jpg) {			//预测图片

    std::string path(jpg);
    std::vector<std::string> img_files_list;
    if (cv::utils::fs::isDirectory(path)) {
        std::vector<cv::String> filenames;
        cv::glob(path, filenames);
        for (auto f : filenames) {
            img_files_list.push_back(f);
        }
    }
    else {
        img_files_list.push_back(path);
    }
    std::cout << "img_file_list length: " << img_files_list.size() << std::endl;

    double elapsed_time = 0.0;
    int warmup_iter = img_files_list.size() > 5 ? 5 : 0;
    for (int idx = 0; idx < img_files_list.size(); ++idx) {
        std::string img_path = img_files_list[idx];
        cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
        cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);

        double run_time = clas->Run(srcimg);
        if (idx >= warmup_iter) {
            elapsed_time += run_time;
            std::cout << "Current image path: " << img_path << std::endl;
            std::cout << "Current time cost: " << run_time << " s, "
                << "average time cost in all: "
                << elapsed_time / (idx + 1 - warmup_iter) << " s." << std::endl;
        }
        else {
            std::cout << "Current time cost: " << run_time << " s." << std::endl;
        }
    }
    return 0;
}
```

<br/>

<img src="https://ai-studio-static-online.cdn.bcebos.com/544686c7ed204aeab89242c2cffb732b973518dd393145769d50e92b82c8c06b" width = "800" height = "400" align=center />

&nbsp; 

#### **完成后重新生成 DLL文件即可**
## 1.2 更改Python
#### **C++部分更改完成后，开始配置python用于调用DLL**
#### **Python传输DLL字符串，需要使用ctypes对字符串进行转换，因为python里面没有string类型 ，代码主要为下：**
```
from ctypes import *
import ctypes


dll=CDLL("G:\\projects\\PaddleClas-release-2.0\\deploy\\cpp_infer\\out\\Release\\clas_system.dll")

config=bytes("G:/projects/PaddleClas-release-2.0/deploy/cpp_infer/tools/config.txt","utf-8")
jpg=bytes('E:/PaddleClas-release-2.0/demo/cat_9.jpg',"utf-8")

LoadModel = dll.LoadModel(config)  	#加载模型
infer = dll.Infer(jpg)			#预测图片

```
## 1.3导入关联DLL
#### **如果出现以下错误，提示找不到DLL文件：（如果没有出现请忽略）**

<img src="https://ai-studio-static-online.cdn.bcebos.com/c53ed93d893a4249b0de7b1aeec4f17a136c975ff0704417992bb2d1e10780cb" width = "800" height = "400" align=center />

&nbsp; 

#### **原因在于生成的DLL再调用其他DLL文件，但是该文件DLL没有找到，可以将预测库内`paddle_inference_install_dir\paddle\lib\`paddle_fluid.dll 文件复制到 生成DLL的Release目录内：**

<img src="https://ai-studio-static-online.cdn.bcebos.com/9621ba8dd6624c7c82487359e2e371df42b7677fd34442519e7aa26e47303a19" width = "800" height = "400" align=center />

&nbsp; 

## 1.3 更改config.txt配置
#### **调用DLL会为其传 config.txt 和 预测图片 的路径 ，在传输前需要对 config.txt的内容进行更改：**
#### **deploy\cpp_infer\tools\config.txt内容如下：**

&nbsp; 

```
# model load config
use_gpu  1				#使用gpu
gpu_id  0
gpu_mem  2000 #4000		#gpu内存，根据自己gpu内存来
cpu_math_library_num_threads  10
use_mkldnn 1
use_tensorrt 0
use_fp16 0

# cls config
cls_model_path  G:\projects\PaddleClas-release-2.0\output\ResNet50_vd\best_model\mp\inference.pdmodel  #inference 模型文件路径
cls_params_path G:\projects\PaddleClas-release-2.0\output\ResNet50_vd\best_model\mp\inference.pdiparams  #inference 权重文件路径
resize_short_size 256
crop_size 224
```
&nbsp; 

#### **inference文件再[部署一](https://aistudio.baidu.com/aistudio/projectdetail/1133588)四章节中有说明和导出**

&nbsp; 

<img src="https://ai-studio-static-online.cdn.bcebos.com/f6616e6a85f34986a618a7066a9b64c4103d3fb9646e4e65bd5385865ba49dec" width = "800" height = "400" align=center />

&nbsp; 

## 1.4 配置准备就绪后运行python

&nbsp; 

<img src="https://ai-studio-static-online.cdn.bcebos.com/d1a0d3418f984934a58e19e2c76a670033e0c356893f445f93aa9a1b24497ac8" width = "800" height = "400" align=center />

&nbsp; 

#### **成功预测（此次使用的模型是再本机训练的，轮数不高只做示例，若提高效果可以尝试使用预训练模型训练后的模型，具体内容请查看[PaddleClasGitHub](https://github.com/PaddlePaddle/PaddleClas/)）**

#### **调用DLL，还有一点，config和jpg路径千万不要写错，否则会出Windows Error -529697949错误**
#### **调用DLL，还有一点，config和jpg路径千万不要写错，否则会出Windows Error -529697949错误**
#### **调用DLL，还有一点，config和jpg路径千万不要写错，否则会出Windows Error -529697949错误，重要的事情说三篇**

# 二、C#调用dll

#### **C#调用DLL前，需要创建一个控制台应用，如果有问题可以参考[部署三](https://aistudio.baidu.com/aistudio/projectdetail/1184186)2.6章节**

## 2.1 C#导入DLL

#### **控制台应用创建完成后，需要将我们之前的DLL以及关联的DLL都导入到C#内（关联DLL也包括预测库内的paddle_fluid.dll）**

&nbsp; 

#### **关联DLL从目录`PaddleClas-release-2.0\deploy\cpp_infer\out\Release`复制到C#项目 `ConsoleApp1\bin\Debug\netcoreapp3.1` 目录内**

&nbsp; 

<img src="https://ai-studio-static-online.cdn.bcebos.com/e9db55534ae2482791179bf602081eefdf78917d6e024919bbc6301988057e17" width = "800" height = "400" align=center />

&nbsp; 

## 2.2 打开C#项目，录入代码如下：
```
using System;
using System.Runtime.InteropServices;

namespace ConsoleApp1
{
    class Program
    {
        [DllImport("clas_system.dll", EntryPoint = "LoadModel", CharSet = CharSet.Ansi)]
        public static extern void LoadModel(string config);//,string jpg
        [DllImport("clas_system.dll", EntryPoint = "Infer", CharSet = CharSet.Ansi)]
        public static extern void Infer(string jpg);//,string jpg
        static void Main()
        {
            string config = "G:/projects/PaddleClas-release-2.0/deploy/cpp_infer/tools/config.txt";
            string jpg = "E:/PaddleClas-release-2.0/demo/cat_9.jpg";
            LoadModel(config);
            Infer(jpg);
            Console.Write("Press any key to continue . . . ");
            Console.ReadKey(true);
        }
    }
}
```

&nbsp; 

<img src="https://ai-studio-static-online.cdn.bcebos.com/3e8d1d5783c74693b0e2c9b23469447d75ba47ece9c540c895a825332e8d6f92" width = "800" height = "400" align=center />

&nbsp; 

## 2.3 运行：

#### **可以F5运行程序，或者在目录`ConsoleApp1\bin\Debug\netcoreapp3.1`内运行程序 ConsoleApp1.exe即可**
&nbsp; 

![](https://ai-studio-static-online.cdn.bcebos.com/bbf3cdcea9254c04b29176058632056ce83101be6c9c4e5ba2ced201150bbcc8)

&nbsp; 

<img src="https://ai-studio-static-online.cdn.bcebos.com/886fa1afdaf8491ab72bb9fa7eae4fc2e2bbfa3ac4294988b045ebbb3123dee2" width = "800" height = "400" align=center />

&nbsp; 

#### **预测程序没有设置返回值，预测结果即显示在控制台内，在此基础上我们还可以添加更多便于预测的方法，比如在控制台内输入图片路径，或者一批图片等等，再或者创建窗体（需要修改c++返回值），更多有趣的功能可以自己尝试~**

&nbsp; 

#### **还有一点，一定要确定 文件目录 是存在的，要不报错会很迷糊**
&nbsp; 

### **如果在部署过程中有任何问题欢迎在评论区提出，或者在[PaddleClas-Github](https://github.com/PaddlePaddle/PaddleClas)提issue**




