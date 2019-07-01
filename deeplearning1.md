# Neural Networks and Deep Learning
Summer School
Kunag yaming Honors School
吴胜俊教授
- Textbook: Michael A. Nielsen "Neural NEtworks and Deep Learning 
- Online version: [链接](http://neuralnetworksanddeeplearning.com)

## Contents:
- Introduction
- Neural networks and learning
-  Backpropagation algorithm
-   Neural nets to compute any function
-  Improving the way neural nets learn
- Deep neural nets
-  Deep learning with ALphaGo
- Quantum neural network and learning

## Introduction
- 解决的问题:
  - 图像识别
  - 声音识别
  - 自然语言处理
  - etc

- 课程例子:
  - 手写数字的识别:
    - 对于人类easy
    - 对于rule-based 算法: hard;
  - 解决的方法:
    - Segmentation: 分段,切片;
    - 分类;
    - Classifying individual digit:
      - 输入数据集:$[0.0,1.0]^{28\times 28}$
      - 函数f(进行分类);
      - 输出数据集:$\{0,1....,9\}$ (Class labels)
  - Segmentation 问题可能的方法;
    - 试图分类
    - 用一个individual digit classfier 对上面的分类进行打分
      - 高分,低分反馈,然后继续分类;
    - 得到最高分的分类

- 神经网络
  -  定义: 神经网络是一个**神经元**的网络。主要有：输入层,隐藏层，输出层;
      



