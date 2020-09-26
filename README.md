'''
	本项目借用cifar10示例代码识别中国象棋棋子
	python 3.8.3 tensorflow 2.3.0 环境调试通过 2020年9月26日
'''

# 一、如何开始训练与预测：
把数据集文件放入cifar10_data目录下即可开始训练。 训练集与测试集各1个。
终端运行：python cifar10_train 即可开始训练
用tensorboard查看结果2步走： 
1、终端运行 "tensorboad --logdir=d:\chess\cifar10_train" 
2、浏览地址：“localhost:6006” 

# 二、如何制作类cifar10数据集：
cout<<"未找到数据目录 train 或 test "<<endl;

cout<<"\n请先准备好分类图像数据并按照以下要求做好准备工作："<<endl;

cout<<"1、在本程序目录下建立子目录train和test"<<endl;
cout<<"2、train下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
cout<<"3、test下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
cout<<"4、运行本程序 dataset.exe 即可分别在train目录下生成训练数据集，test目录下生成测试集。"<<endl;
cout<<"4、当前目录下也3个数据集文件的备份，train和test目录下有几张从数据集中提取出来的图片（用于验证数据集正确性）。\n"<<endl;

# 三、源代码用法
用自己的棋盘图文件制作类Cifar10中国象棋棋子图像数据集，用于棋盘识别等。实现方法如下：
1、用自己的棋盘图文件制作类cifar10棋子数据集。采集样本图像，把不同兵种的棋子图像文件分别存于工作目录 _strWorkingFolder 下不同的子文件夹中，子文件夹依次命名为类别名（如rook、knight共10类），图像文件名则随意。
2、用renBatchImag（）函数按照“类别名_顺序号.jpg”的形式批量重命名各类别子文件夹下所有图像文件，如：”rook_1.jpg”、”rook_2.jpg”和”knight_1.jpg”、”knight_2.jpg”。这步也可省略。
3、用img2bin()函数把全部准备好的样本图像生成想要的二进制数据集(含data_batch_x.bin、test_batch.bin及makeBatchesMeta.txt等文件)。 CCifar10 binData; binData.img2bin( “c:\\dl\\Pieces”);
4、将 tensorflow/models/image/cifar10 模块中获取数据的部分参数修改成为适合自己数据集。
5、完成在自定义数据集上用 tensorflow/models/image/cifar10 模块的源码训练测试。

用法示例：CCifar10 binData;	binData.img2bin( "c:/dl/Pieces"); //即在目录Pieces下生成数据文件与data_meta.txt文件。

作者：潇湘棋士 qq: 77156973 欢迎交流。
