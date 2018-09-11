# Pieces
制作类Cifar10中国象棋棋子图像数据集，用于棋盘识别等。

用自己的图片文件制作类cifar10数据集。实现方法如下：
1、用自己的棋盘图文件制作类cifar10棋子数据集。采集样本图像，把不同兵种的棋子图像文件分别存于工作目录 _strWorkingFolder 下不同的子文件夹中，子文件夹依次命名为类别名（如rook、knight共10类），图像文件名则随意。
2、用renBatchImag（）函数按照“类别名_顺序号.jpg”的形式批量重命名各类别子文件夹下所有图像文件，如：”rook_1.jpg”、”rook_2.jpg”和”knight_1.jpg”、”knight_2.jpg”。这步也可省略。
3、用img2bin()函数把全部准备好的样本图像生成想要的二进制数据集(含data_batch_x.bin、test_batch.bin及makeBatchesMeta.txt等文件)。 CCifar10 binData; binData.img2bin( “c:\\dl\\Pieces”);
4、将 tensorflow/models/image/cifar10 模块中获取数据的部分参数修改成为适合自己数据集。
5、完成在自定义数据集上用 tensorflow/models/image/cifar10 模块的源码训练测试。

用法示例：CCifar10 binData;	binData.img2bin( "c:/dl/Pieces"); //即在目录Pieces下生成数据文件与data_meta.txt文件。
