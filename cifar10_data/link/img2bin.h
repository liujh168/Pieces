//用自己的图片文件制作类cifar10数据集。实现方法如下：
//1、采集样本图像，把不同类别的图像文件分别存于工作目录 _strWorkingFolder 下不同的子文件夹中，子文件夹依次命名为类别名（如cat、dog之类），图像文件名则随意。
//2、用renBatchImag（）函数按照“类别名_顺序号.jpg”的形式批量重命名各类别子文件夹下所有图像文件，如："cat_1.jpg"、"cat_2.jpg"和"dog_1.jpg"、"dog_2.jpg"。
//3、用img2bin()函数把全部准备好的样本图像生成想要的二进制数据集(含data_batch_1.bin及makeBatchesMeta.txt文件)。
//	 CCifar10 binData;	binData.img2bin( "f:\\dl\\images");
//4、将 tensorflow/models/image/cifar10 模块中获取数据的部分参数修改成为适合自己数据集。
//5、完成在自定义数据集上用 tensorflow/models/image/cifar10 模块的源码训练测试

#pragma once
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

class CCifar10
{
public:
	CCifar10()
	{
		_strWorkingFolder = ".";						//当前目录
		_strDataBatchBin = "data_batch_1.bin"; 
		_strBatchesMeta =  "batches.meta.txt";
		_iHeight = 32;	_iWidth  = 32; 
	}

	~CCifar10(){}
	
	//用_strWorkingFolder目录下所有子目录（子目录为分类名）中的图像文件制作类cifar10数据集，数据集相关文件存放在目录 _strWorkingFolder 中
	bool img2bin(void)const;	
	
	//读取_strWorkingFolder目录下数据集文件“_strDataBatchBin”中i_th张图片，放大scale倍，显示3张并全部保存在不同分类目录中(按照类别分别存放)
	bool bin2img(const int i_th, const float scale = 1, const int method=0)const;	

	std::string	_strWorkingFolder;
	std::string	_strDataBatchBin;
	std::string	_strBatchesMeta;

private:
	bool mat2bin( FILE*& fp, std::string& image_file,unsigned char  label)const;//写入单个图像文件
	const std::vector<std::string> getBatchesMeta(void)const;					//从_strWorkingFolder目录下_strBatchesMeta文件中取得分类名列表
	const std::vector<std::string> listFolders( void)const;						//获得工作目录_strWorkingFolder下的所有子目录名
	const std::vector<std::string> listImgFiles( std::string Folder)const;		//获得目录folder下的所有图像文件名
	static const std::string getFileName( std::string & filename );				//从带路径的文件名中分解出单独的文件名

	int			_iHeight;
	int			_iWidth;
};