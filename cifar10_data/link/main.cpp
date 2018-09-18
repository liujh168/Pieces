#include <windows.h>

#include <iostream>
#include <direct.h>

#include "img2bin.h"

void helper(void);

int main(int argc, char* argv[])
{
	if( argc !=2){	helper(); return 0;}
	string commargv(argv[1]);
	char strFile[2054];
	string oldfn;
	CCifar10 binData;  

	//GetModuleFileName(NULL, strFile, MAX_PATH);
	_getcwd(strFile, MAX_PATH);

	if(commargv=="train")
	{
		binData._strWorkingFolder = std::string(strFile)+"/train";
		if(!binData.img2bin())
		{
			cout<<"未找到数据目录 train 或 test "<<endl;

			helper();

			return 0;
		}
		binData.bin2img(10, 3);			//显示验证

		string oldfn = "./train/data_batch_1.bin";
		CopyFile(oldfn.c_str(), "data_batch_1.bin", false);

		oldfn = "./train/batches.meta.txt";
		CopyFile(oldfn.c_str(), "batches.meta.txt", false);

		cout<<"训练集数据文件生成完毕！"<<endl;
		system("pause");

	}

	if(commargv=="test")
	{
		binData._strWorkingFolder = std::string(strFile)+"/test";
		binData._strDataBatchBin = "test_batch.bin";
		if(!binData.img2bin())
		{
			cout<<"未找到数据目录 train 或 test "<<endl;

			helper();

			return 0;
		}
		binData.bin2img(10, 3);			//显示验证

		oldfn = "./test/test_batch.bin";
		CopyFile(oldfn.c_str(), "test_batch.bin", false);

		cout<<"测试集数据生成完毕！"<<endl;
		system("pause");
	}

	return 0 ;
}

void helper(void)
{
	cout<<"本程序用于生成类cifar10数据集文件，用法如下："<<endl;
	cout<<"\n请先准备好分类图像数据并按照以下要求做好准备工作："<<endl;

	cout<<"1、在本程序目录下建立子目录train和test"<<endl;
	cout<<"2、train下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
	cout<<"3、test下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
	cout<<"4、运行本程序，dataset trian 或者 dataset test 分别在train目录下生成训练数据集，test目录下生成测试集。"<<endl;
	cout<<"4、当前目录下也3个数据集文件的备份，train和test目录下有几张从数据集中提取出来的图片（用于验证数据集正确性）。\n"<<endl;
	system("pause");
}