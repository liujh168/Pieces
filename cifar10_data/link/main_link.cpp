#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <conio.h>
#include <string.h>
#include <direct.h>

#include "tools.h"
#include "hotkey.h"
#include "img2bin.h"

using namespace cv;
using namespace std;

DWORD	dwThreadIDMain;
BOOL	bSetHandle;
HANDLE	g_bmEvent;

void	setup(void);						//启动代码
void	clean(void);						//清理代码
BOOL	WINAPI ConsoleCtrlHandler(DWORD dwCtrlType);  //控制台钩子函数	

int main(int argc, char* argv[])
{
	CCifar10 binData;  

	if( argc ==2)
	{
		string commargv = std::string(argv[1]);
		if(commargv=="-h"||commargv=="--help"||commargv=="help"||commargv=="/h"||commargv=="/help")
		{		
			cout<<"本程序用于生成类cifar10数据集文件，用法如下："<<endl;
			cout<<"1、在本程序目录下建立子目录train和test"<<endl;
			cout<<"2、train下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
			cout<<"3、test下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
			cout<<"4、运行本程序，分别在train目录下生成训练数据集，test目录下生成测试集。"<<endl;
			cout<<"4、当前目录下也3个数据集文件的备份，train和test目录下有几张从数据集中提取出来的图片（用于验证数据集正确性）。\n"<<endl;
			system("pause");
			return 0;
		}
	}


	OPENFILENAME myofn;
	char strFile[MAX_PATH];
	memset(&myofn,0,sizeof(OPENFILENAME));
	memset(strFile,0,sizeof(char)*MAX_PATH);
	myofn.nMaxFile = MAX_PATH;
	myofn.lStructSize = sizeof(OPENFILENAME);
	myofn.Flags = OFN_FILEMUSTEXIST;
	//myofn.lpstrFilter="数据集文件(*.*)\0*.png\0\0";	//要选择的文件后缀 
	myofn.lpstrInitialDir = ".";						//默认的文件路径 
	myofn.lpstrFile=strFile;							//存放文件的缓冲区  
	//if(GetOpenFileName(&myofn))							//strFile得到用户所选择文件的路径和文件名 
	{
		binData._strWorkingFolder  = std::string(strFile) ;	//strFile得到文件名
		int iEndIndex;
		iEndIndex = (binData._strWorkingFolder).find_last_of("\\");
		binData._strWorkingFolder = binData._strWorkingFolder.substr( 0, iEndIndex);
	}
	
	//GetModuleFileName(NULL, strFile, MAX_PATH);
	_getcwd(strFile, MAX_PATH);

	binData._strWorkingFolder = std::string(strFile)+"/train";
	if(!binData.img2bin())
	{
			cout<<"未找到数据目录 train 或 test "<<endl;

			cout<<"本程序用于生成类cifar10数据集文件，用法如下："<<endl;
			cout<<"\n请先准备好分类图像数据并按照以下要求做好准备工作："<<endl;

			cout<<"1、在本程序目录下建立子目录train和test"<<endl;
			cout<<"2、train下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
			cout<<"3、test下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
			cout<<"4、运行本程序，分别在train目录下生成训练数据集，test目录下生成测试集。"<<endl;
			cout<<"4、当前目录下也3个数据集文件的备份，train和test目录下有几张从数据集中提取出来的图片（用于验证数据集正确性）。\n"<<endl;
			system("pause");
			return 0;
	}
	binData.bin2img(10, 3);			//显示验证

	cout<<"训练集数据文件生成完毕！"<<endl;
	system("pause");


	binData._strWorkingFolder = std::string(strFile)+"/test";
	binData._strDataBatchBin = "test_batch.bin";
	if(!binData.img2bin())
	{
			cout<<"未找到数据目录 train 或 test "<<endl;

			cout<<"本程序用于生成类cifar10数据集文件，用法如下："<<endl;
			cout<<"\n请先准备好分类图像数据并按照以下要求做好准备工作："<<endl;

			cout<<"1、在本程序目录下建立子目录train和test"<<endl;
			cout<<"2、train下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
			cout<<"3、test下建立不同类别的子目录，子目录下分别放入各类别的图像，文件名随意；"<<endl;
			cout<<"4、运行本程序，分别在train目录下生成训练数据集，test目录下生成测试集。"<<endl;
			cout<<"4、当前目录下也3个数据集文件的备份，train和test目录下有几张从数据集中提取出来的图片（用于验证数据集正确性）。\n"<<endl;
			system("pause");
			return 0;
	}
	binData.bin2img(10, 3);			//显示验证

	string oldfn = "./train/data_batch_1.bin";
	CopyFile(oldfn.c_str(), "data_batch_1.bin", false);

	oldfn = "./test/test_batch.bin";
	CopyFile(oldfn.c_str(), "test_batch.bin", false);

	oldfn = "./train/batches.meta.txt";
	CopyFile(oldfn.c_str(), "batches.meta.txt", false);

	cout<<"数据集生成完毕！"<<endl;
	system("pause");

	return 0 ;

	setup();

	MSG msg = {0};
	while (GetMessage(&msg, NULL, 0, 0) != 0)		//开始消息循环
	{
		switch (msg.message)
		{
		case WM_HOTKEY:
			onhotkey(msg);
			break;
		case WM_QUIT:		//控制台模式这里收不到？
			std::cout << "WM_QUIT! \n"<< std::endl;
			break;
		default:
			break;
		}    
		TranslateMessage( &msg );  
		DispatchMessage( &msg );  
	}   

	if(msg.message==WM_QUIT)
	{
		clean();
	}

	return 0;
}

HWND hwndConsole;
void setup(void)
{
	//放大窗口，设置标题（标题以后可用于显示引擎思考结果）
	SetConsoleTitle("buil my own dataset like cifar10");
	//ShowWindow(::GetConsoleWindow(),SW_MAXIMIZE);

	//在控制台窗口中显示img
	hwndConsole = ::GetConsoleWindow();
	RECT rect;
	GetWindowRect( hwndConsole, &rect );

	//注册热键
	hotkey();

	//创建自动重置事件内核对象
	g_bmEvent = CreateEvent(NULL, FALSE, FALSE, "ucci"); //创建事件对象,注意参数的含义
	
	if (g_bmEvent)//保证应用程序只有一个实例
	{
		if (GetLastError() == ERROR_ALREADY_EXISTS)
		{
			cout<<"本程序已运行!"<<endl;
			return;
		}
	}

	//安装控制台钩子，处理WM_QUIT消息，以便有机会处理退出逻辑
	bSetHandle = SetConsoleCtrlHandler(ConsoleCtrlHandler, true);	
}

void clean(void)
{
	//清理热键
	unhotkey();

	if (bSetHandle) 
	{
		SetConsoleCtrlHandler(ConsoleCtrlHandler, false);   //删除控制台钩子
	}

	CloseHandle(g_bmEvent);
	//MessageBox(NULL, "get wm_quit to exit !  Program being closed!", "CEvent", MB_OK);
}

//BOOL bSetHandle = SetConsoleCtrlHandler(ConsoleCtrlHandler, true);	
//if (bSetHandle) 
//	SetConsoleCtrlHandler(ConsoleCtrlHandler, false);   //删除控制台钩子
BOOL WINAPI ConsoleCtrlHandler(DWORD dwCtrlType)
{
	if(dwCtrlType == CTRL_CLOSE_EVENT)
	{
		PostThreadMessage(dwThreadIDMain, WM_QUIT, 0, 0);
		//MessageBox(NULL, "Program being closed!", "CEvent", MB_OK);
		return TRUE;
	}
	return FALSE;
}
