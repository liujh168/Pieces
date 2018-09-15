#include "tools.h"

#include "opencv2/legacy/legacy.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

#include <io.h>
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
//#include <opencv2\ml.hpp>
using namespace cv;
//using namespace ml;

//自动判断识别OpenCV的版本号，并据此添加对应的依赖库（.lib文件）的方法
#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )
#pragma comment( lib, cvLIB("ml") )

#pragma comment( lib, cvLIB("calib3d") )

int g_biBitCount = 24;

//按行合并如下：
cv::Mat mergeRows(cv::Mat A, cv::Mat B)
{
	// cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	int totalRows = A.rows + B.rows;
	cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
	cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);
	return mergedDescriptors;
}

//按列合并如下：
cv::Mat mergeCols(cv::Mat A, cv::Mat B)
{
	// cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	int totalCols = A.cols + B.cols;
	cv::Mat mergedDescriptors(A.rows,totalCols, A.type());
	cv::Mat submat = mergedDescriptors.colRange(0, A.cols);
	A.copyTo(submat);
	submat = mergedDescriptors.colRange(A.cols, totalCols);
	B.copyTo(submat);
	return mergedDescriptors;
}


//图像模板匹配
//一般而言，源图像与模板图像patch尺寸一样的话，可以直接使用上面介绍的图像相似度测量的方法；
//如果源图像与模板图像尺寸不一样，通常需要进行滑动匹配窗口，扫面个整幅图像获得最好的匹配patch。
//在OpenCV中对应的函数为：matchTemplate()：函数功能是在输入图像中滑动窗口寻找各个位置与模板图像patch的相似度。相似度的评价标准（匹配方法）有：
//CV_TM_SQDIFF 平方差匹配法（相似度越高，值越小），
//CV_TM_CCORR 相关匹配法（采用乘法操作，相似度越高值越大），
//CV_TM_CCOEFF 相关系数匹配法（1表示最好的匹配，-1表示最差的匹配）。
//通常,随着从简单的测量(平方差)到更复杂的测量(相关系数),我们可获得越来越准确的匹配(同时也意味着越来越大的计算代价). /
//最好的办法是对所有这些设置多做一些测试实验,以便为自己的应用选择同时兼顾速度和精度的最佳方案.//

//有一种新的用来计算相似度或者进行距离度量的方法：EMD，Earth Mover‘s Distances
//EMD is defined as the minimal cost that must be paid to transform one histograminto the other, where there is a “ground distance” between the basic featuresthat are aggregated into the histogram。
//光线变化能引起图像颜色值的漂移，尽管漂移没有改变颜色直方图的形状，但漂移引起了颜色值位置的变化，从而可能导致匹配策略失效。而EMD是一种度量准则，度量怎样将一个直方图转变为另一个直方图的形状，包括移动直方图的部分（或全部）到一个新的位置，可以在任意维度的直方图上进行这种度量。
//在OpenCV中有相应的计算方法：cvCalcEMD2()。结合着opencv支持库，计算直方图均衡后与原图的HSV颜色空间直方图之间的EMD距离。

//cv::Point pt;
//Mat image= imread("board.jpg");
//Mat tepl= imread("position.jpg");
//double d = match(image, tepl, &pt,  CV_TM_SQDIFF) //CV_TM_SQDIFF_NORMED  CV_TM_CCORR CV_TM_CCOEFF CV_TM_CCORR_NORMED CV_TM_CCOEFF_NORMED
double match(cv::Mat image, cv::Mat templ, cv::Point &matchLoc, int method)
{
	int result_cols =  image.cols - templ.cols + 1;
	int result_rows = image.rows - templ.rows + 1;

	cv::Mat result = cv::Mat( result_cols, result_rows, CV_32FC1 );
	cv::matchTemplate( image, templ, result, method );
	//cv::normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxVal, matchVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

	//CV_TM_SQDIFF平方差匹配法（相似度越高，值越小）
	//CV_TM_CCOEFF相关系数匹配法（1表示最好的匹配，-1表示最差的匹配）。
	//CV_TM_CCORR相关匹配法（采用乘法操作，相似度越高值越大），
	switch(method)
	{
	case CV_TM_SQDIFF:								//CV_TM_SQDIFF平方差匹配法（相似度越高，值越小）
		matchLoc = minLoc;								//
		matchVal = minVal / (templ.cols * templ.cols);	//去掉模板大小对匹配度的影响：
		break;
	case CV_TM_CCORR:
	case CV_TM_CCOEFF:
		matchLoc = maxLoc;
		matchVal = maxVal / (templ.cols * templ.cols);
		break;
	case CV_TM_SQDIFF_NORMED:		
		matchLoc = minLoc;
		matchVal =  minVal;
		break;
	case CV_TM_CCORR_NORMED:
	case CV_TM_CCOEFF_NORMED:
		matchLoc = maxLoc;
		matchVal =  maxVal;
		break;
	default:
		matchLoc = maxLoc;
		matchVal =  maxVal;
		break;
	}
#ifdef _DEBUG
	// 看看最终结果
	//rectangle(image, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
	//imshow( "image", image );
	//waitKey(0);
#endif
	return matchVal;
}

void StrToClip(char* pstr) //拷贝FEN串到剪贴板
{
	if(::OpenClipboard (NULL))//打开剪贴板
	{
		HANDLE hGlobal;
		char* pGlobal;
		::EmptyClipboard();//清空剪贴板

		//写入数据
		hGlobal=::GlobalAlloc(GHND|GMEM_SHARE, strlen(pstr)+1);
		pGlobal=(char*)GlobalLock(hGlobal);
		//strcpy_s(pGlobal, strlen(pstr)+1, pstr);
		lstrcpy(pGlobal, pstr);
		::GlobalUnlock(hGlobal);//解锁
		
		::SetClipboardData(CF_TEXT,hGlobal);//设置格式  //如果是UNICODE格式，则第一个参数需修改为 UNICODETEXT

		//关闭剪贴板
		::CloseClipboard();
	}
	else
	{
#ifdef _DEBUG
			cout<< "打开剪贴板出错！以下fen串未复制到剪贴板：" << pstr << endl;
#endif
	}
}

void ClipToStr(char* pstr) //从剪贴板拷贝到pstr  这个函数有点问题
{
	//判断剪贴板的数据格式是否可以处理。
	if (!IsClipboardFormatAvailable(CF_TEXT))
	{
		return;
	}

	//打开剪贴板。
	if (!OpenClipboard(NULL))
	{
		return;
	}

	//获取UNICODE的数据。
	HGLOBAL hMem = GetClipboardData(CF_TEXT);
	if (hMem != NULL)
	{
		//获取UNICODE的字符串。
		LPTSTR lpStr = (LPTSTR)GlobalLock(hMem);
		if (lpStr != NULL)
		{
			//显示输出。
			strcpy_s(pstr, strlen(lpStr)+1, lpStr);

			//释放锁内存。
			GlobalUnlock(hMem);
		}
	}

	//关闭剪贴板。
	CloseClipboard();
}

//找到兵河窗口并模拟新建，粘贴局面，开始分析
void start_bh(string fen, bool turn)
{
	HWND bh = FindWindow(NULL,"BHGUI(test) - 新棋局"); 
	if(bh==NULL || !IsWindow(bh))
	{
		return;
		//int iResult = (int)ShellExecute(NULL,"open","c:\\bh\\bh.exe",NULL,NULL,SW_SHOWNORMAL);    //执行应用程序
		//Sleep(3000);
		//HWND bh = FindWindow(NULL,"BHGUI(test) - 新棋局"); 
	}
	BringWindowToTop(bh);  
	SetForegroundWindow(bh);

	keybd_event(VK_CONTROL, 0, 0, 0);	//Alt Pres
	keybd_event('N', 0, 0, 0);			//Alt Pres
	keybd_event('N', 0, KEYEVENTF_KEYUP, 0);		
	keybd_event(VK_CONTROL,0,KEYEVENTF_KEYUP,0);	
	Sleep(1000);        //停顿一秒
	std::cout << "兵河开新局!"<< std::endl;

	StrToClip((char*)fen.c_str()); //拷贝FEN串到剪贴板
	keybd_event(VK_MENU, 0xb8, 0, 0);	//Alt 
	keybd_event('C', 0, 0, 0);			
	keybd_event('C', 0, KEYEVENTF_KEYUP, 0);		
	keybd_event('P', 0, 0, 0);			
	keybd_event('P', 0, KEYEVENTF_KEYUP,  0);		
	Sleep(1000);        //停顿一秒
	std::cout << "兵河粘贴局面:  "<< fen <<  std::endl;

	if(turn)
	{
		keybd_event('E', 0, 0, 0);			//Alt Pres
		keybd_event('A', 0, 0, 0);			//Alt Pres
		keybd_event('E', 0, KEYEVENTF_KEYUP, 0);		
		keybd_event('A', 0, KEYEVENTF_KEYUP, 0);		
		Sleep(1000);        //停顿一秒
		keybd_event(VK_MENU,0xb8, KEYEVENTF_KEYUP,0);		
		std::cout << "兵河开始分析!"<< std::endl;
	}
}


void hwnd2mat(cv::Mat& dst, HWND hwnd)
{
	//HWND hwnd=GetDesktopWindow();
	HDC hwindowDC,hWndCompatibleDC;
	int height,width,srcheight,srcwidth;
	HBITMAP hBitmap;
	BITMAPINFOHEADER  bi;

	hwindowDC = GetDC(NULL);
	hWndCompatibleDC=CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hWndCompatibleDC,COLORONCOLOR);  

	RECT wndsize;    // get the height and width of the screen
	GetWindowRect(hwnd, &wndsize);

	srcheight = wndsize.bottom-wndsize.top;
	srcwidth = wndsize.right-wndsize.left;
	height = srcheight;  //change this to whatever size you want to resize to
	width = srcwidth;

	// create a bitmap
	hBitmap = CreateCompatibleBitmap( hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
	bi.biWidth = width;    
	bi.biHeight = -height;  //this is the line that makes it draw upside down or not
	bi.biPlanes = 1;    
	bi.biBitCount = g_biBitCount;		//此处参数应与CV_8UC4匹配才行。32位时兵河识别时崩溃。如果改用.jpg格式，则此处即使使用32位深度，存盘后也是24位的（win7默认jpg位深24位，PNG32位？）
	bi.biCompression = BI_RGB;    
	bi.biSizeImage = 0;  
	bi.biXPelsPerMeter = 0;    
	bi.biYPelsPerMeter = 0;    
	bi.biClrUsed = 0;    
	bi.biClrImportant = 0;

	// use the previously created device context with the bitmap
	SelectObject(hWndCompatibleDC, hBitmap);
	// copy from the window device context to the bitmap device context
	StretchBlt( hWndCompatibleDC, 0,0, width, height, hwindowDC, wndsize.left, wndsize.top, srcwidth, srcheight, SRCCOPY);	//change SRCCOPY to NOTSRCCOPY for wacky colors !
	
	Mat src;
	src.create(height,width, g_biBitCount==24? CV_8UC3:CV_8UC4);

	GetDIBits(hWndCompatibleDC, hBitmap, 0, height, src.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);			//copy from hWndCompatibleDC to hBitmap

	// avoid memory leak
	DeleteObject (hBitmap); DeleteDC(hWndCompatibleDC); ReleaseDC(hwnd, hwindowDC);

	src.copyTo(dst);
}

//这个方式可以阻挡窗口，参考screenCapture.cpp
void hwnd3mat(cv::Mat& dst, HWND hwnd)
{
	//HWND hwnd=GetDesktopWindow();

	HDC hwindowDC,hWndCompatibleDC;

	int height,width,srcheight,srcwidth;
	HBITMAP hBitmap;
	Mat src;
	BITMAPINFOHEADER  bi;

	hwindowDC=GetDC(hwnd);
	hWndCompatibleDC=CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hWndCompatibleDC,COLORONCOLOR);  

	RECT wndsize;    // get the height and width of the screen
	GetClientRect(hwnd, &wndsize);

	srcheight = wndsize.bottom;
	srcwidth = wndsize.right;
	height = wndsize.bottom;  //change this to whatever size you want to resize to
	width = wndsize.right;

	src.create(height,width,CV_8UC4);
	//src.create(height,width,CV_16UC4);


	// create a bitmap
	hBitmap = CreateCompatibleBitmap( hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
	bi.biWidth = width;    
	bi.biHeight = -height;  //this is the line that makes it draw upside down or not
	bi.biPlanes = 1;    
	bi.biBitCount = 32;    
	bi.biCompression = BI_RGB;    
	bi.biSizeImage = 0;  
	bi.biXPelsPerMeter = 0;    
	bi.biYPelsPerMeter = 0;    
	bi.biClrUsed = 0;    
	bi.biClrImportant = 0;

	// use the previously created device context with the bitmap
	SelectObject(hWndCompatibleDC, hBitmap);
	// copy from the window device context to the bitmap device context
	StretchBlt( hWndCompatibleDC, 0,0, width, height, hwindowDC, 0, 0,srcwidth,srcheight, SRCCOPY);	//change SRCCOPY to NOTSRCCOPY for wacky colors !
	GetDIBits(hWndCompatibleDC,hBitmap,0,height,src.data,(BITMAPINFO *)&bi,DIB_RGB_COLORS);			//copy from hWndCompatibleDC to hBitmap
	
	// avoid memory leak
	DeleteObject (hBitmap); DeleteDC(hWndCompatibleDC); ReleaseDC(hwnd, hwindowDC);
}

void hwnd5mat(cv::Mat& dst, HWND hwnd)//这个最简洁
{
	RECT wndsize;						 // get the height and width of the hwnd
	GetWindowRect(hwnd, &wndsize);

	int wndHeight =wndsize.bottom-wndsize.top;
	int wndWidth = wndsize.right-wndsize.left;

	int screenWidth = GetSystemMetrics(SM_CXSCREEN);		//得到屏幕的分辨率的x
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);		//得到屏幕分辨率的y

	HDC hDesktopDC = GetDC( GetDesktopWindow() );			//得到屏幕的dc
	HDC hDesktopCompatibleDC = CreateCompatibleDC(hDesktopDC);
    HBITMAP hBitmap =CreateCompatibleBitmap(hDesktopDC,wndWidth,wndHeight);	
    SelectObject(hDesktopCompatibleDC,hBitmap); 
	
	StretchBlt( hDesktopCompatibleDC, 0, 0, wndWidth, wndHeight, hDesktopDC, wndsize.left, wndsize.top, wndWidth, wndHeight, SRCCOPY);	//第1种拷贝方法
	//BitBlt(hDesktopCompatibleDC, 0, 0, wndWidth,wndHeight, hDesktopDC, wndsize.left, wndsize.top,SRCCOPY);								//第2种拷贝方法

	Mat src(wndHeight, wndWidth, CV_8UC4);
	
	//第1种获取位图数据的方法, 比第2种简洁
	GetBitmapBits(hBitmap, wndWidth*wndHeight*4, src.data);	
	
	//第2种获取位图数据的方法
	//BITMAPINFO bi;
	//ZeroMemory(&bi, sizeof(BITMAPINFO));
	//bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	//bi.bmiHeader.biWidth = wndWidth;
	//bi.bmiHeader.biHeight = -wndHeight;		//negative so (0,0) is at top left
	//bi.bmiHeader.biPlanes = 1;
	//bi.bmiHeader.biBitCount = 32;			//注意与CV_8UC4这个参数的配套
	//bi.bmiHeader.biCompression = BI_RGB;    
	//bi.bmiHeader.biSizeImage = 0;  
	//bi.bmiHeader.biXPelsPerMeter = 0;    
	//bi.bmiHeader.biYPelsPerMeter = 0;    
	//bi.bmiHeader.biClrUsed = 0;    
	//bi.bmiHeader.biClrImportant = 0;
	//GetDIBits(hDesktopCompatibleDC, hBitmap, 0, wndHeight, src.data, &bi, DIB_RGB_COLORS);	

	//避免内在泄漏
	DeleteObject (hBitmap); ReleaseDC(NULL, hDesktopDC); ReleaseDC(NULL, hDesktopCompatibleDC);//避免内在泄漏

	//深拷贝到引用目标
	src.copyTo(dst);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int hist(  )
{

	//【1】载入素材图并显示
	Mat srcImage;
	srcImage=imread("tt2.jpg");
	imshow( "素材图", srcImage );

	system("color 3F");

	//【2】参数准备
	int bins = 256;
	int hist_size[] = {bins};
	float range[] = { 0, 256 };
	const float* ranges[] = { range};
	MatND redHist,grayHist,blueHist;
	int channels_r[] = {0};

	//【3】进行直方图的计算（红色分量部分）
	calcHist( &srcImage, 1, channels_r, Mat(), //不使用掩膜
		redHist, 1, hist_size, ranges,
		true, false );

	//【4】进行直方图的计算（绿色分量部分）
	int channels_g[] = {1};
	calcHist( &srcImage, 1, channels_g, Mat(), // do not use mask
		grayHist, 1, hist_size, ranges,
		true, // the histogram is uniform
		false );

	//【5】进行直方图的计算（蓝色分量部分）
	int channels_b[] = {2};
	calcHist( &srcImage, 1, channels_b, Mat(), // do not use mask
		blueHist, 1, hist_size, ranges,
		true, // the histogram is uniform
		false );

	//-----------------------绘制出三色直方图------------------------
	//参数准备
	double maxValue_red,maxValue_green,maxValue_blue;
	minMaxLoc(redHist, 0, &maxValue_red, 0, 0);
	minMaxLoc(grayHist, 0, &maxValue_green, 0, 0);
	minMaxLoc(blueHist, 0, &maxValue_blue, 0, 0);
	int scale = 1;
	int histHeight=256;
	Mat histImage = Mat::zeros(histHeight,bins*3, CV_8UC3);

	//正式开始绘制
	for(int i=0;i<bins;i++)
	{
		//参数准备
		float binValue_red = redHist.at<float>(i); 
		float binValue_green = grayHist.at<float>(i);
		float binValue_blue = blueHist.at<float>(i);
		int intensity_red = cvRound(binValue_red*histHeight/maxValue_red);  //要绘制的高度
		int intensity_green = cvRound(binValue_green*histHeight/maxValue_green);  //要绘制的高度
		int intensity_blue = cvRound(binValue_blue*histHeight/maxValue_blue);  //要绘制的高度

		//绘制红色分量的直方图
		rectangle(histImage,Point(i*scale,histHeight-1),
			Point((i+1)*scale - 1, histHeight - intensity_red),
			CV_RGB(255,0,0));

		//绘制绿色分量的直方图
		rectangle(histImage,Point((i+bins)*scale,histHeight-1),
			Point((i+bins+1)*scale - 1, histHeight - intensity_green),
			CV_RGB(0,255,0));

		//绘制蓝色分量的直方图
		rectangle(histImage,Point((i+bins*2)*scale,histHeight-1),
			Point((i+bins*2+1)*scale - 1, histHeight - intensity_blue),
			CV_RGB(0,0,255));

	}

	//在窗口中显示出绘制好的直方图
	imshow( "图像的RGB直方图", histImage );
	waitKey(0);
	return 0;
}


int hist1()
{
	//【1】载入原图并显示
	Mat srcImage = imread("tt2.jpg", 0);
	imshow("原图",srcImage);
	if(!srcImage.data) {cout << "fail to load image" << endl; 	return 0;}

	system("color 1F");

	//【2】定义变量
	MatND dstHist;       // 在cv中用CvHistogram *hist = cvCreateHist
	int dims = 1;
	float hranges[] = {0, 255};
	const float *ranges[] = {hranges};   // 这里需要为const类型
	int size = 256;
	int channels = 0;

	//【3】计算图像的直方图
	calcHist(&srcImage, 1, &channels, Mat(), dstHist, dims, &size, ranges);    // cv 中是cvCalcHist
	int scale = 1;

	Mat dstImage(size * scale, size, CV_8U, Scalar(0));
	//【4】获取最大值和最小值
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dstHist,&minValue, &maxValue, 0, 0);  //  在cv中用的是cvGetMinMaxHistValue

	//【5】绘制出直方图
	int hpt = saturate_cast<int>(0.9 * size);
	for(int i = 0; i < 256; i++)
	{
		float binValue = dstHist.at<float>(i);           //   注意hist中是float类型    而在OpenCV1.0版中用cvQueryHistValue_1D
		int realValue = saturate_cast<int>(binValue * hpt/maxValue);
		rectangle(dstImage,Point(i*scale, size - 1), Point((i+1)*scale - 1, size - realValue), Scalar(255));
	}
	imshow("一维直方图", dstImage);
	waitKey(0);
	return 0;
}

int hist_hsv( )
{

	//【1】载入源图，转化为HSV颜色模型
	Mat srcImage, hsvImage;
	srcImage=imread("tt2.jpg");
	cvtColor(srcImage,hsvImage, CV_BGR2HSV);

	system("color 2F");

	//【2】参数准备
	//将色调量化为30个等级，将饱和度量化为32个等级
	int hueBinNum = 30;//色调的直方图直条数量
	int saturationBinNum = 32;//饱和度的直方图直条数量
	int histSize[ ] = {hueBinNum, saturationBinNum};
	// 定义色调的变化范围为0到179
	float hueRanges[] = { 0, 180 };
	//定义饱和度的变化范围为0（黑、白、灰）到255（纯光谱颜色）
	float saturationRanges[] = { 0, 256 };
	const float* ranges[] = { hueRanges, saturationRanges };
	MatND dstHist;
	//参数准备，calcHist函数中将计算第0通道和第1通道的直方图
	int channels[] = {0, 1};

	//【3】正式调用calcHist，进行直方图计算
	calcHist( &hsvImage,//输入的数组
		1, //数组个数为1
		channels,//通道索引
		Mat(), //不使用掩膜
		dstHist, //输出的目标直方图
		2, //需要计算的直方图的维度为2
		histSize, //存放每个维度的直方图尺寸的数组
		ranges,//每一维数值的取值范围数组
		true, // 指示直方图是否均匀的标识符，true表示均匀的直方图
		false );//累计标识符，false表示直方图在配置阶段会被清零

	//【4】为绘制直方图准备参数
	double maxValue=0;//最大值
	minMaxLoc(dstHist, 0, &maxValue, 0, 0);//查找数组和子数组的全局最小值和最大值存入maxValue中
	int scale = 10;
	Mat histImg = Mat::zeros(saturationBinNum*scale, hueBinNum*10, CV_8UC3);

	//【5】双层循环，进行直方图绘制
	for( int hue = 0; hue < hueBinNum; hue++ )
		for( int saturation = 0; saturation < saturationBinNum; saturation++ )
		{
			float binValue = dstHist.at<float>(hue, saturation);//直方图组距的值
			int intensity = cvRound(binValue*255/maxValue);//强度

			//正式进行绘制
			rectangle( histImg, Point(hue*scale, saturation*scale),
				Point( (hue+1)*scale - 1, (saturation+1)*scale - 1),
				Scalar::all(intensity),
				CV_FILLED );
		}

		//【6】显示效果图
		imshow( "素材图", srcImage );
		imshow( "H-S 直方图", histImg );

		waitKey();
		return 0;
}

double hist_comp(Mat srcImage_base, Mat srcImage_test1)
{
	Mat hsvImage_base;
	Mat hsvImage_test1;

	//srcImage_base = imread( "1.jpg",1 );
	//srcImage_test1 = imread( "2.jpg", 1 );

	//将图像由BGR色彩空间转换到 HSV色彩空间
	cvtColor( srcImage_base, hsvImage_base, CV_BGR2HSV );
	cvtColor( srcImage_test1, hsvImage_test1, CV_BGR2HSV );

	//初始化计算直方图需要的实参
	int h_bins = 50;	int s_bins = 60;	
	int histSize[] = { h_bins, s_bins };
	float h_ranges[] = { 0, 256 };		// hue的取值范围从0到256
	float s_ranges[] = { 0, 180 };		//saturation取值范围从0到180
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };// 使用第0和第1通道

	//创建储存直方图的 MatND 类的实例:
	MatND baseHist;
	MatND testHist1;

	//计算基准图像，两张测试图像，半身基准图像的HSV直方图:
	calcHist( &hsvImage_base, 1, channels, Mat(), baseHist, 2, histSize, ranges, true, false );
	normalize( baseHist, baseHist, 0, 1, NORM_MINMAX, -1, Mat() );

	calcHist( &hsvImage_test1, 1, channels, Mat(), testHist1, 2, histSize, ranges, true, false );
	normalize( testHist1, testHist1, 0, 1, NORM_MINMAX, -1, Mat() );
	
	double base_test1 = compareHist( baseHist, testHist1, CV_COMP_CORREL );			//int compare_method = 0~3

	return base_test1;
}

//vector< Mat > pos_lst;
//vector< int > labels;
//string pos_dir=".";
//load_images( pos_dir, pos_lst, true);
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages)
{
    vector< String > files;
    glob( dirname, files );
    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // load the image
        if ( img.empty() )            // invalid image, skip it.
        {
            cout << files[i] << " is invalid!" << endl;
            continue;
        }
        if ( showImages )
        {
            imshow( "image", img );
            waitKey(0 ); 
        }
        img_lst.push_back( img );
    }
}

//
////编程环境：VS2010 + Opencv2.4.8
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>  
//#include <opencv2/ml/ml.hpp>  
//#include <iostream>  
//#include <string>  
//
//using namespace std;  
//using namespace cv;  

/*
int dnn()  
{  
    //Setup the BPNetwork  
    CvANN_MLP bp;   
    // Set up BPNetwork's parameters  
    CvANN_MLP_TrainParams params;  
    params.train_method=CvANN_MLP_TrainParams::BACKPROP;  //(Back Propagation,BP)反向传播算法
    params.bp_dw_scale=0.1;  
    params.bp_moment_scale=0.1;  

    // Set up training data  
    float labels[10][2] = {{0.9,0.1},{0.1,0.9},{0.9,0.1},{0.1,0.9},{0.9,0.1},{0.9,0.1},{0.1,0.9},{0.1,0.9},{0.9,0.1},{0.9,0.1}};  
    //这里对于样本标记为0.1和0.9而非0和1，主要是考虑到sigmoid函数的输出为一般为0和1之间的数，只有在输入趋近于-∞和+∞才逐渐趋近于0和1，而不可能达到。
    Mat labelsMat(10, 2, CV_32FC1, labels);  

    float trainingData[10][2] = { {11,12},{111,112}, {21,22}, {211,212},{51,32}, {71,42}, {441,412},{311,312}, {41,62}, {81,52} };  
    Mat trainingDataMat(10, 2, CV_32FC1, trainingData);  
    Mat layerSizes=(Mat_<int>(1,5) << 2, 2, 2, 2, 2); //5层：输入层，3层隐藏层和输出层，每层均为两个perceptron
    bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM);//CvANN_MLP::SIGMOID_SYM ,选用sigmoid作为激励函数
    bp.train(trainingDataMat, labelsMat, Mat(),Mat(), params);  //训练

    // Data for visual representation  
    int width = 512, height = 512;  
    Mat image = Mat::zeros(height, width, CV_8UC3);  
    Vec3b green(0,255,0), blue (255,0,0);  
    // Show the decision regions
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)  
        {  
            Mat sampleMat = (Mat_<float>(1,2) << i,j);  
            Mat responseMat;  
            bp.predict(sampleMat,responseMat);  
            float* p=responseMat.ptr<float>(0);  
            //
            if (p[0] > p[1])
            {
                image.at<Vec3b>(j, i)  = green;  
            } 
            else
            {
                image.at<Vec3b>(j, i)  = blue;  
            }
        }  
    }
    // Show the training data  
    int thickness = -1;  
    int lineType = 8;  
    circle( image, Point(111,  112), 5, Scalar(  0,   0,   0), thickness, lineType); 
    circle( image, Point(211,  212), 5, Scalar(  0,   0,   0), thickness, lineType);  
    circle( image, Point(441,  412), 5, Scalar(  0,   0,   0), thickness, lineType);  
    circle( image, Point(311,  312), 5, Scalar(  0,   0,   0), thickness, lineType);  
    circle( image, Point(11,  12), 5, Scalar(255, 255, 255), thickness, lineType);  
    circle( image, Point(21, 22), 5, Scalar(255, 255, 255), thickness, lineType);       
    circle( image, Point(51,  32), 5, Scalar(255, 255, 255), thickness, lineType);  
    circle( image, Point(71, 42), 5, Scalar(255, 255, 255), thickness, lineType);       
    circle( image, Point(41,  62), 5, Scalar(255, 255, 255), thickness, lineType);  
    circle( image, Point(81, 52), 5, Scalar(255, 255, 255), thickness, lineType);       

    imwrite("result.png", image);        // save the image   

    imshow("BP Simple Example", image); // show it to the user  
    waitKey(0);  

    return 0;
}  
*/

//利用cv::addWeighted（）函数结合定义感兴趣区域ROI，实现自定义区域的线性混合
bool  ROI_LinearBlending()
{
	Mat srcImage4= imread("dota_pa.jpg",1);
	Mat logoImage= imread("dota_logo.jpg");
	if( !srcImage4.data || !logoImage.data)
	{
		printf("读取错误！ \n"); 
		return false; 
	}

	Mat imageROI = srcImage4(Rect(200,250,logoImage.cols,logoImage.rows));
	//imageROI= srcImage4(Range(250,250+logoImage.rows),Range(200,200+logoImage.cols));

	addWeighted(imageROI, 0, logoImage, 1, 0, imageROI);  //将logo加到原图上

	imshow("ROI_LinearBlending",srcImage4);

	return true;
}


//-----------------------------【MultiChannelBlending( )函数】--------------------------------
//	描述：多通道混合的实现函数
//-----------------------------------------------------------------------------------------------
bool  MultiChannelBlending()
{
	//【0】定义相关变量
	Mat srcImage;
	Mat logoImage;
	vector<Mat> channels;
	Mat  imageBlueChannel;

	//=================【蓝色通道部分】=================
	//	描述：多通道混合-蓝色分量部分
	//============================================

	// 【1】读入图片
	logoImage= imread("dota_logo.jpg",0);
	srcImage= imread("dota_jugg.jpg");

	if( !logoImage.data ) { printf("Oh，no，读取logoImage错误~！ \n"); return false; }
	if( !srcImage.data ) { printf("Oh，no，读取srcImage错误~！ \n"); return false; }

	//【2】把一个3通道图像转换成3个单通道图像
	split(srcImage,channels);//分离色彩通道

	//【3】将原图的蓝色通道引用返回给imageBlueChannel，注意是引用，相当于两者等价，修改其中一个另一个跟着变
	imageBlueChannel= channels.at(0);
	//【4】将原图的蓝色通道的（500,250）坐标处右下方的一块区域和logo图进行加权操作，将得到的混合结果存到imageBlueChannel中
	addWeighted(imageBlueChannel(Rect(500,250,logoImage.cols,logoImage.rows)), 1.0, logoImage, 0.5, 0, imageBlueChannel(Rect(500,250,logoImage.cols,logoImage.rows)));

	//【5】将三个单通道重新合并成一个三通道
	merge(channels,srcImage);

	//【6】显示效果图
	namedWindow(" <1>游戏原画+logo蓝色通道");
	imshow(" <1>游戏原画+logo蓝色通道",srcImage);

	return true;
}
void test()
{
	vector< Mat > pos_lst;
	vector< int > labels;
	string pos_dir=".";
	load_images( pos_dir, pos_lst, true);
	return;
}


void creatAlphaMat(Mat &mat) // 创建一个图像
{
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            Vec4b&rgba = mat.at<Vec4b>(i, j);
            rgba[0] = 0;//UCHAR_MAX;
            rgba[1] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols)*UCHAR_MAX);
            rgba[2] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows)*UCHAR_MAX);
            //rgba[3] = saturate_cast<uchar>(0.5*(rgba[1] + rgba[2]));
            if(j<200)
				rgba[3] = 0;
			else
				rgba[3]=saturate_cast<uchar>(0.5*(rgba[1] + rgba[2]));
        }
    }
}

int png()
{
    //创建带Alpha通道的 Mat
    Mat mat(480, 640, CV_8UC4);
    creatAlphaMat(mat);

    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    try{
        imwrite("透明值图2.png", mat, compression_params);
        imshow("生成的PNG图", mat);
        fprintf(stdout, "PNG图片文件的数据保存完毕");
        waitKey(0);
    }
    catch (runtime_error& ex){
        fprintf(stderr, "图像转换发生错误:%s\n", ex.what());
        return 1;
    }

    return 0;
}

//opencv3/C++ 机器学习-神经网络ANN_MLP识别数字
//https://blog.csdn.net/akadiao/article/details/79236458
/*********************************************************************************
int train_nn()
{
    ////==========================读取图片创建训练数据==============================////
    //将所有图片大小统一转化为8*16
    const int imageRows = 8;
    const int imageCols = 16;
    //图片共有10类
    const int classSum = 10;
    //每类共50张图片
    const int imagesSum = 50;
    //每一行一个训练图片
    float trainingData[classSum*imagesSum][imageRows*imageCols] = {{0}};
    //训练样本标签
    float labels[classSum*imagesSum][classSum]={{0}};
    Mat src, resizeImg, trainImg;
    for (int i = 0; i < classSum; i++)
    {
        //目标文件夹路径
        std::string inPath = "E:\\image\\image\\charSamples\\";
        char temp[256];
        int k = 0;
        sprintf_s(temp, "%d", i);
        inPath = inPath + temp + "\\*.png";
        //用于查找的句柄
        long handle;
        struct _finddata_t fileinfo;
        //第一次查找
        handle = _findfirst(inPath.c_str(),&fileinfo);
        if(handle == -1)
            return -1;
        do
        {
            //找到的文件的文件名
            std::string imgname = "E:/image/image/charSamples/";
            imgname = imgname + temp + "/" + fileinfo.name;
            src = imread(imgname, 0);
            if (src.empty())
            {
                std::cout<<"can not load image \n"<<std::endl;
                return -1;
            }
            //将所有图片大小统一转化为8*16
            resize(src, resizeImg, Size(imageRows,imageCols), (0,0), (0,0), INTER_AREA);
            threshold(resizeImg, trainImg,0,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
            for(int j = 0; j<imageRows*imageCols; j++)
            {
                trainingData[i*imagesSum + k][j] = (float)resizeImg.data[j];
            }
            // 设置标签数据
            for(int j = 0;j < classSum; j++)
            {
                if(j == i)
                    labels[i*imagesSum + k][j] = 1;
                else 
                    labels[i*imagesSum + k][j] = 0;
            }
            k++;

        } while (!_findnext(handle, &fileinfo));
        Mat labelsMat(classSum*imagesSum, classSum, CV_32FC1,labels);

        _findclose(handle);
    }
    //训练样本数据及对应标签
    Mat trainingDataMat(classSum*imagesSum, imageRows*imageCols, CV_32FC1, trainingData);
    Mat labelsMat(classSum*imagesSum, classSum, CV_32FC1, labels);
    //std::cout<<"trainingDataMat: \n"<<trainingDataMat<<"\n"<<std::endl;
    //std::cout<<"labelsMat: \n"<<labelsMat<<"\n"<<std::endl;
    ////==========================训练部分==============================////

    Ptr<ANN_MLP>model = ANN_MLP::create();
    Mat layerSizes = (Mat_<int>(1,5)<<imageRows*imageCols,128,128,128,classSum);
    model->setLayerSizes(layerSizes);
    model->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
    model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
    model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000,0.0001));

    Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);  
    model->train(trainData); 
    //保存训练结果
    model->save("E:/image/image/MLPModel.xml"); 

    ////==========================预测部分==============================////
    //读取测试图像
    Mat test, dst;
    test = imread("E:/image/image/test.png", 0);;
    if (test.empty())
    {
        std::cout<<"can not load image \n"<<std::endl;
        return -1;
    }
    //将测试图像转化为1*128的向量
    resize(test, test, Size(imageRows,imageCols), (0,0), (0,0), INTER_AREA);
    threshold(test, test, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
    Mat_<float> testMat(1, imageRows*imageCols);
    for (int i = 0; i < imageRows*imageCols; i++)
    {
        testMat.at<float>(0,i) = (float)test.at<uchar>(i/8, i%8);
    }
    //使用训练好的MLP model预测测试图像
    model->predict(testMat, dst);
    std::cout<<"testMat: \n"<<testMat<<"\n"<<std::endl;
    std::cout<<"dst: \n"<<dst<<"\n"<<std::endl; 
    double maxVal = 0;
    Point maxLoc;
    minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc); 
    std::cout<<"测试结果："<<maxLoc.x<<"置信度:"<<maxVal*100<<"%"<<std::endl;
    imshow("test",test); 
    waitKey(0);
    return 0;
}

#include <io.h>
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace ml;
//利用训练完成的神经网络模型进行识别
int predict()
{
    //将所有图片大小统一转化为8*16
    const int imageRows = 8;
    const int imageCols = 16;
    //读取训练结果
    Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>("E:/image/image/MLPModel.xml");
    ////==========================预测部分==============================////
    //读取测试图像
    Mat test, dst;
    test = imread("E:/image/image/test.png", 0);;
    if (test.empty())
    {
        std::cout<<"can not load image \n"<<std::endl;
        return -1;
    }
    //将测试图像转化为1*128的向量
    resize(test, test, Size(imageRows,imageCols), (0,0), (0,0), INTER_AREA);
    threshold(test, test, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
    Mat_<float> testMat(1, imageRows*imageCols);
    for (int i = 0; i < imageRows*imageCols; i++)
    {
        testMat.at<float>(0,i) = (float)test.at<uchar>(i/8, i%8);
    }
    //使用训练好的MLP model预测测试图像
    model->predict(testMat, dst);
    std::cout<<"testMat: \n"<<testMat<<"\n"<<std::endl;
    std::cout<<"dst: \n"<<dst<<"\n"<<std::endl; 
    double maxVal = 0;
    Point maxLoc;
    minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc); 
    std::cout<<"测试结果："<<maxLoc.x<<"置信度:"<<maxVal*100<<"%"<<std::endl;
    imshow("test",test); 
    waitKey(0);
    return 0;
}
*********************************************************************************/


//叠加透明图片
//用法：
//Mat img1 = imread("CycloneGui 优化者：阿♂姚.jpg"),		img2 = imread("src.png", -1);
//Mat img1_t1(img1, cvRect(110, 90, img2.cols, img2.rows));
//copyPNGtoMat(img1_t1, img2, .5);
//imshow("final",img1);
//waitKey(0);

int copyPNGtoMat(cv::Mat &dst, cv::Mat &scr, double scale)  
{  
	//cvtColor(image,imageGRAY,CV_RGB2GRAY);            //RGB转GRAY
	//cvtColor(image,imageRGBA,CV_RGB2BGRA);            //RGB转RGBA	

	if (dst.channels() != 3 || scr.channels() != 4)  
	{  
		return true;  
	}  
	if (scale < 0.01)  
		return false;  
	std::vector<cv::Mat>scr_channels;  
	std::vector<cv::Mat>dstt_channels;  
	split(scr, scr_channels);  
	split(dst, dstt_channels);  
	CV_Assert(scr_channels.size() == 4 && dstt_channels.size() == 3);  
 
	if (scale < 1)  
	{  
		scr_channels[3] *= scale;  
		scale = 1;  
	}  
	for (int i = 0; i < 3; i++)  
	{  
		dstt_channels[i] = dstt_channels[i].mul(255.0 / scale - scr_channels[3], scale / 255.0);  
		dstt_channels[i] += scr_channels[i].mul(scr_channels[3], scale / 255.0);  
	}  
	merge(dstt_channels, dst);  
	return true;  
}  

//增加alpha通道
int addAlpha(cv::Mat& src, cv::Mat& dst, cv::Mat& alpha)
{
	if (src.channels() == 4)
	{
		return -1;
	}
	else if (src.channels() == 1)
	{
		cv::cvtColor(src, src, cv::COLOR_GRAY2RGB);
	}
	
	dst = cv::Mat(src.rows, src.cols, CV_8UC4);
 
	std::vector<cv::Mat> srcChannels;
	std::vector<cv::Mat> dstChannels;

	//分离通道
	cv::split(src, srcChannels);
 
	dstChannels.push_back(srcChannels[0]);
	dstChannels.push_back(srcChannels[1]);
	dstChannels.push_back(srcChannels[2]);


	//添加透明度通道
	dstChannels.push_back(alpha);
	//合并通道
	cv::merge(dstChannels, dst);
 
	return 0;
}

cv::Mat createAlpha(cv::Mat& src)
{
	cv::Mat alpha = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	cv::Mat gray = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
 
	cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);
 
	for (int i = 0; i < src.rows; i++)
	{
		if(i<100)		continue;
		 
		for (int j = 0; j < src.cols; j++)
		{
			alpha.at<uchar>(i, j) =255;// 255 - gray.at<uchar>(i, j);
		}
	}
 
	return alpha;
}

int test_add_alpha()
{
	cv::Mat src = cv::imread("ApowerMirror Main.jpg", 1);
	cv::Mat dst;

	addAlpha(src, dst, createAlpha(src));
 	cv::imwrite("ApowerMirror Main.png", dst);

	Mat redst=imread("ApowerMirror Main.png",-1);
	cv::imshow("ApowerMirror Main.png", redst);
	cv::waitKey(0);
	return 0;
}

//增加透明通道，去掉棋子周边多余的东西
void deledge(cv::Mat& src, float scale)
{
	cv::Mat alpha = cv::Mat(src.rows, src.cols, CV_8UC1);		//建立单通道图像
	for (int i = 0; i < src.rows; i++)							
	{
		for (int j = 0; j < src.cols; j++)
		{
			alpha.at<uchar>(i, j) =0;
		}
	}
	cv::Point center(src.cols/2, src.rows/2);					
	cv::circle(alpha, center, int(src.rows/2*scale), cv::Scalar(255,255,255), -1, 8);		//画个实心圆

	//cv::Mat png = cv::imread("src.png", -1);
	//assert(png.channels() == 4);
	//std::vector<cv::Mat> pngChannels;
	//cv::split(png, pngChannels);
	//alpha = pngChannels[3];

	std::vector<cv::Mat> srcChannels;
	cv::split(src, srcChannels);//分离通道
	srcChannels.push_back(alpha);
	cv::merge(srcChannels, src);//合并通道
	cv::imwrite("dst.png", src);
}