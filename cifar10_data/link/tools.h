#pragma once

//与opencv有关的一些实用函数工具
#include <windows.h>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

extern int g_biBitCount;

void hwnd2mat(cv::Mat& dst, HWND hwnd);		//用GetDIBits取得图像数据。拷贝窗口，win7下开玻璃效果时窗口阻挡了也能截取
void hwnd5mat(cv::Mat& dst, HWND hwnd);		//用GetBitmapBits取得图像数据

//以下函数在link中有用到
double match(cv::Mat image, cv::Mat templ, cv::Point &matchLoc, int method);
cv::Mat mergeRows(cv::Mat A, cv::Mat B);
cv::Mat mergeCols(cv::Mat A, cv::Mat B);

//系统tools(全局函数)
void StrToClip(char* pstr); //拷贝FEN串到剪贴板
void ClipToStr(char* pstr); //从剪贴板拷贝到pstr
void start_bh(string fen, bool turn=true);		//找到兵河窗口并模拟新建，粘贴局面，开始分析(turn==true时)
void deledge(cv::Mat& src, float scale);//增加透明通道（图片半径r以外透明）

int hist();
int hist1();
int hist_hsv();
double hist_comp(Mat srcImage_base, Mat srcImage_test1);

void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false );

int dnn();
bool  MultiChannelBlending();
bool  ROI_LinearBlending();

int png();
int copyPNGtoMat(cv::Mat &dst, cv::Mat &scr, double scale);
int test_add_alpha();

int checkboard();
void checkdxdy(void);
