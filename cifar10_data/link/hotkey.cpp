#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <io.h>
#include <conio.h>
#include <windows.h>
#include <Shlwapi.h>

#include "hotkey.h"

#pragma comment( lib,"shlwapi.lib")

std::string kmess[HOTKEYS] = {					//热键提示信息
	"ALT_F1 红先连接控制开关!",					//保留
	"ALT_F6 断开连接!" ,						//保留
	"ALT_F7 连接测试，打印棋盘！",
	"ALT_F8 初始化并保存方案（按下热键前请将鼠标指向左上角黑车中心位置）！",//保留
	"ALT_F9 黑先连接控制开关!",							//保留
	"ALT_F10 载入方案!",
	"ALT_F11 客户端自动走子标志！",
	"ALT_F12 开启无限分析模式！",						//保留
	"CTRL_ALT_F1 循环修改参数！",						//保留
	"CTRL_ALT_F2 识别棋盘文件图！",
	"CTRL_ALT_F3 兵河分析控制开关！",					//保留
	"CTRL_ALT_F4 引擎停止思考，立即出步！",				//保留
};



//注册HotKey。
void hotkey(void)
{
	if (RegisterHotKey(NULL, ALT_F1, MOD_ALT | MOD_NOREPEAT, VK_F1))	std::cout << ((kmess[ALT_F1-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, ALT_F6, MOD_ALT | MOD_NOREPEAT, VK_F6))	std::cout << ((kmess[ALT_F6-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, ALT_F7, MOD_ALT | MOD_NOREPEAT, VK_F7))	std::cout << ((kmess[ALT_F7-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, ALT_F8, MOD_ALT | MOD_NOREPEAT, VK_F8))	std::cout << ((kmess[ALT_F8-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, ALT_F9, MOD_ALT | MOD_NOREPEAT, VK_F9))	std::cout << ((kmess[ALT_F9-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, ALT_F10, MOD_ALT | MOD_NOREPEAT, VK_F10))	std::cout << ((kmess[ALT_F10-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, ALT_F11, MOD_ALT | MOD_NOREPEAT, VK_F11))	std::cout << ((kmess[ALT_F11-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, ALT_F12, MOD_ALT | MOD_NOREPEAT, VK_F12))	std::cout << ((kmess[ALT_F12-1]).c_str()) << std::endl;

	if (RegisterHotKey(NULL, CTRL_ALT_F1, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_F1))	std::cout << ((kmess[CTRL_ALT_F1-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, CTRL_ALT_F2, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_F2))	std::cout << ((kmess[CTRL_ALT_F2-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, CTRL_ALT_F3, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_F3))	std::cout << ((kmess[CTRL_ALT_F3-1]).c_str()) << std::endl;
	if (RegisterHotKey(NULL, CTRL_ALT_F4, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_F4))	std::cout << ((kmess[CTRL_ALT_F4-1]).c_str()) << std::endl;
}

//注销HotKey, 释放资源。
void unhotkey(void)
{
	::UnregisterHotKey(NULL, ALT_F1);
	::UnregisterHotKey(NULL, ALT_F6);
	::UnregisterHotKey(NULL, ALT_F7);
	::UnregisterHotKey(NULL, ALT_F8);	
	::UnregisterHotKey(NULL, ALT_F9);	
	::UnregisterHotKey(NULL, ALT_F10);
	::UnregisterHotKey(NULL, ALT_F11);
	::UnregisterHotKey(NULL, ALT_F12);

	::UnregisterHotKey(NULL, CTRL_ALT_F1);
	::UnregisterHotKey(NULL, CTRL_ALT_F2);
	::UnregisterHotKey(NULL, CTRL_ALT_F3);
	::UnregisterHotKey(NULL, CTRL_ALT_F4);
}

void onhotkey(MSG msg)
{
	switch(msg.wParam)
	{
	case ALT_F1:	
	case ALT_F9:	
	case ALT_F6:	
		break;
	default:
		break;	
	}
}
