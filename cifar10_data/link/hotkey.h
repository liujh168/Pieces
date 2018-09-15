#pragma once

#include <windows.h>
#include <iostream>

#define ALT_F1     1
#define ALT_F6     2
#define ALT_F7     3
#define ALT_F8     4
#define ALT_F9     5
#define ALT_F10     6
#define ALT_F11     7
#define ALT_F12     8
#define CTRL_ALT_F1     9
#define CTRL_ALT_F2     10
#define CTRL_ALT_F3     11
#define CTRL_ALT_F4     12

#define HOTKEYS  12						//热键数量

extern std::string kmess[HOTKEYS];		//热键提示信息

extern bool bAutoGo;					//自动走子开关	
extern bool bBH;						//控制是否启动兵河分析开关	
extern bool	bConnected;					//连线开关
extern bool bGoInfinite;				//无限分析开关
extern int timeWaitBM;					//等待最佳着法超时时间

void	hotkey(void);					//注册热键
void	unhotkey(void);					//清理热键，释放资源
void	onhotkey(MSG msg);				//热键逻辑