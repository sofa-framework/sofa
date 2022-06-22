/*
\file  win32/adapt.h
\brief Declaration of Win32 adaptation of POSIX functions and types
*/
#ifndef _WIN32_ADAPT_H_
#define _WIN32_ADAPT_H_

#include <windows.h>

typedef DWORD pid_t;

pid_t getpid(void);

#endif  /* _WIN32_ADAPT_H_ */
