
#ifndef _CGoGN_OS_SPEC__
#define _CGoGN_OS_SPEC__

// SPECIFIC FOR WINDOWS

#ifdef WIN32
	#define NOMINMAX 
	#define _CRT_SECURE_NO_WARNINGS
	#include <windows.h>
	#include <stdio.h>
	#include <limits>
	#define _USE_MATH_DEFINES
	#include <cmath>
	#define isnan(X) _isnan(X)

	//#ifndef PI_DEFINED
	//#define PI_DEFINED
	//double M_PI=3.14159265359;
	//#endif
#endif

// SPECIFIC FOR MAC 

#endif