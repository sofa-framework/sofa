/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_SYSTEM_CONFIG_H
#define SOFA_HELPER_SYSTEM_CONFIG_H

#include <sofa/helper/helper.h>

#include <cstddef>              // For NULL

#if defined(_WIN32) || defined(_XBOX)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  define snprintf _snprintf
#endif

#ifdef _WIN32
#  include <windows.h>
#endif

#ifdef _XBOX
#  include <xtl.h>
#endif

#ifdef __PS3__
#  include <cstring>
#  include <ctype.h>
#  include <math.h>
#  include <sys/timer.h>
#  include <typeinfo>

#  define usleep(x) sys_timer_usleep((usecond_t)x)

namespace std
{
	// Stub provided for CImg dependency
	inline int system(const char* command)
	{
		return 1;
	}

	// todo note that this is only used with host serialization from devkit
	inline char* getenv( const char* env_var )
	{
		if (strcmp("TEMP", env_var) == 0)
		{
			return "/SYS_APP_HOME/TEMP";
		}
        else
        {
            return "/SYS_APP_HOME/";
        }
	}
}

#endif // __PS3__


#ifdef BOOST_NO_EXCEPTIONS
#  include<exception>

namespace boost
{
	inline void throw_exception(std::exception const & e)
	{
		return;
	}
}
#endif // BOOST_NO_EXCEPTIONS


#ifdef _MSC_VER
#  ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES 1 // required to get M_PI from math.h
#  endif
// Visual C++ does not include stdint.h
typedef signed __int8		int8_t;
typedef signed __int16		int16_t;
typedef signed __int32		int32_t;
typedef signed __int64		int64_t;
typedef unsigned __int8		uint8_t;
typedef unsigned __int16	uint16_t;
typedef unsigned __int32	uint32_t;
typedef unsigned __int64	uint64_t;
#else
#  include <stdint.h>
#endif

#ifdef SOFA_FLOAT
typedef float SReal;
#else
typedef double SReal;
#endif

#define sofa_do_concat2(a,b) a ## b
#define sofa_do_concat(a,b) sofa_do_concat2(a,b)
#define sofa_concat(a,b) sofa_do_concat(a,b)

#define sofa_tostring(a) sofa_do_tostring(a)
#define sofa_do_tostring(a) #a

#define SOFA_DECL_CLASS(name) extern "C" { int sofa_concat(class_,name) = 0; }
#define SOFA_LINK_CLASS(name) extern "C" { extern int sofa_concat(class_,name); int sofa_concat(link_,name) = sofa_concat(class_,name); }

// Prevent compiler warnings about 'unused variables'.
// This should be used when a parameter name is needed (e.g. for
// documentation purposes) even if it is not used in the code.
#define SOFA_UNUSED(x) (void)(x)

// Use of "extern template" is activated by the following macro
// It must be used for DLLs on windows, and is now also activated
// on other platforms (unless SOFA_NO_EXTERN_TEMPLATE is set), as
// it can fix RTTI issues (typeid / dynamic_cast) on Mac and can
// significantly speed-up compilation and link
#if !defined SOFA_NO_EXTERN_TEMPLATE && !defined SOFA_STATIC_LIBRARY
#  define SOFA_EXTERN_TEMPLATE
#endif

#if defined SOFA_STATIC_LIBRARY || !defined _WIN32
#  define SOFA_EXPORT_DYNAMIC_LIBRARY
#  define SOFA_IMPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec(dllexport)
#  define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec(dllimport)
#  ifdef _MSC_VER
#    pragma warning(disable: 4231) // nonstandard extension used : 'identifier' before template explicit instantiation
#    pragma warning(disable: 4910) // '<identifier>' : '__declspec(dllexport)' and 'extern' are incompatible on an explicit instantiation
#  endif
#endif

// utility for debug tracing
#ifdef _MSC_VER
  #define SOFA_CLASS_METHOD (std::string(this->getClassName()) + "::" + __FUNCTION__ + " ")
#else
  #define SOFA_CLASS_METHOD (std::string(this->getClassName()) + "::" + __func__ + " ")
#endif

#endif // SOFA_HELPER_SYSTEM_CONFIG_H
