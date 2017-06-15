/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
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
#endif

// snprintf() has been provided since MSVC++ 14 (Visual Studio 2015).  For other
// versions, it is simply #defined to _snprintf().
#if (defined(_MSC_VER) && _MSC_VER < 1900) || defined(_XBOX)
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

// utility for debug tracing
#ifdef _MSC_VER
    #define SOFA_CLASS_METHOD ( std::string(this->getClassName()) + "::" + __FUNCTION__ + " " )
#else
    #define SOFA_CLASS_METHOD ( std::string(this->getClassName()) + "::" + __func__ + " " )
#endif

#endif // SOFA_HELPER_SYSTEM_CONFIG_H
