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

#include <sofa/SofaFramework.h>

// to define NULL
#include <cstring>

#ifdef WIN32
#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define snprintf _snprintf
#endif
#include <windows.h>
#endif

#ifdef _XBOX
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define snprintf _snprintf
#include <xtl.h>
#endif

#ifdef __PS3__
#include<typeinfo>
#include<ctype.h>
#include<sys/timer.h>
#include<math.h>

#define usleep(x) sys_timer_usleep((usecond_t)x)

namespace std
{
	// Stub provided for CImg dependecy
	inline int system(const char* command)
	{
		return 1;
	};

	// todo note that this is only used with host serialization from devkit
	inline char* getenv( const char* env_var )
	{
		if(strcmp("TEMP", env_var)==0)
		{
			return "/SYS_APP_HOME/TEMP";
		}
		else if(strcmp("TEMP", env_var)==0)
		{
			return "/SYS_APP_HOME/TEMP";
		}

		return "/SYS_APP_HOME/";
	};
};
/*
#define BOOST_NO_CXX11_EXTERN_TEMPLATE
#define BOOST_NO_CXX11_VARIADIC_MACROS
#  define BOOST_NO_CXX11_DECLTYPE
#  define BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
#  define BOOST_NO_CXX11_RVALUE_REFERENCES
#  define BOOST_NO_CXX11_STATIC_ASSERT
#  define BOOST_NO_CXX11_VARIADIC_TEMPLATES
#  define BOOST_NO_CXX11_AUTO_DECLARATIONS
#  define BOOST_NO_CXX11_AUTO_MULTIDECLARATIONS
#  define BOOST_NO_CXX11_CHAR16_T
#  define BOOST_NO_CXX11_CHAR32_T
#  define BOOST_NO_CXX11_HDR_INITIALIZER_LIST
#  define BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#  define BOOST_NO_CXX11_DELETED_FUNCTIONS
#  define BOOST_MPL_CFG_MSVC_70_ETI_BUG
#  define BOOST_NO_SFINAE_EXPR
#  define BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
#  define BOOST_NO_CXX11_LAMBDAS
#  define BOOST_NO_CXX11_LOCAL_CLASS_TEMPLATE_PARAMETERS
#  define BOOST_NO_CXX11_RAW_LITERALS
#  define BOOST_NO_CXX11_UNICODE_LITERALS
#  define BOOST_NO_CXX11_SCOPED_ENUMS

#define BOOST_NO_CXX11_CONSTEXPR
#define BOOST_NO_CXX11_NOEXCEPT
#define BOOST_NO_CXX11_NULLPTR
#define BOOST_NO_CXX11_RANGE_BASED_FOR
#define BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX

#  define BOOST_NO_CXX11_TEMPLATE_ALIASES
#  define BOOST_NO_CXX11_USER_DEFINED_LITERALS

#  define BOOST_NO_CXX11_DECLTYPE_N3276
#define BOOST_MPL_CFG_NO_ADL_BARRIER_NAMESPACE


#define BOOST_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS
#define BOOST_NO_COMPLETE_VALUE_INITIALIZATION
#define BOOST_NO_IS_ABSTRACT
#define BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION*/

#endif

#ifdef BOOST_NO_EXCEPTIONS
#include<exception>

namespace boost
{
	inline void throw_exception(std::exception const & e)
	{
		return;
	}
}
#endif

#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
# define _USE_MATH_DEFINES 1 // required to get M_PI from math.h
#endif
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
#include <stdint.h>
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

#if !defined(MAKEFOURCC)
#	define MAKEFOURCC(ch0, ch1, ch2, ch3) \
		(uint(uint8_t(ch0)) | (uint(uint8_t(ch1)) << 8) | \
		(uint(uint8_t(ch2)) << 16) | (uint(uint8_t(ch3)) << 24 ))
#endif

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


#endif
