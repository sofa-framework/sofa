/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
//===--------------------------- cxxabi.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __CXXABI_H
#define __CXXABI_H 

/*
 * This header provides the interface to the C++ ABI as defined at:
 *       http://www.codesourcery.com/cxx-abi/
 */

#include <stddef.h>
#include <stdint.h>

#define _LIBCPPABI_VERSION 1001
#define LIBCXXABI_NORETURN  __attribute__((noreturn))

#ifdef __cplusplus

namespace std {
    class type_info; // forward declaration
}


// runtime routines use C calling conventions, but are in __cxxabiv1 namespace
namespace __cxxabiv1 {  
  extern "C"  {

// 3.4 Demangler API
extern char* __cxa_demangle(const char* mangled_name, 
                            char*       output_buffer,
                            size_t*     length, 
                            int*        status);

  } // extern "C"
} // namespace __cxxabiv1

#endif // __cplusplus

namespace abi = __cxxabiv1;

#endif // __CXXABI_H 
