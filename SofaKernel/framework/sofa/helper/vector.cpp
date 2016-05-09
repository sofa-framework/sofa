/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/vector.h>
#include <sofa/helper/vector_device.h>
#include <sofa/helper/integer_id.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <cassert>
#include <iostream>

namespace sofa
{

namespace helper
{

#ifdef DEBUG_OUT_VECTOR
int cptid = 0;
#endif

void SOFA_HELPER_API vector_access_failure(const void* vec, unsigned size, unsigned i, const std::type_info& type)
{
    msg_error("vector") << "in vector<"<<gettypename(type)<<"> " << std::hex << (long)vec << std::dec << " size " << size << " : invalid index " << (int)i;
    BackTrace::dump();
    assert(i < size);
}

void SOFA_HELPER_API vector_access_failure(const void* vec, unsigned size, unsigned i, const std::type_info& type, const char* tindex)
{
    msg_error("vector") << "in vector<"<<gettypename(type)<<", integer_id<"<<tindex<<"> > " << std::hex << (long)vec << std::dec << " size " << size << " : invalid index " << (int)i;
    BackTrace::dump();
    assert(i < size);
}

} // namespace helper

} // namespace sofa
