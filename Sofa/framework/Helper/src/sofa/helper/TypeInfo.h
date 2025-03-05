/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef SOFA_HELPER_TYPEINFO_H
#define SOFA_HELPER_TYPEINFO_H

#include <typeinfo>
#include <sofa/helper/config.h>


namespace sofa::helper
{

class TypeInfo
{
public:
    const std::type_info* pt;
    TypeInfo(const std::type_info& t) : pt(&t) { }
    operator const std::type_info&() const { return *pt; }
    bool operator==(const TypeInfo& t) const { return *pt == *t.pt; }
    bool operator!=(const TypeInfo& t) const { return *pt != *t.pt; }
    bool operator<(const TypeInfo& t) const { return pt->before(*t.pt); }
};

}

#endif /// SOFA_HELPER_TYPEINFO_H
