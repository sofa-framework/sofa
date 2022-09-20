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
#define SOFA_CORE_DATATYPE_DATAINTEGER_DEFINITION
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/core/datatype/Data[Integer].h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Set.h>

DATATYPEINFO_DEFINE(char);
DATATYPEINFO_DEFINE(unsigned char);
DATATYPEINFO_DEFINE(short);
DATATYPEINFO_DEFINE(unsigned short);
DATATYPEINFO_DEFINE(int);
DATATYPEINFO_DEFINE(unsigned int);
DATATYPEINFO_DEFINE(long);
DATATYPEINFO_DEFINE(unsigned long);
DATATYPEINFO_DEFINE(long long);
DATATYPEINFO_DEFINE(unsigned long long);
