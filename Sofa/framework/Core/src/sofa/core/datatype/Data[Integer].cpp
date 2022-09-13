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
#include <sofa/core/datatype/Data[Integer].h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Integer.h>
#include <sofa/core/datatype/DataRegistrationMacro.h>

DATATYPEINFO_REGISTER(char);
DATATYPEINFO_REGISTER(unsigned char);

DATATYPEINFO_REGISTER(short);
DATATYPEINFO_REGISTER(unsigned short);

DATATYPEINFO_REGISTER(int);
DATATYPEINFO_REGISTER(unsigned int);

DATATYPEINFO_REGISTER(long);
DATATYPEINFO_REGISTER(unsigned long);

DATATYPEINFO_REGISTER(long long);
DATATYPEINFO_REGISTER(unsigned long long);

