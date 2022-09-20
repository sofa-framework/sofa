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
#define SOFA_CORE_DATATYPE_DATATAG_DEFINITION
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/core/datatype/Data[Tag].h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Set.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vector.h>

// Custom name type definition 
namespace sofa::defaulttype
{

template<>
struct DataTypeInfo< sofa::core::objectmodel::Tag > : public TextTypeInfo<sofa::core::objectmodel::Tag>
{
    static const char* name() { return "Tag"; }
};

template<>
struct DataTypeInfo< sofa::core::objectmodel::TagSet > : public SetTypeInfo<sofa::core::objectmodel::TagSet>
{
    static const char* name() { return "TagSet"; }
};

} // namespace sofa::defaulttype


DATATYPEINFO_DEFINE(sofa::core::objectmodel::Tag)
DATATYPEINFO_DEFINE(sofa::core::objectmodel::TagSet)
