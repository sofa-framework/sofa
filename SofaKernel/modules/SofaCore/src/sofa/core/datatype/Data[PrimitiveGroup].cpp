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
#include <sofa/core/datatype/Data[PrimitiveGroup].h>
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[vector].h>

namespace sofa::defaulttype
{

template<>
struct DataTypeInfo<sofa::core::loader::PrimitiveGroup> : public IncompleteTypeInfo<sofa::core::loader::PrimitiveGroup>
{};

}



namespace sofa::core::objectmodel
{
template class Data<sofa::core::loader::PrimitiveGroup>;
template class Data<sofa::helper::vector<sofa::core::loader::PrimitiveGroup>>;
}

