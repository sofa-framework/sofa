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
#pragma once
#include <sofa/helper/SelectableItem.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>

namespace sofa::defaulttype
{

template<class TDataType>
struct SelectableItemDataTypeInfo : defaulttype::DefaultDataTypeInfo<TDataType>
{
    enum { ValidInfo       = 1 /**< 1 if this type has valid infos*/ };
    static const std::string name()
    {
        return "SelectableItem";
    }
    static const std::string GetTypeName()
    {
        return "SelectableItem";
    }
};


template<class T>
struct DataTypeInfo<T, std::enable_if_t<std::is_base_of_v<helper::BaseSelectableItem, T>>> :
    SelectableItemDataTypeInfo<T>
{
};

}
