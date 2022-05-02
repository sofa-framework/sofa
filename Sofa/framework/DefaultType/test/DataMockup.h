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

#include <sofa/defaulttype/config.h>

#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
using sofa::defaulttype::IncompleteTypeInfo;

#include <sofa/defaulttype/typeinfo/DataTypeInfoDynamicWrapper.h>
using sofa::defaulttype::DataTypeInfoDynamicWrapper;

#include <sofa/defaulttype/TypeInfoID.h>
using sofa::defaulttype::TypeInfoId;

#include <sofa/defaulttype/TypeInfoRegistry.h>
using sofa::defaulttype::TypeInfoRegistry;
using sofa::defaulttype::TypeInfoType;

#include <iostream>

using sofa::defaulttype::AbstractTypeInfo;

template<class T>
class DataMockup
{
public:
    const AbstractTypeInfo* getTypeInfo()
    {
        static const AbstractTypeInfo* info {};
        if(info!=nullptr && info->ValidInfo())
            return info;

        const AbstractTypeInfo* tmp = TypeInfoRegistry::Get(TypeInfoId::GetTypeId<T>());
        if(tmp == nullptr)
            return nullptr;

        if(tmp->ValidInfo())
            info = tmp;
        return tmp;
    }
};

