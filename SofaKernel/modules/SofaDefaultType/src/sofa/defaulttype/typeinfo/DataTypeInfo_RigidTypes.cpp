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
#include <sofa/defaulttype/typeinfo/DataTypeInfo_RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>

namespace sofa::defaulttype
{


template<typename TT>
int mince2()
{
    DataTypeInfoRegistry::Set(typeid(TT), VirtualTypeInfoA< DataTypeInfo<TT>>::get());
    return 0;
}

int fixedPreLoad2()
{
    mince2<sofa::defaulttype::RigidCoord<2, double>>();
    mince2<sofa::defaulttype::RigidDeriv<2, double>>();
    mince2<sofa::defaulttype::RigidCoord<2, float>>();
    mince2<sofa::defaulttype::RigidDeriv<2, float>>();

    mince2<sofa::defaulttype::RigidCoord<3, double>>();
    mince2<sofa::defaulttype::RigidDeriv<3, double>>();
    mince2<sofa::defaulttype::RigidCoord<3, float>>();
    mince2<sofa::defaulttype::RigidDeriv<3, float>>();
    return 0;
}

static int allFixed = fixedPreLoad2();
} /// namespace sofa::defaulttype

