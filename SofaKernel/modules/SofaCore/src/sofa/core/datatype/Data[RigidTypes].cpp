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
#define SOFA_CORE_DATATYPE_DEFINITION
#include <sofa/core/objectmodel/Data.inl>
#include <sofa/core/datatype/Data[RigidTypes].h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo[RigidTypes].h>

namespace sofa::core::objectmodel
{
template class Data<sofa::defaulttype::Rigid2fTypes::Coord>;
template class Data<sofa::defaulttype::Rigid2dTypes::Coord>;
template class Data<sofa::defaulttype::Rigid3fTypes::Coord>;
template class Data<sofa::defaulttype::Rigid3dTypes::Coord>;

template class Data<sofa::defaulttype::Rigid2fTypes::Deriv>;
template class Data<sofa::defaulttype::Rigid2dTypes::Deriv>;
template class Data<sofa::defaulttype::Rigid3fTypes::Deriv>;
template class Data<sofa::defaulttype::Rigid3dTypes::Deriv>;

template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2fTypes::Coord>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2dTypes::Coord>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3fTypes::Coord>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3dTypes::Coord>>;

template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2fTypes::Deriv>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2dTypes::Deriv>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3fTypes::Deriv>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3dTypes::Deriv>>;

template class Data<sofa::defaulttype::Rigid2fMass>;
template class Data<sofa::defaulttype::Rigid2dMass>;
template class Data<sofa::defaulttype::Rigid3fMass>;
template class Data<sofa::defaulttype::Rigid3dMass>;

template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2fMass>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2dMass>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3fMass>>;
template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3dMass>>;
}

