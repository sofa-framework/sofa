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

#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::core::objectmodel
{
#ifndef SOFA_CORE_DATATYPE_DEFINITION
extern template class Data<sofa::defaulttype::Rigid2fTypes::Coord>;
extern template class Data<sofa::defaulttype::Rigid2dTypes::Coord>;
extern template class Data<sofa::defaulttype::Rigid3fTypes::Coord>;
extern template class Data<sofa::defaulttype::Rigid3dTypes::Coord>;

extern template class Data<sofa::defaulttype::Rigid2fTypes::Deriv>;
extern template class Data<sofa::defaulttype::Rigid2dTypes::Deriv>;
extern template class Data<sofa::defaulttype::Rigid3fTypes::Deriv>;
extern template class Data<sofa::defaulttype::Rigid3dTypes::Deriv>;

extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2fTypes::Coord>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2dTypes::Coord>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3fTypes::Coord>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3dTypes::Coord>>;

extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2fTypes::Deriv>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2dTypes::Deriv>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3fTypes::Deriv>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3dTypes::Deriv>>;

extern template class Data<sofa::defaulttype::Rigid2fMass>;
extern template class Data<sofa::defaulttype::Rigid2dMass>;
extern template class Data<sofa::defaulttype::Rigid3fMass>;
extern template class Data<sofa::defaulttype::Rigid3dMass>;

extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2fMass>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid2dMass>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3fMass>>;
extern template class Data<sofa::helper::vector<sofa::defaulttype::Rigid3dMass>>;
#endif /// SOFA_CORE_DATATYPE_DEFINITION
}


