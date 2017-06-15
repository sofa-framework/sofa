/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "MyMappingPendulumInPlane.inl"

#include <sofa/core/Mapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(MyMappingPendulumInPlane)

int MyMappingPendulumInPlaneClass = core::RegisterObject("Mapping from an angle to a point in 2D")
#ifndef SOFA_FLOAT
    .add< MyMappingPendulumInPlane<Vec1dTypes, Vec3dTypes> >()
    .add< MyMappingPendulumInPlane<Vec1dTypes, Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
    .add< MyMappingPendulumInPlane<Vec1fTypes, Vec3fTypes> >()
#endif
    ;

#ifndef SOFA_FLOAT
template class MyMappingPendulumInPlane<Vec1dTypes, Vec3dTypes>;
template class MyMappingPendulumInPlane<Vec1dTypes, Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class MyMappingPendulumInPlane<Vec1fTypes, Vec3fTypes>;
#endif


}	//mapping

}	//component

}	//sofa
