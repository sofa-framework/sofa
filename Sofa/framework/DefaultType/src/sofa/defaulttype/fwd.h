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
#include <sofa/type/fwd.h>
#include <sofa/linearalgebra/fwd.h>

namespace sofa::defaulttype
{
template<sofa::Size N, typename real>
class RigidDeriv;

template<sofa::Size N, typename real>
class RigidCoord;

template<sofa::Size N, typename real>
class RigidMass;

template<sofa::Size N, typename real>
class StdRigidTypes;

template<typename real>
class StdRigidTypes<3, real>;

typedef StdRigidTypes<2,double> Rigid2dTypes;
typedef RigidMass<2,double> Rigid2dMass;

typedef StdRigidTypes<2,float> Rigid2fTypes;
typedef RigidMass<2,float> Rigid2fMass;

typedef StdRigidTypes<2,SReal> Rigid2Types;
typedef RigidMass<2,SReal> Rigid2Mass;

typedef RigidMass<3,double> Rigid3dMass;
typedef RigidMass<3,float> Rigid3fMass;

typedef StdRigidTypes<3,SReal> Rigid3Types;  ///< un-defined precision type
typedef RigidMass<3,SReal>     Rigid3Mass;   ///< un-defined precision type

typedef StdRigidTypes<2,double> Rigid2dTypes;
typedef RigidMass<2,double> Rigid2dMass;

typedef StdRigidTypes<2,float> Rigid2fTypes;
typedef RigidMass<2,float> Rigid2fMass;

typedef StdRigidTypes<2,SReal> Rigid2Types;
typedef RigidMass<2,SReal> Rigid2Mass;

template<class TCoord, class TDeriv, class TReal = typename TCoord::value_type>
class StdVectorTypes;

typedef StdVectorTypes<sofa::type::Vec3d,sofa::type::Vec3d,double> Vec3dTypes;
typedef StdVectorTypes<sofa::type::Vec2d,sofa::type::Vec2d,double> Vec2dTypes;
typedef StdVectorTypes<sofa::type::Vec1d,sofa::type::Vec1d,double> Vec1dTypes;
typedef StdVectorTypes<sofa::type::Vec6d,sofa::type::Vec6d,double> Vec6dTypes;
typedef StdVectorTypes<sofa::type::Vec3f,sofa::type::Vec3f,float> Vec3fTypes;
typedef StdVectorTypes<sofa::type::Vec2f,sofa::type::Vec2f,float> Vec2fTypes;
typedef StdVectorTypes<sofa::type::Vec1f,sofa::type::Vec1f,float> Vec1fTypes;
typedef StdVectorTypes<sofa::type::Vec6f,sofa::type::Vec6f,float> Vec6fTypes;

}

