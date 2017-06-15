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
#ifndef SOFA_TYPEDEF_Particles_double_H
#define SOFA_TYPEDEF_Particles_double_H


#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/State.h>



typedef sofa::defaulttype::Vec1dTypes   Particles1d;
typedef Particles1d::VecDeriv           VecDeriv1d;
typedef Particles1d::VecCoord           VecCoord1d;
typedef Particles1d::Deriv              Deriv1d;
typedef Particles1d::Coord              Coord1d;
typedef sofa::defaulttype::Vec2dTypes   Particles2d;
typedef Particles2d::VecDeriv           VecDeriv2d;
typedef Particles2d::VecCoord           VecCoord2d;
typedef Particles2d::Deriv              Deriv2d;
typedef Particles2d::Coord              Coord2d;
typedef sofa::defaulttype::Vec3dTypes   Particles3d;
typedef Particles3d::VecDeriv           VecDeriv3d;
typedef Particles3d::VecCoord           VecCoord3d;
typedef Particles3d::Deriv              Deriv3d;
typedef Particles3d::Coord              Coord3d;
typedef sofa::defaulttype::Vec6dTypes   Particles6d;
typedef Particles6d::VecDeriv           VecDeriv6d;
typedef Particles6d::VecCoord           VecCoord6d;
typedef Particles6d::Deriv              Deriv6d;
typedef Particles6d::Coord              Coord6d;

typedef sofa::defaulttype::Rigid2dTypes Rigid2d;
typedef Rigid2d::VecDeriv               VecDerivRigid2d;
typedef Rigid2d::VecCoord               VecCoordRigid2d;
typedef Rigid2d::Deriv                  DerivRigid2d;
typedef Rigid2d::Coord                  CoordRigid2d;
typedef sofa::defaulttype::Rigid3dTypes Rigid3d;
typedef Rigid3d::VecDeriv               VecDerivRigid3d;
typedef Rigid3d::VecCoord               VecCoordRigid3d;
typedef Rigid3d::Quat                   Quat3d;
typedef Rigid3d::Deriv                  DerivRigid3d;
typedef Rigid3d::Coord                  CoordRigid3d;

typedef sofa::defaulttype::ExtVec1dTypes ExtVec1d;
typedef ExtVec1d::VecCoord               VecCoordExtVec1d;
typedef ExtVec1d::VecDeriv               VecDerivExtVec1d;
typedef ExtVec1d::Deriv                  DerivExtVec1d;
typedef ExtVec1d::Coord                  CoordExtVec1d;
typedef sofa::defaulttype::ExtVec2dTypes ExtVec2d;
typedef ExtVec2d::VecCoord               VecCoordExtVec2d;
typedef ExtVec2d::VecDeriv               VecDerivExtVec2d;
typedef ExtVec2d::Deriv                  DerivExtVec2d;
typedef ExtVec2d::Coord                  CoordExtVec2d;
typedef sofa::defaulttype::ExtVec3dTypes ExtVec3d;
typedef ExtVec3d::VecCoord               VecCoordExtVec3d;
typedef ExtVec3d::VecDeriv               VecDerivExtVec3d;
typedef ExtVec3d::Deriv                  DerivExtVec3d;
typedef ExtVec3d::Coord                  CoordExtVec3d;


#ifndef SOFA_FLOAT

typedef Particles1d          Particles1;
typedef VecDeriv1d	     VecDeriv1;
typedef VecCoord1d	     VecCoord1;
typedef Deriv1d	     	     Deriv1;
typedef Coord1d	     	     Coord1;
typedef Particles2d	     Particles2;
typedef VecDeriv2d	     VecDeriv2;
typedef VecCoord2d	     VecCoord2;
typedef Deriv2d	     	     Deriv2;
typedef Coord2d	     	     Coord2;
typedef Particles3d	     Particles3;
typedef VecDeriv3d	     VecDeriv3;
typedef VecCoord3d	     VecCoord3;
typedef Deriv3d	     	     Deriv3;
typedef Coord3d	     	     Coord3;
typedef Particles6d	     Particles6;
typedef VecDeriv6d	     VecDeriv6;
typedef VecCoord6d	     VecCoord6;
typedef Deriv6d	     	     Deriv6;
typedef Coord6d	     	     Coord6;

typedef Rigid2d	     	     Rigid2;
typedef VecDerivRigid2d      VecDerivRigid2;
typedef VecCoordRigid2d      VecCoordRigid2;
typedef DerivRigid2d	     DerivRigid2;
typedef CoordRigid2d	     CoordRigid2;
typedef Rigid3d	     	     Rigid3;
typedef VecDerivRigid3d      VecDerivRigid3;
typedef VecCoordRigid3d      VecCoordRigid3;
typedef Quat3d		     Quat3;
typedef DerivRigid3d	     DerivRigid3;
typedef CoordRigid3d	     CoordRigid3;

typedef ExtVec1d ExtVec1;
typedef ExtVec2d ExtVec2;
typedef ExtVec3d ExtVec3;


typedef sofa::core::State<Particles1> State1;
typedef sofa::core::State<Particles2> State2;
typedef sofa::core::State<Particles3> State3;
typedef sofa::core::State<Particles6> State6;
typedef sofa::core::State<Rigid2>     RigidState2;
typedef sofa::core::State<Rigid3>     RigidState3;

typedef sofa::core::State<ExtVec1>     ExtVecState1;
typedef sofa::core::State<ExtVec2>     ExtVecState2;
typedef sofa::core::State<ExtVec3>     ExtVecState3;

#endif


#endif
