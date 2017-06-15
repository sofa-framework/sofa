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
#ifndef SOFA_TYPEDEF_Particles_float_H
#define SOFA_TYPEDEF_Particles_float_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/State.h>

typedef sofa::defaulttype::Vec1fTypes   Particles1f;
typedef Particles1f::VecDeriv           VecDeriv1f;
typedef Particles1f::VecCoord           VecCoord1f;
typedef Particles1f::Deriv              Deriv1f;
typedef Particles1f::Coord              Coord1f;
typedef sofa::defaulttype::Vec2fTypes   Particles2f;
typedef Particles2f::VecDeriv           VecDeriv2f;
typedef Particles2f::VecCoord           VecCoord2f;
typedef Particles2f::Deriv              Deriv2f;
typedef Particles2f::Coord              Coord2f;
typedef sofa::defaulttype::Vec3fTypes   Particles3f;
typedef Particles3f::VecDeriv           VecDeriv3f;
typedef Particles3f::VecCoord           VecCoord3f;
typedef Particles3f::Deriv              Deriv3f;
typedef Particles3f::Coord              Coord3f;
typedef sofa::defaulttype::Vec6fTypes   Particles6f;
typedef Particles6f::VecDeriv           VecDeriv6f;
typedef Particles6f::VecCoord           VecCoord6f;
typedef Particles6f::Deriv              Deriv6f;
typedef Particles6f::Coord              Coord6f;

typedef sofa::defaulttype::Rigid2fTypes Rigid2f;
typedef Rigid2f::VecDeriv               VecDerivRigid2f;
typedef Rigid2f::VecCoord               VecCoordRigid2f;
typedef Rigid2f::Deriv                  DerivRigid2f;
typedef Rigid2f::Coord                  CoordRigid2f;
typedef sofa::defaulttype::Rigid3fTypes Rigid3f;
typedef Rigid3f::VecDeriv               VecDerivRigid3f;
typedef Rigid3f::VecCoord               VecCoordRigid3f;
typedef Rigid3f::Quat                   Quat3f;
typedef Rigid3f::Deriv                  DerivRigid3f;
typedef Rigid3f::Coord                  CoordRigid3f;


typedef sofa::defaulttype::ExtVec1fTypes ExtVec1f;
typedef ExtVec1f::VecCoord               VecCoordExtVec1f;
typedef ExtVec1f::VecDeriv               VecDerivExtVec1f;
typedef ExtVec1f::Deriv                  DerivExtVec1f;
typedef ExtVec1f::Coord                  CoordExtVec1f;
typedef sofa::defaulttype::ExtVec2fTypes ExtVec2f;
typedef ExtVec2f::VecCoord               VecCoordExtVec2f;
typedef ExtVec2f::VecDeriv               VecDerivExtVec2f;
typedef ExtVec2f::Deriv                  DerivExtVec2f;
typedef ExtVec2f::Coord                  CoordExtVec2f;
typedef sofa::defaulttype::ExtVec3fTypes ExtVec3f;
typedef ExtVec3f::VecCoord               VecCoordExtVec3f;
typedef ExtVec3f::VecDeriv               VecDerivExtVec3f;
typedef ExtVec3f::Deriv                  DerivExtVec3f;
typedef ExtVec3f::Coord                  CoordExtVec3f;

#ifdef SOFA_FLOAT

typedef Particles1f          Particles1;
typedef VecDeriv1f	     VecDeriv1;
typedef VecCoord1f	     VecCoord1;
typedef Deriv1f	     	     Deriv1;
typedef Coord1f	     	     Coord1;
typedef Particles2f	     Particles2;
typedef VecDeriv2f	     VecDeriv2;
typedef VecCoord2f	     VecCoord2;
typedef Deriv2f	     	     Deriv2;
typedef Coord2f	     	     Coord2;
typedef Particles3f	     Particles3;
typedef VecDeriv3f	     VecDeriv3;
typedef VecCoord3f	     VecCoord3;
typedef Deriv3f	     	     Deriv3;
typedef Coord3f	     	     Coord3;
typedef Particles6f	     Particles6;
typedef VecDeriv6f	     VecDeriv6;
typedef VecCoord6f	     VecCoord6;
typedef Deriv6f	     	     Deriv6;
typedef Coord6f	     	     Coord6;

typedef Rigid2f	     	     Rigid2;
typedef VecDerivRigid2f      VecDerivRigid2;
typedef VecCoordRigid2f      VecCoordRigid2;
typedef DerivRigid2f	     DerivRigid2;
typedef CoordRigid2f	     CoordRigid2;
typedef Rigid3f	     	     Rigid3;
typedef VecDerivRigid3f      VecDerivRigid3;
typedef VecCoordRigid3f      VecCoordRigid3;
typedef Quat3f		     Quat3;
typedef DerivRigid3f	     DerivRigid3;
typedef CoordRigid3f	     CoordRigid3;

typedef ExtVec1f ExtVec1;
typedef ExtVec2f ExtVec2;
typedef ExtVec3f ExtVec3;

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
