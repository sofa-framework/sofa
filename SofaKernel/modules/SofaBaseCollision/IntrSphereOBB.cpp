/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_COLLISION_INTRSPHEREOBB_CPP
#include <SofaBaseCollision/IntrSphereOBB.inl>


namespace sofa{
using namespace defaulttype;
namespace component{
namespace collision{

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TIntrSphereOBB<Vec3dTypes,Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API TIntrSphereOBB<Rigid3dTypes,Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TIntrSphereOBB<Vec3fTypes,Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API TIntrSphereOBB<Rigid3fTypes,Rigid3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}
