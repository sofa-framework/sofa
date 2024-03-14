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
#define SOFA_SOFAMISCCOLLISION_OBBCONTACTMAPPER_CPP
#include <CollisionOBBCapsule/response/mapper/OBBContactMapper.h>

#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <CollisionOBBCapsule/geometry/OBBModel.h>

using namespace sofa;
using namespace sofa::core::collision;
using namespace sofa::component::collision;
using namespace collisionobbcapsule::geometry;

namespace sofa::component::collision::response::mapper
{

ContactMapperCreator< ContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, sofa::defaulttype::Vec3Types> > OBBContactMapperClass("PenalityContactForceField", true);
template class ContactMapper<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, sofa::defaulttype::Vec3Types>;

} // namespace sofa::component::collision::response::mapper
