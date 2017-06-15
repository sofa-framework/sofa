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
#define SOFA_COMPONENT_COLLISION_OBBMODEL_CPP
#include <SofaBaseCollision/OBBModel.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(OBB)

int OBBModelClass = core::RegisterObject("Collision model which represents a set of OBBs")
#ifndef SOFA_FLOAT
        .add<  TOBBModel<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add < TOBBModel<Rigid3fTypes> >()
#endif
        .addAlias("OBB")
        .addAlias("OBBModel")
//.addAlias("OBBMesh")
//.addAlias("OBBSet")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TOBBModel<defaulttype::Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API TOBB<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TOBBModel<defaulttype::Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API TOBB<defaulttype::Rigid3fTypes>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa
