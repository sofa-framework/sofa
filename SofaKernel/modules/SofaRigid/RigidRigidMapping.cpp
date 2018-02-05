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
#define SOFA_COMPONENT_MAPPING_RIGIDRIGIDMAPPING_CPP

#include <SofaRigid/RigidRigidMapping.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidRigidMapping)

using namespace defaulttype;

// Register in the Factory
int RigidRigidMappingClass = core::RegisterObject("Set the positions and velocities of points attached to a rigid parent")
#ifndef SOFA_FLOAT
        .add< RigidRigidMapping< Rigid3dTypes, Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< RigidRigidMapping< Rigid3fTypes, Rigid3fTypes > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< RigidRigidMapping< Rigid3dTypes, Rigid3fTypes > >()
        .add< RigidRigidMapping< Rigid3fTypes, Rigid3dTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_RIGID_API RigidRigidMapping< Rigid3dTypes, Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_RIGID_API RigidRigidMapping< Rigid3fTypes, Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_RIGID_API RigidRigidMapping< Rigid3dTypes, Rigid3fTypes >;
template class SOFA_RIGID_API RigidRigidMapping< Rigid3fTypes, Rigid3dTypes >;
#endif
#endif


} // namespace mapping

} // namespace component

} // namespace sofa

