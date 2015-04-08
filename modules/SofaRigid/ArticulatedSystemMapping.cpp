/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_MAPPING_ARTICULATEDSYSTEMMAPPING_CPP

#include <SofaRigid/ArticulatedSystemMapping.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ArticulatedSystemMapping)

// Register in the Factory
int ArticulatedSystemMappingClass = core::RegisterObject("Mapping between a set of 6D DOF's and a set of angles (µ) using an articulated hierarchy container. ")

#ifndef SOFA_FLOAT
        .add< ArticulatedSystemMapping< Vec1dTypes, Rigid3dTypes, Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< ArticulatedSystemMapping< Vec1fTypes, Rigid3fTypes, Rigid3fTypes > >()
#endif
        /*
        #ifndef SOFA_FLOAT
        #ifndef SOFA_DOUBLE
            .add< ArticulatedSystemMapping< Vec1fTypes, Rigid3dTypes > >()
            .add< ArticulatedSystemMapping< Vec1dTypes, Rigid3fTypes > >()
        #endif
        #endif
        */
        ;

#ifndef SOFA_FLOAT
template class SOFA_RIGID_API ArticulatedSystemMapping< Vec1dTypes, Rigid3dTypes, Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_RIGID_API ArticulatedSystemMapping< Vec1fTypes, Rigid3fTypes, Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_RIGID_API ArticulatedSystemMapping< Vec1fTypes, Rigid3fTypes, Rigid3dTypes >;
template class SOFA_RIGID_API ArticulatedSystemMapping< Vec1fTypes, Rigid3dTypes, Rigid3dTypes >;
template class SOFA_RIGID_API ArticulatedSystemMapping< Vec1dTypes, Rigid3fTypes, Rigid3dTypes >;
template class SOFA_RIGID_API ArticulatedSystemMapping< Vec1fTypes, Rigid3dTypes, Rigid3fTypes >;
template class SOFA_RIGID_API ArticulatedSystemMapping< Vec1dTypes, Rigid3fTypes, Rigid3fTypes >;
template class SOFA_RIGID_API ArticulatedSystemMapping< Vec1dTypes, Rigid3dTypes, Rigid3fTypes >;
#endif
#endif


} // namespace mapping

} // namespace component

} // namespace sofa
