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
#define SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_CPP
#include "PersistentContactBarycentricMapping.inl"

#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(PersistentContactBarycentricMapping)

// Register in the Factory
int PersistentContactBarycentricMappingClass = core::RegisterObject("Mapping using barycentric coordinates of the child with respect to cells of its parent")
#ifndef SOFA_FLOAT
        .add< PersistentContactBarycentricMapping< Vec3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< PersistentContactBarycentricMapping< Vec3fTypes, Vec3fTypes > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< PersistentContactBarycentricMapping< Vec3fTypes, Vec3dTypes > >()
        .add< PersistentContactBarycentricMapping< Vec3dTypes, Vec3fTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3fTypes, Vec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3dTypes, Vec3fTypes >;
template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3fTypes, Vec3dTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
