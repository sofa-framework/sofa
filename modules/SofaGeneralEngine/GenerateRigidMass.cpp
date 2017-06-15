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
#define SOFA_COMPONENT_ENGINE_GENERATERIGIDMASS_CPP
#include <SofaGeneralEngine/GenerateRigidMass.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

int GenerateRigidMassClass = core::RegisterObject("An engine computing the RigidMass of a mesh : mass, volume and inertia matrix.")
#ifndef SOFA_FLOAT
        .add< GenerateRigidMass<Rigid3dTypes, Rigid3dMass> >()
#endif
#ifndef SOFA_DOUBLE
        .add< GenerateRigidMass<Rigid3fTypes, Rigid3fMass> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API GenerateRigidMass<Rigid3dTypes, Rigid3dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API GenerateRigidMass<Rigid3fTypes, Rigid3fMass>;
#endif

SOFA_DECL_CLASS(GenerateRigidMass)

} // namespace loader

} // namespace component

} // namespace sofa
