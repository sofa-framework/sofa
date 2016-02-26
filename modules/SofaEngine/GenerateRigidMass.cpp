/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_ENGINE_GENERATERIGIDMASS_CPP
#include <SofaEngine/GenerateRigidMass.inl>

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
template class SOFA_ENGINE_API GenerateRigidMass<Rigid3dTypes, Rigid3dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_ENGINE_API GenerateRigidMass<Rigid3fTypes, Rigid3fMass>;
#endif

SOFA_DECL_CLASS(GenerateRigidMass)

} // namespace loader

} // namespace component

} // namespace sofa
