/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/component/base/container/ArticulatedHierarchyContainer.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace container
{

SOFA_DECL_CLASS(ArticulatedHierarchyContainer)

// Register in the Factory
int ArticulatedHierarchyContainerClass = core::RegisterObject("")
        .add< ArticulatedHierarchyContainer >()
        ;

SOFA_DECL_CLASS(ArticulationCenter)

// Register in the Factory
int ArticulationCenterClass = core::RegisterObject("")
        .add< ArticulatedHierarchyContainer::ArticulationCenter >()
        ;

SOFA_DECL_CLASS(Articulation)

// Register in the Factory
int ArticulationClass = core::RegisterObject("")
        .add< ArticulatedHierarchyContainer::ArticulationCenter::Articulation >()
        ;

} // namespace container

} // namespace component

} // namespace sofa
