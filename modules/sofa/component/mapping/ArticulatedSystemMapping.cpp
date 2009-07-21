/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/mapping/ArticulatedSystemMapping.inl>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;

SOFA_DECL_CLASS(ArticulatedSystemMapping)

// Register in the Factory
int ArticulatedSystemMappingClass = core::RegisterObject("Mapping between a set of 6D DOF's and a set of angles (µ) using an articulated hierarchy container. ")

#ifndef SOFA_FLOAT
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3dTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3fTypes> > > >()
#endif
        /*
        #ifndef SOFA_FLOAT
        #ifndef SOFA_DOUBLE
            .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3dTypes> > > >()
            .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3fTypes> > > >()
        #endif
        #endif
        */
        ;

#ifndef SOFA_FLOAT
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3dTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3fTypes> > >;
#endif
/*
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3dTypes> > >;
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3fTypes> > >;
#endif
#endif
*/
} // namespace mapping

} // namespace component

} // namespace sofa
