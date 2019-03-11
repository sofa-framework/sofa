/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_MISC_MONITOR_CPP

#include <SofaValidation/Monitor.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace misc
{
using namespace sofa::defaulttype;

// Register in the Factory
int MonitorClass = core::RegisterObject("Monitoring of particles")
                .add< Monitor<Vec3Types> >(true)
        .add< Monitor<Vec6Types> >()
        .add< Monitor<Rigid3Types> >()
        
        ;

template class SOFA_VALIDATION_API Monitor<Vec3Types>;
template class SOFA_VALIDATION_API Monitor<Vec6Types>;
template class SOFA_VALIDATION_API Monitor<Rigid3Types>;


} // namespace misc
} // namespace component
} // namespace sofa
