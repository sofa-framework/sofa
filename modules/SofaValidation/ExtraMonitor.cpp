/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_MISC_EXTRAMONITOR_CPP
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaValidation/ExtraMonitor.inl>

namespace sofa
{
namespace component
{
namespace misc
{
using namespace sofa::defaulttype;

// Register in the Factory
int ExtraMonitorClass = core::RegisterObject("Monitoring of particles")
                .add<ExtraMonitor<Vec3Types> >(true)
        .add<ExtraMonitor<Vec6Types> >()
        .add<ExtraMonitor<Rigid3Types> >()
        
        ;

template class ExtraMonitor<Vec3Types>;
template class ExtraMonitor<Vec6Types>;
template class ExtraMonitor<Rigid3Types>;


}  // namespace misc

}  // namespace component

}  // namespace sofa
