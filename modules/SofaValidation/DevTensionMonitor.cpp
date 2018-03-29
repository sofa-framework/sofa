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
#define SOFA_COMPONENT_MISC_DEVTENSIONMONITOR_CPP
#include <SofaValidation/DevTensionMonitor.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace misc
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(DevTensionMonitor)

// Register in the Factory
int DevTensionMonitorClass = core::RegisterObject("Tension monitoring")
#ifndef SOFA_FLOAT
        .add< DevTensionMonitor<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DevTensionMonitor<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_VALIDATION_API DevTensionMonitor<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_VALIDATION_API DevTensionMonitor<Rigid3fTypes>;
#endif

} // namespace misc

} // namespace component

} // namespace sofa
