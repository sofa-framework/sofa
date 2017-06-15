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
#define SOFA_COMPONENT_MISC_EXTRAMONITOR_CPP
#include <SofaValidation/ExtraMonitor.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(ExtraMonitor)

using namespace sofa::defaulttype;

// Register in the Factory
int ExtraMonitorClass = core::RegisterObject("Monitoring of particles")
#ifndef SOFA_FLOAT
        .add< ExtraMonitor<Vec3dTypes> >(true)
        .add< ExtraMonitor<Vec6dTypes> >()
        .add< ExtraMonitor<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ExtraMonitor<Vec3fTypes> >()
        .add< ExtraMonitor<Vec6fTypes> >()
        .add< ExtraMonitor<Rigid3fTypes> >()
#endif
        ;



#ifndef SOFA_FLOAT
template class ExtraMonitor<Vec3dTypes>;
template class ExtraMonitor<Vec6dTypes>;
template class ExtraMonitor<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class ExtraMonitor<Vec3fTypes>;
template class ExtraMonitor<Vec6fTypes>;
template class ExtraMonitor<Rigid3fTypes>;
#endif


} // namespace misc

} // namespace component

} // namespace sofa
