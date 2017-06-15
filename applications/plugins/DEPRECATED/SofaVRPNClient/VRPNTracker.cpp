/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * VRPNTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#define SOFAVRPNCLIENT_VRPNTRACKER_CPP_

#include "VRPNTracker.inl"

#include <sofa/core/ObjectFactory.h>
#include <vrpnclient_config.h>

namespace sofavrpn
{

namespace client
{

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(VRPNTracker)

int VRPNTrackerClass = sofa::core::RegisterObject("VRPN Tracker")
#ifndef SOFA_FLOAT
        .add< VRPNTracker<Vec3dTypes> >()
        .add< VRPNTracker<Rigid3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< VRPNTracker<Vec3fTypes> >()
        .add< VRPNTracker<Rigid3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Vec3dTypes>;
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Vec3fTypes>;
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Rigid3fTypes>;
#endif //SOFA_DOUBLE

}

}
