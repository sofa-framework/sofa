/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
 * IRTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */
#define SOFAVRPNCLIENT_IRTRACKER_CPP_

#include <IRTracker.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vrpnclient_config.h>

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(IRTracker)

int IRTrackerClass = sofa::core::RegisterObject("Infrared Tracker")
#ifndef SOFA_FLOAT
        .add< IRTracker<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IRTracker<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API IRTracker<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API IRTracker<Vec3fTypes>;
#endif //SOFA_DOUBLE
