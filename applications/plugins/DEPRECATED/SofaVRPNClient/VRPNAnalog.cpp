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
 * VRPNAnalog.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */
#define SOFAVRPNCLIENT_VRPNANALOG_CPP_

#include <VRPNAnalog.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vrpnclient_config.h>

namespace sofavrpn
{
namespace client
{

void handle_analog(void *userdata, const vrpn_ANALOGCB a)
{
    VRPNAnalogData* analogData = (VRPNAnalogData*) userdata;

    analogData->data.num_channel = a.num_channel;
    analogData->modified = true;
    //analogData->channel = a.channel;

    //std::cout << ">>>>>> Analog data" << std::endl;
    //std::cout << a.num_channel << " number of channels available" << std::endl;
    for (int i=0 ; i<a.num_channel ; i++)
    {
        //std::cout << i << " " << a.channel[i] << std::endl;
        analogData->data.channel[i] = a.channel[i];
    }

    //std::cout << "------- Analog data" << std::endl;
}

}

}

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(VRPNAnalog)

int VRPNAnalogClass = sofa::core::RegisterObject("VRPN Analog")
#ifndef SOFA_FLOAT
        .add< VRPNAnalog<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< VRPNAnalog<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNAnalog<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNAnalog<Vec3fTypes>;
#endif //SOFA_DOUBLE
