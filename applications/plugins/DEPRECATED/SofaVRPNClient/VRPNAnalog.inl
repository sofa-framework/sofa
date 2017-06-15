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

#ifndef SOFAVRPNCLIENT_VRPNANALOG_INL_
#define SOFAVRPNCLIENT_VRPNANALOG_INL_

#include <VRPNAnalog.h>

#include <sofa/core/ObjectFactory.h>

namespace sofavrpn
{

namespace client
{

template<class Datatypes>
VRPNAnalog<Datatypes>::VRPNAnalog()
    : f_channels(initData(&f_channels, "channels", "Channels"))
{
    anr = NULL;

    rg.initSeed( (long int) this );
}

template<class Datatypes>
VRPNAnalog<Datatypes>::~VRPNAnalog()
{
    // TODO Auto-generated destructor stub
}


template<class Datatypes>
bool VRPNAnalog<Datatypes>::connectToServer()
{
    analogData.data.num_channel = 0;
    analogData.modified = false;
    anr = new vrpn_Analog_Remote(deviceURL.c_str());
    anr->register_change_handler((void*) &analogData, handle_analog);

    anr->mainloop();

    return true;
}

template<class Datatypes>
void VRPNAnalog<Datatypes>::update()
{
    sofa::helper::WriteAccessor< sofa::core::objectmodel::Data<sofa::helper::vector<Real> > > channels = f_channels;
    //std::cout << "read analog " << this->getName() << std::endl;
    if (anr)
    {
        //get infos
        analogData.modified = false;
        anr->mainloop();

        if(analogData.modified)
        {
            channels.clear();

            if (analogData.data.num_channel > 1 && analogData.data.num_channel < 256)
            {
                //std::cout << "Size :" << analogData.data.num_channel << std::endl;
                //put infos in Datas

                channels.resize(analogData.data.num_channel);

                for (int i=0  ; i < analogData.data.num_channel ; i++)
                    channels[i] = analogData.data.channel[i];

            }
            else
            {
                std::cout << "No Channels readable" << std::endl;
            }
        }

    }
//	channels.resize(64);
//	for (unsigned int i=0 ; i<64 ;i++)
//	{
//		//channels[i] = i;
//		channels[i] = (Real)rg.randomDouble(0, 639);
//	}

}

}

}

#endif //SOFAVRPNCLIENT_VRPNANALOG_INL_

