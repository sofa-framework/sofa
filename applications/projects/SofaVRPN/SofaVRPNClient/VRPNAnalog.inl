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
    sofa::helper::WriteAccessor< Data<sofa::helper::vector<Real> > > channels = f_channels;
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

