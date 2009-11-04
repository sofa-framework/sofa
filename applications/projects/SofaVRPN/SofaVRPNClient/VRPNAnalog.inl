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
    addOutput(&f_channels);;

    setDirtyValue();

}

template<class Datatypes>
VRPNAnalog<Datatypes>::~VRPNAnalog()
{
    // TODO Auto-generated destructor stub
}


template<class Datatypes>
bool VRPNAnalog<Datatypes>::connectToServer()
{
    anr = new vrpn_Analog_Remote(deviceURL.c_str());
    anr->register_change_handler(&analogData, handle_analog);

    return true;
}

template<class Datatypes>
void VRPNAnalog<Datatypes>::update()
{
    //sofa::helper::WriteAccessor< Data<unsigned int> > numberOfChannels = f_numberOfChannels;
    sofa::helper::WriteAccessor< Data<sofa::helper::vector<Real> > > channels = f_channels;
    std::cout << "prout" << std::endl;
    //get infos
    anr->mainloop();

    //put infos in Datas
    //numberOfChannels = analogData.num_channel;
    channels.clear();
    channels.resize(analogData.num_channel);
    for (unsigned int i=0  ; i < analogData.num_channel ; i++)
        channels[i] = analogData.channel[i];

    setDirtyOutputs();
    //setDirtyValue();
}

}

}

#endif //SOFAVRPNCLIENT_VRPNANALOG_INL_

