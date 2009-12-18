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
