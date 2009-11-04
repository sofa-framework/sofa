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
// 	  Wiimote
    //    channel[0] = battery level (0-1)
    //    channel[1] = gravity X vector calculation (1 = Earth gravity)
    //    channel[2] = gravity Y vector calculation (1 = Earth gravity)
    //    channel[3] = gravity Z vector calculation (1 = Earth gravity)
    //    channel[4] = X of first sensor spot (0-1023, -1 if not seen)
    //    channel[5] = Y of first sensor spot (0-767, -1 if not seen)
    //    channel[6] = size of first sensor spot (0-15, -1 if not seen)
    //    channel[7] = X of second sensor spot (0-1023, -1 if not seen)
    //    channel[9] = Y of second sensor spot (0-767, -1 if not seen)
    //    channel[9] = size of second sensor spot (0-15, -1 if not seen)
    //    channel[10] = X of third sensor spot (0-1023, -1 if not seen)
    //    channel[11] = Y of third sensor spot (0-767, -1 if not seen)
    //    channel[12] = size of third sensor spot (0-15, -1 if not seen)
    //    channel[13] = X of fourth sensor spot (0-1023, -1 if not seen)
    //    channel[14] = Y of fourth sensor spot (0-767, -1 if not seen)
    //    channel[15] = size of fourth sensor spot (0-15, -1 if not seen)
    // and on the secondary controllers (skipping values to leave room for expansion)
    //    channel[16] = nunchuck gravity X vector
    //    channel[17] = nunchuck gravity Y vector
    //    channel[18] = nunchuck gravity Z vector
    //    channel[19] = nunchuck joystick angle
    //    channel[20] = nunchuck joystick magnitude
    //
    //    channel[32] = classic L button
    //    channel[33] = classic R button
    //    channel[34] = classic L joystick angle
    //    channel[35] = classic L joystick magnitude
    //    channel[36] = classic R joystick angle
    //    channel[37] = classic R joystick magnitude
    //
    //    channel[48] = guitar hero whammy bar
    //    channel[49] = guitar hero joystick angle
    //    channel[50] = guitar hero joystick magnitude

    vrpn_ANALOGCB* analogData = (vrpn_ANALOGCB*) userdata;

    analogData->num_channel = a.num_channel;
    //analogData->channel = a.channel;

//	std::cout << ">>>>>> Analog data" << std::endl;
//	std::cout << a.num_channel << " number of channels available" << std::endl;
    for (int i=0 ; i<a.num_channel ; i++)
    {
        //std::cout << i << " " << a.channel[i] << std::endl;
        analogData->channel[i] = a.channel[i];
    }

//	std::cout << "------- Analog data" << std::endl;
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
