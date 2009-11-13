/*
 * VRPNAnalog.h
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#ifndef SOFAVRPNCLIENT_VRPNANALOG_H_
#define SOFAVRPNCLIENT_VRPNANALOG_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataEngine.h>
#include <sofa/helper/RandomGenerator.h>

#include <VRPNDevice.h>

#include <vrpn/vrpn_Analog.h>

namespace sofavrpn
{

namespace client
{

//typedef	struct _vrpn_ANALOGCB {
//	struct timeval	msg_time;	// Timestamp of analog data
//	vrpn_int32	num_channel;    // how many channels
//	vrpn_float64	channel[vrpn_CHANNEL_MAX];  // analog values
//} vrpn_ANALOGCB;

struct VRPNAnalogData
{
    vrpn_ANALOGCB data;
    bool modified;
};

void handle_analog(void *userdata, const vrpn_ANALOGCB a);

template<class DataTypes>
class VRPNAnalog : public virtual VRPNDevice
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(VRPNAnalog, DataTypes), VRPNDevice);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    Data<sofa::helper::vector<Real> > f_channels;

    VRPNAnalogData analogData;

    VRPNAnalog();
    virtual ~VRPNAnalog();

//	void init();
//	void reinit();
    void update();


private:
    vrpn_Analog_Remote* anr;
    sofa::helper::RandomGenerator rg;

    bool connectToServer();
    //callback
    //static void handle_analog(void *userdata, const vrpn_ANALOGCB a);
};

#if defined(WIN32) && !defined(SOFAVRPNCLIENT_VRPNANALOG_CPP_)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNAnalog<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNAnalog<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_VRPNANALOG_H_ */
