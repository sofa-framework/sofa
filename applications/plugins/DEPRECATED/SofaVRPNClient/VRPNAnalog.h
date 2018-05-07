/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/DataEngine.h>
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

    sofa::core::objectmodel::Data<sofa::helper::vector<Real> > f_channels; ///< Channels

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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFAVRPNCLIENT_VRPNANALOG_CPP_)
#ifndef SOFA_FLOAT
extern template class SOFA_SOFAVRPNCLIENT_API VRPNAnalog<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_SOFAVRPNCLIENT_API VRPNAnalog<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

}

}

#endif /* SOFAVRPNCLIENT_VRPNANALOG_H_ */
