/*
 * VRPNTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#define SOFAVRPNCLIENT_VRPNTRACKER_CPP_

#include "VRPNTracker.inl"

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vrpnclient_config.h>

namespace sofavrpn
{

namespace client
{



void handle_tracker(void *userdata, const vrpn_TRACKERCB t)
{
    VRPNTrackerData* trackerData = (VRPNTrackerData*) userdata;

    trackerData->modified = true;

    if ((int)trackerData->data.size() < t.sensor + 1)
        trackerData->data.resize(t.sensor+1);

    trackerData->data[t.sensor].sensor = t.sensor;

    for (unsigned int i=0 ; i<3 ; i++)
    {
        trackerData->data[t.sensor].pos[i] = t.pos[i];
        trackerData->data[t.sensor].quat[i] = t.quat[i];
    }

    trackerData->data[t.sensor].quat[3] = t.quat[3];
}

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(VRPNTracker)

int VRPNTrackerClass = sofa::core::RegisterObject("VRPN Tracker")
#ifndef SOFA_FLOAT
        .add< VRPNTracker<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< VRPNTracker<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Vec3fTypes>;
#endif //SOFA_DOUBLE

}

}
