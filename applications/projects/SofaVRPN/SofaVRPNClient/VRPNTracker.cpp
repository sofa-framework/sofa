/*
 * VRPNTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */

#define SOFAVRPNCLIENT_VRPNTRACKER_CPP_

#include "VRPNTracker.inl"

#include <sofa/core/ObjectFactory.h>
#include <vrpnclient_config.h>

namespace sofavrpn
{

namespace client
{

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(VRPNTracker)

int VRPNTrackerClass = sofa::core::RegisterObject("VRPN Tracker")
#ifndef SOFA_FLOAT
        .add< VRPNTracker<Vec3dTypes> >()
        .add< VRPNTracker<Rigid3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< VRPNTracker<Vec3fTypes> >()
        .add< VRPNTracker<Rigid3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Vec3dTypes>;
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Vec3fTypes>;
template class SOFA_SOFAVRPNCLIENT_API VRPNTracker<Rigid3fTypes>;
#endif //SOFA_DOUBLE

}

}
