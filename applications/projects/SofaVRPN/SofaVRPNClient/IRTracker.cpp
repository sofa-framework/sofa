/*
 * IRTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */
#define SOFAVRPNCLIENT_IRTRACKER_CPP_

#include <IRTracker.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vrpnclient_config.h>

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(IRTracker)

int IRTrackerClass = sofa::core::RegisterObject("Infrared Tracker")
#ifndef SOFA_FLOAT
        .add< IRTracker<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IRTracker<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API IRTracker<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API IRTracker<Vec3fTypes>;
#endif //SOFA_DOUBLE
