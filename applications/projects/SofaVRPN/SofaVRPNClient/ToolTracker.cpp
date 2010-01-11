/*
 * ToolTracker.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */
#define SOFAVRPNCLIENT_TOOLTRACKER_CPP_

#include <ToolTracker.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vrpnclient_config.h>

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(ToolTracker)

int ToolTrackerClass = sofa::core::RegisterObject("Specific engine to compute positions and orientations from 3 3D points")
#ifndef SOFA_FLOAT
        .add< ToolTracker<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< ToolTracker<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API ToolTracker<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API ToolTracker<Vec3fTypes>;
#endif //SOFA_DOUBLE
