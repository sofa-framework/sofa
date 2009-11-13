/*
 * WiimoteDriver.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */
#define SOFAVRPNCLIENT_WIIMOTEDRIVER_CPP_

#include <WiimoteDriver.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vrpnclient_config.h>

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(WiimoteDriver)

int WiimoteDriverClass = sofa::core::RegisterObject("Wiimote Driver")
#ifndef SOFA_FLOAT
        .add< WiimoteDriver<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< WiimoteDriver<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API WiimoteDriver<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API WiimoteDriver<Vec3fTypes>;
#endif //SOFA_DOUBLE
