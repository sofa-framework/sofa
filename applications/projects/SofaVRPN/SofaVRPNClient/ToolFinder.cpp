/*
 * ToolFinder.cpp
 *
 *  Created on: 8 sept. 2009
 *      Author: froy
 */
#define SOFAVRPNCLIENT_TOOLFINDER_CPP_

#include <ToolFinder.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vrpnclient_config.h>

using namespace sofa::defaulttype;
using namespace sofavrpn::client;

SOFA_DECL_CLASS(ToolFinder)

int ToolFinderClass = sofa::core::RegisterObject("Specific engine to compute positions and orientations from 3 3D points")
#ifndef SOFA_FLOAT
        .add< ToolFinder<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< ToolFinder<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_SOFAVRPNCLIENT_API ToolFinder<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_SOFAVRPNCLIENT_API ToolFinder<Vec3fTypes>;
#endif //SOFA_DOUBLE
