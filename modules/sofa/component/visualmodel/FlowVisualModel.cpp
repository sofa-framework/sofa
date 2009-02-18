/*
 * FlowVisualModel.cpp
 *
 *  Created on: 18 févr. 2009
 *      Author: froy
 */

#include "FlowVisualModel.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FlowVisualModel)

// Register in the Factory
int FlowVisualModelClass = core::RegisterObject("FlowVisualModel")
#ifndef SOFA_FLOAT
        .add< FlowVisualModel<Vec3dTypes> >()
        .add< FlowVisualModel<Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FlowVisualModel<Vec3fTypes> >()
        .add< FlowVisualModel<Vec2fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class FlowVisualModel<defaulttype::Vec3dTypes>;
template class FlowVisualModel<defaulttype::Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class FlowVisualModel<defaulttype::Vec3fTypes>;
template class FlowVisualModel<defaulttype::Vec2fTypes>;
#endif


}

}

}
