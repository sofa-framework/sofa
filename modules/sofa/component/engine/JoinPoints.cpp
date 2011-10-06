#define SOFA_COMPONENT_ENGINE_JOINPOINTS_CPP_

#include "JoinPoints.inl"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(JoinPoints)

int JoinPointsClass = core::RegisterObject("?")
#ifndef SOFA_FLOAT
        .add< JoinPoints<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< JoinPoints<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_ENGINE_API JoinPoints<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_ENGINE_API JoinPoints<Vec3fTypes>;
#endif //SOFA_DOUBLE

} // namespace engine

} // namespace component

} // namespace sofa
