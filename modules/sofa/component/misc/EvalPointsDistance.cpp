#include "EvalPointsDistance.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(EvalPointsDistance)

using namespace defaulttype;


int EvalPointsDistanceClass = core::RegisterObject("Periodically compute the distance between 2 set of points")
#ifndef SOFA_FLOAT
        .add< EvalPointsDistance<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< EvalPointsDistance<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class EvalPointsDistance<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class EvalPointsDistance<Vec3fTypes>;
#endif
} // namespace misc

} // namespace component

} // namespace sofa
