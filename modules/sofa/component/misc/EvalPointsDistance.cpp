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

template class EvalPointsDistance<Vec3dTypes>;
template class EvalPointsDistance<Vec3fTypes>;

int EvalPointsDistanceClass = core::RegisterObject("Periodically compute the distance between 2 set of points")
        .add< EvalPointsDistance<Vec3dTypes> >()
        .add< EvalPointsDistance<Vec3fTypes> >()
        ;

} // namespace misc

} // namespace component

} // namespace sofa
