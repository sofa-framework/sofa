#include "EvalSurfaceDistance.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(EvalSurfaceDistance)

using namespace defaulttype;

template class EvalSurfaceDistance<Vec3dTypes>;
template class EvalSurfaceDistance<Vec3fTypes>;

int EvalSurfaceDistanceClass = core::RegisterObject("Periodically compute the distance between 2 set of points")
        .add< EvalSurfaceDistance<Vec3dTypes> >()
        .add< EvalSurfaceDistance<Vec3fTypes> >()
        ;

} // namespace misc

} // namespace component

} // namespace sofa
