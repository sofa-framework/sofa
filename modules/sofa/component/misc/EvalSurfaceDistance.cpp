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


int EvalSurfaceDistanceClass = core::RegisterObject("Periodically compute the distance between 2 set of points")
#ifndef SOFA_FLOAT
        .add< EvalSurfaceDistance<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< EvalSurfaceDistance<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class EvalSurfaceDistance<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class EvalSurfaceDistance<Vec3fTypes>;
#endif

} // namespace misc

} // namespace component

} // namespace sofa
