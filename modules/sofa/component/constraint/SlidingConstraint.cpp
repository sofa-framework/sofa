#include <sofa/component/constraint/SlidingConstraint.inl>

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/base/container/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(SlidingConstraint)

int SlidingConstraintClass = core::RegisterObject("TODO-SlidingConstraint")
#ifndef SOFA_FLOAT
        .add< SlidingConstraint<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< SlidingConstraint<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SlidingConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SlidingConstraint<Vec3fTypes>;
#endif

} // namespace constraint

} // namespace component

} // namespace sofa

