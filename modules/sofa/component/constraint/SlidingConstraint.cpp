// #include <sofa/component/constraint/SlidingConstraint.inl>
#include "SlidingConstraint.inl"

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
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
        .add< SlidingConstraint<Vec3dTypes> >()
        .add< SlidingConstraint<Vec3fTypes> >()
        ;

template class SlidingConstraint<Vec3dTypes>;
template class SlidingConstraint<Vec3fTypes>;

} // namespace constraint

} // namespace component

} // namespace sofa

