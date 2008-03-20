// #include <sofa/component/constraint/BilateralInteractionConstraint.inl>
#include "BilateralInteractionConstraint.inl"

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

SOFA_DECL_CLASS(BilateralInteractionConstraint)

int BilateralInteractionConstraintClass = core::RegisterObject("TODO-BilateralInteractionConstraint")
        .add< BilateralInteractionConstraint<Vec3dTypes> >()
        .add< BilateralInteractionConstraint<Vec3fTypes> >()
        ;

template class BilateralInteractionConstraint<Vec3dTypes>;
template class BilateralInteractionConstraint<Vec3fTypes>;

} // namespace constraint

} // namespace component

} // namespace sofa

