#include <sofa/component/constraint/UnilateralInteractionConstraint.inl>
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

SOFA_DECL_CLASS(UnilateralInteractionConstraint)

int UnilateralInteractionConstraintClass = core::RegisterObject("TODO-UnilateralInteractionConstraint")
        .add< UnilateralInteractionConstraint<Vec3dTypes> >()
        .add< UnilateralInteractionConstraint<Vec3fTypes> >()
        ;

template class UnilateralInteractionConstraint<Vec3dTypes>;
template class UnilateralInteractionConstraint<Vec3fTypes>;

} // namespace constraint

} // namespace component

} // namespace sofa

