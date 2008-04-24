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
#ifndef SOFA_FLOAT
        .add< UnilateralInteractionConstraint<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< UnilateralInteractionConstraint<Vec3fTypes> >()
#endif
        ;


#ifndef SOFA_FLOAT
template class UnilateralInteractionConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class UnilateralInteractionConstraint<Vec3fTypes>;
#endif


} // namespace constraint

} // namespace component

} // namespace sofa

