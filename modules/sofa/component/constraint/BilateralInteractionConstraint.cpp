#include <sofa/component/constraint/BilateralInteractionConstraint.inl>

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

SOFA_DECL_CLASS(BilateralInteractionConstraint)

int BilateralInteractionConstraintClass = core::RegisterObject("TODO-BilateralInteractionConstraint")
#ifndef SOFA_FLOAT
        .add< BilateralInteractionConstraint<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< BilateralInteractionConstraint<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class BilateralInteractionConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class BilateralInteractionConstraint<Vec3fTypes>;
#endif

} // namespace constraint

} // namespace component

} // namespace sofa

