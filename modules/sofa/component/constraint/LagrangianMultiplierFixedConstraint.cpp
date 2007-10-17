#include <sofa/component/constraint/LagrangianMultiplierFixedConstraint.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(LagrangianMultiplierFixedConstraint)


int LagrangianMultiplierFixedConstraintClass = core::RegisterObject("TODO-LagrangianMultiplierFixedConstraintClass")
        .add< LagrangianMultiplierFixedConstraint<Vec3dTypes> >()
        .add< LagrangianMultiplierFixedConstraint<Vec3fTypes> >()
        ;


template class LagrangianMultiplierFixedConstraint<Vec3dTypes>;
template class LagrangianMultiplierFixedConstraint<Vec3fTypes>;

} // namespace constraint

} // namespace component

} // namespace sofa

