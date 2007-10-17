#include <sofa/component/constraint/LagrangianMultiplierContactConstraint.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraint
{

SOFA_DECL_CLASS(LagrangianMultiplierContactConstraint)

using namespace sofa::defaulttype;
using namespace sofa::helper;

template class LagrangianMultiplierContactConstraint<Vec3dTypes>;
template class LagrangianMultiplierContactConstraint<Vec3fTypes>;

int LagrangianMultiplierContactConstraintClass = core::RegisterObject("TODO-LagrangianMultiplierContactConstraintClass")
        .add< LagrangianMultiplierContactConstraint<Vec3dTypes> >()
        .add< LagrangianMultiplierContactConstraint<Vec3fTypes> >()
        ;

} // namespace constraint

} // namespace component

} // namespace sofa

