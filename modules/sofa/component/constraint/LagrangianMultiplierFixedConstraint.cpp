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
#ifndef SOFA_FLOAT
        .add< LagrangianMultiplierFixedConstraint<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LagrangianMultiplierFixedConstraint<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class LagrangianMultiplierFixedConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class LagrangianMultiplierFixedConstraint<Vec3fTypes>;
#endif

} // namespace constraint

} // namespace component

} // namespace sofa

