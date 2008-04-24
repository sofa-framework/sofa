#include <sofa/component/constraint/LagrangianMultiplierAttachConstraint.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraint
{

SOFA_DECL_CLASS(LagrangianMultiplierAttachConstraint)

using namespace sofa::defaulttype;
using namespace sofa::helper;

int LagrangianMultiplierAttachConstraintClass = core::RegisterObject("TODO-LagrangianMultiplierAttachConstraintClass")
#ifndef SOFA_FLOAT
        .add< LagrangianMultiplierAttachConstraint<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LagrangianMultiplierAttachConstraint<Vec3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class LagrangianMultiplierAttachConstraint<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class LagrangianMultiplierAttachConstraint<Vec3fTypes>;
#endif

} // namespace constraint

} // namespace component

} // namespace sofa

