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

template class LagrangianMultiplierAttachConstraint<Vec3dTypes>;
template class LagrangianMultiplierAttachConstraint<Vec3fTypes>;


int LagrangianMultiplierAttachConstraintClass = core::RegisterObject("TODO-LagrangianMultiplierAttachConstraintClass")
        .add< LagrangianMultiplierAttachConstraint<Vec3dTypes> >()
        .add< LagrangianMultiplierAttachConstraint<Vec3fTypes> >()
        ;

} // namespace constraint

} // namespace component

} // namespace sofa

