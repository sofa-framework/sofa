#include <sofa/component/constraint/LagrangianMultiplierAttachConstraint.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraint
{

SOFA_DECL_CLASS(LagrangianMultiplierAttachConstraint)

using namespace sofa::defaulttype;

template class LagrangianMultiplierAttachConstraint<Vec3dTypes>;
template class LagrangianMultiplierAttachConstraint<Vec3fTypes>;

template<class DataTypes>
void create(LagrangianMultiplierAttachConstraint<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWithParent< LagrangianMultiplierAttachConstraint<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< LagrangianMultiplierAttachConstraint<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
}

Creator<simulation::tree::xml::ObjectFactory, LagrangianMultiplierAttachConstraint<Vec3dTypes> > LagrangianMultiplierAttachConstraintVec3dClass("LagrangianMultiplierAttachConstraint", true);
Creator<simulation::tree::xml::ObjectFactory, LagrangianMultiplierAttachConstraint<Vec3fTypes> > LagrangianMultiplierAttachConstraintVec3fClass("LagrangianMultiplierAttachConstraint", true);

} // namespace constraint

} // namespace component

} // namespace sofa

