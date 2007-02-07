#include <sofa/component/constraint/LagrangianMultiplierContactConstraint.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraint
{

SOFA_DECL_CLASS(LagrangianMultiplierContactConstraint)

using namespace sofa::defaulttype;

template class LagrangianMultiplierContactConstraint<Vec3dTypes>;
template class LagrangianMultiplierContactConstraint<Vec3fTypes>;

template<class DataTypes>
void create(LagrangianMultiplierContactConstraint<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< LagrangianMultiplierContactConstraint<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        simulation::tree::xml::createWith2Objects< LagrangianMultiplierContactConstraint<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
}

Creator<simulation::tree::xml::ObjectFactory, LagrangianMultiplierContactConstraint<Vec3dTypes> > LagrangianMultiplierContactConstraintVec3dClass("LagrangianMultiplierContactConstraint", true);
Creator<simulation::tree::xml::ObjectFactory, LagrangianMultiplierContactConstraint<Vec3fTypes> > LagrangianMultiplierContactConstraintVec3fClass("LagrangianMultiplierContactConstraint", true);

} // namespace constraint

} // namespace component

} // namespace sofa

