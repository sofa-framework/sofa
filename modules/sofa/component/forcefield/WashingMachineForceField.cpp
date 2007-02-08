#include <sofa/component/forcefield/WashingMachineForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

template class WashingMachineForceField<Vec3dTypes>;
template class WashingMachineForceField<Vec3fTypes>;

SOFA_DECL_CLASS(WashingMachineForceField)


template<class DataTypes>
void create(WashingMachineForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< WashingMachineForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    obj->parseFields( arg->getAttributeMap() );
}

Creator<simulation::tree::xml::ObjectFactory, WashingMachineForceField<Vec3dTypes> > WashingMachineForceFieldVec3dClass("WashingMachineForceField", true);
Creator<simulation::tree::xml::ObjectFactory, WashingMachineForceField<Vec3fTypes> > WashingMachineForceFieldVec3fClass("WashingMachineForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa

