#include <sofa/component/forcefield/PenalityContactForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(PenalityContactForceField)

using namespace sofa::defaulttype;

template class PenalityContactForceField<Vec3dTypes>;
template class PenalityContactForceField<Vec3fTypes>;

template<class DataTypes>
void create(PenalityContactForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWithParent< PenalityContactForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< PenalityContactForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
}

Creator<simulation::tree::xml::ObjectFactory, PenalityContactForceField<Vec3dTypes> > PenalityContactForceFieldVec3dClass("PenalityContactForceField", true);
Creator<simulation::tree::xml::ObjectFactory, PenalityContactForceField<Vec3fTypes> > PenalityContactForceFieldVec3fClass("PenalityContactForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa

