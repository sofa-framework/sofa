#include <sofa/component/interactionforcefield/RepulsiveSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(RepulsiveSpringForceField)

template class RepulsiveSpringForceField<Vec3dTypes>;
template class RepulsiveSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(RepulsiveSpringForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< RepulsiveSpringForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        simulation::tree::xml::createWith2Objects< RepulsiveSpringForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj != NULL)
    {
        if (arg->getAttribute("filename"))
            obj->load(arg->getAttribute("filename"));
    }
}

Creator<simulation::tree::xml::ObjectFactory, RepulsiveSpringForceField<Vec3dTypes> > RepulsiveSpringInteractionForceFieldVec3dClass("RepulsiveSpringForceField", true);
Creator<simulation::tree::xml::ObjectFactory, RepulsiveSpringForceField<Vec3fTypes> > RepulsiveSpringInteractionForceFieldVec3fClass("RepulsiveSpringForceField", true);

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

