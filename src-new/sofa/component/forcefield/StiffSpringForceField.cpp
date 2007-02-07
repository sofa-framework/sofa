// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/component/forcefield/StiffSpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(StiffSpringForceField)

using namespace sofa::defaulttype;

template class StiffSpringForceField<Vec3dTypes>;
template class StiffSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(StiffSpringForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< StiffSpringForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        simulation::tree::xml::createWith2Objects< StiffSpringForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj != NULL)
    {
        if (arg->getAttribute("filename"))
            obj->load(arg->getAttribute("filename"));
        if (arg->getAttribute("stiffness"))
            obj->setStiffness(atof(arg->getAttribute("stiffness")));
        if (arg->getAttribute("damping"))
            obj->setDamping(atof(arg->getAttribute("damping")));
    }
}

Creator<simulation::tree::xml::ObjectFactory, StiffSpringForceField<Vec3dTypes> > StiffSpringInteractionForceFieldVec3dClass("StiffSpringForceField", true);
Creator<simulation::tree::xml::ObjectFactory, StiffSpringForceField<Vec3fTypes> > StiffSpringInteractionForceFieldVec3fClass("StiffSpringForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa

