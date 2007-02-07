// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/component/forcefield/SpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
//#include <typeinfo>


namespace sofa
{

namespace component
{

namespace forcefield
{

SOFA_DECL_CLASS(SpringForceField)

using namespace sofa::defaulttype;

template class SpringForceField<Vec3dTypes>;
template class SpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(SpringForceField<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWithParent< SpringForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< SpringForceField<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes>, core::componentmodel::behavior::MechanicalState<DataTypes> >(obj, arg);
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

Creator<simulation::tree::xml::ObjectFactory, SpringForceField<Vec3dTypes> > SpringInteractionForceFieldVec3dClass("SpringForceField", true);
Creator<simulation::tree::xml::ObjectFactory, SpringForceField<Vec3fTypes> > SpringInteractionForceFieldVec3fClass("SpringForceField", true);

} // namespace forcefield

} // namespace component

} // namespace sofa

