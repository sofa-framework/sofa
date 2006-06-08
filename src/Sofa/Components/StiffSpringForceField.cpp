// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include "StiffSpringForceField.inl"
#include "Common/Vec3Types.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(StiffSpringForceField)

using namespace Common;

template class StiffSpringForceField<Vec3dTypes>;
template class StiffSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(StiffSpringForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParentAndFilename< StiffSpringForceField<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2ObjectsAndFilename< StiffSpringForceField<DataTypes>, Core::MechanicalModel<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
}

Creator< ObjectFactory, StiffSpringForceField<Vec3dTypes> > StiffSpringInteractionForceFieldVec3dClass("StiffSpringForceField", true);
Creator< ObjectFactory, StiffSpringForceField<Vec3fTypes> > StiffSpringInteractionForceFieldVec3fClass("StiffSpringForceField", true);

} // namespace Components

} // namespace Sofa
