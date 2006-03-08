#include "StiffSpringForceField.inl"
#include "Common/Vec3Types.h"
#include "XML/DynamicNode.h"
#include "Sofa/Core/MechanicalObject.h"
#include "XML/ForceFieldNode.h"
#include "XML/InteractionForceFieldNode.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(StiffSpringForceField)

using namespace Common;

template class StiffSpringForceField<Vec3dTypes>;
template class StiffSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(StiffSpringForceField<DataTypes>*& obj, XML::Node<Core::ForceField>* arg)
{
    XML::createWithParentAndFilename< StiffSpringForceField<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
}

Creator< XML::ForceFieldNode::Factory, StiffSpringForceField<Vec3dTypes> > StiffSpringForceFieldVec3dClass("StiffSpringForceField", true);
Creator< XML::ForceFieldNode::Factory, StiffSpringForceField<Vec3fTypes> > StiffSpringForceFieldVec3fClass("StiffSpringForceField", true);

template<class DataTypes>
void create(StiffSpringForceField<DataTypes>*& obj, XML::Node<Core::InteractionForceField>* arg)
{
    XML::createWith2ObjectsAndFilename< StiffSpringForceField<DataTypes>, Core::MechanicalObject<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
}

Creator< XML::InteractionForceFieldNode::Factory, StiffSpringForceField<Vec3dTypes> > StiffSpringInteractionForceFieldVec3dClass("StiffSpringForceField", true);
Creator< XML::InteractionForceFieldNode::Factory, StiffSpringForceField<Vec3fTypes> > StiffSpringInteractionForceFieldVec3fClass("StiffSpringForceField", true);

} // namespace Components

} // namespace Sofa
