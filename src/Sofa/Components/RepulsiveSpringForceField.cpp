#include "RepulsiveSpringForceField.inl"
#include "Common/Vec3Types.h"
#include "XML/DynamicNode.h"
#include "Sofa/Core/MechanicalObject.h"
#include "XML/ForceFieldNode.h"
#include "XML/InteractionForceFieldNode.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template class RepulsiveSpringForceField<Vec3dTypes>;
template class RepulsiveSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(RepulsiveSpringForceField<DataTypes>*& obj, XML::Node<Core::ForceField>* arg)
{
    XML::createWithParentAndFilename< RepulsiveSpringForceField<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
}

Creator< XML::ForceFieldNode::Factory, RepulsiveSpringForceField<Vec3dTypes> > RepulsiveSpringForceFieldVec3dClass("RepulsiveSpringForceField", true);
Creator< XML::ForceFieldNode::Factory, RepulsiveSpringForceField<Vec3fTypes> > RepulsiveSpringForceFieldVec3fClass("RepulsiveSpringForceField", true);

template<class DataTypes>
void create(RepulsiveSpringForceField<DataTypes>*& obj, XML::Node<Core::InteractionForceField>* arg)
{
    XML::createWith2ObjectsAndFilename< RepulsiveSpringForceField<DataTypes>, Core::MechanicalObject<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
}

Creator< XML::InteractionForceFieldNode::Factory, RepulsiveSpringForceField<Vec3dTypes> > RepulsiveSpringInteractionForceFieldVec3dClass("RepulsiveSpringForceField", true);
Creator< XML::InteractionForceFieldNode::Factory, RepulsiveSpringForceField<Vec3fTypes> > RepulsiveSpringInteractionForceFieldVec3fClass("RepulsiveSpringForceField", true);

} // namespace Components

} // namespace Sofa
