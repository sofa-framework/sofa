#include "SpringForceField.inl"
#include "Common/Vec3Types.h"
#include "XML/DynamicNode.h"
#include "Sofa/Core/MechanicalObject.h"
#include "XML/ForceFieldNode.h"
#include "XML/InteractionForceFieldNode.h"

//#include <typeinfo>

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(SpringForceField)

using namespace Common;

template class SpringForceField<Vec3dTypes>;
template class SpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(SpringForceField<DataTypes>*& obj, XML::Node<Core::ForceField>* arg)
{
    XML::createWithParentAndFilename< SpringForceField<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
}

Creator< XML::ForceFieldNode::Factory, SpringForceField<Vec3dTypes> > SpringForceFieldVec3dClass("SpringForceField", true);
Creator< XML::ForceFieldNode::Factory, SpringForceField<Vec3fTypes> > SpringForceFieldVec3fClass("SpringForceField", true);

template<class DataTypes>
void create(SpringForceField<DataTypes>*& obj, XML::Node<Core::InteractionForceField>* arg)
{
    XML::createWith2ObjectsAndFilename< SpringForceField<DataTypes>, Core::MechanicalObject<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
}

Creator< XML::InteractionForceFieldNode::Factory, SpringForceField<Vec3dTypes> > SpringInteractionForceFieldVec3dClass("SpringForceField", true);
Creator< XML::InteractionForceFieldNode::Factory, SpringForceField<Vec3fTypes> > SpringInteractionForceFieldVec3fClass("SpringForceField", true);

} // namespace Components

} // namespace Sofa
