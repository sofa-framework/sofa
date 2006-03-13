#include "RigidMapping.inl"
#include "Common/Vec3Types.h"
#include "XML/MappingNode.h"
#include "Sofa/Core/MappedModel.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(RigidMapping)

using namespace Common;

template<class BaseMapping>
void create(RigidMapping<BaseMapping>*& obj, XML::Node<Core::BasicMapping>* arg)
{
    if (arg->getAttribute("filename"))
        XML::createWith2ObjectsAndFilename< RigidMapping<BaseMapping>, typename RigidMapping<BaseMapping>::In, typename RigidMapping<BaseMapping>::Out>(obj, arg);
    else
        XML::createWith2Objects< RigidMapping<BaseMapping>, typename RigidMapping<BaseMapping>::In, typename RigidMapping<BaseMapping>::Out>(obj, arg);
}

Creator< XML::MappingNode::Factory, RigidMapping< MechanicalMapping< RigidObject, MechanicalModel<Vec3dTypes> > > > RigidMapping3dClass("RigidMapping", true);
Creator< XML::MappingNode::Factory, RigidMapping< MechanicalMapping< RigidObject, MechanicalModel<Vec3fTypes> > > > RigidMapping3fClass("RigidMapping", true);

Creator< XML::MappingNode::Factory, RigidMapping< Mapping< RigidObject, MappedModel<Vec3dTypes> > > > RigidMappingMapped3dClass("RigidMapping", true);
Creator< XML::MappingNode::Factory, RigidMapping< Mapping< RigidObject, MappedModel<Vec3fTypes> > > > RigidMappingMapped3fClass("RigidMapping", true);

Creator< XML::MappingNode::Factory, RigidMapping< Mapping< RigidObject, MappedModel<ExtVec3dTypes> > > > RigidMappingMappedExt3dClass("RigidMapping", true);
Creator< XML::MappingNode::Factory, RigidMapping< Mapping< RigidObject, MappedModel<ExtVec3fTypes> > > > RigidMappingMappedExt3fClass("RigidMapping", true);

template class RigidMapping< MechanicalMapping<RigidObject, MechanicalModel<Vec3dTypes> > >;
template class RigidMapping< MechanicalMapping<RigidObject, MechanicalModel<Vec3fTypes> > >;

template class RigidMapping< Mapping<RigidObject, MappedModel<Vec3dTypes> > >;
template class RigidMapping< Mapping<RigidObject, MappedModel<Vec3fTypes> > >;

template class RigidMapping< Mapping<RigidObject, MappedModel<ExtVec3dTypes> > >;
template class RigidMapping< Mapping<RigidObject, MappedModel<ExtVec3fTypes> > >;

} // namespace Components

} // namespace Sofa
