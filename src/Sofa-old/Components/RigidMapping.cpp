#include "RigidMapping.inl"
#include "Common/Vec3Types.h"
#include "Common/RigidTypes.h"
#include "Common/ObjectFactory.h"
#include "Sofa/Core/MappedModel.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Core/MechanicalMapping.inl"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(RigidMapping)

using namespace Common;
using namespace Core;

template<class BaseMapping>
void create(RigidMapping<BaseMapping>*& obj, ObjectDescription* arg)
{
    if (arg->getAttribute("filename"))
        XML::createWith2ObjectsAndFilename< RigidMapping<BaseMapping>, typename RigidMapping<BaseMapping>::In, typename RigidMapping<BaseMapping>::Out>(obj, arg);
    else
        XML::createWith2Objects< RigidMapping<BaseMapping>, typename RigidMapping<BaseMapping>::In, typename RigidMapping<BaseMapping>::Out>(obj, arg);
}

Creator< ObjectFactory, RigidMapping< MechanicalMapping< MechanicalModel<RigidTypes>, MechanicalModel<Vec3dTypes> > > > RigidMapping3dClass("RigidMapping", true);
Creator< ObjectFactory, RigidMapping< MechanicalMapping< MechanicalModel<RigidTypes>, MechanicalModel<Vec3fTypes> > > > RigidMapping3fClass("RigidMapping", true);

Creator< ObjectFactory, RigidMapping< Mapping< MechanicalModel<RigidTypes>, MappedModel<Vec3dTypes> > > > RigidMappingMapped3dClass("RigidMapping", true);
Creator< ObjectFactory, RigidMapping< Mapping< MechanicalModel<RigidTypes>, MappedModel<Vec3fTypes> > > > RigidMappingMapped3fClass("RigidMapping", true);

Creator< ObjectFactory, RigidMapping< Mapping< MechanicalModel<RigidTypes>, MappedModel<ExtVec3dTypes> > > > RigidMappingMappedExt3dClass("RigidMapping", true);
Creator< ObjectFactory, RigidMapping< Mapping< MechanicalModel<RigidTypes>, MappedModel<ExtVec3fTypes> > > > RigidMappingMappedExt3fClass("RigidMapping", true);

template class RigidMapping< MechanicalMapping<MechanicalModel<RigidTypes>, MechanicalModel<Vec3dTypes> > >;
template class RigidMapping< MechanicalMapping<MechanicalModel<RigidTypes>, MechanicalModel<Vec3fTypes> > >;

template class RigidMapping< Mapping<MechanicalModel<RigidTypes>, MappedModel<Vec3dTypes> > >;
template class RigidMapping< Mapping<MechanicalModel<RigidTypes>, MappedModel<Vec3fTypes> > >;

template class RigidMapping< Mapping<MechanicalModel<RigidTypes>, MappedModel<ExtVec3dTypes> > >;
template class RigidMapping< Mapping<MechanicalModel<RigidTypes>, MappedModel<ExtVec3fTypes> > >;

} // namespace Components

} // namespace Sofa
