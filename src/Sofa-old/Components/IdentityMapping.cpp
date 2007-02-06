#include "IdentityMapping.inl"
#include "Common/Vec3Types.h"
#include "Common/ObjectFactory.h"
#include "Sofa/Core/MappedModel.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Core/MechanicalMapping.h"

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Core;

SOFA_DECL_CLASS(IdentityMapping)

template<class BaseMapping>
void create(IdentityMapping<BaseMapping>*& obj, ObjectDescription* arg)
{
    XML::createWith2Objects< IdentityMapping<BaseMapping>, typename IdentityMapping<BaseMapping>::In, typename IdentityMapping<BaseMapping>::Out>(obj, arg);
}

// Mech -> Mech
Creator< ObjectFactory, IdentityMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3dTypes> > > > IdentityMapping3d3dClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3fTypes> > > > IdentityMapping3f3fClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3fTypes> > > > IdentityMapping3d3fClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3dTypes> > > > IdentityMapping3f3dClass("IdentityMapping", true);

// Mech -> Mapped
Creator< ObjectFactory, IdentityMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3dTypes> > > > IdentityMapping3dM3dClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3fTypes> > > > IdentityMapping3fM3fClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3fTypes> > > > IdentityMapping3dM3fClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3dTypes> > > > IdentityMapping3fM3dClass("IdentityMapping", true);

// Mech -> ExtMapped

Creator< ObjectFactory, IdentityMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > > IdentityMapping3dME3dClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > > IdentityMapping3fME3fClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > > IdentityMapping3dME3fClass("IdentityMapping", true);
Creator< ObjectFactory, IdentityMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > > IdentityMapping3fME3dClass("IdentityMapping", true);


// Mech -> Mech
template class IdentityMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3dTypes> > >;
template class IdentityMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3fTypes> > >;
template class IdentityMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3fTypes> > >;
template class IdentityMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3dTypes> > >;

// Mech -> Mapped
template class IdentityMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class IdentityMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class IdentityMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class IdentityMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3dTypes> > >;

// Mech -> ExtMapped
template class IdentityMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class IdentityMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class IdentityMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class IdentityMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;

} // namespace Components

} // namespace Sofa
