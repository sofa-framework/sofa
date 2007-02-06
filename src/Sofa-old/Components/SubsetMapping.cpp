#include "SubsetMapping.inl"

#include "Sofa-old/Core/MechanicalMapping.inl"

#include "Common/Vec3Types.h"
#include "Common/ObjectFactory.h"
#include "Sofa-old/Core/MappedModel.h"
#include "Sofa-old/Core/MechanicalModel.h"

namespace Sofa
{

namespace Components
{


using namespace Common;
using namespace Core;

SOFA_DECL_CLASS(SubsetMapping)

template<class BaseMapping>
void create(SubsetMapping<BaseMapping>*& obj, ObjectDescription* arg)
{
    XML::createWith2Objects< SubsetMapping<BaseMapping>, typename SubsetMapping<BaseMapping>::In, typename SubsetMapping<BaseMapping>::Out>(obj, arg);
    if (obj != NULL)
    {
        obj->parseFields( arg->getAttributeMap() );
    }
}

// Mech -> Mech
Creator< ObjectFactory, SubsetMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3dTypes> > > > SubsetMapping3d3dClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3fTypes> > > > SubsetMapping3f3fClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3fTypes> > > > SubsetMapping3d3fClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3dTypes> > > > SubsetMapping3f3dClass("SubsetMapping", true);

// Mech -> Mapped
Creator< ObjectFactory, SubsetMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3dTypes> > > > SubsetMapping3dM3dClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3fTypes> > > > SubsetMapping3fM3fClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3fTypes> > > > SubsetMapping3dM3fClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3dTypes> > > > SubsetMapping3fM3dClass("SubsetMapping", true);

// Mech -> ExtMapped

Creator< ObjectFactory, SubsetMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > > SubsetMapping3dME3dClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > > SubsetMapping3fME3fClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > > SubsetMapping3dME3fClass("SubsetMapping", true);
Creator< ObjectFactory, SubsetMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > > SubsetMapping3fME3dClass("SubsetMapping", true);


// Mech -> Mech
template class SubsetMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3dTypes> > >;
template class SubsetMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3fTypes> > >;
template class SubsetMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3fTypes> > >;
template class SubsetMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3dTypes> > >;

// Mech -> Mapped
template class SubsetMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class SubsetMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class SubsetMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class SubsetMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3dTypes> > >;

// Mech -> ExtMapped
template class SubsetMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class SubsetMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class SubsetMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class SubsetMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;


} // namespace Components

} // namespace Sofa

