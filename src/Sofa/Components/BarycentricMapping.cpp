#include "BarycentricMapping.inl"
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

SOFA_DECL_CLASS(BarycentricMapping)

template<class BaseMapping>
void create(BarycentricMapping<BaseMapping>*& obj, ObjectDescription* arg)
{
    XML::createWith2Objects< BarycentricMapping<BaseMapping>, typename BarycentricMapping<BaseMapping>::In, typename BarycentricMapping<BaseMapping>::Out>(obj, arg);
}


// Mech -> Mech
Creator< ObjectFactory, BarycentricMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3dTypes> > > > BarycentricMapping3d3dClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3fTypes> > > > BarycentricMapping3f3fClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3fTypes> > > > BarycentricMapping3d3fClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3dTypes> > > > BarycentricMapping3f3dClass("BarycentricMapping", true);

// Mech -> Mapped
Creator< ObjectFactory, BarycentricMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3dTypes> > > > BarycentricMapping3dM3dClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3fTypes> > > > BarycentricMapping3fM3fClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3fTypes> > > > BarycentricMapping3dM3fClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3dTypes> > > > BarycentricMapping3fM3dClass("BarycentricMapping", true);

// Mech -> ExtMapped
Creator< ObjectFactory, BarycentricMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > > BarycentricMapping3dME3dClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > > BarycentricMapping3fME3fClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > > BarycentricMapping3dME3fClass("BarycentricMapping", true);
Creator< ObjectFactory, BarycentricMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > > BarycentricMapping3fME3dClass("BarycentricMapping", true);

// Mech -> Mech
template class BarycentricMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3dTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalModel<Vec3dTypes>, MechanicalModel<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalModel<Vec3fTypes>, MechanicalModel<Vec3dTypes> > >;

// Mech -> Mapped
template class BarycentricMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class BarycentricMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<Vec3dTypes> > >;

// Mech -> ExtMapped
template class BarycentricMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class BarycentricMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalModel<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalModel<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;

} // namespace Components

} // namespace Sofa
