#include <sofa/component/mapping/IdentityMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;

SOFA_DECL_CLASS(IdentityMapping)

template<class BaseMapping>
void create(IdentityMapping<BaseMapping>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWith2Objects< IdentityMapping<BaseMapping>, typename IdentityMapping<BaseMapping>::In, typename IdentityMapping<BaseMapping>::Out>(obj, arg);
}

// Mech -> Mech
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > > IdentityMapping3d3dClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > > IdentityMapping3f3fClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > > IdentityMapping3d3fClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > > IdentityMapping3f3dClass("IdentityMapping", true);

// Mech -> Mapped
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> > > > IdentityMapping3dM3dClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> > > > IdentityMapping3fM3fClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> > > > IdentityMapping3dM3fClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> > > > IdentityMapping3fM3dClass("IdentityMapping", true);

// Mech -> ExtMapped

Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > > IdentityMapping3dME3dClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > > IdentityMapping3fME3fClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > > IdentityMapping3dME3fClass("IdentityMapping", true);
Creator<simulation::tree::xml::ObjectFactory, IdentityMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > > IdentityMapping3fME3dClass("IdentityMapping", true);


// Mech -> Mech
template class IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;

// Mech -> Mapped
template class IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class IdentityMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class IdentityMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> > >;

// Mech -> ExtMapped
template class IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class IdentityMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class IdentityMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class IdentityMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa

