#include <sofa/component/mapping/RigidMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidMapping)

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

template<class BaseMapping>
void create(RigidMapping<BaseMapping>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    if (arg->getAttribute("filename"))
        XML::createWith2ObjectsAndFilename< RigidMapping<BaseMapping>, typename RigidMapping<BaseMapping>::In, typename RigidMapping<BaseMapping>::Out>(obj, arg);
    else
        XML::createWith2Objects< RigidMapping<BaseMapping>, typename RigidMapping<BaseMapping>::In, typename RigidMapping<BaseMapping>::Out>(obj, arg);
}

Creator<simulation::tree::xml::ObjectFactory, RigidMapping< MechanicalMapping< MechanicalState<RigidTypes>, MechanicalState<Vec3dTypes> > > > RigidMapping3dClass("RigidMapping", true);
Creator<simulation::tree::xml::ObjectFactory, RigidMapping< MechanicalMapping< MechanicalState<RigidTypes>, MechanicalState<Vec3fTypes> > > > RigidMapping3fClass("RigidMapping", true);

Creator<simulation::tree::xml::ObjectFactory, RigidMapping< Mapping< MechanicalState<RigidTypes>, MappedModel<Vec3dTypes> > > > RigidMappingMapped3dClass("RigidMapping", true);
Creator<simulation::tree::xml::ObjectFactory, RigidMapping< Mapping< MechanicalState<RigidTypes>, MappedModel<Vec3fTypes> > > > RigidMappingMapped3fClass("RigidMapping", true);

Creator<simulation::tree::xml::ObjectFactory, RigidMapping< Mapping< MechanicalState<RigidTypes>, MappedModel<ExtVec3dTypes> > > > RigidMappingMappedExt3dClass("RigidMapping", true);
Creator<simulation::tree::xml::ObjectFactory, RigidMapping< Mapping< MechanicalState<RigidTypes>, MappedModel<ExtVec3fTypes> > > > RigidMappingMappedExt3fClass("RigidMapping", true);

template class RigidMapping< MechanicalMapping<MechanicalState<RigidTypes>, MechanicalState<Vec3dTypes> > >;
template class RigidMapping< MechanicalMapping<MechanicalState<RigidTypes>, MechanicalState<Vec3fTypes> > >;

template class RigidMapping< Mapping<MechanicalState<RigidTypes>, MappedModel<Vec3dTypes> > >;
template class RigidMapping< Mapping<MechanicalState<RigidTypes>, MappedModel<Vec3fTypes> > >;

template class RigidMapping< Mapping<MechanicalState<RigidTypes>, MappedModel<ExtVec3dTypes> > >;
template class RigidMapping< Mapping<MechanicalState<RigidTypes>, MappedModel<ExtVec3fTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa

