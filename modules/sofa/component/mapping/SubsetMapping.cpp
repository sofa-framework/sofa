#include "SubsetMapping.inl"

#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;

SOFA_DECL_CLASS(SubsetMapping)

int SubsetMappingClass = core::RegisterObject("TODO-SubsetMappingClass")
        .add< SubsetMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        ;

// Mech -> Mech
template class SubsetMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SubsetMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class SubsetMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class SubsetMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;

// Mech -> Mapped
template class SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > >;

// Mech -> ExtMapped
template class SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa
