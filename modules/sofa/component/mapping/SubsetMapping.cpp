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
#ifndef SOFA_FLOAT
        .add< SubsetMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > > >()
// .add< SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< SubsetMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > > >()
// .add< SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< SubsetMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > > >()
#endif
#endif
        .addAlias("SurfaceIdentityMapping")
        ;


#ifndef SOFA_FLOAT
template class SubsetMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class SubsetMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
// template class SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SubsetMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class SubsetMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;
template class SubsetMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class SubsetMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif

// Mech -> Mech

// Mech -> Mapped

// Mech -> ExtMapped

} // namespace mapping

} // namespace component

} // namespace sofa
