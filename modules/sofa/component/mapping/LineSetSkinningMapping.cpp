#include <sofa/component/mapping/LineSetSkinningMapping.inl>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
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

SOFA_DECL_CLASS(LineSetSkinningMapping)

// Register in the Factory
int HandMappingClass = core::RegisterObject("skin a model from a set of rigid lines")
#ifndef SOFA_FLOAT
        .add< LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > > >()
// .add< LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
// .add< LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > > >()
#endif
#endif
//.add< HandMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< HandMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
//.add< HandMapping< Mapping< MappedModel<Rigid3fTypes>, MappedModel<Vec3dTypes> > > >()
//.add< HandMapping< Mapping< MappedModel<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
//.add< HandMapping< Mapping< MappedModel<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< HandMapping< Mapping< MappedModel<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        ;


#ifndef SOFA_FLOAT
template class LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
template class LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// template class LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
template class LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
template class LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
// template class LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;
template class LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
template class LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;
template class LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif
// Mech -> Mech
//template class HandMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
//template class HandMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;

// Mech -> Mapped
//template class HandMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
//template class HandMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;

// Mech -> ExtMapped
//template class HandMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
//template class HandMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa
