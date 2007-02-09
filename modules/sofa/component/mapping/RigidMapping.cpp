#include <sofa/component/mapping/RigidMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
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

using namespace defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;


// Register in the Factory
int RigidMappingClass = core::RegisterObject("TODO")
        .add< RigidMapping< MechanicalMapping< MechanicalState<RigidTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<RigidTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< MechanicalState<RigidTypes>, MappedModel<Vec3dTypes> > > >()
        .add< RigidMapping< Mapping< MechanicalState<RigidTypes>, MappedModel<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< MechanicalState<RigidTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< RigidMapping< Mapping< MechanicalState<RigidTypes>, MappedModel<ExtVec3fTypes> > > >()
        ;

template class RigidMapping< MechanicalMapping<MechanicalState<RigidTypes>, MechanicalState<Vec3dTypes> > >;
template class RigidMapping< MechanicalMapping<MechanicalState<RigidTypes>, MechanicalState<Vec3fTypes> > >;

template class RigidMapping< Mapping<MechanicalState<RigidTypes>, MappedModel<Vec3dTypes> > >;
template class RigidMapping< Mapping<MechanicalState<RigidTypes>, MappedModel<Vec3fTypes> > >;

template class RigidMapping< Mapping<MechanicalState<RigidTypes>, MappedModel<ExtVec3dTypes> > >;
template class RigidMapping< Mapping<MechanicalState<RigidTypes>, MappedModel<ExtVec3fTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa

