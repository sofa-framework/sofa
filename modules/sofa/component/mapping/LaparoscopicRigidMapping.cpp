#include <sofa/component/mapping/LaparoscopicRigidMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
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

SOFA_DECL_CLASS(LaparoscopicRigidMapping)

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;

// Register in the Factory
int LaparoscopicRigidMappingClass = core::RegisterObject("TODO-LaparoscopicRigidMappingClass")
        .add< LaparoscopicRigidMapping< MechanicalMapping< MechanicalState<LaparoscopicRigidTypes>, MechanicalState<RigidTypes> > > >()
        .add< LaparoscopicRigidMapping< Mapping< MechanicalState<LaparoscopicRigidTypes>, MappedModel<RigidTypes> > > >()
        ;

template class LaparoscopicRigidMapping< MechanicalMapping<MechanicalState<LaparoscopicRigidTypes>, MechanicalState<RigidTypes> > >;
template class LaparoscopicRigidMapping< Mapping<MechanicalState<LaparoscopicRigidTypes>, MappedModel<RigidTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa

