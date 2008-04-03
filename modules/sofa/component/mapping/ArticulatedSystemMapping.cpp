#include <sofa/component/mapping/ArticulatedSystemMapping.inl>
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

SOFA_DECL_CLASS(ArticulatedSystemMapping)

// Register in the Factory
int ArticulatedSystemMappingClass = core::RegisterObject("")
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3fTypes> > > >()
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3fTypes> > > >()
        ;

// Mech -> Mech
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3dTypes> > >;
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3dTypes> > >;
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3fTypes> > >;
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3fTypes> > >;
} // namespace mapping

} // namespace component

} // namespace sofa
