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

#ifndef SOFA_FLOAT
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3dTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3fTypes> > > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3fTypes> > > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3dTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3fTypes> > >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Rigid3dTypes> > >;
template class ArticulatedSystemMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Rigid3fTypes> > >;
#endif
#endif
} // namespace mapping

} // namespace component

} // namespace sofa
