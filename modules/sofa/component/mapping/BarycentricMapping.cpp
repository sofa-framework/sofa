#include <sofa/component/mapping/BarycentricMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
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

SOFA_DECL_CLASS(BarycentricMapping)

// Register in the Factory
int BarycentricMappingClass = core::RegisterObject("TODO")
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        ;

// Mech -> Mech
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;

// Mech -> Mapped
template class BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> > >;

// Mech -> ExtMapped
template class BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa

