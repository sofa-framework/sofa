#include <sofa/component/mapping/SPHFluidSurfaceMapping.inl>
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
using namespace core::componentmodel::behavior;

SOFA_DECL_CLASS(SPHFluidSurfaceMapping)

// Register in the Factory
int SPHFluidSurfaceMappingClass = core::RegisterObject("TODO-SPHFluidSurfaceMappingClass")
        .add< SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> > >()
        .add< SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> > >()
        .add< SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> > >()
        .add< SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> > >()
        .add< SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >()
        .add< SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >()
        .add< SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >()
        .add< SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >()
        ;

// Mech -> Mapped
template class SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3dTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<Vec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<Vec3dTypes> >;

// Mech -> ExtMapped
template class SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3dTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3dTypes>, MappedModel<ExtVec3fTypes> >;
template class SPHFluidSurfaceMapping< MechanicalState<Vec3fTypes>, MappedModel<ExtVec3dTypes> >;

} // namespace mapping

} // namespace component

} // namespace sofa

