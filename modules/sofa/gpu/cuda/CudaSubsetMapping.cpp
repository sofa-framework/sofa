#include "CudaTypes.h"
#include <CudaSubsetMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::component::mapping;

SOFA_DECL_CLASS(CudaSubsetMapping)

int SubsetMappingCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
#ifndef SOFA_FLOAT
#endif
#ifndef SOFA_DOUBLE
        .add< SubsetMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3fTypes>, MappedModel<CudaVec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
// .add< SubsetMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3f1Types> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3fTypes> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3f1Types> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3f1Types> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3fTypes>, MappedModel<CudaVec3f1Types> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3fTypes> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< SubsetMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< SubsetMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3dTypes> > > >()
        .add< SubsetMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3dTypes> > > >()
// .add< SubsetMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3dTypes> > > >()
#endif
#endif
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
