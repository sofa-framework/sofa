#include "CudaTypes.h"
#include <CudaRigidMapping.inl>
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

SOFA_DECL_CLASS(CudaRigidMapping)

int RigidMappingCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< RigidMapping< MechanicalMapping< MechanicalState<CudaRigid3fTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< RigidMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<CudaVec3fTypes> > > >()
//.add< RigidMapping< MechanicalMapping< MechanicalState<CudaRigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< RigidMapping< MechanicalMapping< MechanicalState<CudaRigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<CudaRigid3fTypes>, MappedModel<CudaVec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<CudaVec3fTypes> > > >()
        .add< RigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<CudaVec3fTypes> > > >()
//.add< RigidMapping< Mapping< State<CudaRigid3fTypes>, MappedModel<Vec3dTypes> > > >()
//.add< RigidMapping< Mapping< State<CudaRigid3fTypes>, MappedModel<Vec3fTypes> > > >()
//.add< RigidMapping< Mapping< State<CudaRigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< RigidMapping< Mapping< State<CudaRigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
