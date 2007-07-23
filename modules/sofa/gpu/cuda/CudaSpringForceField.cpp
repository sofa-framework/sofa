#include "CudaTypes.h"
#include "CudaSpringForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template class SpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class StiffSpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class MeshSpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;

} // namespace forcefield

} // namespace component

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaSpringForceField)

int SpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::SpringForceField<CudaVec3fTypes> >()
        ;

int StiffSpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::StiffSpringForceField<CudaVec3fTypes> >()
        ;

int MeshSpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::MeshSpringForceField<CudaVec3fTypes> >()
        ;

int TriangleBendingSpringsCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::TriangleBendingSprings<CudaVec3fTypes> >()
        ;

int QuadBendingSpringsCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::QuadBendingSprings<CudaVec3fTypes> >()
        ;


} // namespace cuda

} // namespace gpu

} // namespace sofa
