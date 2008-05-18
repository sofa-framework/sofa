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

template class SpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;
template class StiffSpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;
template class MeshSpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
template class SpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class StiffSpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class MeshSpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;

template class SpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
template class StiffSpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
template class MeshSpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

} // namespace forcefield

} // namespace component

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaSpringForceField)

int SpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::SpringForceField<CudaVec3fTypes> >()
        .add< component::forcefield::SpringForceField<CudaVec3f1Types> >()
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::forcefield::SpringForceField<CudaVec3dTypes> >()
        .add< component::forcefield::SpringForceField<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        ;

int StiffSpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::StiffSpringForceField<CudaVec3fTypes> >()
        .add< component::forcefield::StiffSpringForceField<CudaVec3f1Types> >()
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::forcefield::StiffSpringForceField<CudaVec3dTypes> >()
        .add< component::forcefield::StiffSpringForceField<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        ;

int MeshSpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::MeshSpringForceField<CudaVec3fTypes> >()
        .add< component::forcefield::MeshSpringForceField<CudaVec3f1Types> >()
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::forcefield::MeshSpringForceField<CudaVec3dTypes> >()
        .add< component::forcefield::MeshSpringForceField<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        ;

int TriangleBendingSpringsCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::TriangleBendingSprings<CudaVec3fTypes> >()
        .add< component::forcefield::TriangleBendingSprings<CudaVec3f1Types> >()
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::forcefield::TriangleBendingSprings<CudaVec3dTypes> >()
        .add< component::forcefield::TriangleBendingSprings<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        ;

int QuadBendingSpringsCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::QuadBendingSprings<CudaVec3fTypes> >()
        .add< component::forcefield::QuadBendingSprings<CudaVec3f1Types> >()
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::forcefield::QuadBendingSprings<CudaVec3dTypes> >()
        .add< component::forcefield::QuadBendingSprings<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        ;


} // namespace cuda

} // namespace gpu

} // namespace sofa
