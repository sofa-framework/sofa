#include "CudaTypes.h"
#include "CudaSpringForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

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
