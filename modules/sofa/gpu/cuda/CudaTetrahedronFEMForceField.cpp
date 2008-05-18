#include "CudaTypes.h"
#include "CudaTetrahedronFEMForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaTetrahedronFEMForceField)

int TetrahedronFEMForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::TetrahedronFEMForceField<CudaVec3fTypes> >()
        .add< component::forcefield::TetrahedronFEMForceField<CudaVec3f1Types> >()
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::forcefield::TetrahedronFEMForceField<CudaVec3dTypes> >()
        .add< component::forcefield::TetrahedronFEMForceField<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
