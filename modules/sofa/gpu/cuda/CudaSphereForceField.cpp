#include "CudaTypes.h"
#include "CudaSphereForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaSphereForceField)

int SphereForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::SphereForceField<CudaVec3fTypes> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
