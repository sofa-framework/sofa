#include "CudaTypes.h"
#include "CudaUniformMass.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaUniformMass)

int UniformMassCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::mass::UniformMass<CudaVec3fTypes,float> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
