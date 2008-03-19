#include "CudaTypes.h"
#include "CudaEllipsoidForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaEllipsoidForceField)

int EllipsoidForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::EllipsoidForceField<CudaVec3fTypes> >()
        .add< component::forcefield::EllipsoidForceField<CudaVec3f1Types> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
