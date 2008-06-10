#include "CudaTypes.h"
#include "CudaPenalityContactForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaPenalityContactForceField)

int PenalityContactForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::forcefield::PenalityContactForceField<CudaVec3fTypes> >()
        .add< component::forcefield::PenalityContactForceField<CudaVec3f1Types> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
