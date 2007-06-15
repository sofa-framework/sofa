#include "CudaTypes.h"
#include "CudaExternalForceField.inl"
#include <sofa/core/ObjectFactory.h>
//#include <typeinfo>


namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaExternalForceField)
int ExternalForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")

        .add< component::interactionforcefield::ExternalForceField<CudaVec3fTypes> >();



} // namespace interactionforcefield

} // namespace component

} // namespace sofa

