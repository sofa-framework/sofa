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
        .add< component::mass::UniformMass<CudaVec3f1Types,float> >()
        .add< component::mass::UniformMass<CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass> >()
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::mass::UniformMass<CudaVec3dTypes,double> >()
        .add< component::mass::UniformMass<CudaVec3d1Types,double> >()
        .add< component::mass::UniformMass<CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
