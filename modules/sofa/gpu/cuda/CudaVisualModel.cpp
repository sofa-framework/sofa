#include "CudaTypes.h"
#include "CudaVisualModel.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaCudaVisualModel)

int CudaVisualModelClass = core::RegisterObject("Rendering of meshes based on CUDA")
        .add< component::visualmodel::CudaVisualModel<CudaVec3fTypes> >()
        .add< component::visualmodel::CudaVisualModel<CudaVec3f1Types> >()
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::visualmodel::CudaVisualModel<CudaVec3dTypes> >()
        .add< component::visualmodel::CudaVisualModel<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
