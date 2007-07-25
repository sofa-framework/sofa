#ifndef SOFA_GPU_CUDA_CUDASPHEREMODEL_H
#define SOFA_GPU_CUDA_CUDASPHEREMODEL_H

#include "CudaTypes.h"
#include <sofa/component/collision/SphereModel.h>
#include "CudaMechanicalObject.h"

namespace sofa
{

namespace gpu
{

namespace cuda
{

typedef sofa::component::collision::TSphereModel<gpu::cuda::CudaVec3fTypes> CudaSphereModel;

typedef sofa::component::collision::TSphere<gpu::cuda::CudaVec3fTypes> CudaSphere;

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
