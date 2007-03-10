#ifndef SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/PlaneForceField.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

struct GPUPlane
{
    defaulttype::Vec3f normal;
    float d;
    float stiffness;
    float damping;
};

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

template <>
class PlaneForceFieldInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::GPUPlane plane;
    gpu::cuda::CudaVector<float> penetration;
};

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx);

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
