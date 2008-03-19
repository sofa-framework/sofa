#ifndef SOFA_GPU_CUDA_CUDAELLIPSOIDFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDAELLIPSOIDFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/EllipsoidForceField.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

struct GPUEllipsoid
{
    defaulttype::Vec3f center;
    defaulttype::Vec3f inv_r2;
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
class EllipsoidForceFieldInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::GPUEllipsoid ellipsoid;
    gpu::cuda::CudaVector<float> tmp;
};

template <>
void EllipsoidForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void EllipsoidForceField<gpu::cuda::CudaVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx);

template <>
class EllipsoidForceFieldInternalData<gpu::cuda::CudaVec3f1Types>
{
public:
    gpu::cuda::GPUEllipsoid ellipsoid;
    gpu::cuda::CudaVector<float> tmp;
};

template <>
void EllipsoidForceField<gpu::cuda::CudaVec3f1Types>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void EllipsoidForceField<gpu::cuda::CudaVec3f1Types>::addDForce (VecDeriv& df, const VecDeriv& dx);

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
