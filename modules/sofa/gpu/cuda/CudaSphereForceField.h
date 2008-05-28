#ifndef SOFA_GPU_CUDA_CUDASPHEREFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDASPHEREFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/SphereForceField.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

struct GPUSphere
{
    defaulttype::Vec3f center;
    float r;
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
class SphereForceFieldInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::GPUSphere sphere;
    gpu::cuda::CudaVector<defaulttype::Vec4f> penetration;
};

template <>
void SphereForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void SphereForceField<gpu::cuda::CudaVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

template <>
class SphereForceFieldInternalData<gpu::cuda::CudaVec3f1Types>
{
public:
    gpu::cuda::GPUSphere sphere;
    gpu::cuda::CudaVector<defaulttype::Vec4f> penetration;
};

template <>
void SphereForceField<gpu::cuda::CudaVec3f1Types>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void SphereForceField<gpu::cuda::CudaVec3f1Types>::addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
