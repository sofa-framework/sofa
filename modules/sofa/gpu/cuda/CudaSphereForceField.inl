#ifndef SOFA_GPU_CUDA_CUDASPHEREFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDASPHEREFORCEFIELD_INL

#include "CudaSphereForceField.h"
#include <sofa/component/forcefield/SphereForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void SphereForceFieldCuda3f_addForce(unsigned int size, GPUSphere* sphere, void* penetration, void* f, const void* x, const void* v);
    void SphereForceFieldCuda3f_addDForce(unsigned int size, GPUSphere* sphere, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void SphereForceFieldCuda3f1_addForce(unsigned int size, GPUSphere* sphere, void* penetration, void* f, const void* x, const void* v);
    void SphereForceFieldCuda3f1_addDForce(unsigned int size, GPUSphere* sphere, const void* penetration, void* f, const void* dx); //, const void* dfdx);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;


template <>
void SphereForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.sphere.center = sphereCenter.getValue();
    data.sphere.r = sphereRadius.getValue();
    data.sphere.stiffness = stiffness.getValue();
    data.sphere.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    SphereForceFieldCuda3f_addForce(x.size(), &data.sphere, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void SphereForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.sphere.stiffness;
    data.sphere.stiffness *= kFactor;
    SphereForceFieldCuda3f_addDForce(dx.size(), &data.sphere, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.sphere.stiffness = stiff;
}


template <>
void SphereForceField<gpu::cuda::CudaVec3f1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.sphere.center = sphereCenter.getValue();
    data.sphere.r = sphereRadius.getValue();
    data.sphere.stiffness = stiffness.getValue();
    data.sphere.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    SphereForceFieldCuda3f1_addForce(x.size(), &data.sphere, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void SphereForceField<gpu::cuda::CudaVec3f1Types>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.sphere.stiffness;
    data.sphere.stiffness *= kFactor;
    SphereForceFieldCuda3f1_addDForce(dx.size(), &data.sphere, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.sphere.stiffness = stiff;
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
