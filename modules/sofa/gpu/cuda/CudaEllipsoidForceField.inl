#ifndef SOFA_GPU_CUDA_CUDAELLIPSOIDFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAELLIPSOIDFORCEFIELD_INL

#include "CudaEllipsoidForceField.h"
#include <sofa/component/forcefield/EllipsoidForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void EllipsoidForceFieldCuda3f_addForce(unsigned int size, GPUEllipsoid* ellipsoid, void* tmp, void* f, const void* x, const void* v);
    void EllipsoidForceFieldCuda3f_addDForce(unsigned int size, GPUEllipsoid* ellipsoid, const void* tmp, void* f, const void* dx); //, const void* dfdx);

    void EllipsoidForceFieldCuda3f1_addForce(unsigned int size, GPUEllipsoid* ellipsoid, void* tmp, void* f, const void* x, const void* v);
    void EllipsoidForceFieldCuda3f1_addDForce(unsigned int size, GPUEllipsoid* ellipsoid, const void* tmp, void* f, const void* dx); //, const void* dfdx);

    int EllipsoidForceFieldCuda3f_getNTmp();
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;


template <>
void EllipsoidForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    const Coord center = this->center.getValue();
    const Coord r = this->vradius.getValue();
    const Real stiffness = this->stiffness.getValue();
    const Real damping = this->damping.getValue();
    data.ellipsoid.center = center;
    for (int i=0; i<3; ++i)
        data.ellipsoid.inv_r2[i] = 1/(r[i]*r[i]);
    data.ellipsoid.stiffness = stiffness;
    data.ellipsoid.damping = damping;
    f.resize(x.size());
    data.tmp.resize(x.size()*EllipsoidForceFieldCuda3f_getNTmp());
    EllipsoidForceFieldCuda3f_addForce(x.size(), &data.ellipsoid, data.tmp.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void EllipsoidForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx)
{
    df.resize(dx.size());
    EllipsoidForceFieldCuda3f_addDForce(dx.size(), &data.ellipsoid, data.tmp.deviceRead(), df.deviceWrite(), dx.deviceRead());
}


template <>
void EllipsoidForceField<gpu::cuda::CudaVec3f1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    const Coord center = this->center.getValue();
    const Coord r = this->vradius.getValue();
    const Real stiffness = this->stiffness.getValue();
    const Real damping = this->damping.getValue();
    data.ellipsoid.center = center;
    for (int i=0; i<3; ++i)
        data.ellipsoid.inv_r2[i] = 1/(r[i]*r[i]);
    data.ellipsoid.stiffness = stiffness;
    data.ellipsoid.damping = damping;
    f.resize(x.size());
    data.tmp.resize(x.size()*EllipsoidForceFieldCuda3f_getNTmp());
    EllipsoidForceFieldCuda3f1_addForce(x.size(), &data.ellipsoid, data.tmp.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void EllipsoidForceField<gpu::cuda::CudaVec3f1Types>::addDForce(VecDeriv& df, const VecCoord& dx)
{
    df.resize(dx.size());
    EllipsoidForceFieldCuda3f1_addDForce(dx.size(), &data.ellipsoid, data.tmp.deviceRead(), df.deviceWrite(), dx.deviceRead());
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
