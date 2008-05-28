#ifndef SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_INL

#include "CudaPlaneForceField.h"
#include <sofa/component/forcefield/PlaneForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{

    void PlaneForceFieldCuda3f_addForce(unsigned int size, GPUPlane<float>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f_addDForce(unsigned int size, GPUPlane<float>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void PlaneForceFieldCuda3f1_addForce(unsigned int size, GPUPlane<float>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f1_addDForce(unsigned int size, GPUPlane<float>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

    void PlaneForceFieldCuda3d_addForce(unsigned int size, GPUPlane<double>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d_addDForce(unsigned int size, GPUPlane<double>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void PlaneForceFieldCuda3d1_addForce(unsigned int size, GPUPlane<double>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d1_addDForce(unsigned int size, GPUPlane<double>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;


template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3f_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= kFactor;
    PlaneForceFieldCuda3f_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = stiff;
}


template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3f1_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= kFactor;
    PlaneForceFieldCuda3f1_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = stiff;
}

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3d_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= kFactor;
    PlaneForceFieldCuda3d_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = stiff;
}


template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3d1_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= kFactor;
    PlaneForceFieldCuda3d1_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = stiff;
}

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
