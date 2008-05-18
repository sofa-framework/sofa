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

template<class real>
struct GPUPlane
{
    defaulttype::Vec<3,real> normal;
    real d;
    real stiffness;
    real damping;
};

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

template<class TCoord, class TDeriv, class TReal>
class PlaneForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef TReal Real;
    gpu::cuda::GPUPlane<Real> plane;
    gpu::cuda::CudaVector<Real> penetration;
};

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx);

template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addDForce (VecDeriv& df, const VecDeriv& dx);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addDForce (VecDeriv& df, const VecDeriv& dx);

template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addDForce (VecDeriv& df, const VecDeriv& dx);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
