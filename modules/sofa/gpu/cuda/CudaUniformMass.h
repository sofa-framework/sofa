#ifndef SOFA_GPU_CUDA_CUDAUNIFORMMASS_H
#define SOFA_GPU_CUDA_CUDAUNIFORMMASS_H

#ifndef SOFA_DOUBLE //cuda only operates with float

#include "CudaTypes.h"
#include <sofa/component/mass/UniformMass.h>

namespace sofa
{

namespace component
{

namespace mass
{

// -- Mass interface
template <>
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template <>
bool UniformMass<gpu::cuda::CudaVec3fTypes, float>::addBBox(double* minBBox, double* maxBBox);

template <>
double UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getPotentialEnergy( const VecCoord& x );

template <>
double UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getElementMass(unsigned int );

template <>
void UniformMass<gpu::cuda::CudaRigid3fTypes, Rigid3fMass>::draw();

template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template <>
bool UniformMass<gpu::cuda::CudaVec3f1Types, float>::addBBox(double* minBBox, double* maxBBox);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

// -- Mass interface
template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, double>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, double>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, double>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template <>
bool UniformMass<gpu::cuda::CudaVec3dTypes, double>::addBBox(double* minBBox, double* maxBBox);

template <>
double UniformMass<gpu::cuda::CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getPotentialEnergy( const VecCoord& x );

template <>
double UniformMass<gpu::cuda::CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getElementMass(unsigned int );

template <>
void UniformMass<gpu::cuda::CudaRigid3dTypes, Rigid3dMass>::draw();

template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template <>
bool UniformMass<gpu::cuda::CudaVec3d1Types, double>::addBBox(double* minBBox, double* maxBBox);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

} // namespace mass

} // namespace component

} // namespace sofa

#endif

#endif
