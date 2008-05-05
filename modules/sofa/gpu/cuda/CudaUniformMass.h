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


} // namespace mass

} // namespace component

} // namespace sofa

#endif

#endif
