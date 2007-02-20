#ifndef SOFA_CONTRIB_CUDA_CUDAUNIFORMMASS_H
#define SOFA_CONTRIB_CUDA_CUDAUNIFORMMASS_H

#include "CudaTypes.h"
#include "Sofa-old/Components/UniformMass.h"

namespace Sofa
{

namespace Components
{

// -- Mass interface
template <>
void UniformMass<Contrib::CUDA::CudaVec3fTypes, float>::addMDx(VecDeriv& res, const VecDeriv& dx);

template <>
void UniformMass<Contrib::CUDA::CudaVec3fTypes, float>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<Contrib::CUDA::CudaVec3fTypes, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

} // namespace Components

} // namespace Sofa

#endif
