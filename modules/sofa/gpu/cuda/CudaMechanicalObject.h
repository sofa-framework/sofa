#ifndef SOFA_CONTRIB_CUDA_CUDAMECHANICALOBJECT_H
#define SOFA_CONTRIB_CUDA_CUDAMECHANICALOBJECT_H

#include "CudaTypes.h"
#include "Sofa-old/Core/MechanicalObject.h"

namespace Sofa
{

namespace Core
{

template <>
class MechanicalObjectInternalData<Contrib::CUDA::CudaVec3fTypes>
{
public:
    /// Temporary storate for dot product operation
    Contrib::CUDA::CudaVec3fTypes::VecDeriv tmpdot;
};

template <>
void MechanicalObject<Contrib::CUDA::CudaVec3fTypes>::accumulateForce();

template <>
void MechanicalObject<Contrib::CUDA::CudaVec3fTypes>::vOp(VecId v, VecId a, VecId b, double f);

template <>
double MechanicalObject<Contrib::CUDA::CudaVec3fTypes>::vDot(VecId a, VecId b);

template <>
void MechanicalObject<Contrib::CUDA::CudaVec3fTypes>::resetForce();

} // namespace Core

} // namespace Sofa

#endif
