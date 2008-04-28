#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_H

#include "CudaTypes.h"
#include <sofa/component/MechanicalObject.h>

namespace sofa
{

namespace component
{

template <>
class MechanicalObjectInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    /// Temporary storate for dot product operation
    gpu::cuda::CudaVec3fTypes::VecDeriv tmpdot;
};

template <>
void MechanicalObject<gpu::cuda::CudaVec3fTypes>::accumulateForce();

template <>
void MechanicalObject<gpu::cuda::CudaVec3fTypes>::vOp(VecId v, VecId a, VecId b, double f);

template <>
void MechanicalObject<gpu::cuda::CudaVec3fTypes>::vMultiOp(const VMultiOp& ops);

template <>
double MechanicalObject<gpu::cuda::CudaVec3fTypes>::vDot(VecId a, VecId b);

template <>
void MechanicalObject<gpu::cuda::CudaVec3fTypes>::resetForce();

template <>
void MechanicalObject<gpu::cuda::CudaVec3fTypes>::getIndicesInSpace(helper::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const;

template <>
class MechanicalObjectInternalData<gpu::cuda::CudaVec3f1Types>
{
public:
    /// Temporary storate for dot product operation
    gpu::cuda::CudaVec3f1Types::VecDeriv tmpdot;
};

template <>
void MechanicalObject<gpu::cuda::CudaVec3f1Types>::accumulateForce();

template <>
void MechanicalObject<gpu::cuda::CudaVec3f1Types>::vOp(VecId v, VecId a, VecId b, double f);

template <>
void MechanicalObject<gpu::cuda::CudaVec3f1Types>::vMultiOp(const VMultiOp& ops);

template <>
double MechanicalObject<gpu::cuda::CudaVec3f1Types>::vDot(VecId a, VecId b);

template <>
void MechanicalObject<gpu::cuda::CudaVec3f1Types>::resetForce();

template <>
void MechanicalObject<gpu::cuda::CudaVec3f1Types>::getIndicesInSpace(helper::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const;

} // namespace component

} // namespace sofa

#endif
