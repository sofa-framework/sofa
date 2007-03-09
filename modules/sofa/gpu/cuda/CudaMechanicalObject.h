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
double MechanicalObject<gpu::cuda::CudaVec3fTypes>::vDot(VecId a, VecId b);

template <>
void MechanicalObject<gpu::cuda::CudaVec3fTypes>::resetForce();

template <>
void MechanicalObject<gpu::cuda::CudaVec3fTypes>::getIndicesInSpace(std::vector<unsigned>& indices,Real xmin,Real xmax,Real ymin,Real ymax,Real zmin,Real zmax) const;

} // namespace component

} // namespace sofa

#endif
