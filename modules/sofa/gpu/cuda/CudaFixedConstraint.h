#ifndef SOFA_GPU_CUDA_CUDAFIXEDCONSTRAINT_H
#define SOFA_GPU_CUDA_CUDAFIXEDCONSTRAINT_H

#include "CudaTypes.h"
#include <sofa/component/constraint/FixedConstraint.h>

namespace sofa
{

namespace component
{

namespace constraint
{

template <>
class FixedConstraintInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    // min/max fixed indices for contiguous constraints
    int minIndex;
    int maxIndex;
    // vector of indices for general case
    gpu::cuda::CudaVector<int> cudaIndices;
};

template <>
void FixedConstraint<gpu::cuda::CudaVec3fTypes>::init();

template <>
void FixedConstraint<gpu::cuda::CudaVec3fTypes>::addConstraint(unsigned int index);

template <>
void FixedConstraint<gpu::cuda::CudaVec3fTypes>::removeConstraint(unsigned int index);

// -- Constraint interface
template <>
void FixedConstraint<gpu::cuda::CudaVec3fTypes>::projectResponse(VecDeriv& dx);

template <>
class FixedConstraintInternalData<gpu::cuda::CudaVec3f1Types>
{
public:
    // min/max fixed indices for contiguous constraints
    int minIndex;
    int maxIndex;
    // vector of indices for general case
    gpu::cuda::CudaVector<int> cudaIndices;
};

template <>
void FixedConstraint<gpu::cuda::CudaVec3f1Types>::init();

template <>
void FixedConstraint<gpu::cuda::CudaVec3f1Types>::addConstraint(unsigned int index);

template <>
void FixedConstraint<gpu::cuda::CudaVec3f1Types>::removeConstraint(unsigned int index);

// -- Constraint interface
template <>
void FixedConstraint<gpu::cuda::CudaVec3f1Types>::projectResponse(VecDeriv& dx);

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
