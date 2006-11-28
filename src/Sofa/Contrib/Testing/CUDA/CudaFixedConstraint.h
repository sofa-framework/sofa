#ifndef SOFA_CONTRIB_CUDA_CUDAFIXEDCONSTRAINT_H
#define SOFA_CONTRIB_CUDA_CUDAFIXEDCONSTRAINT_H

#include "CudaTypes.h"
#include "Sofa/Components/FixedConstraint.h"

namespace Sofa
{

namespace Components
{

template <>
class FixedConstraintInternalData<Contrib::CUDA::CudaVec3fTypes>
{
public:
    // min/max fixed indices for contiguous constraints
    int minIndex;
    int maxIndex;
    // vector of indices for general case
    Contrib::CUDA::CudaVector<int> cudaIndices;
};

template <>
void FixedConstraint<Contrib::CUDA::CudaVec3fTypes>::init();

// -- Constraint interface
template <>
void FixedConstraint<Contrib::CUDA::CudaVec3fTypes>::projectResponse(VecDeriv& dx);

} // namespace Components

} // namespace Sofa

#endif
