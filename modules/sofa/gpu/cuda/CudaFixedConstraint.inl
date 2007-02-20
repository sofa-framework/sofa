#ifndef SOFA_CONTRIB_CUDA_CUDAFIXEDCONSTRAINT_INL
#define SOFA_CONTRIB_CUDA_CUDAFIXEDCONSTRAINT_INL

#include "CudaFixedConstraint.h"
#include "Sofa-old/Components/FixedConstraint.inl"

namespace Sofa
{

namespace Contrib
{

namespace CUDA
{

extern "C"
{
    void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
}

} // namespace CUDA

} // namespace Contrib

namespace Components
{

using namespace Contrib::CUDA;

template <>
void FixedConstraint<CudaVec3fTypes>::init()
{
    this->Core::Constraint<CudaVec3fTypes>::init();
    const SetIndex& indices = f_indices.getValue();
    data.minIndex = -1;
    data.maxIndex = -1;
    data.cudaIndices.clear();
    if (!indices.empty())
    {
        // put indices in a set to sort them and remove duplicates
        std::set<int> sortedIndices;
        for (SetIndex::const_iterator it = indices.begin(); it!=indices.end(); it++)
            sortedIndices.insert(*it);
        // check if the indices are contiguous
        if (*sortedIndices.begin() + (int)sortedIndices.size()-1 == *sortedIndices.rbegin())
        {
            data.minIndex = *sortedIndices.begin();
            data.maxIndex = *sortedIndices.rbegin();
        }
        else
        {
            data.cudaIndices.reserve(sortedIndices.size());
            for (std::set<int>::const_iterator it = sortedIndices.begin(); it!=sortedIndices.end(); it++)
                data.cudaIndices.push_back(*it);
        }
    }
}

// -- Constraint interface
template <>
void FixedConstraint<CudaVec3fTypes>::projectResponse(VecDeriv& dx)
{
    if (data.minIndex >= 0)
        FixedConstraintCuda3f_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+3*data.minIndex);
    else
        FixedConstraintCuda3f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

} // namespace Components

} // namespace Sofa

#endif
