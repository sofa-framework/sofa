#ifndef SOFA_GPU_CUDA_CUDAFIXEDCONSTRAINT_INL
#define SOFA_GPU_CUDA_CUDAFIXEDCONSTRAINT_INL

#include "CudaFixedConstraint.h"
#include <sofa/component/constraint/FixedConstraint.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace constraint
{

using namespace gpu::cuda;

template <>
void FixedConstraint<CudaVec3fTypes>::init()
{
    this->core::componentmodel::behavior::Constraint<CudaVec3fTypes>::init();
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

template <>
void FixedConstraint<gpu::cuda::CudaVec3fTypes>::addConstraint(unsigned int index)
{
    //std::cout << "CudaFixedConstraint::addConstraint("<<index<<")\n";
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
    if (data.cudaIndices.empty())
    {
        if (data.minIndex == -1)
        {
            //std::cout << "CudaFixedConstraint: single index "<<index<<"\n";
            data.minIndex = index;
            data.maxIndex = index;
        }
        else if (data.minIndex == (int)index+1)
        {
            data.minIndex = index;
            //std::cout << "CudaFixedConstraint: new min index "<<index<<"\n";
        }
        else if (data.maxIndex == (int)index-1)
        {
            data.maxIndex = index;
            //std::cout << "CudaFixedConstraint: new max index "<<index<<"\n";
        }
        else
        {
            data.cudaIndices.reserve(data.maxIndex-data.minIndex+2);
            for (int i=data.minIndex; i<data.maxIndex; i++)
                data.cudaIndices.push_back(i);
            data.cudaIndices.push_back(index);
            data.minIndex = -1;
            data.maxIndex = -1;
            //std::cout << "CudaFixedConstraint: new indices array size "<<data.cudaIndices.size()<<"\n";
        }
    }
    else
    {
        data.cudaIndices.push_back(index);
        //std::cout << "CudaFixedConstraint: indices array size "<<data.cudaIndices.size()<<"\n";
    }
}

template <>
void FixedConstraint<gpu::cuda::CudaVec3fTypes>::removeConstraint(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
    if (data.cudaIndices.empty())
    {
        if (data.minIndex <= (int)index && (int)index <= data.maxIndex)
        {
            if (data.minIndex == (int)index)
            {
                if (data.maxIndex == (int)index)
                {
                    // empty set
                    data.minIndex = -1;
                    data.maxIndex = -1;
                }
                else
                    ++data.minIndex;
            }
            else if (data.maxIndex == (int)index)
                --data.maxIndex;
            else
            {
                data.cudaIndices.reserve(data.maxIndex-data.minIndex);
                for (int i=data.minIndex; i<data.maxIndex; i++)
                    if (i != (int)index)
                        data.cudaIndices.push_back(i);
                data.minIndex = -1;
                data.maxIndex = -1;
            }
        }
    }
    else
    {
        bool found = false;
        for (unsigned int i=0; i<data.cudaIndices.size(); i++)
        {
            if (found)
                data.cudaIndices[i-1] = data.cudaIndices[i];
            else if (data.cudaIndices[i] == (int)index)
                found = true;
        }
        if (found)
            data.cudaIndices.resize(data.cudaIndices.size()-1);
    }
}

// -- Constraint interface
template <>
void FixedConstraint<gpu::cuda::CudaVec3fTypes>::projectResponse(VecDeriv& dx)
{
    if (data.minIndex >= 0)
        FixedConstraintCuda3f_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+3*data.minIndex);
    else
        FixedConstraintCuda3f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
