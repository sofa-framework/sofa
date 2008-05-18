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
    void FixedConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

    void FixedConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace constraint
{

using namespace gpu::cuda;

template<class TCoord, class TDeriv, class TReal>
void FixedConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::init(Main* m)
{
    Data& data = m->data;
    data.minIndex = -1;
    data.maxIndex = -1;
    data.cudaIndices.clear();
    m->core::componentmodel::behavior::Constraint<DataTypes>::init();
    const SetIndex& indices = m->f_indices.getValue();
    if (!indices.empty())
    {
        // put indices in a set to sort them and remove duplicates
        std::set<int> sortedIndices;
        for (typename SetIndex::const_iterator it = indices.begin(); it!=indices.end(); it++)
            sortedIndices.insert(*it);
        // check if the indices are contiguous
        if (*sortedIndices.begin() + (int)sortedIndices.size()-1 == *sortedIndices.rbegin())
        {
            data.minIndex = *sortedIndices.begin();
            data.maxIndex = *sortedIndices.rbegin();
            //std::cout << "CudaFixedConstraint: "<<sortedIndices.size()<<" contiguous fixed indices, "<<data.minIndex<<" - "<<data.maxIndex<<std::endl;
        }
        else
        {
            //std::cout << "CudaFixedConstraint: "<<sortedIndices.size()<<" non-contiguous fixed indices"<<std::endl;
            data.cudaIndices.reserve(sortedIndices.size());
            for (std::set<int>::const_iterator it = sortedIndices.begin(); it!=sortedIndices.end(); it++)
                data.cudaIndices.push_back(*it);
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
void FixedConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addConstraint(Main* m, unsigned int index)
{
    Data& data = m->data;
    //std::cout << "CudaFixedConstraint::addConstraint("<<index<<")\n";
    m->f_indices.beginEdit()->push_back(index);
    m->f_indices.endEdit();
    if (data.cudaIndices.empty())
    {
        if (data.minIndex == -1)
        {
            //std::cout << "CudaFixedConstraint: single index "<<index<<"\n";
            data.minIndex = index;
            data.maxIndex = index;
        }
        else if ((int)index >= data.minIndex && (int)index <= data.maxIndex)
        {
            // point already fixed
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
            std::cout << "CudaFixedConstraint: new indices array size "<<data.cudaIndices.size()<<"\n";
        }
    }
    else
    {
        data.cudaIndices.push_back(index);
        //std::cout << "CudaFixedConstraint: indices array size "<<data.cudaIndices.size()<<"\n";
    }
}

template<class TCoord, class TDeriv, class TReal>
void FixedConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::removeConstraint(Main* m, unsigned int index)
{
    Data& data = m->data;
    removeValue(*m->f_indices.beginEdit(),index);
    m->f_indices.endEdit();
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


template <>
void FixedConstraintInternalData<gpu::cuda::CudaVec3fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintCuda3f_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda3f_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+3*data.minIndex);
    else
        FixedConstraintCuda3f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedConstraintInternalData<gpu::cuda::CudaVec3f1Types>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintCuda3f1_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda3f1_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+4*data.minIndex);
    else
        FixedConstraintCuda3f1_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void FixedConstraintInternalData<gpu::cuda::CudaVec3dTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintCuda3d_projectResponseContiguous(dx.size(), ((double*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda3d_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((double*)dx.deviceWrite())+3*data.minIndex);
    else
        FixedConstraintCuda3d_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedConstraintInternalData<gpu::cuda::CudaVec3d1Types>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintCuda3d1_projectResponseContiguous(dx.size(), ((double*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda3d1_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((double*)dx.deviceWrite())+4*data.minIndex);
    else
        FixedConstraintCuda3d1_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaFixedConstraint_ImplMethods(T) \
    template<> void FixedConstraint< T >::init() \
    { data.init(this); } \
    template<> void FixedConstraint< T >::addConstraint(unsigned int index) \
    { data.addConstraint(this, index); } \
    template<> void FixedConstraint< T >::removeConstraint(unsigned int index) \
    { data.removeConstraint(this, index); } \
    template<> void FixedConstraint< T >::projectResponse(VecDeriv& dx) \
    { data.projectResponse(this, dx); }

CudaFixedConstraint_ImplMethods(gpu::cuda::CudaVec3fTypes);
CudaFixedConstraint_ImplMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

CudaFixedConstraint_ImplMethods(gpu::cuda::CudaVec3dTypes);
CudaFixedConstraint_ImplMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

#undef CudaFixedConstraint_ImplMethods

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
