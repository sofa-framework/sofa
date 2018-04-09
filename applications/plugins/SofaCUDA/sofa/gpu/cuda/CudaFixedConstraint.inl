/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDAFIXEDCONSTRAINT_INL
#define SOFA_GPU_CUDA_CUDAFIXEDCONSTRAINT_INL

#include "CudaFixedConstraint.h"
#include <SofaBoundaryCondition/FixedConstraint.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void FixedConstraintCuda1f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda1f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCudaRigid3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCudaRigid3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void FixedConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedConstraintCudaRigid3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedConstraintCudaRigid3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace projectiveconstraintset
{

using namespace gpu::cuda;

template<class TCoord, class TDeriv, class TReal>
void FixedConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::init(Main* m)
{
    Data& data = *m->data;
    data.minIndex = -1;
    data.maxIndex = -1;
    data.cudaIndices.clear();
    m->core::behavior::template ProjectiveConstraintSet<DataTypes>::init();
    const SetIndexArray& indices = m->d_indices.getValue();
    if (!indices.empty())
    {
        // put indices in a set to sort them and remove duplicates
        std::set<int> sortedIndices;
        for (typename SetIndex::const_iterator it = indices.begin(); it!=indices.end(); ++it)
            sortedIndices.insert(*it);
        // check if the indices are contiguous
        if (*sortedIndices.begin() + (int)sortedIndices.size()-1 == *sortedIndices.rbegin())
        {
            data.minIndex = *sortedIndices.begin();
            data.maxIndex = *sortedIndices.rbegin();
            //std::cout << "CudaFixedConstraint: "<<sortedIndices.size()<<" contiguous fixed indices, "<<data.minIndex<<" - "<<data.maxIndex<<sendl;
        }
        else
        {
            //std::cout << "CudaFixedConstraint: "<<sortedIndices.size()<<" non-contiguous fixed indices"<<sendl;
            data.cudaIndices.reserve(sortedIndices.size());
            for (std::set<int>::const_iterator it = sortedIndices.begin(); it!=sortedIndices.end(); ++it)
                data.cudaIndices.push_back(*it);
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
void FixedConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addConstraint(Main* m, unsigned int index)
{
    Data& data = *m->data;
    //std::cout << "CudaFixedConstraint::addConstraint("<<index<<")\n";
    m->d_indices.beginEdit()->push_back(index);
    m->d_indices.endEdit();
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
            for (int i=data.minIndex; i<data.maxIndex; ++i)
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
    Data& data = *m->data;
    removeValue(*m->d_indices.beginEdit(),index);
    m->d_indices.endEdit();
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
                for (int i=data.minIndex; i<data.maxIndex; ++i)
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
        for (unsigned int i=0; i<data.cudaIndices.size(); ++i)
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

template<int N, class real>
void FixedConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::init(Main* m)
{
    Data& data = *m->data;
    data.minIndex = -1;
    data.maxIndex = -1;
    data.cudaIndices.clear();
    m->core::behavior::template ProjectiveConstraintSet<DataTypes>::init();
    const SetIndexArray& indices = m->d_indices.getValue();
    if (!indices.empty())
    {
        // put indices in a set to sort them and remove duplicates
        std::set<int> sortedIndices;
        for (typename SetIndex::const_iterator it = indices.begin(); it!=indices.end(); ++it)
            sortedIndices.insert(*it);
        // check if the indices are contiguous
        if (*sortedIndices.begin() + (int)sortedIndices.size()-1 == *sortedIndices.rbegin())
        {
            data.minIndex = *sortedIndices.begin();
            data.maxIndex = *sortedIndices.rbegin();
            //std::cout << "CudaFixedConstraint: "<<sortedIndices.size()<<" contiguous fixed indices, "<<data.minIndex<<" - "<<data.maxIndex<<sendl;
        }
        else
        {
            //std::cout << "CudaFixedConstraint: "<<sortedIndices.size()<<" non-contiguous fixed indices"<<sendl;
            data.cudaIndices.reserve(sortedIndices.size());
            for (std::set<int>::const_iterator it = sortedIndices.begin(); it!=sortedIndices.end(); ++it)
                data.cudaIndices.push_back(*it);
        }
    }
}

template<int N, class real>
void FixedConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::addConstraint(Main* m, unsigned int index)
{
    Data& data = *m->data;
    //std::cout << "CudaFixedConstraint::addConstraint("<<index<<")\n";
    m->d_indices.beginEdit()->push_back(index);
    m->d_indices.endEdit();
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
            for (int i=data.minIndex; i<data.maxIndex; ++i)
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

template<int N, class real>
void FixedConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::removeConstraint(Main* m, unsigned int index)
{
    Data& data = *m->data;
    removeValue(*m->d_indices.beginEdit(),index);
    m->d_indices.endEdit();
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
                for (int i=data.minIndex; i<data.maxIndex; ++i)
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
        for (unsigned int i=0; i<data.cudaIndices.size(); ++i)
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
void FixedConstraintInternalData<gpu::cuda::CudaVec1fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedConstraintCuda1f_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda1f_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+data.minIndex);
    else
        FixedConstraintCuda1f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}


template <>
void FixedConstraintInternalData<gpu::cuda::CudaVec3fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedConstraintCuda3f_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda3f_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+3*data.minIndex);
    else
        FixedConstraintCuda3f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}


template <>
void FixedConstraintInternalData<gpu::cuda::CudaVec3f1Types>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedConstraintCuda3f1_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda3f1_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+4*data.minIndex);
    else
        FixedConstraintCuda3f1_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedConstraintInternalData<gpu::cuda::CudaRigid3fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedConstraintCudaRigid3f_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCudaRigid3f_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+6*data.minIndex);
    else
        FixedConstraintCudaRigid3f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}



#ifdef SOFA_GPU_CUDA_DOUBLE

// // Handle topological changes
// template <>
// void FixedConstraint<gpu::cuda::CudaVec3dTypes>::handleTopologyChange() {
// // 	std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
// // 	std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();
// //
// // 	d_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());
// //printf("WARNING handleTopologyChange<gpu::cuda::CudaVec3dTypes> not implemented\n");
// }

template <>
void FixedConstraintInternalData<gpu::cuda::CudaVec3dTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedConstraintCuda3d_projectResponseContiguous(dx.size(), ((double*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda3d_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((double*)dx.deviceWrite())+3*data.minIndex);
    else
        FixedConstraintCuda3d_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedConstraintInternalData<gpu::cuda::CudaVec3d1Types>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedConstraintCuda3d1_projectResponseContiguous(dx.size(), ((double*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCuda3d1_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((double*)dx.deviceWrite())+4*data.minIndex);
    else
        FixedConstraintCuda3d1_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedConstraintInternalData<gpu::cuda::CudaRigid3dTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedConstraintCudaRigid3d_projectResponseContiguous(dx.size(), ((double*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintCudaRigid3d_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((double*)dx.deviceWrite())+6*data.minIndex);
    else
        FixedConstraintCudaRigid3d_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

#endif // SOFA_GPU_CUDA_DOUBLE

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaFixedConstraint_ImplMethods(T) \
    template<> void FixedConstraint< T >::init() \
    { data->init(this); } \
    template<> void FixedConstraint< T >::addConstraint(unsigned int index) \
    { data->addConstraint(this, index); } \
    template<> void FixedConstraint< T >::removeConstraint(unsigned int index) \
    { data->removeConstraint(this, index); } \
    template<> void FixedConstraint< T >::projectResponse(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_resData) \
    {  \
		VecDeriv &resData = *d_resData.beginEdit(); \
		data->projectResponse(this, resData);  \
		d_resData.endEdit(); \
	}

CudaFixedConstraint_ImplMethods(gpu::cuda::CudaVec3fTypes);
CudaFixedConstraint_ImplMethods(gpu::cuda::CudaVec3f1Types);
CudaFixedConstraint_ImplMethods(gpu::cuda::CudaRigid3fTypes);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaFixedConstraint_ImplMethods(gpu::cuda::CudaVec3dTypes);
CudaFixedConstraint_ImplMethods(gpu::cuda::CudaVec3d1Types);
CudaFixedConstraint_ImplMethods(gpu::cuda::CudaRigid3dTypes);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaFixedConstraint_ImplMethods

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
