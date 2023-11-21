/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <SofaCUDA/component/constraint/projective/CudaFixedProjectiveConstraint.h>
#include <sofa/component/constraint/projective/FixedProjectiveConstraint.inl>

namespace sofa::gpu::cuda
{

extern "C"
{
    void FixedProjectiveConstraintCuda1f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda1f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCuda3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCuda3f1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda3f1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCudaRigid3f_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCudaRigid3f_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void FixedProjectiveConstraintCuda3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCuda3d1_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCuda3d1_projectResponseIndexed(unsigned int size, const void* indices, void* dx);
    void FixedProjectiveConstraintCudaRigid3d_projectResponseContiguous(unsigned int size, void* dx);
    void FixedProjectiveConstraintCudaRigid3d_projectResponseIndexed(unsigned int size, const void* indices, void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

} // namespace sofa::gpu::cuda

namespace sofa::component::constraint::projective
{

using namespace gpu::cuda;

template<class TCoord, class TDeriv, class TReal>
void FixedProjectiveConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::init(Main* m)
{
    Data& data = *m->data;
    data.minIndex = sofa::InvalidID;
    data.maxIndex = sofa::InvalidID;
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
            msg_info("CudaFixedProjectiveConstraint") << "init: " << sortedIndices.size() << " contiguous fixed indices, " << data.minIndex << " - " << data.maxIndex;
        }
        else
        {
            msg_info("CudaFixedProjectiveConstraint") << "init: " << sortedIndices.size() << " non-contiguous fixed indices";
            data.cudaIndices.reserve(sortedIndices.size());
            for (std::set<int>::const_iterator it = sortedIndices.begin(); it!=sortedIndices.end(); ++it)
                data.cudaIndices.push_back(*it);
        }
    }

    m->d_componentState.setValue(ComponentState::Valid);
}

template<class TCoord, class TDeriv, class TReal>
void FixedProjectiveConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addConstraint(Main* m, Index index)
{
    Data& data = *m->data;
    //std::cout << "CudaFixedProjectiveConstraint::addConstraint("<<index<<")\n";
    m->d_indices.beginEdit()->push_back(index);
    m->d_indices.endEdit();
    if (data.cudaIndices.empty())
    {
        if (data.minIndex == sofa::InvalidID)
        {
            //std::cout << "CudaFixedProjectiveConstraint: single index "<<index<<"\n";
            data.minIndex = index;
            data.maxIndex = index;
        }
        else if (index >= data.minIndex && index <= data.maxIndex)
        {
            // point already fixed
        }
        else if (data.minIndex == index + 1)
        {
            data.minIndex = index;
            //std::cout << "CudaFixedProjectiveConstraint: new min index "<<index<<"\n";
        }
        else if (data.maxIndex == sofa::InvalidID || data.maxIndex == index - 1)
        {
            data.maxIndex = index;
            //std::cout << "CudaFixedProjectiveConstraint: new max index "<<index<<"\n";
        }
        else
        {
            data.cudaIndices.reserve(data.maxIndex - data.minIndex + 2);
            for (int i = data.minIndex; i < data.maxIndex; ++i)
                data.cudaIndices.push_back(i);
            data.cudaIndices.push_back(index);
            data.minIndex = sofa::InvalidID;
            data.maxIndex = sofa::InvalidID;
            msg_info("CudaFixedProjectiveConstraint") << "new indices array size " << data.cudaIndices.size() << "\n";
        }
    }
    else
    {
        data.cudaIndices.push_back(index);
        //std::cout << "CudaFixedProjectiveConstraint: indices array size "<<data.cudaIndices.size()<<"\n";
    }
}

template<class TCoord, class TDeriv, class TReal>
void FixedProjectiveConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::removeConstraint(Main* m, Index index)
{
    Data& data = *m->data;
    removeValue(*m->d_indices.beginEdit(),index);
    m->d_indices.endEdit();
    if (data.cudaIndices.empty())
    {
        if (data.minIndex <= index && index <= data.maxIndex)
        {
            if (data.minIndex == index)
            {
                if (data.maxIndex == index)
                {
                    // empty set
                    data.minIndex = sofa::InvalidID;
                    data.maxIndex = sofa::InvalidID;
                }
                else
                    ++data.minIndex;
            }
            else if (data.maxIndex == index)
                --data.maxIndex;
            else
            {
                data.cudaIndices.reserve(data.maxIndex-data.minIndex);
                for (int i=data.minIndex; i<data.maxIndex; ++i)
                    if (i != index)
                        data.cudaIndices.push_back(i);
                data.minIndex = sofa::InvalidID;
                data.maxIndex = sofa::InvalidID;
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
            else if (data.cudaIndices[i] == index)
                found = true;
        }
        if (found)
            data.cudaIndices.resize(data.cudaIndices.size()-1);
    }
}

template<int N, class real>
void FixedProjectiveConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::init(Main* m)
{
    Data& data = *m->data;
    data.minIndex = sofa::InvalidID;
    data.maxIndex = sofa::InvalidID;
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
            msg_info("CudaFixedProjectiveConstraint") << "init: " << sortedIndices.size() << " contiguous fixed indices, " << data.minIndex << " - " << data.maxIndex;
        }
        else
        {
            msg_info("CudaFixedProjectiveConstraint") << "init: " << sortedIndices.size() << " non-contiguous fixed indices";
            data.cudaIndices.reserve(sortedIndices.size());
            for (std::set<int>::const_iterator it = sortedIndices.begin(); it!=sortedIndices.end(); ++it)
                data.cudaIndices.push_back(*it);
        }
    }

    m->d_componentState.setValue(ComponentState::Valid);
}

template<int N, class real>
void FixedProjectiveConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::addConstraint(Main* m, Index index)
{
    Data& data = *m->data;
    //std::cout << "CudaFixedProjectiveConstraint::addConstraint("<<index<<")\n";
    m->d_indices.beginEdit()->push_back(index);
    m->d_indices.endEdit();
    if (data.cudaIndices.empty())
    {
        if (data.minIndex == sofa::InvalidID)
        {
            //std::cout << "CudaFixedProjectiveConstraint: single index "<<index<<"\n";
            data.minIndex = index;
            data.maxIndex = index;
        }
        else if (index >= data.minIndex && index <= data.maxIndex)
        {
            // point already fixed
        }
        else if (data.minIndex == index+1)
        {
            data.minIndex = index;
            //std::cout << "CudaFixedProjectiveConstraint: new min index "<<index<<"\n";
        }
        else if (data.maxIndex == sofa::InvalidID || data.maxIndex == index-1)
        {
            data.maxIndex = index;
            //std::cout << "CudaFixedProjectiveConstraint: new max index "<<index<<"\n";
        }
        else
        {
            data.cudaIndices.reserve(data.maxIndex-data.minIndex+2);
            for (int i=data.minIndex; i<data.maxIndex; ++i)
                data.cudaIndices.push_back(i);
            data.cudaIndices.push_back(index);
            data.minIndex = sofa::InvalidID;
            data.maxIndex = sofa::InvalidID;
            msg_info("CudaFixedProjectiveConstraint") << "new indices array size "<<data.cudaIndices.size()<<"\n";
        }
    }
    else
    {
        data.cudaIndices.push_back(index);
        //std::cout << "CudaFixedProjectiveConstraint: indices array size "<<data.cudaIndices.size()<<"\n";
    }
}

template<int N, class real>
void FixedProjectiveConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::removeConstraint(Main* m, Index index)
{
    Data& data = *m->data;
    removeValue(*m->d_indices.beginEdit(),index);
    m->d_indices.endEdit();
    if (data.cudaIndices.empty())
    {
        if (data.minIndex <= index && index <= data.maxIndex)
        {
            if (data.minIndex == index)
            {
                if (data.maxIndex == index)
                {
                    // empty set
                    data.minIndex = sofa::InvalidID;
                    data.maxIndex = sofa::InvalidID;
                }
                else
                    ++data.minIndex;
            }
            else if (data.maxIndex == index)
                --data.maxIndex;
            else
            {
                data.cudaIndices.reserve(data.maxIndex-data.minIndex);
                for (int i=data.minIndex; i<data.maxIndex; ++i)
                    if (i != index)
                        data.cudaIndices.push_back(i);
                data.minIndex = sofa::InvalidID;
                data.maxIndex = sofa::InvalidID;
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
            else if (data.cudaIndices[i] == index)
                found = true;
        }
        if (found)
            data.cudaIndices.resize(data.cudaIndices.size()-1);
    }
}



template <>
void FixedProjectiveConstraintInternalData<gpu::cuda::CudaVec1fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    const Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedProjectiveConstraintCuda1f_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedProjectiveConstraintCuda1f_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+data.minIndex);
    else
        FixedProjectiveConstraintCuda1f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}


template <>
void FixedProjectiveConstraintInternalData<gpu::cuda::CudaVec3fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    const Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedProjectiveConstraintCuda3f_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));        
    else if (data.minIndex != sofa::InvalidID)
        FixedProjectiveConstraintCuda3f_projectResponseContiguous(data.maxIndex - data.minIndex + 1, ((float*)dx.deviceWrite()) + 3 * data.minIndex);
    else
        FixedProjectiveConstraintCuda3f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}


template <>
void FixedProjectiveConstraintInternalData<gpu::cuda::CudaVec3f1Types>::projectResponse(Main* m, VecDeriv& dx)
{
    const Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedProjectiveConstraintCuda3f1_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex != sofa::InvalidID)
        FixedProjectiveConstraintCuda3f1_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+4*data.minIndex);
    else
        FixedProjectiveConstraintCuda3f1_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedProjectiveConstraintInternalData<gpu::cuda::CudaRigid3fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    const Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedProjectiveConstraintCudaRigid3f_projectResponseContiguous(dx.size(), ((float*)dx.deviceWrite()));
    else if (data.minIndex != sofa::InvalidID)
        FixedProjectiveConstraintCudaRigid3f_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((float*)dx.deviceWrite())+6*data.minIndex);
    else
        FixedProjectiveConstraintCudaRigid3f_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}



#ifdef SOFA_GPU_CUDA_DOUBLE

// // Handle topological changes
// template <>
// void FixedProjectiveConstraint<gpu::cuda::CudaVec3dTypes>::handleTopologyChange() {
// // 	std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
// // 	std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();
// //
// // 	d_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());
// //printf("WARNING handleTopologyChange<gpu::cuda::CudaVec3dTypes> not implemented\n");
// }

template <>
void FixedProjectiveConstraintInternalData<gpu::cuda::CudaVec3dTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedProjectiveConstraintCuda3d_projectResponseContiguous(dx.size(), ((double*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedProjectiveConstraintCuda3d_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((double*)dx.deviceWrite())+3*data.minIndex);
    else
        FixedProjectiveConstraintCuda3d_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedProjectiveConstraintInternalData<gpu::cuda::CudaVec3d1Types>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedProjectiveConstraintCuda3d1_projectResponseContiguous(dx.size(), ((double*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedProjectiveConstraintCuda3d1_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((double*)dx.deviceWrite())+4*data.minIndex);
    else
        FixedProjectiveConstraintCuda3d1_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedProjectiveConstraintInternalData<gpu::cuda::CudaRigid3dTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->d_fixAll.getValue())
        FixedProjectiveConstraintCudaRigid3d_projectResponseContiguous(dx.size(), ((double*)dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedProjectiveConstraintCudaRigid3d_projectResponseContiguous(data.maxIndex-data.minIndex+1, ((double*)dx.deviceWrite())+6*data.minIndex);
    else
        FixedProjectiveConstraintCudaRigid3d_projectResponseIndexed(data.cudaIndices.size(), data.cudaIndices.deviceRead(), dx.deviceWrite());
}

#endif // SOFA_GPU_CUDA_DOUBLE

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaFixedProjectiveConstraint_ImplMethods(T) \
    template<> void FixedProjectiveConstraint< T >::init() \
    { data->init(this); } \
    template<> void FixedProjectiveConstraint< T >::addConstraint(Index index) \
    { data->addConstraint(this, index); } \
    template<> void FixedProjectiveConstraint< T >::removeConstraint(Index index) \
    { data->removeConstraint(this, index); } \
    template<> void FixedProjectiveConstraint< T >::projectResponse(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_resData) \
    {  \
		VecDeriv &resData = *d_resData.beginEdit(); \
		data->projectResponse(this, resData);  \
		d_resData.endEdit(); \
	}

CudaFixedProjectiveConstraint_ImplMethods(gpu::cuda::CudaVec3fTypes);
CudaFixedProjectiveConstraint_ImplMethods(gpu::cuda::CudaVec3f1Types);
CudaFixedProjectiveConstraint_ImplMethods(gpu::cuda::CudaRigid3fTypes);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaFixedProjectiveConstraint_ImplMethods(gpu::cuda::CudaVec3dTypes);
CudaFixedProjectiveConstraint_ImplMethods(gpu::cuda::CudaVec3d1Types);
CudaFixedProjectiveConstraint_ImplMethods(gpu::cuda::CudaRigid3dTypes);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaFixedProjectiveConstraint_ImplMethods

} // namespace sofa::component::constraint::projective
