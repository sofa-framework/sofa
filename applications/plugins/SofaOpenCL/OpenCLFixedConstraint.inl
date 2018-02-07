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
#ifndef SOFAOPENCL_OPENCLFIXEDCONSTRAINT_INL
#define SOFAOPENCL_OPENCLFIXEDCONSTRAINT_INL

#include "OpenCLFixedConstraint.h"
#include <SofaBoundaryCondition/FixedConstraint.inl>

namespace sofa
{

namespace gpu
{

namespace opencl
{


extern "C"
{
    extern void FixedConstraintOpenCL3f_projectResponseContiguous(unsigned int size, _device_pointer dx);
    extern void FixedConstraintOpenCL3f_projectResponseIndexed(unsigned int size, const _device_pointer indices, _device_pointer dx);
    extern void FixedConstraintOpenCL3f1_projectResponseContiguous(unsigned int size, _device_pointer dx);
    extern void FixedConstraintOpenCL3f1_projectResponseIndexed(unsigned int size, const _device_pointer indices, _device_pointer dx);
#ifdef SOFA_DEV
    extern void FixedConstraintOpenCLRigid3f_projectResponseContiguous(unsigned int size, _device_pointer dx);
    extern void FixedConstraintOpenCLRigid3f_projectResponseIndexed(unsigned int size, const _device_pointer indices, _device_pointer dx);
#endif // SOFA_DEV



    extern void FixedConstraintOpenCL3d_projectResponseContiguous(unsigned int size, _device_pointer dx);
    extern void FixedConstraintOpenCL3d_projectResponseIndexed(unsigned int size, const _device_pointer indices, _device_pointer dx);
    extern void FixedConstraintOpenCL3d1_projectResponseContiguous(unsigned int size, _device_pointer dx);
    extern void FixedConstraintOpenCL3d1_projectResponseIndexed(unsigned int size, const _device_pointer indices, _device_pointer dx);
#ifdef SOFA_DEV
    extern void FixedConstraintOpenCLRigid3d_projectResponseContiguous(unsigned int size, _device_pointer dx);
    extern void FixedConstraintOpenCLRigid3d_projectResponseIndexed(unsigned int size, const _device_pointer indices, _device_pointer dx);
#endif // SOFA_DEV


}

} // namespace opencl

} // namespace gpu

namespace component
{

namespace projectiveconstraintset
{


using namespace gpu::opencl;

template<class TCoord, class TDeriv, class TReal>
void FixedConstraintInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::init(Main* m)
{
    Data& data = *m->data;
    data.minIndex = -1;
    data.maxIndex = -1;
    data.OpenCLIndices.clear();
    m->core::behavior::template ProjectiveConstraintSet<DataTypes>::init();
    const SetIndexArray& indices = m->d_indices.getValue();
    if (!indices.empty())
    {
        // put indices in a set to sort them and remove duplicates
        std::set<int> sortedIndices;
        for (typename SetIndexArray::const_iterator it = indices.begin(); it!=indices.end(); it++)
            sortedIndices.insert(*it);
        // check if the indices are contiguous
        if (*sortedIndices.begin() + (int)sortedIndices.size()-1 == *sortedIndices.rbegin())
        {
            data.minIndex = *sortedIndices.begin();
            data.maxIndex = *sortedIndices.rbegin();
            //std::cout << "OpenCLFixedConstraint: "<<sortedIndices.size()<<" contiguous fixed indices, "<<data.minIndex<<" - "<<data.maxIndex<<sendl;
        }
        else
        {
            //std::cout << "OpenCLFixedConstraint: "<<sortedIndices.size()<<" non-contiguous fixed indices"<<sendl;
            data.OpenCLIndices.reserve(sortedIndices.size());
            for (std::set<int>::const_iterator it = sortedIndices.begin(); it!=sortedIndices.end(); it++)
                data.OpenCLIndices.push_back(*it);
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
void FixedConstraintInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::addConstraint(Main* m, unsigned int index)
{
    Data& data = *m->data;
    //std::cout << "OpenCLFixedConstraint::addConstraint("<<index<<")\n";
    m->d_indices.beginEdit()->push_back(index);
    m->d_indices.endEdit();
    if (data.OpenCLIndices.empty())
    {
        if (data.minIndex == -1)
        {
            //std::cout << "OpenCLFixedConstraint: single index "<<index<<"\n";
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
            //std::cout << "OpenCLFixedConstraint: new min index "<<index<<"\n";
        }
        else if (data.maxIndex == (int)index-1)
        {
            data.maxIndex = index;
            //std::cout << "OpenCLFixedConstraint: new max index "<<index<<"\n";
        }
        else
        {
            data.OpenCLIndices.reserve(data.maxIndex-data.minIndex+2);
            for (int i=data.minIndex; i<data.maxIndex; i++)
                data.OpenCLIndices.push_back(i);
            data.OpenCLIndices.push_back(index);
            data.minIndex = -1;
            data.maxIndex = -1;
            std::cout << "OpenCLFixedConstraint: new indices array size "<<data.OpenCLIndices.size()<<"\n";
        }
    }
    else
    {
        data.OpenCLIndices.push_back(index);
        //std::cout << "OpenCLFixedConstraint: indices array size "<<data.OpenCLIndices.size()<<"\n";
    }
}

template<class TCoord, class TDeriv, class TReal>
void FixedConstraintInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::removeConstraint(Main* m, unsigned int index)
{
    Data& data = *m->data;
    removeValue(*m->d_indices.beginEdit(),index);
    m->d_indices.endEdit();
    if (data.OpenCLIndices.empty())
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
                data.OpenCLIndices.reserve(data.maxIndex-data.minIndex);
                for (int i=data.minIndex; i<data.maxIndex; i++)
                    if (i != (int)index)
                        data.OpenCLIndices.push_back(i);
                data.minIndex = -1;
                data.maxIndex = -1;
            }
        }
    }
    else
    {
        bool found = false;
        for (unsigned int i=0; i<data.OpenCLIndices.size(); i++)
        {
            if (found)
                data.OpenCLIndices[i-1] = data.OpenCLIndices[i];
            else if (data.OpenCLIndices[i] == (int)index)
                found = true;
        }
        if (found)
            data.OpenCLIndices.resize(data.OpenCLIndices.size()-1);
    }
}

#ifdef SOFA_DEV
template<int N, class real>
void FixedConstraintInternalData< gpu::opencl::OpenCLRigidTypes<N, real> >::init(Main* m)
{
    Data& data = *m->data;
    data.minIndex = -1;
    data.maxIndex = -1;
    data.OpenCLIndices.clear();
    m->core::behavior::ProjectiveConstraintSet<DataTypes>::init();
    const SetIndexArray& indices = m->d_indices.getValue();
    if (!indices.empty())
    {
        // put indices in a set to sort them and remove duplicates
        std::set<int> sortedIndices;
        for (typename SetIndexArray::const_iterator it = indices.begin(); it!=indices.end(); it++)
            sortedIndices.insert(*it);
        // check if the indices are contiguous
        if (*sortedIndices.begin() + (int)sortedIndices.size()-1 == *sortedIndices.rbegin())
        {
            data.minIndex = *sortedIndices.begin();
            data.maxIndex = *sortedIndices.rbegin();
            //std::cout << "OpenCLFixedConstraint: "<<sortedIndices.size()<<" contiguous fixed indices, "<<data.minIndex<<" - "<<data.maxIndex<<sendl;
        }
        else
        {
            //std::cout << "OpenCLFixedConstraint: "<<sortedIndices.size()<<" non-contiguous fixed indices"<<sendl;
            data.OpenCLIndices.reserve(sortedIndices.size());
            for (std::set<int>::const_iterator it = sortedIndices.begin(); it!=sortedIndices.end(); it++)
                data.OpenCLIndices.push_back(*it);
        }
    }
}

template<int N, class real>
void FixedConstraintInternalData< gpu::opencl::OpenCLRigidTypes<N, real> >::addConstraint(Main* m, unsigned int index)
{
    Data& data = *m->data;
    //std::cout << "OpenCLFixedConstraint::addConstraint("<<index<<")\n";
    m->d_indices.beginEdit()->push_back(index);
    m->d_indices.endEdit();
    if (data.OpenCLIndices.empty())
    {
        if (data.minIndex == -1)
        {
            //std::cout << "OpenCLFixedConstraint: single index "<<index<<"\n";
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
            //std::cout << "OpenCLFixedConstraint: new min index "<<index<<"\n";
        }
        else if (data.maxIndex == (int)index-1)
        {
            data.maxIndex = index;
            //std::cout << "OpenCLFixedConstraint: new max index "<<index<<"\n";
        }
        else
        {
            data.OpenCLIndices.reserve(data.maxIndex-data.minIndex+2);
            for (int i=data.minIndex; i<data.maxIndex; i++)
                data.OpenCLIndices.push_back(i);
            data.OpenCLIndices.push_back(index);
            data.minIndex = -1;
            data.maxIndex = -1;
            std::cout << "OpenCLFixedConstraint: new indices array size "<<data.OpenCLIndices.size()<<"\n";
        }
    }
    else
    {
        data.OpenCLIndices.push_back(index);
        //std::cout << "OpenCLFixedConstraint: indices array size "<<data.OpenCLIndices.size()<<"\n";
    }
}

template<int N, class real>
void FixedConstraintInternalData< gpu::opencl::OpenCLRigidTypes<N, real> >::removeConstraint(Main* m, unsigned int index)
{
    Data& data = *m->data;
    removeValue(*m->d_indices.beginEdit(),index);
    m->d_indices.endEdit();
    if (data.OpenCLIndices.empty())
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
                data.OpenCLIndices.reserve(data.maxIndex-data.minIndex);
                for (int i=data.minIndex; i<data.maxIndex; i++)
                    if (i != (int)index)
                        data.OpenCLIndices.push_back(i);
                data.minIndex = -1;
                data.maxIndex = -1;
            }
        }
    }
    else
    {
        bool found = false;
        for (unsigned int i=0; i<data.OpenCLIndices.size(); i++)
        {
            if (found)
                data.OpenCLIndices[i-1] = data.OpenCLIndices[i];
            else if (data.OpenCLIndices[i] == (int)index)
                found = true;
        }
        if (found)
            data.OpenCLIndices.resize(data.OpenCLIndices.size()-1);
    }
}


#endif // SOFA_DEV

template <>
void FixedConstraintInternalData<gpu::opencl::OpenCLVec3fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintOpenCL3f_projectResponseContiguous(dx.size(), dx.deviceWrite());
    else if (data.minIndex >= 0)
        FixedConstraintOpenCL3f_projectResponseContiguous(data.maxIndex-data.minIndex+1, OpenCLMemoryManager<float>::deviceOffset(dx.deviceWrite(),3*data.minIndex));
    else
        FixedConstraintOpenCL3f_projectResponseIndexed(data.OpenCLIndices.size(), data.OpenCLIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedConstraintInternalData<gpu::opencl::OpenCLVec3f1Types>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintOpenCL3f1_projectResponseContiguous(dx.size(), dx.deviceWrite());
    else if (data.minIndex >= 0)
        FixedConstraintOpenCL3f1_projectResponseContiguous(data.maxIndex-data.minIndex+1,  OpenCLMemoryManager<float>::deviceOffset(dx.deviceWrite(),4*data.minIndex));
    else
        FixedConstraintOpenCL3f1_projectResponseIndexed(data.OpenCLIndices.size(), data.OpenCLIndices.deviceRead(), dx.deviceWrite());
}

#ifdef SOFA_DEV
template <>
void FixedConstraintInternalData<gpu::opencl::OpenCLRigid3fTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintOpenCLRigid3f_projectResponseContiguous(dx.size(), dx.deviceWrite());
    else if (data.minIndex >= 0)
        FixedConstraintOpenCLRigid3f_projectResponseContiguous(data.maxIndex-data.minIndex+1, OpenCLMemoryManager<float>::deviceOffset(dx.deviceWrite(),6*data.minIndex));
    else
        FixedConstraintOpenCLRigid3f_projectResponseIndexed(data.OpenCLIndices.size(), data.OpenCLIndices.deviceRead(), dx.deviceWrite());
}
#endif // SOFA_DEV

template <>
void FixedConstraintInternalData<gpu::opencl::OpenCLVec3dTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintOpenCL3d_projectResponseContiguous(dx.size(), dx.deviceWrite());
    else if (data.minIndex >= 0)
        FixedConstraintOpenCL3d_projectResponseContiguous(data.maxIndex-data.minIndex+1,  OpenCLMemoryManager<double>::deviceOffset(dx.deviceWrite(),3*data.minIndex));
    else
        FixedConstraintOpenCL3d_projectResponseIndexed(data.OpenCLIndices.size(), data.OpenCLIndices.deviceRead(), dx.deviceWrite());
}

template <>
void FixedConstraintInternalData<gpu::opencl::OpenCLVec3d1Types>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintOpenCL3d1_projectResponseContiguous(dx.size(), dx.deviceWrite());
    else if (data.minIndex >= 0)
        FixedConstraintOpenCL3d1_projectResponseContiguous(data.maxIndex-data.minIndex+1,  OpenCLMemoryManager<double>::deviceOffset(dx.deviceWrite(),4*data.minIndex));
    else
        FixedConstraintOpenCL3d1_projectResponseIndexed(data.OpenCLIndices.size(), data.OpenCLIndices.deviceRead(), dx.deviceWrite());
}

#ifdef SOFA_DEV
template <>
void FixedConstraintInternalData<gpu::opencl::OpenCLRigid3dTypes>::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;
    if (m->f_fixAll.getValue())
        FixedConstraintOpenCLRigid3d_projectResponseContiguous(dx.size(), (dx.deviceWrite()));
    else if (data.minIndex >= 0)
        FixedConstraintOpenCLRigid3d_projectResponseContiguous(data.maxIndex-data.minIndex+1,  OpenCLMemoryManager<double>::deviceOffset(dx.deviceWrite(),6*data.minIndex));
    else
        FixedConstraintOpenCLRigid3d_projectResponseIndexed(data.OpenCLIndices.size(), data.OpenCLIndices.deviceRead(), dx.deviceWrite());
}
#endif // SOFA_DEV


// I know using macros is bad design but this is the only way not to repeat the code for all OpenCL types
#define OpenCLFixedConstraint_ImplMethods(T) \
	template<> void FixedConstraint< T >::init() \
	{ data->init(this); } \
	template<> void FixedConstraint< T >::addConstraint(unsigned int index) \
	{ data->addConstraint(this, index); } \
	template<> void FixedConstraint< T >::removeConstraint(unsigned int index) \
	{ data->removeConstraint(this, index); } \
    template<> void FixedConstraint< T >::projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_resData) \
    {  \
        VecDeriv &resData = *d_resData.beginEdit(mparams); \
        data->projectResponse(this, resData);               \
        d_resData.endEdit(mparams);                        \
    }

OpenCLFixedConstraint_ImplMethods(gpu::opencl::OpenCLVec3fTypes);
OpenCLFixedConstraint_ImplMethods(gpu::opencl::OpenCLVec3f1Types);
#ifdef SOFA_DEV
OpenCLFixedConstraint_ImplMethods(gpu::opencl::OpenCLRigid3fTypes);
#endif // SOFA_DEV



OpenCLFixedConstraint_ImplMethods(gpu::opencl::OpenCLVec3dTypes);
OpenCLFixedConstraint_ImplMethods(gpu::opencl::OpenCLVec3d1Types);
#ifdef SOFA_DEV
OpenCLFixedConstraint_ImplMethods(gpu::opencl::OpenCLRigid3dTypes);
#endif // SOFA_DEV

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
