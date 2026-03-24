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

#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <sofa/component/collision/response/mapper/SubsetContactMapper.inl>
#include <SofaCUDA/component/collision/geometry/CudaSphereModel.h>
#include <SofaCUDA/component/collision/geometry/CudaPointModel.h>
#include <sofa/gpu/cuda/GPUDetectionOutputVector.h>
#include <SofaCUDA/component/mapping/nonlinear/CudaRigidMapping.h>
#include <SofaCUDA/component/mapping/linear/CudaSubsetMapping.h>
#include <sofa/gpu/cuda/CudaTypes.h>


namespace sofa::gpu::cuda
{

extern "C"
{
    void SubsetContactMapperCuda3f_setPoints1(unsigned int size, unsigned int nbTests, unsigned int maxPoints, unsigned int nbPointsPerElem, const void* tests, const void* contacts, void* map);
}

} // namespace sofa::gpu::cuda


namespace sofa::component::collision
{

using namespace sofa::defaulttype;
using namespace sofa::gpu::cuda;
using sofa::core::collision::GPUDetectionOutputVector;

/// Mapper for CudaPointDistanceGridCollisionModel
template <class DataTypes>
class response::mapper::ContactMapper<sofa::gpu::cuda::CudaPointCollisionModel,DataTypes> : public response::mapper::SubsetContactMapper<sofa::gpu::cuda::CudaPointCollisionModel,DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef SubsetContactMapper<sofa::gpu::cuda::CudaPointCollisionModel,DataTypes> Inherit;
    typedef typename Inherit::MMechanicalState MMechanicalState;
    typedef typename Inherit::MCollisionModel MCollisionModel;
    typedef typename Inherit::MMapping MMapping;

    int addPoint(const Coord& P, int index, Real& r)
    {
        int i = Inherit::addPoint(P, index, r);
        return i;
    }

    void setPoints1(GPUDetectionOutputVector* outputs)
    {
        int n = outputs->size();
        int nt = outputs->nbTests();
        int maxp = 0;
        for (int i=0; i<nt; i++)
            if (outputs->rtest(i).curSize > maxp) maxp = outputs->rtest(i).curSize;
        typename MMapping::IndexArray& map = *this->mapping->d_indices.beginEdit();
        map.fastResize(n);
        gpu::cuda::SubsetContactMapperCuda3f_setPoints1(n, nt, maxp, this->model->groupSize.getValue(), outputs->tests.deviceRead(), outputs->results.deviceRead(), map.deviceWrite());
        this->mapping->d_indices.endEdit();
    }
};


template <class DataTypes>
class response::mapper::ContactMapper<CudaSphereCollisionModel, DataTypes> : public response::mapper::SubsetContactMapper<CudaSphereCollisionModel, DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef SubsetContactMapper<CudaSphereCollisionModel, DataTypes> Inherit;
    typedef typename Inherit::MMechanicalState MMechanicalState;
    typedef typename Inherit::MCollisionModel MCollisionModel;
    typedef typename Inherit::MMapping MMapping;

    int addPoint(const Coord& P, int index, Real& r)
    {
        int i = this->Inherit::addPoint(P, index, r);
        return i;
    }

    void setPoints1(GPUDetectionOutputVector* outputs)
    {
        int n = outputs->size();
        int nt = outputs->nbTests();
        int maxp = 0;
        for (int i=0; i<nt; i++)
            if (outputs->rtest(i).curSize > maxp) maxp = outputs->rtest(i).curSize;
        typename MMapping::IndexArray& map = *this->mapping->d_indices.beginEdit();
        map.fastResize(n);
        gpu::cuda::SubsetContactMapperCuda3f_setPoints1(n, nt, maxp, 0, outputs->tests.deviceRead(), outputs->results.deviceRead(), map.deviceWrite());
        this->mapping->d_indices.endEdit();
    }
};

#if !defined(SOFACUDA_CUDACONTACTMAPPER_CPP)
extern template class SOFA_GPU_CUDA_API response::mapper::ContactMapper<sofa::gpu::cuda::CudaPointCollisionModel, sofa::gpu::cuda::CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API response::mapper::ContactMapper<sofa::gpu::cuda::CudaSphereCollisionModel, sofa::gpu::cuda::CudaVec3fTypes>;
#endif

} // namespace sofa::component::collision
