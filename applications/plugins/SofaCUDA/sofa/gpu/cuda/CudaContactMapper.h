/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GPU_CUDA_CUDACONTACTMAPPER_H
#define SOFA_GPU_CUDA_CUDACONTACTMAPPER_H

#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/SubsetContactMapper.inl>
#include <sofa/gpu/cuda/CudaDistanceGridCollisionModel.h>
#include <sofa/gpu/cuda/CudaPointModel.h>
#include <sofa/gpu/cuda/CudaSphereModel.h>
#include <sofa/gpu/cuda/CudaCollisionDetection.h>
#include <sofa/gpu/cuda/CudaRigidMapping.h>
#include <sofa/gpu/cuda/CudaSubsetMapping.h>


namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void RigidContactMapperCuda3f_setPoints2(unsigned int size, unsigned int nbTests, unsigned int maxPoints, const void* tests, const void* contacts, void* map);
    void SubsetContactMapperCuda3f_setPoints1(unsigned int size, unsigned int nbTests, unsigned int maxPoints, unsigned int nbPointsPerElem, const void* tests, const void* contacts, void* map);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using sofa::core::collision::GPUDetectionOutputVector;


/// Mapper for CudaRigidDistanceGridCollisionModel
template <class DataTypes>
class ContactMapper<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,DataTypes> : public RigidContactMapper<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef RigidContactMapper<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,DataTypes> Inherit;
    typedef typename Inherit::MMechanicalState MMechanicalState;
    typedef typename Inherit::MCollisionModel MCollisionModel;

    int addPoint(const Coord& P, int index, Real& r)
    {
        int i = this->Inherit::addPoint(P, index, r);
        if (!this->mapping)
        {
            MCollisionModel* model = this->model;
            MMechanicalState* outmodel = this->outmodel.get();
            Data<VecCoord>* d_x = outmodel->write(core::VecCoordId::position());
            VecDeriv& vx = *d_x->beginEdit();
            Data<VecDeriv>* d_v = outmodel->write(core::VecDerivId::velocity());
            VecCoord& vv = *d_v->beginEdit();

            typename DataTypes::Coord& x = vx[i];
            typename DataTypes::Deriv& v = vv[i];
            if (model->isTransformed(index))
            {
                x = model->getTranslation(index) + model->getRotation(index) * P;
            }
            else
            {
                x = P;
            }
            v = typename DataTypes::Deriv();

            d_x->endEdit();
            d_v->endEdit();
        }
        return i;
    }

    void setPoints2(GPUDetectionOutputVector* outputs)
    {
        int n = outputs->size();
        int nt = outputs->nbTests();
        int maxp = 0;
        for (int i=0; i<nt; i++)
            if (outputs->rtest(i).curSize > maxp) maxp = outputs->rtest(i).curSize;
        if (this->outmodel)
            this->outmodel->resize(n);
        if (this->mapping)
        {
            this->mapping->points.beginEdit()->fastResize(n);
            this->mapping->rotatedPoints.fastResize(n);
            gpu::cuda::RigidContactMapperCuda3f_setPoints2(n, nt, maxp, outputs->tests.deviceRead(), outputs->results.deviceRead(), this->mapping->points.beginEdit()->deviceWrite());
        }
        else
        {
            Data<VecCoord>* d_x = this->outmodel->write(core::VecCoordId::position());
            VecCoord& vx = *d_x->beginEdit();
            gpu::cuda::RigidContactMapperCuda3f_setPoints2(n, nt, maxp, outputs->tests.deviceRead(), outputs->results.deviceRead(), vx.deviceWrite());
            d_x->endEdit();
        }
    }
};


/// Mapper for CudaPointDistanceGridCollisionModel
template <class DataTypes>
class ContactMapper<sofa::gpu::cuda::CudaPointModel,DataTypes> : public SubsetContactMapper<sofa::gpu::cuda::CudaPointModel,DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef SubsetContactMapper<sofa::gpu::cuda::CudaPointModel,DataTypes> Inherit;
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
        typename MMapping::IndexArray& map = *this->mapping->f_indices.beginEdit();
        map.fastResize(n);
        gpu::cuda::SubsetContactMapperCuda3f_setPoints1(n, nt, maxp, this->model->groupSize.getValue(), outputs->tests.deviceRead(), outputs->results.deviceRead(), map.deviceWrite());
        this->mapping->f_indices.endEdit();
    }
};


template <class DataTypes>
class ContactMapper<sofa::gpu::cuda::CudaSphereModel,DataTypes> : public SubsetContactMapper<sofa::gpu::cuda::CudaSphereModel,DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef SubsetContactMapper<sofa::gpu::cuda::CudaSphereModel,DataTypes> Inherit;
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
        typename MMapping::IndexArray& map = *this->mapping->f_indices.beginEdit();
        map.fastResize(n);
        gpu::cuda::SubsetContactMapperCuda3f_setPoints1(n, nt, maxp, 0, outputs->tests.deviceRead(), outputs->results.deviceRead(), map.deviceWrite());
        this->mapping->f_indices.endEdit();
    }
};



} // namespace collision

} // namespace component

} // namespace sofa

#endif
