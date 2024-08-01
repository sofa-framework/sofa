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
#include <sofa/component/collision/response/mapper/RigidContactMapper.h>
#include <sofa/gpu/cuda/GPUDetectionOutputVector.h>
#include <SofaDistanceGrid/CUDA/CudaDistanceGridCollisionModel.h>

namespace sofa::gpu::cuda
{
extern "C"
{
    void RigidContactMapperCuda3f_setPoints2(unsigned int size, unsigned int nbTests, unsigned int maxPoints, const void* tests, const void* contacts, void* map);
}
}

/// Mapper for CudaRigidDistanceGridCollisionModel
template <class DataTypes>
class sofa::component::collision::response::mapper::ContactMapper<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,DataTypes> :
    public sofa::component::collision::response::mapper::RigidContactMapper<sofa::gpu::cuda::CudaRigidDistanceGridCollisionModel,DataTypes>
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

    void setPoints2(sofacuda::GPUDetectionOutputVector* outputs)
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
            this->mapping->d_points.beginEdit()->fastResize(n);
            this->mapping->m_rotatedPoints.fastResize(n);
            gpu::cuda::RigidContactMapperCuda3f_setPoints2(n, nt, maxp, outputs->tests.deviceRead(), outputs->results.deviceRead(), this->mapping->d_points.beginEdit()->deviceWrite());
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
