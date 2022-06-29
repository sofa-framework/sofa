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

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::gpu::cuda
{

using namespace sofa::defaulttype;

class CudaPointCollisionModel;

class CudaPoint : public core::TCollisionElementIterator<CudaPointCollisionModel>
{
public:
    CudaPoint(CudaPointCollisionModel* model, Index index);

    Index i0();
    std::size_t getSize();

    explicit CudaPoint(const core::CollisionElementIterator& i);
};

class CudaPointCollisionModel : public core::CollisionModel
{
public:
    SOFA_CLASS(CudaPointCollisionModel,core::CollisionModel);
    typedef CudaVec3fTypes InDataTypes;
    typedef CudaVec3fTypes DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef CudaPoint Element;
    friend class CudaPoint;

    Data<std::size_t> groupSize; ///< number of point per collision element

    CudaPointCollisionModel();

    virtual void init() override;

    // -- CollisionModel interface

    virtual void resize(Size size) override;

    virtual void computeBoundingTree(int maxDepth=0) override;

    //virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(const core::visual::VisualParams*,Index index) override;

    void draw(const core::visual::VisualParams*) override;

    core::behavior::MechanicalState<InDataTypes>* getMechanicalState() { return mstate; }

protected:

    core::behavior::MechanicalState<InDataTypes>* mstate;
};

inline CudaPoint::CudaPoint(CudaPointCollisionModel* model, Index index)
    : core::TCollisionElementIterator<CudaPointCollisionModel>(model, index)
{}

inline CudaPoint::CudaPoint(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<CudaPointCollisionModel>(static_cast<CudaPointCollisionModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline CudaPoint::Index CudaPoint::i0()
{
    return model->groupSize.getValue()*index;
}

inline std::size_t CudaPoint::getSize()
{
    if (index == model->getSize()-1)
        return model->getMechanicalState()->getSize();
    else
        return model->groupSize.getValue();
}

} // namespace sofa::gpu::cuda
