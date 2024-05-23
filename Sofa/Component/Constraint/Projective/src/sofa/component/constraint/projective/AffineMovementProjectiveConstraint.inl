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

#include <sofa/simulation/fwd.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <iostream>
#include <sofa/helper/cast.h>
#include <sofa/type/vector_algorithm.h>

#include <sofa/component/constraint/projective/AffineMovementProjectiveConstraint.h>

namespace sofa::component::constraint::projective
{

template <class DataTypes>
AffineMovementProjectiveConstraint<DataTypes>::AffineMovementProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr)
    , data(new AffineMovementProjectiveConstraintInternalData<DataTypes>)
    , d_meshIndices(initData(&d_meshIndices, "meshIndices", "Indices of the mesh") )
    , d_indices(initData(&d_indices, "indices", "Indices of the constrained points") )
    , d_beginConstraintTime(initData(&d_beginConstraintTime, "beginConstraintTime", "Begin time of the bilinear constraint") )
    , d_endConstraintTime(initData(&d_endConstraintTime, "endConstraintTime", "End time of the bilinear constraint") )
    , d_rotation(initData(&d_rotation, "rotation", "rotation applied to border points") )
    , d_quaternion(initData(&d_quaternion, "quaternion", "quaternion applied to border points") )
    , d_translation(initData(&d_translation, "translation", "translation applied to border points") )
    , d_drawConstrainedPoints(initData(&d_drawConstrainedPoints, "drawConstrainedPoints", "draw constrained points") )
    , l_topology(initLink("topology", "link to the topology container"))
{
    if(!d_beginConstraintTime.isSet())
        d_beginConstraintTime = 0;
    if(!d_endConstraintTime.isSet())
        d_endConstraintTime = 20;

    m_meshIndices.setParent(&d_meshIndices);
    m_indices.setParent(&d_indices);
    m_beginConstraintTime.setParent(&d_beginConstraintTime);
    m_endConstraintTime.setParent(&d_endConstraintTime);
    m_rotation.setParent(&d_rotation);
    m_quaternion.setParent(&d_quaternion);
    m_translation.setParent(&d_translation);
    m_drawConstrainedPoints.setParent(&d_drawConstrainedPoints);
}



template <class DataTypes>
AffineMovementProjectiveConstraint<DataTypes>::~AffineMovementProjectiveConstraint()
{

}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::clearConstraints()
{
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::addConstraint(Index index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::removeConstraint(Index index)
{
    removeValue(*d_indices.beginEdit(), index);
    d_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        // Initialize topological changes support
        d_indices.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }

    const SetIndexArray & indices = d_indices.getValue();

    auto maxIndex=this->mstate->getSize();
    for (Size i=0; i<indices.size(); ++i)
    {
        const Index index=indices[i];
        if (index >= maxIndex)
        {
            msg_error() << "Index " << index << " not valid!";
            removeConstraint(index);
        }
    }

}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::projectResponseImpl(VecDeriv& dx)
{
    const SetIndexArray & indices = d_indices.getValue();
    for (size_t i = 0; i< indices.size(); ++i)
    {
        dx[indices[i]]=Deriv();
    }
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseImpl(res.wref());
}



template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = vData;
    projectResponseImpl(res.wref());
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    const SetIndexArray & indices = d_indices.getValue();

    // Time
    SReal beginTime = d_beginConstraintTime.getValue();
    SReal endTime = d_endConstraintTime.getValue();
    SReal totalTime = endTime - beginTime;

    //initialize initial mesh Dofs positions, if it's not done
    if(meshPointsX0.size()==0)
        this->initializeInitialPositions(d_meshIndices.getValue(), xData, meshPointsX0);

    //initialize final mesh Dofs positions, if it's not done
    if(meshPointsXf.size()==0)
        this->initializeFinalPositions(d_meshIndices.getValue(), xData, meshPointsX0, meshPointsXf);

    //initialize initial constrained Dofs positions, if it's not done
    if(x0.size() == 0)
        this->initializeInitialPositions(indices,xData,x0);

    //initialize final constrained Dofs positions, if it's not done
    if (xf.size() == 0)
        this->initializeFinalPositions(indices,xData,x0,xf);
    // Update the intermediate Dofs positions computed by linear interpolation
    SReal time = sofa::core::objectmodel::basecontext::getTime(this->getContext()->getRootContext());
    if( time > beginTime && time <= endTime && totalTime > 0)
    {
        for (auto index : indices)
        {
            DataTypes::setCPos( x[index],
                                ((DataTypes::getCPos(xf[index])-DataTypes::getCPos(x0[index]))*time +
                                 (DataTypes::getCPos(x0[index])*endTime - DataTypes::getCPos(xf[index])*beginTime))/totalTime);
        }
    }
    else if (time > endTime)
    {
        for (auto index : indices)
        {
            x[index] = xf[index];
        }
    }
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned /*offset*/ )
{
    // clears the rows and columns associated with constrained particles
    const unsigned blockSize = DataTypes::deriv_total_size;

    for (const auto id : d_indices.getValue())
    {
        M->clearRowsCols( id * blockSize, (id+1) * blockSize );
    }
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::getFinalPositions( VecCoord& finalPos,DataVecCoord& xData)
{
    // Indices of mesh points
    const SetIndexArray & meshIndices = d_meshIndices.getValue();

    // Initialize final positions
    if(meshPointsXf.size()==0)
    {this->initializeFinalPositions(meshIndices,xData,meshPointsX0,meshPointsXf);}

    // Set final positions
    finalPos.resize(meshIndices.size());
    for (size_t i=0; i < meshIndices.size() ; ++i)
    {
        finalPos[meshIndices[i]] = meshPointsXf[meshIndices[i]];
    }
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::initializeInitialPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0)
{
    helper::WriteAccessor<DataVecCoord> x = xData;

    x0.resize(x.size());
    for (size_t i=0; i < indices.size() ; ++i)
    {
        x0[indices[i]] = x[indices[i]];
    }
}


template <>
void AffineMovementProjectiveConstraint<defaulttype::Rigid3Types>::transform(const SetIndexArray & indices,
                                                                   defaulttype::Rigid3Types::VecCoord& x0,
                                                                   defaulttype::Rigid3Types::VecCoord& xf)
{
    // Get quaternion and translation values
    RotationMatrix rotationMat(0);
    const Quat quat =  d_quaternion.getValue();
    quat.toMatrix(rotationMat);
    const Vec3 translation = d_translation.getValue();

    // Apply transformation
    for (size_t i=0; i < indices.size() ; ++i)
    {
        // Translation
        xf[indices[i]].getCenter() = rotationMat*(x0[indices[i]].getCenter()) + translation;
        // Rotation
        xf[indices[i]].getOrientation() = (quat)+x0[indices[i]].getOrientation();
    }
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::transform(const SetIndexArray & indices, VecCoord& x0, VecCoord& xf)
{
    Vec3 translation = d_translation.getValue();

    for (size_t i=0; i < indices.size() ; ++i)
    {
        DataTypes::setCPos(xf[indices[i]], (d_rotation.getValue()) * DataTypes::getCPos(x0[indices[i]]) + translation);
    }
}


template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::initializeFinalPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0, VecCoord& xf)
{
    helper::WriteAccessor<DataVecCoord> x = xData;

    xf.resize(x.size());

    // if the positions were not initialized
    if(x0.size() == 0)
        this->initializeInitialPositions(indices,xData,x0);

    transform(indices,x0,xf);
}

template <class DataTypes>
void AffineMovementProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = d_indices.getValue();
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Vec3 point;

    if(d_drawConstrainedPoints.getValue())
    {
        std::vector< Vec3 > points;
        for( auto& index : indices )
        {
            point = DataTypes::getCPos(x[index]);
            points.push_back(point);
        }
        constexpr sofa::type::RGBAColor color(1,0.5,0.5,1);
        vparams->drawTool()->drawPoints(points, 10, color);
    }
}

} // namespace sofa::component::constraint::projective
