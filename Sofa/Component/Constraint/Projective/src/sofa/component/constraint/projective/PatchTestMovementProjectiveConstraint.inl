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

#include <sofa/simulation/Node.h>
#include <sofa/component/constraint/projective/PatchTestMovementProjectiveConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/Simulation.h>
#include <iostream>
#include <sofa/type/vector_algorithm.h>
#include <sofa/helper/cast.h>

namespace sofa::component::constraint::projective
{

template <class DataTypes>
PatchTestMovementProjectiveConstraint<DataTypes>::PatchTestMovementProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr)
    , data(new PatchTestMovementProjectiveConstraintInternalData<DataTypes>)
    , d_meshIndices( initData(&d_meshIndices,"meshIndices","Indices of the mesh") )
    , d_indices( initData(&d_indices,"indices","Indices of the constrained points") )
    , d_beginConstraintTime( initData(&d_beginConstraintTime,"beginConstraintTime","Begin time of the bilinear constraint") )
    , d_endConstraintTime( initData(&d_endConstraintTime,"endConstraintTime","End time of the bilinear constraint") )
    , d_constrainedPoints( initData(&d_constrainedPoints,"constrainedPoints","Coordinates of the constrained points") )
    , d_cornerMovements(  initData(&d_cornerMovements,"cornerMovements","movements of the corners of the grid") )
    , d_cornerPoints(  initData(&d_cornerPoints,"cornerPoints","corner points for computing constraint") )
    , d_drawConstrainedPoints(  initData(&d_drawConstrainedPoints,"drawConstrainedPoints","draw constrained points") )
    , l_topology(initLink("topology", "link to the topology container"))
{
    if(!d_beginConstraintTime.isSet())
     d_beginConstraintTime = 0;
    if(!d_endConstraintTime.isSet())
        d_endConstraintTime = 20;
}



template <class DataTypes>
PatchTestMovementProjectiveConstraint<DataTypes>::~PatchTestMovementProjectiveConstraint()
{

}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::clearConstraints()
{
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::addConstraint(Index index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::removeConstraint(Index index)
{
    sofa::type::removeValue(*d_indices.beginEdit(),index);
    d_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::init()
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

    const Index maxIndex=this->mstate->getSize();
    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const Index index=indices[i];
        if (index >= maxIndex)
        {
            msg_error() <<"Index " << index << " not valid!";
            removeConstraint(index);
        }
    }

     // Find the 4 corners of the grid topology
    this->findCornerPoints();
}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::findCornerPoints()
{
    Coord corner0, corner1, corner2, corner3,corner4,corner5,corner6,corner7;
    // Write accessor
    helper::WriteAccessor< Data<VecCoord > > cornerPositions = d_cornerPoints;
    helper::WriteAccessor< Data<VecCoord > > constrainedPoints = d_constrainedPoints;
    bool isMeshin3D = false;

    if (!constrainedPoints.empty())
    {
        const Coord& point = constrainedPoints[0];

        // Search if the constrained points are in the same plane
        for(Size i = 0; i < constrainedPoints.size() ; i++)
        {
            if(CoordSize > 2 && constrainedPoints[i][2]!=point[2])
            {
                isMeshin3D = true;
            }
        }
    }

    if(constrainedPoints.size() > 0)
    {
        corner0 = constrainedPoints[0];
        corner1 = constrainedPoints[0];
        corner2 = constrainedPoints[0];
        corner3 = constrainedPoints[0];
        corner4 = constrainedPoints[0];
        corner5 = constrainedPoints[0];
        corner6 = constrainedPoints[0];
        corner7 = constrainedPoints[0];

        for (size_t i = 0; i < constrainedPoints.size() ; i++)
        {
            if(constrainedPoints[i][0] < corner0[0] || constrainedPoints[i][1] < corner0[1] || ( CoordSize>2 && constrainedPoints[i][2] < corner0[2] ) )
            {
                corner0 = constrainedPoints[i];
            }

            if(constrainedPoints[i][0] > corner2[0] || constrainedPoints[i][1] > corner2[1] || ( CoordSize>2 && constrainedPoints[i][2] < corner2[2] ) )
            {
                 corner2 = constrainedPoints[i];
            }

            if(constrainedPoints[i][1] < corner1[1] || constrainedPoints[i][0] > corner1[0] || ( CoordSize>2 && constrainedPoints[i][2] < corner1[2] ))
            {
                 corner1 = constrainedPoints[i];
            }

            if(constrainedPoints[i][0] < corner3[0] || constrainedPoints[i][1] > corner3[1] || ( CoordSize>2 && constrainedPoints[i][2] < corner3[2] ))
            {
                 corner3 = constrainedPoints[i];
            }

            if(isMeshin3D && (constrainedPoints[i][0] < corner4[0] || constrainedPoints[i][1] < corner4[1] || (CoordSize>2 && constrainedPoints[i][2] > corner0[2] )) )
            {
                corner4 = constrainedPoints[i];
            }

            if(isMeshin3D && (constrainedPoints[i][0] > corner6[0] || constrainedPoints[i][1] > corner6[1] || (CoordSize>2 && constrainedPoints[i][2] > corner2[2] )) )
            {
                corner6 = constrainedPoints[i];
            }

            if(isMeshin3D && (constrainedPoints[i][1] < corner5[1] || constrainedPoints[i][0] > corner5[0] || (CoordSize>2 && constrainedPoints[i][2] > corner5[2] )) )
            {
                corner5 = constrainedPoints[i];
            }

            else if(isMeshin3D && (constrainedPoints[i][0] < corner7[0] || constrainedPoints[i][1] > corner7[1] || (CoordSize>2 && constrainedPoints[i][2] > corner7[2] )) )
            {
                corner7 = constrainedPoints[i];
            }
         }

        cornerPositions.push_back(corner0);
        cornerPositions.push_back(corner1);
        cornerPositions.push_back(corner2);
        cornerPositions.push_back(corner3);

        // 3D
        if(isMeshin3D)
        {
            cornerPositions.push_back(corner4);
            cornerPositions.push_back(corner5);
            cornerPositions.push_back(corner6);
            cornerPositions.push_back(corner7);
        }
    }
}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::projectResponseImpl(VecDeriv& dx)
{
    const SetIndexArray & indices = d_indices.getValue();
    for (size_t i = 0; i< indices.size(); ++i)
    {
        dx[indices[i]]=Deriv();
    }
}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseImpl(res.wref());
}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = vData;
    projectResponseImpl(res.wref());
}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    const sofa::simulation::Node::SPtr root = down_cast<sofa::simulation::Node>( this->getContext()->getRootContext() );
    helper::WriteAccessor<DataVecCoord> x = xData;
    const SetIndexArray & indices = d_indices.getValue();

    // Time
    double beginTime = d_beginConstraintTime.getValue();
    double endTime = d_endConstraintTime.getValue();
    double totalTime = endTime - beginTime;

    //initialize initial mesh Dofs positions, if it's not done
    if(meshPointsX0.size()==0)
        this->initializeInitialPositions(d_meshIndices.getValue(),xData,meshPointsX0);

    //initialize final mesh Dofs positions, if it's not done
    if(meshPointsXf.size()==0)
       this->initializeFinalPositions(d_meshIndices.getValue(),xData, meshPointsX0, meshPointsXf);

    //initialize initial constrained Dofs positions, if it's not done
    if(x0.size() == 0)
        this->initializeInitialPositions(indices,xData,x0);

     //initialize final constrained Dofs positions, if it's not done
    if (xf.size() == 0)
        this->initializeFinalPositions(indices,xData,x0,xf);

    // Update the intermediate Dofs positions computed by linear interpolation
   double time = root->getTime();
   if( time > beginTime && time <= endTime && totalTime > 0)
    {
        for (size_t i = 0; i< indices.size(); ++i)
        {
            x[indices[i]] = ((xf[indices[i]]-x0[indices[i]])*time + (x0[indices[i]]*endTime - xf[indices[i]]*beginTime))/totalTime;
        }
    }
   else if (time > endTime)
   {
        for (size_t i = 0; i< indices.size(); ++i)
        {
             x[indices[i]] = xf[indices[i]];
        }
   }
}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset )
{
    // clears the rows and columns associated with constrained particles
    const unsigned blockSize = DataTypes::deriv_total_size;

    for (const auto id : d_indices.getValue())
    {
        M->clearRowsCols( offset + id * blockSize, offset + (id+1) * blockSize );
    }

}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::getFinalPositions( VecCoord& finalPos,DataVecCoord& xData)
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
void PatchTestMovementProjectiveConstraint<DataTypes>::initializeInitialPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0)
{
    helper::WriteAccessor<DataVecCoord> x = xData;

    x0.resize(x.size());
    for (size_t i=0; i < indices.size() ; ++i)
    {
        x0[indices[i]] = x[indices[i]];
    }

}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::initializeFinalPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0, VecCoord& xf)
{
     Deriv displacement;
     helper::WriteAccessor<DataVecCoord> x = xData;

    xf.resize(x.size());

    // if the positions were not initialized
    if(x0.size() == 0)
        this->initializeInitialPositions(indices,xData,x0);

    for (size_t i=0; i < indices.size() ; ++i)
    {
        this->computeInterpolatedDisplacement(indices[i],xData,displacement);
        xf[indices[i]] = x0[indices[i]] + displacement ;
    }

}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::computeInterpolatedDisplacement(int pointIndice,const DataVecCoord& xData, Deriv& displacement)
{
    // For each mesh point compute the associated displacement

    // The 3 barycentric coefficients along x, y and z axis
    Real alpha, beta, gamma;

    // Corner points
    const VecCoord& cornerPoints = d_cornerPoints.getValue();
    if(cornerPoints.size()==0)
        this->findCornerPoints();

    if(cornerPoints.size() == 4)
    {
        Coord corner0 = cornerPoints[0];
        Coord corner1 = cornerPoints[1];
        Coord corner3 = cornerPoints[3];

        // Coord of the point
         helper::ReadAccessor<DataVecCoord> x = xData;
         Coord point = x[pointIndice];

         // Compute alpha = barycentric coefficient along the x axis
         alpha = fabs(point[0]-corner0[0])/fabs(corner1[0]-corner0[0]);

         // Compute beta = barycentric coefficient along the y axis
         beta = fabs(point[1]-corner0[1])/fabs(corner3[1]-corner0[1]);

         // cornerMovements
         const VecDeriv& cornerMovements = d_cornerMovements.getValue();

         // Compute displacement by linear interpolation
         displacement = cornerMovements[0]*(1-alpha)*(1-beta) + cornerMovements[1]*alpha*(1-beta)+ cornerMovements[2]*alpha*beta+cornerMovements[3]*(1-alpha)*beta;
    }

    else if(cornerPoints.size() == 8)
    {
        Coord corner0 = cornerPoints[0];
        Coord corner1 = cornerPoints[1];
        Coord corner3 = cornerPoints[3];
        Coord corner4 = cornerPoints[4];

        // Coord of the point
        helper::ReadAccessor<DataVecCoord> x = xData;
        Coord point = x[pointIndice];

        // Compute alpha = barycentric coefficient along the x axis
        alpha = fabs(point[0]-corner0[0])/fabs(corner1[0]-corner0[0]);

        // Compute beta = barycentric coefficient along the y axis
        beta = fabs(point[1]-corner0[1])/fabs(corner3[1]-corner0[1]);

        // Compute gamma = barycentric coefficient along the z axis
        if( CoordSize>2 )
            gamma = fabs(point[2]-corner0[2])/fabs(corner4[2]-corner0[2]); // 3D
        else
            gamma = 0; // 2D

        // cornerMovements
        const VecDeriv& cornerMovements = d_cornerMovements.getValue();

        // Compute displacement by linear interpolation
        displacement = (cornerMovements[0]*(1-alpha)*(1-beta) + cornerMovements[1]*alpha*(1-beta)+ cornerMovements[2]*alpha*beta+cornerMovements[3]*(1-alpha)*beta) * (1-gamma)
        + (cornerMovements[4]*(1-alpha)*(1-beta) + cornerMovements[5]*alpha*(1-beta)+ cornerMovements[6]*alpha*beta+cornerMovements[7]*(1-alpha)*beta) * gamma;
    }
    else
    {
        msg_info() << "error don't find the corner points" ;
    }

}

template <class DataTypes>
void PatchTestMovementProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = d_indices.getValue();
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    type::Vec3 point;

    if(d_drawConstrainedPoints.getValue())
    {
        std::vector< type::Vec3 > points;
        for (unsigned int index : indices)
        {
            point = DataTypes::getCPos(x[index]);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, sofa::type::RGBAColor(1,0.5,0.5,1));
    }
}

} // namespace sofa::component::constraint::projective
