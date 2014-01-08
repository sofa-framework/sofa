/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_BILINEARMOVEMENTCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_BILINEARMOVEMENTCONSTRAINT_INL

#include <sofa/component/projectiveconstraintset/BilinearMovementConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <sofa/component/topology/TopologySubsetData.inl>



namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace core::topology;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::behavior;

// Define TestFunction
template< class DataTypes>
bool BilinearMovementConstraint<DataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    return fc != 0;
}


// Define RemovalFunction
template< class DataTypes>
void BilinearMovementConstraint<DataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, value_type &)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
}


template <class DataTypes>
BilinearMovementConstraint<DataTypes>::BilinearMovementConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , data(new BilinearMovementConstraintInternalData<DataTypes>)
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_beginConstraintTime( initData(&m_beginConstraintTime,"beginConstraintTime","Begin time of the bilinear constraint") )
    , m_endConstraintTime( initData(&m_endConstraintTime,"endConstraintTime","End time of the bilinear constraint") )
    , m_meshIndices( initData(&m_meshIndices,"meshIndices","Indices of the mesh") )
    , m_constrainedPoints( initData(&m_constrainedPoints,"constrainedPoints","Coordinates of the constrained points") )
    , m_cornerMovements(  initData(&m_cornerMovements,"cornerMovements","movements of the corners of the grid") )
    , m_cornerPoints(  initData(&m_cornerPoints,"cornerPoints","corner points for computing constraint") )
    , m_drawConstrainedPoints(  initData(&m_drawConstrainedPoints,"drawConstrainedPoints","draw constrained points") )
{
    pointHandler = new FCPointHandler(this, &m_indices);

    if(!m_beginConstraintTime.isSet())
     m_beginConstraintTime = 0; 
    if(!m_endConstraintTime.isSet())
        m_endConstraintTime = 200;

}



template <class DataTypes>
BilinearMovementConstraint<DataTypes>::~BilinearMovementConstraint()
{
    if (pointHandler)
        delete pointHandler;
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::clearConstraints()
{
    m_indices.beginEdit()->clear();
    m_indices.endEdit();
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::addConstraint(unsigned int index)
{
    m_indices.beginEdit()->push_back(index);
    m_indices.endEdit();
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*m_indices.beginEdit(),index);
    m_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    m_indices.createTopologicalEngine(topology, pointHandler);
    m_indices.registerTopologicalData();

    const SetIndexArray & indices = m_indices.getValue();

    unsigned int maxIndex=this->mstate->getSize();
    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int index=indices[i];
        if (index >= maxIndex)
        {
            serr << "Index " << index << " not valid!" << sendl;
            removeConstraint(index);
        }
    }

     // Find the 4 corners of the grid topology
    this->findCornerPoints();

}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::findCornerPoints()
{
    Coord corner0, corner1, corner2, corner3;
    // Write accessor 
    helper::WriteAccessor< Data<VecCoord > > cornerPositions = m_cornerPoints;
    helper::WriteAccessor< Data<VecCoord > > constrainedPoints = m_constrainedPoints;
    if(constrainedPoints.size() > 0)
    {
        corner0 = constrainedPoints[0];
        corner1 = constrainedPoints[0];
        corner2 = constrainedPoints[0];
        corner3 = constrainedPoints[0];
        for (size_t i = 0; i < constrainedPoints.size() ; i++)
        {
            if(constrainedPoints[i][0] < corner0[0] || constrainedPoints[i][1] < corner0[1] || constrainedPoints[i][2] < corner0[2])
            {
                corner0 = constrainedPoints[i];
            }

            if(constrainedPoints[i][0] > corner2[0] || constrainedPoints[i][1] > corner2[1] || constrainedPoints[i][2] > corner2[2])
            {   
                 corner2 = constrainedPoints[i];
            }

            if(constrainedPoints[i][1] < corner1[1] || constrainedPoints[i][0] > corner1[0] )
            {   
                 corner1 = constrainedPoints[i];
            }

            else if(constrainedPoints[i][0] < corner3[0] || constrainedPoints[i][1] > corner3[1])
            {   
                 corner3 = constrainedPoints[i];
            }
         }
          
        cornerPositions.push_back(corner0);
        cornerPositions.push_back(corner1);
        cornerPositions.push_back(corner2);
        cornerPositions.push_back(corner3);
    }
}

template <class DataTypes>
template <class DataDeriv>
void BilinearMovementConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataDeriv& dx)
{
    const SetIndexArray & indices = m_indices.getValue();
    for (size_t i = 0; i< indices.size(); ++i)
    {
        dx[indices[i]]=Deriv();
    }
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(mparams /* PARAMS FIRST */, res.wref());
}



template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> res = vData;
    projectResponseT<VecDeriv>(mparams /* PARAMS FIRST */, res.wref());
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecCoord& xData)
{
    sofa::simulation::Node::SPtr root =sofa::simulation::getSimulation()->GetRoot();
    helper::WriteAccessor<DataVecCoord> x = xData;
    const SetIndexArray & indices = m_indices.getValue();
    
    // Time
    double beginTime = m_beginConstraintTime.getValue();
    double endTime = m_endConstraintTime.getValue();
    double totalTime = endTime - beginTime;
   
    //initialize initial mesh Dofs positions, if it's not done
    if(meshPointsX0.size()==0)
        this->initializeInitialPositions(m_meshIndices.getValue(),xData,meshPointsX0);

    //initialize final mesh Dofs positions, if it's not done
    if(meshPointsXf.size()==0)
       this->initializeFinalPositions(m_meshIndices.getValue(),xData, meshPointsX0, meshPointsXf);

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
void BilinearMovementConstraint<DataTypes>::getFinalPositions( VecCoord& finalPos,DataVecCoord& xData)
{
    // Indices of mesh points
    const SetIndexArray & meshIndices = m_meshIndices.getValue();

    // Initialize final positions
    if(meshPointsXf.size()==0)
    {this->initializeFinalPositions(meshIndices,xData,meshPointsX0,meshPointsXf);}
    
    // Set final positions
    finalPos.resize(meshPointsXf.size());
    for (size_t i=0; i < meshPointsXf.size() ; ++i)
    {
        finalPos[meshIndices[i]] = meshPointsXf[meshIndices[i]];
        //std::cout << "theoretical final positions of indice" <<meshIndices[i] << " = " << meshPointsXf[meshIndices[i]][0] << " , " <<meshPointsXf[meshIndices[i]][1] << " , " << meshPointsXf[meshIndices[i]][2] << std::endl;
    }
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::initializeInitialPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0)
{
    helper::WriteAccessor<DataVecCoord> x = xData;

    x0.resize(x.size());
    for (size_t i=0; i < indices.size() ; ++i)
    {
        x0[indices[i]] = x[indices[i]];
    }
    
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::initializeFinalPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0, VecCoord& xf)
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
void BilinearMovementConstraint<DataTypes>::computeInterpolatedDisplacement(int pointIndice,const DataVecCoord& xData, Deriv& displacement)
{
    // For each mesh point compute the associated displacement

    // The 2 barycentric coefficients for x and y
    Real alpha, beta;

    // Corner points
    const VecCoord& cornerPoints = m_cornerPoints.getValue();
    if(cornerPoints.size()==0)
        this->findCornerPoints();
    Coord corner0 = cornerPoints[0];
    Coord corner1 = cornerPoints[1];
    Coord corner2 = cornerPoints[2];
    Coord corner3 = cornerPoints[3];

    // Coord of the point
     helper::ReadAccessor<DataVecCoord> x = xData;
     Coord point = x[pointIndice];
   
     // Compute alpha = barycentric coefficient along the x axis
     alpha = fabs(point[0]-corner0[0])/fabs(corner1[0]-corner0[0]);
     
     // Compute beta = barycentric coefficient along the y axis
     beta = fabs(point[1]-corner0[1])/fabs(corner3[1]-corner0[1]);

     // cornerMovements
     const VecDeriv& cornerMovements = m_cornerMovements.getValue();

     // Compute displacement by linear interpolation
     displacement = cornerMovements[0]*(1-alpha)*(1-beta) + cornerMovements[1]*alpha*(1-beta)+ cornerMovements[2]*alpha*beta+cornerMovements[3]*(1-alpha)*beta;
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = m_indices.getValue();
    std::vector< Vector3 > points;
    const VecCoord& x = *this->mstate->getX();
    Vector3 point;

    if(m_drawConstrainedPoints.getValue())
    {
        for (SetIndexArray::const_iterator it = indices.begin();it != indices.end();++it)
        {
            point = DataTypes::getCPos(x[*it]);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }  
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif

