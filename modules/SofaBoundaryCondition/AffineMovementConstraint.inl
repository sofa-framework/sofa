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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_AFFINEMOVEMENTCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_AFFINEMOVEMENTCONSTRAINT_INL

#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TopologySubsetData.inl>
#include <sofa/simulation/Simulation.h>
#include <iostream>
#include <sofa/helper/cast.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{


// Define TestFunction
template< class DataTypes>
bool AffineMovementConstraint<DataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    return fc != 0;
}


// Define RemovalFunction
template< class DataTypes>
void AffineMovementConstraint<DataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, core::objectmodel::Data<value_type>&)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
}


template <class DataTypes>
AffineMovementConstraint<DataTypes>::AffineMovementConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , data(new AffineMovementConstraintInternalData<DataTypes>)
    , m_meshIndices( initData(&m_meshIndices,"meshIndices","Indices of the mesh") )
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_beginConstraintTime( initData(&m_beginConstraintTime,"beginConstraintTime","Begin time of the bilinear constraint") )
    , m_endConstraintTime( initData(&m_endConstraintTime,"endConstraintTime","End time of the bilinear constraint") )
    , m_rotation(  initData(&m_rotation,"rotation","rotation applied to border points") )
    , m_quaternion( initData(&m_quaternion,"quaternion","quaternion applied to border points") )
    , m_translation(  initData(&m_translation,"translation","translation applied to border points") )
    , m_drawConstrainedPoints(  initData(&m_drawConstrainedPoints,"drawConstrainedPoints","draw constrained points") )
{
    pointHandler = new FCPointHandler(this, &m_indices);

    if(!m_beginConstraintTime.isSet())
     m_beginConstraintTime = 0; 
    if(!m_endConstraintTime.isSet())
        m_endConstraintTime = 20;
}



template <class DataTypes>
AffineMovementConstraint<DataTypes>::~AffineMovementConstraint()
{
    if (pointHandler)
        delete pointHandler;
}
 
template <class DataTypes>
void AffineMovementConstraint<DataTypes>::clearConstraints()
{
    m_indices.beginEdit()->clear();
    m_indices.endEdit();
}

template <class DataTypes>
void AffineMovementConstraint<DataTypes>::addConstraint(unsigned int index)
{
    m_indices.beginEdit()->push_back(index);
    m_indices.endEdit();
}

template <class DataTypes>
void AffineMovementConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*m_indices.beginEdit(),index);
    m_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void AffineMovementConstraint<DataTypes>::init()
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

}

template <class DataTypes>
template <class DataDeriv>
void AffineMovementConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/, DataDeriv& dx)
{
    const SetIndexArray & indices = m_indices.getValue();
    for (size_t i = 0; i< indices.size(); ++i)
    {
        dx[indices[i]]=Deriv();
    }
}

template <class DataTypes>
void AffineMovementConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(mparams, res.wref());
}



template <class DataTypes>
void AffineMovementConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> res = vData;
    projectResponseT<VecDeriv>(mparams, res.wref());
}

template <class DataTypes>
void AffineMovementConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    sofa::simulation::Node::SPtr root = down_cast<sofa::simulation::Node>( this->getContext()->getRootContext() );
    helper::WriteAccessor<DataVecCoord> x = xData;
    const SetIndexArray & indices = m_indices.getValue();
    
    // Time
    SReal beginTime = m_beginConstraintTime.getValue();
    SReal endTime = m_endConstraintTime.getValue();
    SReal totalTime = endTime - beginTime;
   
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
   SReal time = root->getTime();
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
void AffineMovementConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned /*offset*/ )
{
    // clears the rows and columns associated with constrained particles
    unsigned blockSize = DataTypes::deriv_total_size;
 
    for(SetIndexArray::const_iterator it= m_indices.getValue().begin(), iend=m_indices.getValue().end(); it!=iend; it++ )
    {
        M->clearRowsCols((*it) * blockSize,(*it+1) * (blockSize) );
    }
   
}

template <class DataTypes>
void AffineMovementConstraint<DataTypes>::getFinalPositions( VecCoord& finalPos,DataVecCoord& xData)
{
    // Indices of mesh points
    const SetIndexArray & meshIndices = m_meshIndices.getValue();
  
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
void AffineMovementConstraint<DataTypes>::initializeInitialPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0)
{
    helper::WriteAccessor<DataVecCoord> x = xData;

    x0.resize(x.size());
    for (size_t i=0; i < indices.size() ; ++i)
    {
        x0[indices[i]] = x[indices[i]];
    }
    
}

template <class DataTypes>
void AffineMovementConstraint<DataTypes>::transform(const SetIndexArray & indices, VecCoord& x0, VecCoord& xf)
{
    Vector3 translation = m_translation.getValue();
 
    for (size_t i=0; i < indices.size() ; ++i)
    {
         DataTypes::setCPos(xf[indices[i]], (m_rotation.getValue())*DataTypes::getCPos(x0[indices[i]]) + translation);
    }
  
}


template <>
void AffineMovementConstraint<defaulttype::Rigid3Types>::transform(const SetIndexArray & indices, defaulttype::Rigid3Types::VecCoord& x0, defaulttype::Rigid3Types::VecCoord& xf)
{
    // Get quaternion and translation values
    RotationMatrix rotationMat(0);
    Quat quat =  m_quaternion.getValue(); 
    quat.toMatrix(rotationMat);
    Vector3 translation = m_translation.getValue();

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
void AffineMovementConstraint<DataTypes>::initializeFinalPositions (const SetIndexArray & indices, DataVecCoord& xData, VecCoord& x0, VecCoord& xf)
{
     Deriv displacement; 
     helper::WriteAccessor<DataVecCoord> x = xData;
   
    xf.resize(x.size());
    
    // if the positions were not initialized
    if(x0.size() == 0)
        this->initializeInitialPositions(indices,xData,x0);
    
    transform(indices,x0,xf);
}

template <class DataTypes>
void AffineMovementConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const SetIndexArray & indices = m_indices.getValue();
    std::vector< Vector3 > points;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Vector3 point;

    if(m_drawConstrainedPoints.getValue())
    {
        for (SetIndexArray::const_iterator it = indices.begin();it != indices.end();++it)
        {
            point = DataTypes::getCPos(x[*it]);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, defaulttype::Vec<4,float>(1,0.5,0.5,1));
    }  
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif

