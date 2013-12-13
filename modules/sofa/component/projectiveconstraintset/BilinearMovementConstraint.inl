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

// Define TestNewPointFunction
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
    , m_constrainedPoints( initData(&m_constrainedPoints,"constrainedPoints","Coordinates of the constrained points") )
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_cornerMovements(  initData(&m_cornerMovements,"movements","movements corresponding to the key times") )
    , m_cornerPoints(  initData(&m_cornerPoints,"cornerPoints","corner points for computing constraint") )
{
    pointHandler = new FCPointHandler(this, &m_indices);
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

}

template <class DataTypes>
template <class DataDeriv>
void BilinearMovementConstraint<DataTypes>::projectResponseT(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataDeriv& dx)
{
   
}

template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(mparams /* PARAMS FIRST */, res.wref());
}



template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& vData)
{
    
}


template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecCoord& xData)
{
    sofa::simulation::Node::SPtr root =sofa::simulation::getSimulation()->GetRoot();

    // Apply movements only when animate is activated
    if(root->getAnimate())
    {
        
        helper::WriteAccessor<DataVecCoord> x = xData;
        Real alpha = 1;
        Deriv displacement;
        const SetIndexArray & indices = m_indices.getValue();
        const VecDeriv& cornerMovements = m_cornerMovements.getValue();
        const VecCoord& constrainedPoints = m_constrainedPoints.getValue();
        const VecCoord& cornerPoints = m_cornerPoints.getValue();
        
        // Compute displacements of the edge points with displacements of the corner points
        for (int i = 0; i< constrainedPoints.size(); ++i)
        {
            Coord p = constrainedPoints[i];
           
            // Edge 0 beteen P0 and P1
            if(p[1] == cornerPoints[0][1])
            {
                alpha=(p-cornerPoints[0]).norm()/(cornerPoints[1]-cornerPoints[0]).norm();
                displacement = cornerMovements[0]*(1-alpha) + cornerMovements[1]*alpha;  
                x[indices[i]] = x[indices[i]]  + displacement;

            }
            // Edge 1 beteen P1 and P2
            else if(p[0] == cornerPoints[1][0])
            {
                alpha = (p-cornerPoints[1]).norm()/(cornerPoints[2]-cornerPoints[1]).norm();
                displacement = cornerMovements[1]*(1-alpha) + cornerMovements[2]*alpha;
                x[indices[i]] = x[indices[i]]  + displacement;
            }
            // Edge 2 beteen P2 and P3
            else if(p[1] == cornerPoints[2][1])
            {
                alpha = (p-cornerPoints[2]).norm()/(cornerPoints[2]-cornerPoints[3]).norm();
                displacement = cornerMovements[2]*(1-alpha) + cornerMovements[3]*alpha;
                x[indices[i]] = x[indices[i]]  + displacement;
            }
            // Edge 3 beteen P3 and P0
            else if(p[0] == cornerPoints[3][0])
            {
                alpha = (p-cornerPoints[3]).norm()/(cornerPoints[3]-cornerPoints[0]).norm();
                displacement = cornerMovements[3]*(1-alpha) + cornerMovements[0]*alpha;
                x[indices[i]] = x[indices[i]]  + displacement;
            }
       }
    
    }
 
}

//display the path the constrained dofs will go through
template <class DataTypes>
void BilinearMovementConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif

