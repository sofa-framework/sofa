/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONSTRAINT_LINEARMOVEMENTCONSTRAINT_INL
#define SOFA_COMPONENT_CONSTRAINT_LINEARMOVEMENTCONSTRAINT_INL

#include <sofa/component/constraint/PartialLinearMovementConstraint.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/core/componentmodel/behavior/Constraint.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>

#include <sofa/component/topology/PointSubset.h>



namespace sofa
{

namespace component
{

namespace constraint
{

using namespace core::componentmodel::topology;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::componentmodel::behavior;


// Define TestNewPointFunction
template< class DataTypes>
bool PartialLinearMovementConstraint<DataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    PartialLinearMovementConstraint<DataTypes> *fc= (PartialLinearMovementConstraint<DataTypes> *)param;
    if (fc)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Define RemovalFunction
template< class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    PartialLinearMovementConstraint<DataTypes> *fc= (PartialLinearMovementConstraint<DataTypes> *)param;
    if (fc)
    {
        fc->removeIndex((unsigned int) pointIndex);
    }
    return;
}

template <class DataTypes>
PartialLinearMovementConstraint<DataTypes>::PartialLinearMovementConstraint()
    : core::componentmodel::behavior::Constraint<DataTypes>(NULL)
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_keyTimes(  initData(&m_keyTimes,"keyTimes","key times for the movements") )
    , m_keyMovements(  initData(&m_keyMovements,"movements","movements corresponding to the key times") )
    , movedDirections( initData(&movedDirections,"movedDirections","for each direction, 1 if moved, 0 if free") )
{
    // default to indice 0
    m_indices.beginEdit()->push_back(0);
    m_indices.endEdit();

    //default valueEvent to 0
    m_keyTimes.beginEdit()->push_back( 0.0 );
    m_keyTimes.endEdit();
    m_keyMovements.beginEdit()->push_back( Deriv() );
    m_keyMovements.endEdit();
    Vec6Bool movedDirection;
    for( unsigned i=0; i<NumDimensions; i++)
        movedDirection[i] = true;
    movedDirections.setValue(movedDirection);
}


// Handle topological changes
template <class DataTypes> void PartialLinearMovementConstraint<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=topology->firstChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->lastChange();

    m_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());

}

template <class DataTypes>
PartialLinearMovementConstraint<DataTypes>::~PartialLinearMovementConstraint()
{
}

template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::clearIndices()
{
    m_indices.beginEdit()->clear();
    m_indices.endEdit();
}

template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::addIndex(unsigned int index)
{
    m_indices.beginEdit()->push_back(index);
    m_indices.endEdit();
}

template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::removeIndex(unsigned int index)
{
    removeValue(*m_indices.beginEdit(),index);
    m_indices.endEdit();
}

template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::clearKeyMovements()
{
    m_keyTimes.beginEdit()->clear();
    m_keyTimes.endEdit();
    m_keyMovements.beginEdit()->clear();
    m_keyMovements.endEdit();
}

template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::addKeyMovement(Real time, Deriv movement)
{
    m_keyTimes.beginEdit()->push_back( time );
    m_keyTimes.endEdit();
    m_keyMovements.beginEdit()->push_back( movement );
    m_keyMovements.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::init()
{
    this->core::componentmodel::behavior::Constraint<DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    topology::PointSubset my_subset = m_indices.getValue();

    my_subset.setTestFunction(FCTestNewPointFunction);
    my_subset.setRemovalFunction(FCRemovalFunction);

    my_subset.setTestParameter( (void *) this );
    my_subset.setRemovalParameter( (void *) this );

    x0.resize(0);
    nextM = prevM = Deriv();

    currentTime = -1.0;
    finished = false;
}


template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::reset()
{
    nextT = prevT = 0.0;
    nextM = prevM = Deriv();

    currentTime = -1.0;
    finished = false;
}


template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::projectResponse(VecDeriv& dx)
{
    Real cT = (Real) this->getContext()->getTime();
    Vec6Bool movedDirection = movedDirections.getValue();
    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    if (finished && nextT != prevT)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();

        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for( unsigned j=0; j<NumDimensions; j++)
                if(movedDirection[j]) dx[*it][j] = (Real) 0.0;
        }
    }
}

template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::projectVelocity(VecDeriv& dx)
{
    Real cT = (Real) this->getContext()->getTime();
    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    if (finished && nextT != prevT)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();

        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            dx[*it] = (nextM - prevM)*(1.0 / (nextT - prevT));
        }
    }
}


template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::projectPosition(VecCoord& x)
{
    Real cT = (Real) this->getContext()->getTime();

    //initialize initial Dofs positions, if it's not done
    if (x0.size() == 0)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();
        x0.resize( x.size() );
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            x0[*it] = x[*it];
    }

    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }
    Vec6Bool movedDirection = movedDirections.getValue();
    //if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
    if(finished && nextT != prevT)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();

        Real dt = (cT - prevT) / (nextT - prevT);
        Deriv m = prevM + (nextM-prevM)*dt;

        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for( unsigned j=0; j< NumDimensions; j++)
                if(movedDirection[j]) x[*it][j] = x0[*it][j] + m[j] ;
        }
    }
}

template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::findKeyTimes()
{
    Real cT = (Real) this->getContext()->getTime();
    finished = false;

    if(m_keyTimes.getValue().size() != 0 && cT >= *m_keyTimes.getValue().begin() && cT <= *m_keyTimes.getValue().rbegin())
    {
        nextT = *m_keyTimes.getValue().begin();
        prevT = nextT;

        typename helper::vector<Real>::const_iterator it_t = m_keyTimes.getValue().begin();
        typename VecDeriv::const_iterator it_m = m_keyMovements.getValue().begin();

        //WARNING : we consider that the key-events are in chronological order
        //here we search between which keyTimes we are, to know which are the motion to interpolate
        while( it_t != m_keyTimes.getValue().end() && !finished)
        {
            if( *it_t <= cT)
            {
                prevT = *it_t;
                prevM = *it_m;
            }
            else
            {
                nextT = *it_t;
                nextM = *it_m;
                finished = true;
            }
            it_t++;
            it_m++;
        }
    }
}


//display the path the constrained dofs will go through
template <class DataTypes>
void PartialLinearMovementConstraint<DataTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels() || m_keyTimes.getValue().size() == 0 ) return;
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_LINES);
    const SetIndexArray & indices = m_indices.getValue().getArray();
    for (unsigned int i=0 ; i<m_keyMovements.getValue().size()-1 ; i++)
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            gl::glVertexT(x0[*it]+m_keyMovements.getValue()[i]);
            gl::glVertexT(x0[*it]+m_keyMovements.getValue()[i+1]);
        }
    }
    glEnd();
}

// Specialization for rigids
template <>
void PartialLinearMovementConstraint<Rigid3dTypes >::draw();
template <>
void PartialLinearMovementConstraint<Rigid3fTypes >::draw();

} // namespace constraint

} // namespace component

} // namespace sofa

#endif

