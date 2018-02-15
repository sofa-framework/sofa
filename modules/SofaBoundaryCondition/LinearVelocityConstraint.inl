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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARVELOCITYCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARVELOCITYCONSTRAINT_INL

#include <SofaBoundaryCondition/LinearVelocityConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <SofaBaseTopology/TopologySubsetData.inl>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{


// Define TestNewPointFunction
template< class TDataTypes>
bool LinearVelocityConstraint<TDataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    return lc != 0;
}

// Define RemovalFunction
template< class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, value_type &)
{
    if (lc)
    {
        lc->removeIndex((unsigned int) pointIndex);
    }
}

template <class TDataTypes>
LinearVelocityConstraint<TDataTypes>::LinearVelocityConstraint()
    : core::behavior::ProjectiveConstraintSet<TDataTypes>(NULL)
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_keyTimes(  initData(&m_keyTimes,"keyTimes","key times for the movements") )
    , m_keyVelocities(  initData(&m_keyVelocities,"velocities","velocities corresponding to the key times") )
    , m_coordinates( initData(&m_coordinates, "coordinates", "coordinates on which to apply velocities") )
{
    // default to indice 0
    m_indices.beginEdit()->push_back(0);
    m_indices.endEdit();

    //default valueEvent to 0
    m_keyTimes.beginEdit()->push_back( 0.0 );
    m_keyTimes.endEdit();
    m_keyVelocities.beginEdit()->push_back( Deriv() );
    m_keyVelocities.endEdit();

    pointHandler = new FCPointHandler(this, &m_indices);
}


template <class TDataTypes>
LinearVelocityConstraint<TDataTypes>::~LinearVelocityConstraint()
{
    if (pointHandler)
        delete pointHandler;
}

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::clearIndices()
{
    m_indices.beginEdit()->clear();
    m_indices.endEdit();
}

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::addIndex(unsigned int index)
{
    m_indices.beginEdit()->push_back(index);
    m_indices.endEdit();
}

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::removeIndex(unsigned int index)
{
    removeValue(*m_indices.beginEdit(),index);
    m_indices.endEdit();
}

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::clearKeyVelocities()
{
    m_keyTimes.beginEdit()->clear();
    m_keyTimes.endEdit();
    m_keyVelocities.beginEdit()->clear();
    m_keyVelocities.endEdit();
}

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::addKeyVelocity(Real time, Deriv movement)
{
    m_keyTimes.beginEdit()->push_back( time );
    m_keyTimes.endEdit();
    m_keyVelocities.beginEdit()->push_back( movement );
    m_keyVelocities.endEdit();
}

// -- Constraint interface


template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<TDataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    m_indices.createTopologicalEngine(topology, pointHandler);
    m_indices.registerTopologicalData();

    m_coordinates.createTopologicalEngine(topology);
    m_coordinates.registerTopologicalData();

    x0.resize(0);
    xP.resize(0);
    nextV = prevV = Deriv();

    currentTime = -1.0;
    finished = false;
}

template <class DataTypes>
void LinearVelocityConstraint<DataTypes>::reset()
{
    nextT = prevT = 0.0;
    nextV = prevV = Deriv();

    currentTime = -1.0;
    finished = false;
}


template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::projectResponse(const core::MechanicalParams* /*mparams*/, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res = resData;
    VecDeriv& dx = res.wref();

    Real cT = (Real) this->getContext()->getTime();
    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    if (finished && nextT != prevT)
    {
        const SetIndexArray & indices = m_indices.getValue();

        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            dx[*it] = Deriv();
        }
    }

}

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real cT = (Real) this->getContext()->getTime();

    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    if (finished && nextT != prevT)
    {
        //if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
        Deriv v = ((nextV - prevV)*((cT - prevT)/(nextT - prevT))) + prevV;

        const SetIndexArray & indices = m_indices.getValue();
        const SetIndexArray & coordinates = m_coordinates.getValue();

        if (coordinates.size() == 0)
        {
            //set the motion to the Dofs
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                dx[*it] = v;
            }
        }
        else
        {
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                for(SetIndexArray::const_iterator itInd = coordinates.begin(); itInd != coordinates.end(); ++itInd)
                {
                    dx[*it][*itInd] = v[*itInd];
                }
            }
        }
    }
}


template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;

    //initialize initial Dofs positions, if it's not done
    if (x0.size() == 0)
    {
        const SetIndexArray & indices = m_indices.getValue();
        x0.resize( x.size() );
        xP.resize( x.size() );
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x0[*it] = x[*it];
            xP[*it] = x0[*it];
        }
    }

    Real cT = (Real) this->getContext()->getTime();

    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }


    Real dTsimu = (Real) this->getContext()->getDt();


    if(finished)
    {
        Real dt = (cT - prevT) / (nextT - prevT);
        Deriv m = (nextV-prevV)*dt + prevV;

        const SetIndexArray & indices = m_indices.getValue();
        const SetIndexArray & coordinates = m_coordinates.getValue();

        if (coordinates.size() == 0)
        {
            //set the motion to the Dofs
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                x[*it] = xP[*it] + m*dTsimu;
                xP[*it] = x[*it];
            }
        }
        else
        {
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                for(SetIndexArray::const_iterator itInd = coordinates.begin(); itInd != coordinates.end(); ++itInd)
                {
                    x[*it][*itInd] = xP[*it][*itInd] + m[*itInd]*dTsimu;
                    xP[*it] = x[*it];
                }
            }
        }
    }
}

template <class DataTypes>
void LinearVelocityConstraint<DataTypes>::findKeyTimes()
{
    Real cT = (Real) this->getContext()->getTime();
    finished = false;

    if(m_keyTimes.getValue().size() != 0 && cT >= *m_keyTimes.getValue().begin() && cT <= *m_keyTimes.getValue().rbegin())
    {
        nextT = *m_keyTimes.getValue().begin();
        prevT = nextT;

        typename helper::vector<Real>::const_iterator it_t = m_keyTimes.getValue().begin();
        typename VecDeriv::const_iterator it_v = m_keyVelocities.getValue().begin();

        //WARNING : we consider that the key-events are in chronological order
        //here we search between which keyTimes we are, to know which are the motion to interpolate
        while( it_t != m_keyTimes.getValue().end() && !finished)
        {
            if( *it_t <= cT)
            {
                prevT = *it_t;
                prevV = *it_v;
            }
            else
            {
                nextT = *it_t;
                nextV = *it_v;
                finished = true;
            }
            it_t++;
            it_v++;
        }
    }
}// LinearVelocityConstraint::findKeyTimes

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /*cData*/)
{

}

//display the path the constrained dofs will go through
template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels() || m_keyTimes.getValue().size() == 0 ) return;
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_LINES);
    const SetIndexArray & indices = m_indices.getValue();
    for (unsigned int i=0 ; i<m_keyVelocities.getValue().size()-1 ; i++)
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            sofa::helper::gl::glVertexT(x0[*it]+m_keyVelocities.getValue()[i]);
            sofa::helper::gl::glVertexT(x0[*it]+m_keyVelocities.getValue()[i+1]);
        }
    }
    glEnd();
#endif /* SOFA_NO_OPENGL */
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif


