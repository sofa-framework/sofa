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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARVELOCITYCONSTRAINT_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_LINEARVELOCITYCONSTRAINT_INL

#include <sofa/component/projectiveconstraintset/LinearVelocityConstraint.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/helper/gl/template.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>

#include <sofa/component/topology/PointSubset.h>


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
using namespace sofa::core::objectmodel;


// Define TestNewPointFunction
template< class TDataTypes>
bool LinearVelocityConstraint<TDataTypes>::FCTestNewPointFunction(int /*nbPoints*/, void* param, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& )
{
    LinearVelocityConstraint<TDataTypes> *fc= (LinearVelocityConstraint<TDataTypes> *)param;
    return fc != 0;
}

// Define RemovalFunction
template< class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::FCRemovalFunction(int pointIndex, void* param)
{
    LinearVelocityConstraint<TDataTypes> *fc= (LinearVelocityConstraint<TDataTypes> *)param;
    if (fc)
    {
        fc->removeIndex((unsigned int) pointIndex);
    }
}

template <class TDataTypes>
LinearVelocityConstraint<TDataTypes>::LinearVelocityConstraint()
    : core::behavior::ProjectiveConstraintSet<TDataTypes>(NULL)
    , m_indices( BaseObject::initData(&m_indices,"indices","Indices of the constrained points") )
    , m_keyTimes(  BaseObject::initData(&m_keyTimes,"keyTimes","key times for the movements") )
    , m_keyVelocities(  BaseObject::initData(&m_keyVelocities,"velocities","velocities corresponding to the key times") )
    , m_coordinates( BaseObject::initData(&m_coordinates, "coordinates", "coordinates on which to apply velocities") )
{
    // default to indice 0
    m_indices.beginEdit()->push_back(0);
    m_indices.endEdit();

    //default valueEvent to 0
    m_keyTimes.beginEdit()->push_back( 0.0 );
    m_keyTimes.endEdit();
    m_keyVelocities.beginEdit()->push_back( Deriv() );
    m_keyVelocities.endEdit();
}


// Handle topological changes
template <class TDataTypes> void LinearVelocityConstraint<TDataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=topology->endChange();

    m_indices.beginEdit()->handleTopologyEvents(itBegin,itEnd,this->getMState()->getSize());
}

template <class TDataTypes>
LinearVelocityConstraint<TDataTypes>::~LinearVelocityConstraint()
{
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
    topology::PointSubset my_subset = m_indices.getValue();

    my_subset.setTestFunction(FCTestNewPointFunction);
    my_subset.setRemovalFunction(FCRemovalFunction);

    my_subset.setTestParameter( (void *) this );
    my_subset.setRemovalParameter( (void *) this );

    x0.resize(0);
    nextV = prevV = Deriv();
}


template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::projectResponse(DataVecDeriv& /*resData*/, const core::MechanicalParams* /*mparams*/)
{

}

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::projectVelocity(DataVecDeriv& vData, const core::MechanicalParams* /*mparams*/)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real cT = (Real) this->getContext()->getTime();

    if(m_keyTimes.getValue().size() != 0 && cT >= *m_keyTimes.getValue().begin() && cT <= *m_keyTimes.getValue().rbegin() && nextT!=prevT)
    {
        nextT = *m_keyTimes.getValue().begin();
        prevT = nextT;

        bool finished=false;

        typename helper::vector<Real>::const_iterator it_t = m_keyTimes.getValue().begin();
        typename VecDeriv::const_iterator it_m = m_keyVelocities.getValue().begin();

        //WARNING : we consider that the key-events are in chronological order
        //here we search between which keyTimes we are, to know which are the motion to interpolate
        while( it_t != m_keyTimes.getValue().end() && !finished)
        {
            if( *it_t <= cT)
            {
                prevT = *it_t;
                prevV = *it_m;
            }
            else
            {
                nextT = *it_t;
                nextV = *it_m;
                finished = true;
            }
            it_t++;
            it_m++;
        }
        const SetIndexArray & indices = m_indices.getValue().getArray();
        const SetIndexArray & coordinates = m_coordinates.getValue().getArray();

        if (finished)
        {
            //if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
            Deriv v = ((nextV - prevV)*((cT - prevT)/(nextT - prevT))) + prevV;

#if 0
            std::cout<<"LinearVelocityConstraint::projectVelocity, TIME = "<<cT<<", v = "<<v<<std::endl;
#endif

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
}


template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::projectPosition(DataVecCoord& xData, const core::MechanicalParams* /*mparams*/)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    //initialize initial Dofs positions, if it's not done
    if (x0.size() == 0)
    {
        const SetIndexArray & indices = m_indices.getValue().getArray();
        x0.resize( x.size() );
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            x0[*it] = x[*it];
    }

    Real cT = (Real) this->getContext()->getTime();

    //if we found 2 keyTimes, we have to interpolate a position (linear interpolation)
    if(m_keyTimes.getValue().size() != 0 && cT >= *m_keyTimes.getValue().begin() && cT <= *m_keyTimes.getValue().rbegin())
    {

        nextT = *m_keyTimes.getValue().begin();
        prevT = nextT;

        bool finished=false;

        typename helper::vector<Real>::const_iterator it_t = m_keyTimes.getValue().begin();
        typename VecDeriv::const_iterator it_m = m_keyVelocities.getValue().begin();

        //WARNING : we consider that the key-events are in chronological order
        //here we search between which keyTimes we are, to know which are the motion to interpolate
        while( it_t != m_keyTimes.getValue().end() && !finished)
        {
            if( *it_t <= cT)
            {
                prevT = *it_t;
                prevV = *it_m;
            }
            else
            {
                nextT = *it_t;
                nextV = *it_m;
                finished = true;
            }
            it_t++;
            it_m++;
        }

        const SetIndexArray & indices = m_indices.getValue().getArray();
        const SetIndexArray & coordinates = m_coordinates.getValue().getArray();
        Real dTsimu = (Real) this->getContext()->getDt();


        if(finished)
        {
            Real dt = (cT - prevT) / (nextT - prevT);
            Deriv m = (nextV-prevV)*dt + prevV;

#if 0
            std::cout<<"LinearVelocityConstraint::projectPosition, TIME = "<<cT<<", m = "<<m<<", dTsimu = "<<dTsimu<<std::endl;
#endif

            if (coordinates.size() == 0)
            {
                //set the motion to the Dofs
                for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
                {
                    x[*it] = x[*it] + m*dTsimu;
                }
            }
            else
            {
                for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
                {
                    for(SetIndexArray::const_iterator itInd = coordinates.begin(); itInd != coordinates.end(); ++itInd)
                    {
                        x[*it][*itInd] = x[*it][*itInd] + m[*itInd]*dTsimu;
#if 0
                        std::cout<<"LinearVelocityConstraint::projectPosition x["<<*it<<"] = "<<x[*it][*itInd]<<std::endl;
#endif
                    }
                }
            }
        }
    }
}

template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::projectJacobianMatrix(DataMatrixDeriv& /*cData*/, const core::MechanicalParams* /*mparams*/)
{

}

//display the path the constrained dofs will go through
template <class TDataTypes>
void LinearVelocityConstraint<TDataTypes>::draw()
{
    if (!this->getContext()->getShowBehaviorModels() || m_keyTimes.getValue().size() == 0 ) return;
    glDisable (GL_LIGHTING);
    glPointSize(10);
    glColor4f (1,0.5,0.5,1);
    glBegin (GL_LINES);
    const SetIndexArray & indices = m_indices.getValue().getArray();
    for (unsigned int i=0 ; i<m_keyVelocities.getValue().size()-1 ; i++)
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            gl::glVertexT(x0[*it]+m_keyVelocities.getValue()[i]);
            gl::glVertexT(x0[*it]+m_keyVelocities.getValue()[i+1]);
        }
    }
    glEnd();
}

// Specialization for rigids
template <>
void LinearVelocityConstraint<Rigid3dTypes >::draw();
template <>
void LinearVelocityConstraint<Rigid3fTypes >::draw();

} // namespace constraint

} // namespace component

} // namespace sofa

#endif


