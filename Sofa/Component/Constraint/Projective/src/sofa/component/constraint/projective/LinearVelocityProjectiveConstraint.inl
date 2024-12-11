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

#include <sofa/component/constraint/projective/LinearVelocityProjectiveConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <sofa/type/vector_algorithm.h>


namespace sofa::component::constraint::projective
{

template <class TDataTypes>
LinearVelocityProjectiveConstraint<TDataTypes>::LinearVelocityProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<TDataTypes>(nullptr)
    , d_indices( initData(&d_indices,"indices","Indices of the constrained points") )
    , d_keyTimes(  initData(&d_keyTimes,"keyTimes","key times for the movements") )
    , d_keyVelocities(  initData(&d_keyVelocities,"velocities","velocities corresponding to the key times") )
    , d_coordinates( initData(&d_coordinates, "coordinates", "coordinates on which to apply velocities") )
    , d_continueAfterEnd( initData(&d_continueAfterEnd, false, "continueAfterEnd", "If set to true then the last velocity will still be applied after all the key events") )
    , l_topology(initLink("topology", "link to the topology container"))
    , m_finished(false)
{
    d_indices.beginEdit()->push_back(0);
    d_indices.endEdit();

    d_keyTimes.beginEdit()->push_back( 0.0 );
    d_keyTimes.endEdit();
    d_keyVelocities.beginEdit()->push_back( Deriv() );
    d_keyVelocities.endEdit();
}


template <class TDataTypes>
LinearVelocityProjectiveConstraint<TDataTypes>::~LinearVelocityProjectiveConstraint()
{

}

template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::clearIndices()
{
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::addIndex(Index index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::removeIndex(Index index)
{
    sofa::type::removeValue(*d_indices.beginEdit(),index);
    d_indices.endEdit();
}

template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::clearKeyVelocities()
{
    d_keyTimes.beginEdit()->clear();
    d_keyTimes.endEdit();
    d_keyVelocities.beginEdit()->clear();
    d_keyVelocities.endEdit();
}

template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::addKeyVelocity(Real time, Deriv movement)
{
    d_keyTimes.beginEdit()->push_back( time );
    d_keyTimes.endEdit();
    d_keyVelocities.beginEdit()->push_back( movement );
    d_keyVelocities.endEdit();
}

// -- Constraint interface


template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<TDataTypes>::init();

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
        d_coordinates.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }

    m_x0.resize(0);
    m_xP.resize(0);
    m_nextV = m_prevV = Deriv();

    m_finished = false;
}

template <class DataTypes>
void LinearVelocityProjectiveConstraint<DataTypes>::reset()
{
    m_nextT = m_prevT = 0.0;
    m_nextV = m_prevV = Deriv();

    m_finished = false;
}


template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::projectResponse(const core::MechanicalParams* /*mparams*/, DataVecDeriv& resData)
{
    if(!m_finished)
    {
        m_finished = findKeyTimes();
    }

    if (isConstraintActive())
    {
        helper::WriteAccessor<DataVecDeriv> res = resData;
        VecDeriv& dx = res.wref();

        const SetIndexArray & indices = d_indices.getValue();

        //set the motion to the Dofs
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            dx[*it] = Deriv();
        }
    }

}

template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    if(!m_finished)
    {
        m_finished = findKeyTimes();
    }

    if (isConstraintActive())
    {
        helper::WriteAccessor<DataVecDeriv> dx = vData;
        Real cT = (Real) this->getContext()->getTime();

        //if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
        Deriv v = ((m_nextV - m_prevV)*((cT - m_prevT)/(m_nextT - m_prevT))) + m_prevV;

        //If we finished the key times but continue after end is on
        if(m_finished)
        {
            v = m_prevV;
        }

        const SetIndexArray & indices = d_indices.getValue();
        const SetIndexArray & coordinates = d_coordinates.getValue();

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
void LinearVelocityProjectiveConstraint<TDataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{

    helper::WriteAccessor<DataVecCoord> x = xData;

    //initialize initial Dofs positions, if it's not done
    if (m_x0.size() == 0)
    {
        const SetIndexArray & indices = d_indices.getValue();
        m_x0.resize( x.size() );
        m_xP.resize( x.size() );
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            m_x0[*it] = x[*it];
            m_xP[*it] = m_x0[*it];
        }
    }


    if(!m_finished)
    {
        m_finished = findKeyTimes();
    }

    if(isConstraintActive())
    {
        const Real cT = (Real) this->getContext()->getTime();
        const Real dTsimu = (Real) this->getContext()->getDt();
        const Real dt = (cT - m_prevT) / (m_nextT - m_prevT);
        Deriv m = (m_nextV-m_prevV)*dt + m_prevV;

        //If we finished the key times but continue after end is on
        if(m_finished)
        {
            m = m_prevV;
        }

        const SetIndexArray & indices = d_indices.getValue();
        const SetIndexArray & coordinates = d_coordinates.getValue();

        if (coordinates.size() == 0)
        {
            //set the motion to the Dofs
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                x[*it] = m_xP[*it] + m*dTsimu;
                m_xP[*it] = x[*it];
            }
        }
        else
        {
            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                for(SetIndexArray::const_iterator itInd = coordinates.begin(); itInd != coordinates.end(); ++itInd)
                {
                    x[*it][*itInd] = m_xP[*it][*itInd] + m[*itInd]*dTsimu;
                    m_xP[*it] = x[*it];
                }
            }
        }
    }
}

template <class DataTypes>
bool LinearVelocityProjectiveConstraint<DataTypes>::isConstraintActive() const
{
    return (!m_finished || d_continueAfterEnd.getValue()) && (m_nextT != m_prevT);
}

template <class DataTypes>
bool LinearVelocityProjectiveConstraint<DataTypes>::findKeyTimes()
{
    const Real cT = (Real) this->getContext()->getTime();

    if(cT > *d_keyTimes.getValue().rbegin())
    {
        m_prevV = m_nextV;
        m_nextT = INFINITY;
        return true;
    }

    if(d_keyTimes.getValue().size() != 0 && cT >= *d_keyTimes.getValue().begin() && cT <= *d_keyTimes.getValue().rbegin())
    {
        m_nextT = *d_keyTimes.getValue().begin();
        m_prevT = m_nextT;

        typename type::vector<Real>::const_iterator it_t = d_keyTimes.getValue().begin();
        typename VecDeriv::const_iterator it_v = d_keyVelocities.getValue().begin();

        //WARNING : we consider that the key-events are in chronological order
        //here we search between which keyTimes we are, to know which are the motion to interpolate
        while( it_t != d_keyTimes.getValue().end())
        {
            if( *it_t <= cT)
            {
                m_prevT = *it_t;
                m_prevV = *it_v;
            }
            else
            {
                m_nextT = *it_t;
                m_nextV = *it_v;
                return false;
            }
            ++it_t;
            ++it_v;
        }
    }
    return false;
}// LinearVelocityProjectiveConstraint::findKeyTimes

template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /*cData*/)
{

}

template <class DataTypes>
void LinearVelocityProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);

    if(isConstraintActive())
    {
        const int o = matrix->getGlobalOffset(this->mstate.get());
        if (o >= 0)
        {
            unsigned int offset = (unsigned int)o;
            constexpr unsigned int N = Deriv::size();

            const SetIndexArray& indices = this->d_indices.getValue();
            for (const unsigned int index : indices)
            {
                for (unsigned int c = 0; c < N; ++c)
                {
                    vector->clear(offset + N * index + c);
                }
            }
        }
    }
}

template <class DataTypes>
void LinearVelocityProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    if(isConstraintActive())
    {
        if (const core::behavior::MultiMatrixAccessor::MatrixRef r =
                matrix->getMatrix(this->mstate.get()))
        {
            constexpr unsigned int N = Deriv::size();
            const SetIndexArray& indices = this->d_indices.getValue();

            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                // Reset Fixed Row and Col
                for (unsigned int c = 0; c < N; ++c)
                {
                    r.matrix->clearRowCol(r.offset + N * (*it) + c);
                }
                // Set Fixed Vertex
                for (unsigned int c = 0; c < N; ++c)
                {
                    r.matrix->set(r.offset + N * (*it) + c, r.offset + N * (*it) + c, 1.0);
                }
            }
        }
    }
}

template <class DataTypes>
void LinearVelocityProjectiveConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset )
{
    static const unsigned blockSize = DataTypes::deriv_total_size;

    if(isConstraintActive())
    {
        const SetIndexArray & indices = this->d_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for (unsigned int c = 0; c < blockSize; ++c)
            {
                M->clearRowCol( offset + (*it) * blockSize + c);
            }
        }
    }
}

template <class DataTypes>
void LinearVelocityProjectiveConstraint<DataTypes>::applyConstraint(
    sofa::core::behavior::ZeroDirichletCondition* matrix)
{
    static constexpr unsigned int N = Deriv::size();
    if(isConstraintActive())
    {
        const SetIndexArray& indices = this->d_indices.getValue();

        for (const auto index : indices)
        {
            for (unsigned int c = 0; c < N; ++c)
            {
                matrix->discardRowCol(N * index + c, N * index + c);
            }
        }
    }
}

//display the path the constrained dofs will go through
template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels() || d_keyTimes.getValue().size() == 0 ) return;
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    vparams->drawTool()->disableLighting();

    std::vector<sofa::type::Vec3> vertices;
    constexpr sofa::type::RGBAColor color(1, 0.5, 0.5, 1);
    const VecDeriv& keyVelocities = d_keyVelocities.getValue();
    const SetIndexArray & indices = d_indices.getValue();
    for (unsigned int i=0 ; i<keyVelocities.size()-1 ; i++)
    {
        for(const auto index : indices)
        {
            const auto p0 = m_x0[index]+keyVelocities[i];
            const auto p1 = m_x0[index]+keyVelocities[i+1];

            const typename DataTypes::CPos& cpos0 = DataTypes::getCPos(p0);
            const typename DataTypes::CPos& cpos1 = DataTypes::getCPos(p1);

            vertices.push_back(sofa::type::Vec3(cpos0));
            vertices.push_back(sofa::type::Vec3(cpos1));
        }

    }

    vparams->drawTool()->drawLines(vertices, 1.0, color);


}

} // namespace sofa::component::constraint::projective
