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

#include <sofa/component/constraint/projective/LinearMovementProjectiveConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/type/vector_algorithm.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <algorithm>

namespace sofa::component::constraint::projective
{

template <class DataTypes>
LinearMovementProjectiveConstraint<DataTypes>::LinearMovementProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr)
    , data(new LinearMovementProjectiveConstraintInternalData<DataTypes>)
    , m_indices( initData(&m_indices,"indices","Indices of the constrained points") )
    , m_keyTimes(  initData(&m_keyTimes,"keyTimes","key times for the movements") )
    , m_keyMovements(  initData(&m_keyMovements,"movements","movements corresponding to the key times") )
    , d_relativeMovements( initData(&d_relativeMovements, bool(true), "relativeMovements", "If true, movements are relative to first position, absolute otherwise") )
    , showMovement( initData(&showMovement, bool(false), "showMovement", "Visualization of the movement to be applied to constrained dofs."))
    , l_topology(initLink("topology", "link to the topology container"))
    , finished(false)
{
    // default to indice 0
    m_indices.beginEdit()->push_back(0);
    m_indices.endEdit();

    //default valueEvent to 0
    m_keyTimes.beginEdit()->push_back( 0.0 );
    m_keyTimes.endEdit();
    m_keyMovements.beginEdit()->push_back( Deriv() );
    m_keyMovements.endEdit();
}



template <class DataTypes>
LinearMovementProjectiveConstraint<DataTypes>::~LinearMovementProjectiveConstraint()
{

}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::clearIndices()
{
    m_indices.beginEdit()->clear();
    m_indices.endEdit();
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::addIndex(Index index)
{
    m_indices.beginEdit()->push_back(index);
    m_indices.endEdit();
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::removeIndex(Index index)
{
    sofa::type::removeValue(*m_indices.beginEdit(),index);
    m_indices.endEdit();
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::clearKeyMovements()
{
    m_keyTimes.beginEdit()->clear();
    m_keyTimes.endEdit();
    m_keyMovements.beginEdit()->clear();
    m_keyMovements.endEdit();
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::addKeyMovement(Real time, Deriv movement)
{
    m_keyTimes.beginEdit()->push_back( time );
    m_keyTimes.endEdit();
    m_keyMovements.beginEdit()->push_back( movement );
    m_keyMovements.endEdit();
}

// -- Constraint interface
template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::init()
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
        m_indices.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }

    x0.resize(0);
    nextM = prevM = Deriv();

    currentTime = -1.0;
    finished = false;
}


template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::reset()
{
    nextT = prevT = 0.0;
    nextM = prevM = Deriv();

    currentTime = -1.0;
    finished = false;
}


template <class DataTypes>
template <class DataDeriv>
void LinearMovementProjectiveConstraint<DataTypes>::projectResponseT(DataDeriv& dx,
    const std::function<void(DataDeriv&, const unsigned int)>& clear)
{
    Real cT = static_cast<Real>(this->getContext()->getTime());
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
            clear(dx, *it);
        }
    }
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataVecDeriv> res = resData;
    projectResponseT<VecDeriv>(res.wref(), [](auto& dx, const unsigned int index) {dx[index].clear(); });
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& vData)
{
    helper::WriteAccessor<DataVecDeriv> dx = vData;
    Real cT = static_cast<Real>(this->getContext()->getTime());
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
            dx[*it] = (nextM - prevM)*(1.0 / (nextT - prevT));
        }
    }
}


template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
{
    helper::WriteAccessor<DataVecCoord> x = xData;
    Real cT = static_cast<Real>(this->getContext()->getTime());

    //initialize initial Dofs positions, if it's not done
    if (x0.size() == 0)
    {
        const SetIndexArray & indices = m_indices.getValue();
        x0.resize(x.size());
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x0[*it] = x[*it];
        }
    }

    if ((cT != currentTime) || !finished)
    {
        findKeyTimes();
    }

    //if we found 2 keyTimes, we have to interpolate a velocity (linear interpolation)
    if(finished && nextT != prevT)
    {
        interpolatePosition<Coord>(cT, x.wref());
    }
}

template <class DataTypes>
template <class MyCoord>
void LinearMovementProjectiveConstraint<DataTypes>::interpolatePosition(Real cT, typename std::enable_if<!std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x)
{
    const SetIndexArray & indices = m_indices.getValue();

    Real dt = (cT - prevT) / (nextT - prevT);
    Deriv m = prevM + (nextM-prevM)*dt;

    //set the motion to the Dofs
    if (d_relativeMovements.getValue())
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it] = x0[*it] + m ;
        }
    }
    else
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it] = m ;
        }
    }
}

template <class DataTypes>
template <class MyCoord>
void LinearMovementProjectiveConstraint<DataTypes>::interpolatePosition(Real cT, typename std::enable_if<std::is_same<MyCoord, defaulttype::RigidCoord<3, Real> >::value, VecCoord>::type& x)
{
    const SetIndexArray & indices = m_indices.getValue();

    Real dt = (cT - prevT) / (nextT - prevT);
    Deriv m = prevM + (nextM-prevM)*dt;
    type::Quat<Real> prevOrientation = type::Quat<Real>::createQuaterFromEuler(getVOrientation(prevM));
    type::Quat<Real> nextOrientation = type::Quat<Real>::createQuaterFromEuler(getVOrientation(nextM));

    //set the motion to the Dofs
    if (d_relativeMovements.getValue())
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it].getCenter() = x0[*it].getCenter() + getVCenter(m) ;
            x[*it].getOrientation() = x0[*it].getOrientation() * prevOrientation.slerp2(nextOrientation, dt);
        }
    }
    else
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            x[*it].getCenter() =  getVCenter(m) ;
            x[*it].getOrientation() = prevOrientation.slerp2(nextOrientation, dt);
        }
    }
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);
    helper::WriteAccessor<DataMatrixDeriv> c = cData;

    projectResponseT<MatrixDeriv>(c.wref(), [](MatrixDeriv& res, const unsigned int index) { res.clearColBlock(index); });
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::findKeyTimes()
{
    Real cT = static_cast<Real>(this->getContext()->getTime());
    finished = false;

    if(m_keyTimes.getValue().size() != 0 && cT >= *m_keyTimes.getValue().begin() && cT <= *m_keyTimes.getValue().rbegin())
    {
        nextT = *m_keyTimes.getValue().begin();
        prevT = nextT;

        typename type::vector<Real>::const_iterator it_t = m_keyTimes.getValue().begin();
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
            ++it_t;
            ++it_m;
        }
    }
}


template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset )
{
    static const unsigned blockSize = DataTypes::deriv_total_size;

    // clears the rows and columns associated with fixed particles
    for (const auto id : m_indices.getValue())
    {
        M->clearRowsCols( offset + id * blockSize, offset + (id+1) * blockSize );
    }
    
}


// Matrix Integration interface
template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    if(const core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate.get()))
    {
        const unsigned int N = Deriv::size();
        const SetIndexArray & indices = m_indices.getValue();

        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            // Reset Fixed Row and Col
            for (unsigned int c=0; c<N; ++c)
                r.matrix->clearRowCol(r.offset + N * (*it) + c);
            // Set Fixed Vertex
            for (unsigned int c=0; c<N; ++c)
                r.matrix->set(r.offset + N * (*it) + c, r.offset + N * (*it) + c, 1.0);
        }
    }
}

template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    const int o = matrix->getGlobalOffset(this->mstate.get());
    if (o >= 0)
    {
        const unsigned int offset = (unsigned int)o;
        const unsigned int N = Deriv::size();

        const SetIndexArray & indices = m_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            for (unsigned int c=0; c<N; ++c)
                vector->clear(offset + N * (*it) + c);
        }
    }
}

template <class TDataTypes>
void LinearMovementProjectiveConstraint<TDataTypes>::applyConstraint(
    sofa::core::behavior::ZeroDirichletCondition* matrix)
{
    constexpr unsigned int N = Deriv::size();

    for (const auto index : m_indices.getValue())
    {
        for (unsigned int c = 0; c < N; ++c)
        {
            matrix->discardRowCol(N * index + c, N * index + c);
        }
    }
}

//display the path the constrained dofs will go through
template <class DataTypes>
void LinearMovementProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels() || m_keyTimes.getValue().size() == 0)
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    constexpr sofa::type::RGBAColor color(1, 0.5, 0.5, 1);

    if (showMovement.getValue())
    {
        vparams->drawTool()->disableLighting();

        std::vector<sofa::type::Vec3> vertices;

        constexpr auto minDimensions = std::min<sofa::Size>(DataTypes::spatial_dimensions, 3u);

        const SetIndexArray & indices = m_indices.getValue();
        const VecDeriv& keyMovements = m_keyMovements.getValue();
        if (d_relativeMovements.getValue()) 
        {
            for (unsigned int i = 0; i < m_keyMovements.getValue().size() - 1; i++)
            {
                for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
                {
                    const auto& tmp0 = DataTypes::getCPos(x0[*it]) + DataTypes::getDPos(keyMovements[i]);
                    const auto& tmp1 = DataTypes::getCPos(x0[*it]) + DataTypes::getDPos(keyMovements[i + 1]);
                    sofa::type::Vec3 v0, v1;
                    std::copy_n(tmp0.begin(), minDimensions, v0.begin());
                    std::copy_n(tmp1.begin(), minDimensions, v1.begin());
                    vertices.push_back(v0);
                    vertices.push_back(v1);
                }
            }
        } 
        else 
        {
            for (unsigned int i = 0; i < keyMovements.size() - 1; i++)
            {
                for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
                {
                    const auto& tmp0 = DataTypes::getDPos(keyMovements[i]);
                    const auto& tmp1 = DataTypes::getDPos(keyMovements[i + 1]);
                    sofa::type::Vec3 v0, v1;
                    std::copy_n(tmp0.begin(), minDimensions, v0.begin());
                    std::copy_n(tmp1.begin(), minDimensions, v1.begin());
                    vertices.push_back(v0);
                    vertices.push_back(v1);
                }
            }
        }
        vparams->drawTool()->drawLines(vertices, 10, color);
    }
    else
    {
        const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

        sofa::type::vector<type::Vec3> points;
        type::Vec3 point;
        const SetIndexArray & indices = m_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            point = DataTypes::getCPos(x[*it]);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, color);
    }


}

} // namespace sofa::component::constraint::projective
