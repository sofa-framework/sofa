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
    , l_topology(initLink("topology", "link to the topology container"))
    , finished(false)
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

    x0.resize(0);
    xP.resize(0);
    nextV = prevV = Deriv();

    currentTime = -1.0;
    finished = false;
}

template <class DataTypes>
void LinearVelocityProjectiveConstraint<DataTypes>::reset()
{
    nextT = prevT = 0.0;
    nextV = prevV = Deriv();

    currentTime = -1.0;
    finished = false;
}


template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::projectResponse(const core::MechanicalParams* /*mparams*/, DataVecDeriv& resData)
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
    if (x0.size() == 0)
    {
        const SetIndexArray & indices = d_indices.getValue();
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

        const SetIndexArray & indices = d_indices.getValue();
        const SetIndexArray & coordinates = d_coordinates.getValue();

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
void LinearVelocityProjectiveConstraint<DataTypes>::findKeyTimes()
{
    Real cT = (Real) this->getContext()->getTime();
    finished = false;

    if(d_keyTimes.getValue().size() != 0 && cT >= *d_keyTimes.getValue().begin() && cT <= *d_keyTimes.getValue().rbegin())
    {
        nextT = *d_keyTimes.getValue().begin();
        prevT = nextT;

        typename type::vector<Real>::const_iterator it_t = d_keyTimes.getValue().begin();
        typename VecDeriv::const_iterator it_v = d_keyVelocities.getValue().begin();

        //WARNING : we consider that the key-events are in chronological order
        //here we search between which keyTimes we are, to know which are the motion to interpolate
        while( it_t != d_keyTimes.getValue().end() && !finished)
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
            ++it_t;
            ++it_v;
        }
    }
}// LinearVelocityProjectiveConstraint::findKeyTimes

template <class TDataTypes>
void LinearVelocityProjectiveConstraint<TDataTypes>::projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /*cData*/)
{

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
            const auto p0 = x0[index]+keyVelocities[i];
            const auto p1 = x0[index]+keyVelocities[i+1];

            const typename DataTypes::CPos& cpos0 = DataTypes::getCPos(p0);
            const typename DataTypes::CPos& cpos1 = DataTypes::getCPos(p1);

            vertices.push_back(sofa::type::Vec3(cpos0));
            vertices.push_back(sofa::type::Vec3(cpos1));
        }

    }

    vparams->drawTool()->drawLines(vertices, 1.0, color);


}

} // namespace sofa::component::constraint::projective
