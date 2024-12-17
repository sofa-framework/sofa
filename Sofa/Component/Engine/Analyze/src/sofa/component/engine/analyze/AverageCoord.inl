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
#include <sofa/component/engine/analyze/AverageCoord.h>
#include <iostream>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::component::engine::analyze
{

template <class DataTypes>
AverageCoord<DataTypes>::AverageCoord()
    : d_indices( initData (&d_indices, "indices", "indices of the coordinates to average") )
    , d_vecId(initData (&d_vecId, sofa::core::vec_id::write_access::position.getIndex(), "vecId", "index of the vector (default value corresponds to core::vec_id::write_access::position )") )
    , d_average( initData (&d_average, "average", "average of the values with the given indices in the given coordinate vector \n"
                                                   "(default value corresponds to the average coord of the mechanical context)") )
{
    addInput(&d_indices);
    addInput(&d_vecId);
    addOutput(&d_average);
}

template <class DataTypes>
void AverageCoord<DataTypes>::init()
{
    core::behavior::SingleStateAccessor<DataTypes>::init();

    setDirtyValue();
}

template <class DataTypes>
void AverageCoord<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void AverageCoord<DataTypes>::doUpdate()
{
    if(this->mstate==nullptr)
    {
        msg_info(this) << "This component requires a mechanical state in its context.";
        return;
    }

    const Data<VecCoord>* coordPtr = this->mstate->read(core::VecCoordId(d_vecId.getValue()));

    if (!coordPtr)
    {
        msg_error() << "Cannot get coordinates from VecId " << d_vecId.getValue();
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }

    helper::ReadAccessor< Data<VecCoord> > coord = *coordPtr;
    const VecIndex& indices = d_indices.getValue();

    Coord c;
    const auto n = (indices.empty()) ? coord.size() : indices.size();

    if (n == 0)
    {
        msg_error() << "Trying to average an empty list of coordinates";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
        return;
    }

    for( std::size_t i = 0; i < n; ++i )
    {
        c += coord[ (indices.empty()) ? i : indices[i]];
    }
    c *= 1./ static_cast<typename Coord::value_type>(n);

    d_average.setValue(c);
}

template<class DataTypes>
void AverageCoord<DataTypes>::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
        this->onBeginAnimationStep(this->getContext()->getDt());
}

template <class DataTypes>
void AverageCoord<DataTypes>::onBeginAnimationStep(const double dt)
{
    SOFA_UNUSED(dt);
    update();
}

} //namespace sofa::component::engine::analyze
