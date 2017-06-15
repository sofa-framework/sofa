/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_AverageCoord_INL
#define SOFA_COMPONENT_ENGINE_AverageCoord_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "AverageCoord.h"
#include <sofa/helper/gl/template.h>
#include <iostream>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
AverageCoord<DataTypes>::AverageCoord()
    : d_indices( initData (&d_indices, "indices", "indices of the coordinates to average") )
    , d_vecId(initData (&d_vecId, sofa::core::VecCoordId::position().getIndex(), "vecId", "index of the vector (default value corresponds to core::VecCoordId::position() )") )
    , d_average( initData (&d_average, "average", "average of the values with the given indices in the given coordinate vector \n"
                                                   "(default value corresponds to the average coord of the mechanical context)") )
{
}

template <class DataTypes>
void AverageCoord<DataTypes>::init()
{
    mstate = dynamic_cast< sofa::core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    addInput(&d_indices);
    addInput(&d_vecId);
    addOutput(&d_average);
    setDirtyValue();
}

template <class DataTypes>
void AverageCoord<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void AverageCoord<DataTypes>::update()
{
    if(mstate==NULL)
    {
        msg_info(this) << "This component requires a mechanical state in its context.";
        return;
    }

    helper::ReadAccessor< Data<VecCoord> > coord = *mstate->read(core::VecCoordId(d_vecId.getValue()));
    const VecIndex& indices = d_indices.getValue();

    Coord c;
    unsigned int n = (indices.empty()) ? coord.size() : indices.size();

    for( unsigned i=0; i< n; ++i )
    {
        c += coord[ (indices.empty()) ? i : indices[i]];
    }
    c *= 1./n;

    cleanDirty();

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

} // namespace engine

} // namespace component

} // namespace sofa

#endif
