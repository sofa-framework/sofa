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
#ifndef SOFA_COMPONENT_ENGINE_AverageCoord_INL
#define SOFA_COMPONENT_ENGINE_AverageCoord_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "AverageCoord.h"
#include <sofa/helper/gl/template.h>
#include <iostream>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
AverageCoord<DataTypes>::AverageCoord()
    : f_indices( initData (&f_indices, "indices", "indices of the coordinates to average") )
    , f_vecId(initData (&f_vecId, sofa::core::VecCoordId::position().getIndex(), "vecId", "index of the vector (default value corresponds to core::VecCoordId::position() )") )
    , f_average( initData (&f_average, "average", "average of the values with the given indices in the given coordinate vector") )
{
}

template <class DataTypes>
void AverageCoord<DataTypes>::init()
{
    mstate = dynamic_cast< sofa::core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    addInput(&f_indices);
    addInput(&f_vecId);
    addOutput(&f_average);
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
    cleanDirty();

    helper::ReadAccessor< Data<VecCoord> > coord = *mstate->read(core::VecCoordId(f_vecId.getValue()));
    const VecIndex& indices = f_indices.getValue();

    Coord c;
    unsigned int n = (indices.empty()) ? coord.size() : indices.size();

    for( unsigned i=0; i< n; ++i )
    {
        c += coord[ (indices.empty()) ? i : indices[i]];
//        cerr<<"AverageCoord<DataTypes>::update, coord = "<< coord[indices[i]] << ", new average = " << c << endl;
    }
    c *= 1./n;

//    cerr<<"AverageCoord<DataTypes>::update, c= "<< c << endl;

    f_average.setValue(c,true);
}

template<class DataTypes>
void AverageCoord<DataTypes>::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
        this->onBeginAnimationStep(this->getContext()->getDt());
}

template <class DataTypes>
void AverageCoord<DataTypes>::onBeginAnimationStep(const double /*dt*/)
{
    update();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
