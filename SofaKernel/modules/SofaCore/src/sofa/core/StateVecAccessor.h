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

#include <sofa/core/fwd.h>
#include <sofa/core/State.h>
#include <sofa/core/MultiVecId[V_COORD].h>
#include <sofa/core/MultiVecId[V_DERIV].h>
#include <sofa/core/MultiVecId[V_MATDERIV].h>

namespace sofa::core
{

template<class Type, class DataType=typename State<Type>::VecCoord>
const sofa::core::objectmodel::Data<DataType>* getRead(const State<Type>* state, const TMultiVecId<V_COORD, V_READ>& id)
{
    return state->read(id.getId(state));
}

template<class Type, class DataType=typename State<Type>::VecDeriv>
const sofa::core::objectmodel::Data<DataType>* getRead(const State<Type>* state, const TMultiVecId<V_DERIV, V_READ>& id)
{
    return state->read(id.getId(state));
}

template<class Type, class DataType=typename State<Type>::MatrixDeriv>
const sofa::core::objectmodel::Data<DataType>* getRead(const State<Type>* state, const TMultiVecId<V_MATDERIV, V_READ>& id)
{
    return state->read(id.getId(state));
}

template<class Type, class DataType=typename State<Type>::VecCoord>
sofa::core::objectmodel::Data<DataType>* getWrite(State<Type>* state, const TMultiVecId<V_COORD, V_WRITE>& id)
{
    return state->write(id.getId(state));
}

template<class Type, class DataType=typename State<Type>::VecDeriv>
sofa::core::objectmodel::Data<DataType>* getWrite(State<Type>* state, const TMultiVecId<V_DERIV, V_WRITE>& id)
{
    return state->write(id.getId(state));
}

template<class Type, class DataType=typename State<Type>::MatrixDeriv>
sofa::core::objectmodel::Data<DataType>* getWrite(State<Type>* state, const TMultiVecId<V_MATDERIV, V_WRITE>& id)
{
    return state->write(id.getId(state));
}

} // namespace sofa::core

