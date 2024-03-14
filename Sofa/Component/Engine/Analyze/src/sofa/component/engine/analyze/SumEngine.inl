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
#include <sofa/component/engine/analyze/SumEngine.h>
#include <sofa/helper/logging/Messaging.h>
#include <numeric>


namespace sofa::component::engine::analyze
{

template <class DataTypes>
SumEngine<DataTypes>::SumEngine()
    : d_input ( initData (&d_input, "input", "input vector") )
    , d_output( initData (&d_output, "output", "output sum") )
{
    addInput(&d_input);
    addOutput(&d_output);
}

template <class DataType>
void SumEngine<DataType>::init()
{
    setDirtyValue();
}

template <class DataType>
void SumEngine<DataType>::reinit()
{
    update();
}

template <class DataType>
void SumEngine<DataType>::doUpdate()
{
    helper::ReadAccessor<Data<VecData> > in = d_input;

    helper::WriteOnlyAccessor<Data<DataType> > out = d_output;
    out.wref() = std::accumulate(in.begin(), in.end(), DataType() );
}

} //namespace sofa::component::engine::analyze
