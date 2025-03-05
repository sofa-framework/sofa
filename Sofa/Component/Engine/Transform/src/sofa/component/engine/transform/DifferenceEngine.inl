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
#include <sofa/component/engine/transform/DifferenceEngine.h>
#include <sofa/helper/logging/Messaging.h>


namespace sofa::component::engine::transform
{

template <class DataTypes>
DifferenceEngine<DataTypes>::DifferenceEngine()
    : d_input ( initData (&d_input, "input", "input vector") )
    , d_substractor ( initData (&d_substractor, "substractor", "vector to subtract to input") )
    , d_output( initData (&d_output, "output", "output vector = input-substractor") )
{
    addInput(&d_input);
    addInput(&d_substractor);
    addOutput(&d_output);
}

template <class DataType>
void DifferenceEngine<DataType>::init()
{
    setDirtyValue();
}

template <class DataType>
void DifferenceEngine<DataType>::reinit()
{
    update();
}

template <class DataType>
void DifferenceEngine<DataType>::doUpdate()
{
    helper::ReadAccessor<Data<VecData> > in = d_input;
    helper::ReadAccessor<Data<VecData> > sub = d_substractor;

    helper::WriteOnlyAccessor<Data<VecData> > out = d_output;

    if(in.size() != sub.size())
    {
        msg_warning() << "Input vector and vector to subtract should have same size. Abort.";
        return;
    }

    out.resize( in.size() );

    for( size_t i=0 ; i<in.size() ; ++i )
        out[i] = in[i] - sub[i];
}

} //namespace sofa::component::engine::transform
