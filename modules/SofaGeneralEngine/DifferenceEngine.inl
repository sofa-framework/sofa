/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_DifferenceEngine_INL
#define SOFA_COMPONENT_ENGINE_DifferenceEngine_INL


#include "DifferenceEngine.h"
#include <sofa/helper/logging/Messaging.h>


namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
DifferenceEngine<DataTypes>::DifferenceEngine()
    : d_input ( initData (&d_input, "input", "input vector") )
    , d_substractor ( initData (&d_substractor, "substractor", "vector to substract to input") )
    , d_output( initData (&d_output, "output", "output vector = input-substractor") )
{

}

template <class DataType>
void DifferenceEngine<DataType>::init()
{
    addInput(&d_input);
    addInput(&d_substractor);
    addOutput(&d_output);
    setDirtyValue();
}

template <class DataType>
void DifferenceEngine<DataType>::reinit()
{
    update();
}

template <class DataType>
void DifferenceEngine<DataType>::update()
{
    helper::ReadAccessor<Data<VecData> > in = d_input;
    helper::ReadAccessor<Data<VecData> > sub = d_substractor;

    cleanDirty();

    helper::WriteOnlyAccessor<Data<VecData> > out = d_output;

    if(in.size() != sub.size())
    {
        msg_warning(this) << "Input vector and vector to substract should have same size. Abort.";
        return;
    }

    out.resize( in.size() );

    for( size_t i=0 ; i<in.size() ; ++i )
        out[i] = in[i] - sub[i];
}

} // namespace engine

} // namespace component

} // namespace sofa


#endif
