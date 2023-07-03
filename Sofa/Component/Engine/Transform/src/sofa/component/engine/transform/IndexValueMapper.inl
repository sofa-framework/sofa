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
#include <sofa/component/engine/transform/IndexValueMapper.h>
#include <sofa/helper/narrow_cast.h>

namespace sofa::component::engine::transform
{

template <class DataTypes>
IndexValueMapper<DataTypes>::IndexValueMapper()
    : f_inputValues(initData(&f_inputValues, "inputValues", "Already existing values (can be empty) "))
    , f_indices(initData(&f_indices, "indices", "Indices to map value on "))
    , f_value(initData(&f_value, "value", "Value to map indices on "))
    , f_outputValues(initData(&f_outputValues, "outputValues", "New map between indices and values"))
    , p_defaultValue(initData(&p_defaultValue, (Real) 1.0, "defaultValue", "Default value for indices without any value"))
{
    addInput(&f_inputValues);
    addInput(&f_indices);
    addInput(&f_value);

    addOutput(&f_outputValues);
}

template <class DataTypes>
void IndexValueMapper<DataTypes>::init()
{
    setDirtyValue();
}

template <class DataTypes>
void IndexValueMapper<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void IndexValueMapper<DataTypes>::doUpdate()
{
    helper::ReadAccessor< Data< type::vector<Real> > > inputValues = f_inputValues;
    const helper::ReadAccessor< Data< type::vector<Index> > > indices = f_indices;
    const Real& value = f_value.getValue();

    helper::WriteOnlyAccessor< Data< type::vector<Real> > > outputValues = f_outputValues;

    const Real& defaultValue = p_defaultValue.getValue();

    //copy existing values
    outputValues.clear();
    outputValues.resize(inputValues.size());
    for(unsigned int i=0 ; i<inputValues.size() ; i++)
        outputValues[i] = inputValues[i];

    //add new value
    for(unsigned int i=0 ; i<indices.size() ; i++)
    {
        Index ind = indices[i];
        //new index may be bigger than the current existing map, so have to resize it
        if (ind >= outputValues.size())
        {
            const auto oldSize = sofa::helper::narrow_cast<Index>(outputValues.size());
            outputValues.resize(ind+1);
            //if there is hole when resizing, put default value
            for (Index j=oldSize ; j<ind+1 ; ++j)
                outputValues[j] = defaultValue;
        }
        outputValues[ind] = value;
    }
}


} //namespace sofa::component::engine::transform
