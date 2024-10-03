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
#include <sofa/component/engine/transform/Indices2ValuesMapper.h>

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::component::engine::transform
{

using namespace sofa;
using namespace sofa::core::topology;

template <class DataTypes>
Indices2ValuesMapper<DataTypes>::Indices2ValuesMapper()
    : f_inputValues(initData(&f_inputValues, "inputValues", "Already existing values (can be empty) "))
    , f_indices(initData(&f_indices, "indices", "Indices to map value on "))
    , f_values(initData(&f_values, "values", "Values to map indices on "))
    , f_outputValues(initData(&f_outputValues, "outputValues", "New map between indices and values"))
    , p_defaultValue(initData(&p_defaultValue, (Real) 1.0, "defaultValue", "Default value for indices without any value"))
{
    addInput(&f_inputValues);
    addInput(&f_indices);
    addInput(&f_values);

    addOutput(&f_outputValues);
}

template <class DataTypes>
void Indices2ValuesMapper<DataTypes>::init()
{
    setDirtyValue();
}

template <class DataTypes>
void Indices2ValuesMapper<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void Indices2ValuesMapper<DataTypes>::doUpdate()
{
    helper::ReadAccessor< Data< type::vector<Real> > > inputValues = f_inputValues;

    helper::ReadAccessor< Data< type::vector<Real> > > indices = f_indices;
    helper::ReadAccessor< Data< type::vector<Real> > > values = f_values;

    helper::WriteAccessor< Data< type::vector<Real> > > outputValues = f_outputValues;

    const Real& defaultValue = p_defaultValue.getValue();

    //copy existing values
    outputValues.clear();
    outputValues.resize(inputValues.size());

    //add new value
    for(unsigned int i=0 ; i<inputValues.size() ; i++)
    {
        bool found = false;
        for (size_t j = 0; j < indices.size(); j++) {
            if (inputValues[i] == indices[j]) {
                outputValues[i] = values[j];
                found = true;
                break;
            }
        }
        if (!found)
            outputValues[i] = defaultValue;
    }
}


} //namespace sofa::component::engine::transform
