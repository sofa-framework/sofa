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
#ifndef SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_INL
#define SOFA_COMPONENT_ENGINE_INDICESFROMVALUES_INL

#include <SofaGeneralEngine/IndicesFromValues.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class T>
IndicesFromValues<T>::IndicesFromValues()
    : f_values( initData (&f_values, "values", "input values") )
    , f_global( initData (&f_global, "global", "Global values, in which the input values are searched") )
    , f_indices( initData(&f_indices, "indices","Output indices of the given values, searched in global") )
    , f_otherIndices( initData(&f_otherIndices, "otherIndices","Output indices of the other values, (NOT the given ones) searched in global") )
    , f_recursiveSearch( initData(&f_recursiveSearch, false, "recursiveSearch", "if set to true, output are indices of the \"global\" data matching with one of the values"))
{
}

template <class T>
IndicesFromValues<T>::~IndicesFromValues()
{
}

template <class T>
void IndicesFromValues<T>::init()
{
    addInput(&f_values);
    addInput(&f_global);
    addOutput(&f_indices);
    addOutput(&f_otherIndices);
    setDirtyValue();
}

template <class T>
void IndicesFromValues<T>::reinit()
{
    update();
}

template <class T>
void IndicesFromValues<T>::update()
{
    helper::ReadAccessor<Data<VecValue> > global = f_global;
    helper::ReadAccessor<Data<VecValue> > values = f_values;

    cleanDirty();

    helper::WriteOnlyAccessor<Data<VecIndex> > indices = f_indices;
    helper::WriteOnlyAccessor<Data<VecIndex> > otherIndices = f_otherIndices;

    indices.clear();
    otherIndices.clear();

    if(f_recursiveSearch.getValue()) {
        for (unsigned int i=0; i<values.size(); i++)
        {
            const Value v = values[i];
            int index=-1;
            for (unsigned int j=0; j<global.size(); j++)
            {
                //if (global[j] == v)
                /// @todo: add operator== to helper::fixed_array and defaulttype::RididCoord/Deriv
                if (!(global[j] < v) && !(v < global[j]))
                {
                    index = j;
                    indices.push_back(j);
                } else {
                    otherIndices.push_back(j);
                }
            }
            if (index < 0) {
                sout << "Input value " << values[i] <<" not found"<< sendl;
            }
        }
    } else {
        indices.reserve(values.size());
        for (unsigned int i=0; i<values.size(); ++i)
        {
            const Value v = values[i];
            int index=-1;
            for (unsigned int j=0; j<global.size(); ++j)
            {
                //if (global[j] == v)
                /// @todo: add operator== to helper::fixed_array and defaulttype::RididCoord/Deriv
                if (!(global[j] < v) && !(v < global[j]))
                {
                    index = j;
                    break;
                }
            }
            if (index >= 0)
                indices.push_back(index);
            else
                serr << "Input value " << i <<" not found : " << v << sendl;
        }
    }
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
