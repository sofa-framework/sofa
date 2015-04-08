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
#ifndef SOFA_COMPONENT_ENGINE_VALUESFROMINDICES_INL
#define SOFA_COMPONENT_ENGINE_VALUESFROMINDICES_INL

#include <SofaEngine/ValuesFromIndices.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class T>
ValuesFromIndices<T>::ValuesFromIndices()
    : f_in( initData (&f_in, "in", "input values") )
    , f_indices( initData(&f_indices, "indices","Indices of the values") )
    , f_out( initData (&f_out, "out", "Output values corresponding to the indices"))
    , f_outStr( initData (&f_outStr, "outStr", "Output values corresponding to the indices, converted as a string"))
{
}

template <class T>
ValuesFromIndices<T>::~ValuesFromIndices()
{
}

template <class T>
void ValuesFromIndices<T>::init()
{
    f_outStr.setParent(&f_out);
    addInput(&f_in);
    addInput(&f_indices);
    addOutput(&f_out);
    setDirtyValue();
}

template <class T>
void ValuesFromIndices<T>::reinit()
{
    update();
}

template <class T>
void ValuesFromIndices<T>::update()
{
    cleanDirty();
    helper::ReadAccessor<Data<VecValue> > in = f_in;
    helper::ReadAccessor<Data<VecIndex> > indices = f_indices;
    helper::WriteAccessor<Data<VecValue> > out = f_out;

    out.clear();
    out.reserve(indices.size());
    for (unsigned int i=0; i<indices.size(); ++i)
    {
        if ((unsigned)indices[i] < in.size())
            out.push_back(in[indices[i]]);
        else
            serr << "Invalid input index " << i <<": " << indices[i] << " >= " << in.size() << sendl;
    }
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
