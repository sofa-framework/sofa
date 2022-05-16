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
#include <sofa/component/engine/generate/MergeSets.h>
#include <sofa/core/visual/VisualParams.h>

#include <algorithm>

namespace sofa::component::engine::generate
{

template <class T>
MergeSets<T>::MergeSets()
    : f_in1( initData (&f_in1, "in1", "first set of indices") )
    , f_in2( initData (&f_in2, "in2", "second set of indices") )
    , f_out( initData (&f_out, "out", "merged set of indices") )
    , f_op( initData (&f_op, std::string("union"), "op", "name of operation to compute (union, intersection, difference, symmetric_difference)") )
{
    addInput(&f_in1);
    addInput(&f_in2);
    addInput(&f_op);
    addOutput(&f_out);
}

template <class T>
MergeSets<T>::~MergeSets()
{
}

template <class T>
void MergeSets<T>::init()
{
    setDirtyValue();
}

template <class T>
void MergeSets<T>::reinit()
{
    update();
}

template <class T>
void MergeSets<T>::doUpdate()
{
    std::string op = f_op.getValue();
    if (op.empty()) op = "union";

    helper::ReadAccessor<Data<VecIndex> > in1 = f_in1;
    helper::ReadAccessor<Data<VecIndex> > in2 = f_in2;

    helper::WriteOnlyAccessor<Data<VecIndex> > out = f_out;

    out.clear();

    SetIndex set1(in1.begin(), in1.end());
    SetIndex set2(in2.begin(), in2.end());
    SetIndex tmp;
    switch (op[0])
    {
    case 'u':
    case 'U': // union
        std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(tmp, tmp.begin()));
        break;
    case 'i':
    case 'I': // intersection
        std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(tmp, tmp.begin()));
        break;
    case 'd':
    case 'D': // difference
        std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(tmp, tmp.begin()));
        break;
    case 's':
    case 'S': // symmetric_difference
        std::set_symmetric_difference(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(tmp, tmp.begin()));
        break;
    }
    out.clear();
    out.reserve(tmp.size());
    for (typename SetIndex::const_iterator it=tmp.begin(), itend=tmp.end(); it!=itend; ++it)
        out.push_back(*it);
}

} //namespace sofa::component::engine::generate
