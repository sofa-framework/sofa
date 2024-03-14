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
#include <sofa/config.h>
#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>

namespace sofa::defaulttype
{

/// Function resetting all the element of a container with its default constructor value type
template<class Vec>
void resetDataTypeVec(Vec& vec)
{
    for (auto& v : vec)
    {
        v = typename Vec::value_type{};
    }
}

using sofa::type::Vec;
using sofa::type::vector;

/// In case of a vector<Vec>, zero can be set directly with memset on all the memory space for a faster reset
template < sofa::Size N, typename ValueType>
void resetVecTypeVec(vector<Vec<N, ValueType> >& vec)
{
    std::memset(static_cast<void*>(vec.data()), 0, sizeof(ValueType) * N * vec.size());
}

template <>
inline void resetDataTypeVec<vector<Vec<6, float> > >(vector<Vec<6, float> >& vec)
{
    resetVecTypeVec(vec);
}

template <>
inline void resetDataTypeVec<vector<Vec<6, double> > >(vector<Vec<6, double> >& vec)
{
    resetVecTypeVec(vec);
}

template <>
inline void resetDataTypeVec<vector<Vec<3, float> > >(vector<Vec<3, float> >& vec)
{
    resetVecTypeVec(vec);
}

template <>
inline void resetDataTypeVec<vector<Vec<3, double> > >(vector<Vec<3, double> >& vec)
{
    resetVecTypeVec(vec);
}

template <>
inline void resetDataTypeVec<vector<Vec<2, float> > >(vector<Vec<2, float> >& vec)
{
    resetVecTypeVec(vec);
}

template <>
inline void resetDataTypeVec<vector<Vec<2, double> > >(vector<Vec<2, double> >& vec)
{
    resetVecTypeVec(vec);
}

template <>
inline void resetDataTypeVec<vector<Vec<1, float> > >(vector<Vec<1, float> >& vec)
{
    resetVecTypeVec(vec);
}

template <>
inline void resetDataTypeVec<vector<Vec<1, double> > >(vector<Vec<1, double> >& vec)
{
    resetVecTypeVec(vec);
}

}
