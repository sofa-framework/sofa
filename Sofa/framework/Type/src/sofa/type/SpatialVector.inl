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

#include <sofa/type/SpatialVector.h>

namespace sofa::type
{

template<class TReal>
SpatialVector<TReal>::SpatialVector( const Vec& l, const Vec& f )
    : lineVec(l)
    , freeVec(f)
{}

template<class TReal>
void SpatialVector<TReal>::clear()
{
    lineVec = freeVec = Vec(0, 0, 0);
}

template<class TReal>
SpatialVector<TReal>& SpatialVector<TReal>::operator+= (const SpatialVector<TReal>& v)
{
    lineVec += v.lineVec;
    freeVec += v.freeVec;
    return *this;
}

template<class TReal>
SpatialVector<TReal> SpatialVector<TReal>::operator+( const SpatialVector<TReal>& v ) const
{
    return SpatialVector(lineVec + v.lineVec, freeVec + v.freeVec);
}

template<class TReal>
SpatialVector<TReal> SpatialVector<TReal>::operator-( const SpatialVector<TReal>& v ) const
{
    return SpatialVector<TReal>(lineVec - v.lineVec, freeVec - v.freeVec);
}

template<class TReal>
SpatialVector<TReal> SpatialVector<TReal>::operator- ( ) const
{
    return SpatialVector<TReal>(-lineVec, -freeVec);
}

/// Spatial dot product (cross terms)
template<class TReal>
TReal SpatialVector<TReal>::operator* ( const SpatialVector<TReal>& v ) const
{
    return lineVec * v.freeVec + freeVec * v.lineVec;
}

/// Spatial cross product
template<class TReal>
SpatialVector<TReal> SpatialVector<TReal>::cross( const SpatialVector<TReal>& v ) const
{
    return SpatialVector<TReal>(
        type::cross(lineVec,v.lineVec),
        type::cross(freeVec,v.lineVec) + type::cross(lineVec,v.freeVec)
    );
}

template<class TReal>
SpatialVector<TReal> SpatialVector<TReal>::operator* (const Mat66& m) const
{
    SpatialVector<TReal> result;
    for (int i = 0; i < 3; i++)
    {
        result.lineVec[i] = 0;
        result.freeVec[i] = 0;
        for (int j = 0; j < 3; j++)
        {
            result.lineVec[i] += lineVec[j] * m(i,j) + freeVec[j] * m(i,j + 3);
            result.freeVec[i] += lineVec[j] * m(i + 3,j) + freeVec[j] * m(i + 3,j + 3);
        }
    }
    return result;
}

}
