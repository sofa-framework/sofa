/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_VECTOR_ALGEBRA_H
#define SOFA_HELPER_VECTOR_ALGEBRA_H

#include <sofa/helper/vector.h>

namespace sofa
{

namespace helper
{

// -----------------------------------------------------------
//
/*! @name linear algebra on standard vectors

*/
//
// -----------------------------------------------------------
//@{

/// Dot product of two vectors
template<class V1, class V2>
SReal dot( const V1& vector1, const V2& vector2 )
{
    assert(vector1.size()==vector2.size());
    SReal result=0;
    for(std::size_t i=0; i<vector1.size(); i++)
        result += vector1[i] * vector2[i];
    return result;
}

/// Norm of a vector
template<class V>
SReal norm( const V& v )
{
    return sqrt(dot(v,v));
}

/// Vector operation: result = ax + y
template<class V1, class Scalar, class V2, class V3>
void axpy( V1& result, Scalar a, const V2& x, const V3& y )
{
    std::size_t n = x.size();
    assert(n==y.size());
    result.resize(n);
    for(std::size_t i=0; i<n; i++)
        result[i] = x[i]*a + y[i];
}

} // namespace helper

} // namespace sofa

#endif //SOFA_HELPER_VECTOR_H
