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

#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/rmath.h>

#define EQUALITY_THRESHOLD 1e-6

namespace sofa::defaulttype
{

/// Euclidean norm.
template<std::size_t N, typename real>
real Vec<N,real>::norm() const
{
    return helper::rsqrt(norm2());
}

/// l-norm of the vector
/// The type of norm is set by parameter l.
/// Use l<0 for the infinite norm.
template<std::size_t N, typename real>
real Vec<N,real>::lNorm( int l ) const
{
    if( l==2 ) return norm(); // euclidian norm
    else if( l<0 ) // infinite norm
    {
        real n=0;
        for( std::size_t i=0; i<N; i++ )
        {
            real a = helper::rabs( this->elems[i] );
            if( a>n ) n=a;
        }
        return n;
    }
    else if( l==1 ) // Manhattan norm
    {
        real n=0;
        for( std::size_t i=0; i<N; i++ )
        {
            n += helper::rabs( this->elems[i] );
        }
        return n;
    }
    else if( l==0 ) // counting not null
    {
        real n=0;
        for( std::size_t i=0; i<N; i++ )
            if( this->elems[i] ) n+=1;
        return n;
    }
    else // generic implementation
    {
        real n = 0;
        for( std::size_t i=0; i<N; i++ )
            n += pow( helper::rabs( this->elems[i] ), l );
        return pow( n, real(1.0)/(real)l );
    }
}

} /// namespace sofa::defaulttype
