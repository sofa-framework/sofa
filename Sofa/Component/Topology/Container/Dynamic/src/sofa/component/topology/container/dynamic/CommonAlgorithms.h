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
#include <sofa/component/topology/container/dynamic/config.h>

#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>

namespace sofa::component::topology::container::dynamic
{


/// Cross product for 3-elements Vectors.
template< class Real>
inline Real areaProduct(const sofa::type::Vec<3,Real>& a, const sofa::type::Vec<3,Real>& b)
{
    return sofa::type::Vec<3,Real>(a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x()).norm();
}

/// area for 2-elements sofa::type::vectors.
template< class Real>
inline Real areaProduct(const type::Vec<2,Real>& a, const type::Vec<2,Real>& b )
{
    return a[0]*b[1] - a[1]*b[0];
}

/// area invalid for 1-elements sofa::type::vectors.
template< class Real>
inline Real areaProduct(const type::Vec<1,Real>&, const type::Vec<1,Real>&)
{
    assert(false);
    return (Real)0;
}
/// orthogonal of a 2D vector
template< class Real>
type::Vec<2,Real> ortho(const type::Vec<2,Real> &in)
{
    sofa::type::Vec<2,Real> out(-in[1],in[0]);
    return(out);
}

/// cross product  for 2-elements sofa::type::vectors.
template< class Real>
type::Vec<2,Real> cross(const type::Vec<2,Real>&, const type::Vec<2,Real>&)
{
    assert(false);
    return(sofa::type::Vec<2,Real>());
}

/// cross product  for 1-elements sofa::type::vectors.
template< class Real>
type::Vec<1,Real> cross(const type::Vec<1,Real>&, const type::Vec<1,Real>&)
{
    assert(false);
    return(sofa::type::Vec<1,Real>());
}

/// Volume (triple product) for 3-elements sofa::type::vectors.
template<typename real>
inline real tripleProduct(const sofa::type::Vec<3,real>& a, const sofa::type::Vec<3,real>& b,const sofa::type::Vec<3,real> &c)
{
    return dot(a,cross(b,c));
}

/// Volume invalid for 2-elements sofa::type::vectors.
template <typename real>
inline real tripleProduct(const sofa::type::Vec<2,real>&, const sofa::type::Vec<2,real>&, const sofa::type::Vec<2,real> &)
{
    assert(false);
    return (real)0;
}

/// Volume invalid for 1-elements sofa::type::vectors.
template <typename real>
inline real tripleProduct(const sofa::type::Vec<1,real>&, const sofa::type::Vec<1,real>&, const sofa::type::Vec<1,real> &)
{
    assert(false);
    return (real)0;
}
/// this function is only valid for small value of n which should be sufficient for a regular use.
inline size_t lfactorial(size_t n)
{
    size_t retval = 1;
    for (int i = (int)n; i > 1; --i)
        retval *= (size_t) i;
    return retval;
}
template < class Real >
Real binomial(const size_t p, const size_t q) {
    size_t ival=1;
    size_t i;
    if (p>q) {
        for (i=p+q;i>p;--i){
            ival*=i;
        }
        return((Real)ival)/lfactorial(q);
    } else {
        for (i=p+q;i>q;--i){
            ival*=i;
        }
        return((Real)ival)/lfactorial(p);
    }
}
template <class Real>
Real multinomial(const size_t n,type::vector<unsigned char> valArray)
{
    size_t i,ival,N;
    N=valArray.size();
    // divide n! with the largest of the multinomial coefficient
    std::sort(valArray.begin(),valArray.end());
    ival=1;
    for (i=n;i>valArray[N-1];--i){
        ival*=i;
    }
    Real val=1;
    for (i=0;i<(N-1);++i)
        val*=lfactorial(valArray[i]);
    return((Real)ival)/(val);
}
template <size_t N, class Real>
Real multinomial(const size_t n,const sofa::type::Vec<N,unsigned char> tbi)
{
    sofa::type::vector<unsigned char> valArray;
    for (size_t j=0;j<N;++j) {
        valArray.push_back(tbi[j]);
    }
    return(multinomial<Real>(n,valArray));
}

template <size_t N, class Real>
Real multinomialVector(const sofa::type::vector< sofa::type::Vec<N,unsigned char> > tbiArray)
{
    size_t i,j;
    Real result=(Real)1;
    type::vector<unsigned char> valArray;
    size_t totalDegree;
    for (j=0;j<N;++j) {
        valArray.clear();
        totalDegree=0;
        for (i=0;i<tbiArray.size();++i)
        {
            valArray.push_back(tbiArray[i][j]);
            totalDegree+=tbiArray[i][j];
        }
        result*=multinomial<Real>(totalDegree,valArray);
    }
    return(result);
}
template <size_t N, class Real>
Real binomialVector(const sofa::type::Vec<N,unsigned char>  tbi1,const sofa::type::Vec<N,unsigned char>  tbi2)
{
    size_t j;
    Real result=(Real)1;
    for (j=0;j<N;++j) {
        result*=binomial<Real>(tbi1[j],tbi2[j]);
    }
    return(result);
}

} //namespace sofa::component::topology::container::dynamic
