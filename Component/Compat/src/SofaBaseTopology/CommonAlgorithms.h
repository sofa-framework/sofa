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
#include <sofa/component/topology/dynamiccontainer/CommonAlgorithms.h>

// SOFA_DEPRECATED_HEADER("v22.06", "v23.06", "sofa/component/topology/dynamiccontainer/CommonAlgorithms.h")

namespace sofa::component::topology
{

template< class Real>
inline Real areaProduct(const sofa::type::Vec<3,Real>& a, const sofa::type::Vec<3,Real>& b)
{
    return dynamiccontainer::areaProduct(a, b);
}

template< class Real>
inline Real areaProduct(const type::Vec<2,Real>& a, const type::Vec<2,Real>& b )
{
    return dynamiccontainer::areaProduct(a, b);
}

template< class Real>
inline Real areaProduct(const type::Vec<1,Real>& a, const type::Vec<1,Real>& b)
{
    return dynamiccontainer::areaProduct(a, b);
}

template< class Real>
type::Vec<2,Real> ortho(const type::Vec<2,Real> &in)
{
    return dynamiccontainer::ortho(in);
}

template< class Real>
type::Vec<2,Real> cross(const type::Vec<2,Real>& a, const type::Vec<2,Real>& b)
{
    return dynamiccontainer::cross(a, b);
}

template< class Real>
type::Vec<1,Real> cross(const type::Vec<1,Real>& a, const type::Vec<1,Real>& b)
{
    return dynamiccontainer::cross(a, b);
}

template<typename real>
inline real tripleProduct(const sofa::type::Vec<3,real>& a, const sofa::type::Vec<3,real>& b,const sofa::type::Vec<3,real> &c)
{
    return dynamiccontainer::tripleProduct(a, b, c);
}

template <typename real>
inline real tripleProduct(const sofa::type::Vec<2,real>& a, const sofa::type::Vec<2,real>& b, const sofa::type::Vec<2,real> &c)
{
    return dynamiccontainer::tripleProduct(a, b, c);
}

template <typename real>
inline real tripleProduct(const sofa::type::Vec<1,real>& a, const sofa::type::Vec<1,real>& b, const sofa::type::Vec<1,real> &c)
{
    return dynamiccontainer::tripleProduct(a, b, c);
}

inline size_t lfactorial(size_t n)
{
    return dynamiccontainer::lfactorial(n);
}

template < class Real >
Real binomial(const size_t p, const size_t q) 
{
    return dynamiccontainer::binomial(p,q);
}

template <class Real>
Real multinomial(const size_t n,type::vector<unsigned char> valArray)
{
    return dynamiccontainer::binomial(n, valArray);
}

template <size_t N, class Real>
Real multinomial(const size_t n,const sofa::type::Vec<N,unsigned char> tbi)
{
    return dynamiccontainer::multinomial(n, tbi);
}

template <size_t N, class Real>
Real multinomialVector(const sofa::type::vector< sofa::type::Vec<N,unsigned char> > tbiArray)
{
    return dynamiccontainer::multinomialVector(tbiArray);
}
template <size_t N, class Real>
Real binomialVector(const sofa::type::Vec<N,unsigned char>  tbi1,const sofa::type::Vec<N,unsigned char>  tbi2)
{
    return dynamiccontainer::binomialVector(tbi1, tbi2);
}

} // namespace sofa::component::topology
