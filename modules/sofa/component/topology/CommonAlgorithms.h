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
#ifndef SOFA_COMPONENT_TOPOLOGY_COMMONALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_COMMONALGORITHMS_H

#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

/// Cross product for 3-elements vectors.
template< class Real>
Real areaProduct(const Vec<3,Real>& a, const Vec<3,Real>& b)
{
    return Vec<3,Real>(a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x()).norm();
}

/// area for 2-elements vectors.
template< class Real>
Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b )
{
    return a[0]*b[1] - a[1]*b[0];
}

/// area invalid for 1-elements vectors.
template< class Real>
Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>& )
{
    assert(false);
    return (Real)0;
}


/// Volume (triple product) for 3-elements vectors.
template<typename real>
inline real tripleProduct(const Vec<3,real>& a, const Vec<3,real>& b,const Vec<3,real> &c)
{
    return dot(a,cross(b,c));
}

/// Volume invalid for 2-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<2,real>& , const Vec<2,real>& , const Vec<2,real> &)
{
    assert(false);
    return (real)0;
}

/// Volume invalid for 1-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<1,real>& , const Vec<1,real>& , const Vec<1,real> &)
{
    assert(false);
    return (real)0;
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif
