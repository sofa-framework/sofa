/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#define SOFA_COMPONENT_LINEARSOLVER_FULLVECTOR_CPP
#include <sofa/component/linearsolver/FullVector.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{
/*
template<> FullVector<bool>::FullVector()
: data(NULL), cursize(0), allocsize(0)
{
}
*/
template<> void FullVector<bool>::set(int i, SReal v)
{
    data[i] = (v!=0);
}

template<> void FullVector<bool>::add(int i, SReal v)
{
    data[i] |= (v!=0);
}

template<> bool FullVector<bool>::dot(const FullVector<Real>& a) const
{
    Real r = false;
    for(int i=0; i<cursize && !r; ++i)
        r = (*this)[i] && a[i];
    return r;
}

template<> double FullVector<bool>::norm() const
{
    double r = 0.0;
    for(int i=0; i<cursize; ++i)
        r += (*this)[i] ? 1.0 : 0.0;
    return helper::rsqrt(r);
}

template SOFA_COMPONENT_LINEARSOLVER_API class FullVector<bool>;

} // namespace linearsolver

} // namespace component

} // namespace sofa
