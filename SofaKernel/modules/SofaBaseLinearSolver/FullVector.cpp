/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_LINEARSOLVER_FULLVECTOR_CPP
#include <SofaBaseLinearSolver/FullVector.h>
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
template<> void FullVector<bool>::set(Index i, SReal v)
{
    data[i] = (v!=0);
}

template<> void FullVector<bool>::add(Index i, SReal v)
{
    data[i] |= (v!=0);
}

template<> bool FullVector<bool>::dot(const FullVector<Real>& a) const
{
    Real r = false;
    for(Index i=0; i<cursize && !r; ++i)
        r = (*this)[i] && a[i];
    return r;
}

template<> double FullVector<bool>::norm() const
{
    double r = 0.0;
    for(Index i=0; i<cursize; ++i)
        r += (*this)[i] ? 1.0 : 0.0;
    return helper::rsqrt(r);
}

template SOFA_BASE_LINEAR_SOLVER_API class FullVector<bool>;

} // namespace linearsolver

} // namespace component

} // namespace sofa
