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
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/type/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>

namespace sofa::linearalgebra
{

template<typename Real>
void FullVector<Real>::checkIndex(Index n) const
{
    if (n >= cursize)
    {
        msg_error("FullVector") << "in vector<" << sofa::helper::gettypename(typeid(*this)) << "> " << std::hex << this << std::dec << " size " << cursize << " : invalid index " << (int)n;
        sofa::helper::BackTrace::dump();
        assert(n < cursize);
    }
}

template<typename Real>
double FullVector<Real>::norm() const
{
    return helper::rsqrt(dot(*this));
}

/// v = a
template<typename Real>
void FullVector<Real>::operator=(const FullVector<Real>& a)
{
    fastResize(a.size());
    std::copy(a.begin(), a.end(), begin());
}

template<typename Real>
void FullVector<Real>::operator=(const Real& a)
{
    std::fill(begin(), end(), a);
}

template<typename Real>
void FullVector<Real>::fastResize(Index dim)
{
    if (dim == cursize) return;
    if (allocsize >= 0)
    {
        if (dim > allocsize)
        {
            if (allocsize > 0)
                delete[] data;
            allocsize = dim;
            data = new Real[dim];
        }
    }
    else
    {
        if (dim > -allocsize)
        {
            msg_error("FullVector") << "Cannot resize preallocated vector to size "<<dim ;
            return;
        }
    }
    cursize = dim;
}

template<typename Real>
std::ostream& readFromStream(std::ostream& out, const FullVector<Real>& v )
{
    for (Index i=0,s=v.size(); i<s; ++i)
    {
        if (i) out << ' ';
        out << v[i];
    }
    return out;
}

template<typename Real>
void FullVector<Real>::resize(Index dim)
{
    fastResize(dim);
    clear();
}

template<typename Real>
void FullVector<Real>::clear()
{
    if (cursize > 0)
        std::fill( this->begin(), this->end(), Real() );
}

template<typename Real>
void FullVector<Real>::swap(FullVector<Real>& v)
{
    Index t;
    t = cursize; cursize = v.cursize; v.cursize = t;
    t = allocsize; allocsize = v.allocsize; v.allocsize = t;
    Real* d;
    d = data; data = v.data; v.data = d;
}

// for compatibility with baseVector
template<typename Real>
void FullVector<Real>::clear(Index dim)
{
    resize(dim);
}

/// v += a
template<typename Real>
void FullVector<Real>::operator+=(const FullVector<Real>& a)
{
    for(Index i=0; i<cursize; ++i)
        (*this)[i] += (Real)a[i];
}

/// v -= a
template<typename Real>
void FullVector<Real>::operator-=(const FullVector<Real>& a)
{
    for(Index i=0; i<cursize; ++i)
        (*this)[i] -= (Real)a[i];
}

/// v = a*f
template<typename Real>
void FullVector<Real>::eq(const FullVector<Real>& a, Real f)
{
    for(Index i=0; i<cursize; ++i)
        (*this)[i] = (Real)(a[i]*f);
}

/// v = a+b*f
template<typename Real>
void FullVector<Real>::eq(const FullVector<Real>& a, const FullVector<Real>& b, Real f)
{
    for(Index i=0; i<cursize; ++i)
        (*this)[i] = (Real)(a[i]+b[i]*f);
}

/// v += a*f
template<typename Real>
void FullVector<Real>::peq(const FullVector<Real>& a, Real f)
{
    for(Index i=0; i<cursize; ++i)
        (*this)[i] += (Real)(a[i]*f);
}

/// v *= f
template<typename Real>
void FullVector<Real>::operator*=(Real f)
{
    for(Index i=0; i<cursize; ++i)
        (*this)[i] *= (Real)f;
}

/// \return v.a
template<typename Real>
Real FullVector<Real>::dot(const FullVector<Real>& a) const
{
    Real r = 0;
    for(Index i=0; i<cursize; ++i)
        r += (*this)[i]*a[i];
    return r;
}


} // namespace sofa::linearalgebra
