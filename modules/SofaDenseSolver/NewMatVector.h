/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_NEWMATVECTOR_H
#define SOFA_COMPONENT_LINEARSOLVER_NEWMATVECTOR_H
#include "config.h"

#include <newmat/newmat.h>
#define WANT_STREAM
#include <newmat/newmatio.h>
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

class NewMatVector : public NEWMAT::ColumnVector, public defaulttype::BaseVector
{
public:

    typedef NEWMAT::ColumnVector SubVector;
    typedef NewMatVector SubVectorType;
    typedef SReal Real;

    NewMatVector()
    {
    }

    virtual ~NewMatVector()
    {
    }

    virtual void resize(Index dim)
    {
        ReSize(dim);
        (*this) = 0.0;
    }

    virtual SReal element(Index i) const
    {
        return NEWMAT::ColumnVector::element(i);
    }

    void set(Index i, SReal v)
    {
        NEWMAT::ColumnVector::element(i) = v;
    }

    void add(Index i, SReal v)
    {
        NEWMAT::ColumnVector::element(i) += v;
    }

    SReal& operator[](Index i)
    {
        return NEWMAT::ColumnVector::element(i);
    }

    SReal operator[](Index i) const
    {
        return NEWMAT::ColumnVector::element(i);
    }

    Index size() const
    {
        return Nrows();
    }

    NEWMAT::GetSubMatrix sub(Index i, Index n)
    {
        return NEWMAT::ColumnVector::SubMatrix(i+1,i+n,1,1);
    }

    template<class T>
    void getsub(Index i, Index n, T& v)
    {
        v = NEWMAT::ColumnVector::SubMatrix(i+1,i+n,1,1);
    }

    template<class T>
    void setsub(Index i, Index n, const T& v)
    {
        NEWMAT::ColumnVector::SubMatrix(i+1,i+n,1,1) = v;
    }

    /// v = 0
    void clear()
    {
        (*this) = 0.0;
    }

    /// v = a
    void eq(const NewMatVector& a)
    {
        (*this) = a;
    }

    /// v = a+b*f
    void eq(const NewMatVector& a, const NewMatVector& b, double f=1.0)
    {
        (*this) = a + b*f;
    }

    /// v += a*f
    void peq(const NewMatVector& a, double f=1.0)
    {
        (*this) += a*f;
    }
    /// v *= f
    void teq(double f)
    {
        (*this) *= f;
    }
    /// \return v.a
    double dot(const NewMatVector& a) const
    {
        return NEWMAT::DotProduct(*this,a);
    }

    /// \return sqrt(v.v)
    double norm() const
    {
        return NormFrobenius();
    }

    //void operator=(double f) { NEWMAT::ColumnVector::operator=(f); }

    template<class T>
    void operator=(const T& m) { NEWMAT::ColumnVector::operator=(m); }

    friend std::ostream& operator << (std::ostream& out, const NewMatVector& v )
    {
        for (Index i=0,s=v.Nrows(); i<s; ++i)
        {
            if (i) out << ' ';
            out << v[i];
        }
        return out;
    }

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
