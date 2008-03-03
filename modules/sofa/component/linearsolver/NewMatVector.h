/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_NEWMATVECTOR_H
#define SOFA_COMPONENT_LINEARSOLVER_NEWMATVECTOR_H

#include "NewMAT/newmat.h"
//#define WANT_STREAM
//#include "NewMAT/newmatio.h"
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

class NewMatVector : public NewMAT::ColumnVector, public defaulttype::BaseVector
{
    friend class NewMatMatrix;
public:

    NewMatVector()
    {
    }

    virtual ~NewMatVector()
    {
    }

    virtual void resize(int dim)
    {
        ReSize(dim);
        (*this) = 0.0;
    }

    virtual double &element(int i)
    {
        return NewMAT::ColumnVector::element(i);
    }

    double& operator[](int i)
    {
        return NewMAT::ColumnVector::element(i);
    }

    double operator[](int i) const
    {
        return NewMAT::ColumnVector::element(i);
    }

    virtual int size(void)
    {
        return Nrows();
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
        return NewMAT::DotProduct(*this,a);
    }

    /// \return sqrt(v.v)
    double norm() const
    {
        return NormFrobenius();
    }

    //void operator=(double f) { NewMAT::ColumnVector::operator=(f); }

    template<class T>
    void operator=(const T& m) { NewMAT::ColumnVector::operator=(m); }

    friend std::ostream& operator << (std::ostream& out, const NewMatVector& v )
    {
        for (int i=0,s=v.Nrows(); i<s; ++i)
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
