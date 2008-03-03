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
#ifndef SOFA_COMPONENT_LINEARSOLVER_NEWMATMATRIX_H
#define SOFA_COMPONENT_LINEARSOLVER_NEWMATMATRIX_H

#include <sofa/defaulttype/BaseMatrix.h>
#include "NewMatVector.h"

namespace sofa
{

namespace component
{

namespace linearsolver
{

class NewMatMatrix : public NewMAT::Matrix, public defaulttype::BaseMatrix
{
public:

    virtual void resize(int nbRow, int nbCol)
    {
        ReSize(nbRow, nbCol);
        (*this) = 0.0;
    }

    virtual int rowSize(void)
    {
        return Nrows();
    }

    virtual int colSize(void)
    {
        return Ncols();
    }

    virtual double &element(int i, int j)
    {
        return NewMAT::Matrix::element(i,j);
    }

    void solve(NewMatVector *rv, NewMatVector *ov)
    {
        *rv = this->i() * *ov;
    }

    virtual void solve(defaulttype::BaseVector *op, defaulttype::BaseVector *res)
    {
        NewMatVector *rv = dynamic_cast<NewMatVector *>(res);
        NewMatVector *ov = dynamic_cast<NewMatVector *>(op);

        assert((ov!=NULL) && (rv!=NULL));
        solve(rv,ov);
    }

    template<class T>
    void operator=(const T& m) { NewMAT::Matrix::operator=(m); }

    void clear() { (*this) = 0.0; }

    static const char* Name() { return "NewMat"; }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
