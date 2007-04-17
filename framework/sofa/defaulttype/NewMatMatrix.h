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
#ifndef SOFA_DEFAULTTYPE_NEWMATMATRIX_H
#define SOFA_DEFAULTTYPE_NEWMATMATRIX_H

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/NewMatVector.h>

#include "NewMAT/newmat.h"

namespace sofa
{

namespace defaulttype
{

class NewMatMatrix : public BaseMatrix
{
public:

    NewMatMatrix()
    {
        impl = new NewMAT::Matrix;
    }

    virtual ~NewMatMatrix()
    {
        delete impl;
    }

    virtual void resize(int nbRow, int nbCol)
    {
        impl->ReSize(nbRow, nbCol);
        (*impl) = 0.0;
    };

    virtual int rowSize(void)
    {
        return impl->Nrows();
    };

    virtual int colSize(void)
    {
        return impl->Ncols();
    };

    virtual double &element(int i, int j)
    {
        return impl->element(i,j);
    };

    virtual void solve(BaseVector *op, BaseVector *res)
    {
        NewMatVector *rv = dynamic_cast<NewMatVector *>(res);
        NewMatVector *ov = dynamic_cast<NewMatVector *>(op);

        assert((ov!=NULL) && (rv!=NULL));
        *(rv->impl) = impl->i() * (*(ov->impl));
    };

private:
    NewMAT::Matrix *impl;
};


} // namespace defaulttype

} // namespace sofa
#endif
