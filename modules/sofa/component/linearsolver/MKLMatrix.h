/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_MKLMATRIX_H
#define SOFA_COMPONENT_LINEARSOLVER_MKLMATRIX_H

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/MKLVector.h>

#include <MKL/mat_dyn.h>
#include <mkl_lapack.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

class MKLMatrix : public defaulttype::BaseMatrix
{
public:

    MKLMatrix()
    {
        impl = new Dynamic_Matrix<double>;
    }

    virtual ~MKLMatrix()
    {
        delete impl;
    }

    virtual void resize(Index nbRow, Index nbCol)
    {
        impl->resize((int)nbRow, (int)nbCol);
        //	(*impl) = 0.0;
    };

    virtual unsigned int rowSize(void)
    {
        return impl->rows;
    };

    virtual unsigned int colSize(void)
    {
        return impl->columns;
    };

    virtual SReal &element(Index i, Index j)
    {
        return *(impl->operator[]((unsigned int)j) + i);
    };

    virtual void solve(MKLVector *rHTerm)
    {
        int n= (int)impl->rows;
        int nrhs=1;
        int lda = n;
        int ldb = n;
        int info;
        int *ipiv = new int[n];

        // solve Ax=b
        // b is overwritten by the linear system solution
        dgesv(&n,&nrhs,impl->m,&lda,ipiv,rHTerm->impl->v,&ldb,&info);
        delete ipiv;
    };


private:
    Dynamic_Matrix<double> *impl;
};


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
