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
#ifndef SOFA_DEFAULTTYPE_MAT_SOLVE_CHOLESKY_H
#define SOFA_DEFAULTTYPE_MAT_SOLVE_CHOLESKY_H

#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace defaulttype
{

/** Cholesky decomposition: compute triangular matrix L such that M=L.Lt
  \pre M must be symmetric positive definite
  returns false is the decomposition fails
  */
template<int n, class real>
bool cholDcmp(Mat<n,n,real>& L, const Mat<n,n,real>& M)
{
    if( M[0][0] <= 0 ) return false;
    real d = 1.0 / sqrt(M[0][0]);
    for (int i=0; i<n; i++)
        L[0][i] = M[i][0] * d;
    for (int j=1; j<n; j++)
    {
        real ss=0;
        for (int k=0; k<j; k++)
            ss+=L[k][j]*L[k][j];
        if( M[j][j]-ss <= 0 ) return false;
        d = 1.0 / sqrt(M[j][j]-ss);
        L[j][j] = (M[j][j]-ss) * d;
        for (int i=j+1; i<n; i++)
        {
            ss=0;
            for (int k=0; k<j; k++) ss+=L[k][i]*L[k][j];
            L[j][i] = (M[i][j]-ss) * d;
        }
    }
    return true;
}

/** Cholesky back-substitution: solve the system Mx=b using the triangular matrix L such that M=L.Lt
  \pre L was computed using the Cholesky decomposition of L
  */
template<int n, class real>
void cholBksb(Vec<n,real>& x, const Mat<n,n,real>& L, const Vec<n,real>& b)
{
    //Solve L u = b
    for (int j=0; j<n; j++)
    {
        double temp = 0.0;
        double d = 1.0 / L[j][j];
        for (int i=0; i<j; i++)
            temp += x[i] * L[i][j];
        x[j] = (b[j] - temp) * d ;
    }

    //Solve L^t x = u
    for (int j=n-1; j>=0; j--)
    {
        double temp = 0.0;
        double d = 1.0 / L[j][j];
        for (int i=j+1; i<n; i++)
        {
            temp += x[i] * L[j][i];
        }
        x[j] = (x[j] - temp) * d;
    }
}

/** Cholesky solution: solve the system Mx=b using a Cholesky decomposition.
  \pre M must be symmetric positive definite
  Returns false is the decomposition fails.
  If you have several solutions to perform with the same matrix M and different vectors b, it is more efficient to factor the matrix once and then use back-substitution for each vector.
  */
template<int n, class real>
bool cholSlv(Vec<n,real>& x, const Mat<n,n,real>& M, const Vec<n,real>& b)
{
    Mat<n,n,real> L;
    if( !cholDcmp(L,M) ) return false;
    cholBksb(x, L, b);
    return true;
}

/** Inversion of a positive symmetric definite (PSD) matrix using a Cholesky decomposition.
  Returns false if the matrix is not PSD.
  */
template<int n, class real>
bool cholInv(Mat<n,n,real>& Inv, const Mat<n,n,real>& M )
{
    Mat<n,n,real> L;
    if( !cholDcmp(L,M) ) return false;
    for(unsigned i=0; i<n; i++ )
    {
        Vec<n,real> v; // initialized to 0
        v[i]=1;
        cholBksb(Inv[i], L, v);
    }
    return true;
}

} // namespace defaulttype

} // namespace sofa

#endif
