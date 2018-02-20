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
#ifndef SOFA_DEFAULTTYPE_MAT_SOLVE_LU_H
#define SOFA_DEFAULTTYPE_MAT_SOLVE_LU_H

#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace defaulttype
{

/// Compute the LU decomposition of matrix a. a is replaced by its pivoted LU decomposition. indx stores pivoting indices.
template< int n, typename Real>
void ludcmp(Mat<n,n,Real> &a, Vec<n,int> &indx)
{
    const Real TINY=(Real)1.0e-20;
    int i,imax=0,j,k;
    Real big,dum,sum,temp;

    Vec<n,Real> vv;
    for (i=0; i<n; i++)
    {
        big=0.0;
        for (j=0; j<n; j++)
            if ((temp=fabs(a[i][j])) > big) big=temp;
        assert (big != 0.0);
        vv[i]=(Real)1.0/big;
    }
    for (j=0; j<n; j++)
    {
        for (i=0; i<j; i++)
        {
            sum=a[i][j];
            for (k=0; k<i; k++) sum -= a[i][k]*a[k][j];
            a[i][j]=sum;
        }
        big=0.0;
        for (i=j; i<n; i++)
        {
            sum=a[i][j];
            for (k=0; k<j; k++) sum -= a[i][k]*a[k][j];
            a[i][j]=sum;
            if ((dum=vv[i]*fabs(sum)) >= big)
            {
                big=dum;
                imax=i;
            }
        }
        if (j != imax)
        {
            for (k=0; k<n; k++)
            {
                dum=a[imax][k];
                a[imax][k]=a[j][k];
                a[j][k]=dum;
            }
            vv[imax]=vv[j];
        }
        indx[j]=imax;
        if (a[j][j] == 0.0) a[j][j]=TINY;
        if (j != n-1)
        {
            dum=(Real)1.0/(a[j][j]);
            for (i=j+1; i<n; i++) a[i][j] *= dum;
        }
    }
}

/// Compute the solution of Mx=b. b is replaced by x. a and indx together represent the LU decomposition of m, as given my method ludcmp.
template< int n, typename Real>
void lubksb(const Mat<n,n,Real> &a, const Vec<n,int> &indx, Vec<n,Real> &b)
{
    int i,ii=0,ip,j;
    Real sum;

    for (i=0; i<n; i++)
    {
        ip=indx[i];
        sum=b[ip];
        b[ip]=b[i];
        if (ii != 0)
            for (j=ii-1; j<i; j++) sum -= a[i][j]*b[j];
        else if (sum != 0.0)
            ii=i+1;
        b[i]=sum;
    }
    for (i=n-1; i>=0; i--)
    {
        sum=b[i];
        for (j=i+1; j<n; j++) sum -= a[i][j]*b[j];
        b[i]=sum/a[i][i];
    }
}

/** Compute the inverse of matrix m.
\warning Matrix m is replaced by its LU decomposition.
*/
template< int n, typename Real>
void luinv( Mat<n,n,Real> &inv, Mat<n,n,Real> &m )
{
    Vec<n,int> idx;
    Vec<n,Real> col;

    ludcmp(m,idx);

    for( int i=0; i<n; i++ )
    {
        for( int j=0; j<n; j++ )
            col[j] = 0;
        col[i] = 1;
        lubksb(m,idx,col);
        for( int j=0; j<n; j++ )
            inv[j][i] = col[j];
    }
}

} // namespace defaulttype

} // namespace sofa

#endif
