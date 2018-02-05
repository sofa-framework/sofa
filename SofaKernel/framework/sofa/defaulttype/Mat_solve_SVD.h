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
#ifndef SOFA_DEFAULTTYPE_MAT_SOLVE_SVD_H
#define SOFA_DEFAULTTYPE_MAT_SOLVE_SVD_H

#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace defaulttype
{

/// return the max of two values
template<class T1,class T2>
inline const T1 S_MAX(const T1 &a, const T2 &b)
{
    return b > a ? (b) : (a);
}

/// return the min of two values
template<class T1,class T2>
inline const T1 S_MIN(const T1 &a, const T2 &b)
{
    return b < a ? (b) : (a);
}

template<class T1,class T2>
inline const T1 S_SIGN(const T1 &a, const T2 &b)
{
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

template<class T>
inline const T S_SQR(const T a)
{
    return a*a;
}

///Computes sqrt(a^2 + b^2) without destructive underflow or overflow.
template <class T1, class T2>
T1 pythag(const T1 a, const T2 b)
{
    T1 absa,absb;
    absa=fabs(a);
    absb=fabs(b);
    if (absa > absb) return absa*sqrt(1.0+SQR(absb/absa));
    else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
}

/// Compute the SVD decomposition of matrix a (from nr). a is replaced by its pivoted LU decomposition. indx stores pivoting indices.
/** SVD decomposition   a = u.w.vt
\pre a: original matrix, destroyed to become u
\pre w: diagonal vector
\pre v: matrix
 */
template< int m, int n, typename Real>
void svddcmp(Mat<m,n,Real> &a, Vec<n,Real> &w, Mat<n,m,Real> &v)
{
    bool flag;
    int i,its,j,jj,k,l,nm;
    Real anorm,c,f,g,h,s,scale,x,y,z;

    Vec<n,Real> rv1;
    g=scale=anorm=0.0;
    for (i=0; i<n; i++)
    {
        l=i+2;
        rv1[i]=scale*g;
        g=s=scale=0.0;
        if (i < m)
        {
            for (k=i; k<m; k++) scale += fabs(a[k][i]);
            if (scale != 0.0)
            {
                for (k=i; k<m; k++)
                {
                    a[k][i] /= scale;
                    s += a[k][i]*a[k][i];
                }
                f=a[i][i];
                g = -S_SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][i]=f-g;
                for (j=l-1; j<n; j++)
                {
                    for (s=0.0,k=i; k<m; k++) s += a[k][i]*a[k][j];
                    f=s/h;
                    for (k=i; k<m; k++) a[k][j] += f*a[k][i];
                }
                for (k=i; k<m; k++) a[k][i] *= scale;
            }
        }
        w[i]=scale *g;
        g=s=scale=0.0;
        if (i+1 <= m && i != n)
        {
            for (k=l-1; k<n; k++) scale += fabs(a[i][k]);
            if (scale != 0.0)
            {
                for (k=l-1; k<n; k++)
                {
                    a[i][k] /= scale;
                    s += a[i][k]*a[i][k];
                }
                f=a[i][l-1];
                g = -S_SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][l-1]=f-g;
                for (k=l-1; k<n; k++) rv1[k]=a[i][k]/h;
                for (j=l-1; j<m; j++)
                {
                    for (s=0.0,k=l-1; k<n; k++) s += a[j][k]*a[i][k];
                    for (k=l-1; k<n; k++) a[j][k] += s*rv1[k];
                }
                for (k=l-1; k<n; k++) a[i][k] *= scale;
            }
        }
        anorm=S_MAX(anorm,(fabs(w[i])+fabs(rv1[i])));
    }
    for (i=n-1; i>=0; i--)
    {
        if (i < n-1)
        {
            if (g != 0.0)
            {
                for (j=l; j<n; j++)
                    v[j][i]=(a[i][j]/a[i][l])/g;
                for (j=l; j<n; j++)
                {
                    for (s=0.0,k=l; k<n; k++) s += a[i][k]*v[k][j];
                    for (k=l; k<n; k++) v[k][j] += s*v[k][i];
                }
            }
            for (j=l; j<n; j++) v[i][j]=v[j][i]=0.0;
        }
        v[i][i]=1.0;
        g=rv1[i];
        l=i;
    }
    for (i=S_MIN(m,n)-1; i>=0; i--)
    {
        l=i+1;
        g=w[i];
        for (j=l; j<n; j++) a[i][j]=0.0;
        if (g != 0.0)
        {
            g=1.0/g;
            for (j=l; j<n; j++)
            {
                for (s=0.0,k=l; k<m; k++) s += a[k][i]*a[k][j];
                f=(s/a[i][i])*g;
                for (k=i; k<m; k++) a[k][j] += f*a[k][i];
            }
            for (j=i; j<m; j++) a[j][i] *= g;
        }
        else for (j=i; j<m; j++) a[j][i]=0.0;
        ++a[i][i];
    }
    for (k=n-1; k>=0; k--)
    {
        for (its=0; its<30; its++)
        {
            flag=true;
            for (l=k; l>=0; l--)
            {
                nm=l-1;
                if (fabs(rv1[l])+anorm == anorm)
                {
                    flag=false;
                    break;
                }
                if (fabs(w[nm])+anorm == anorm) break;
            }
            if (flag)
            {
                c=0.0;
                s=1.0;
                for (i=l-1; i<k+1; i++)
                {
                    f=s*rv1[i];
                    rv1[i]=c*rv1[i];
                    if (fabs(f)+anorm == anorm) break;
                    g=w[i];
                    h=pythag(f,g);
                    w[i]=h;
                    h=1.0/h;
                    c=g*h;
                    s = -f*h;
                    for (j=0; j<m; j++)
                    {
                        y=a[j][nm];
                        z=a[j][i];
                        a[j][nm]=y*c+z*s;
                        a[j][i]=z*c-y*s;
                    }
                }
            }
            z=w[k];
            if (l == k)
            {
                if (z < 0.0)
                {
                    w[k] = -z;
                    for (j=0; j<n; j++) v[j][k] = -v[j][k];
                }
                break;
            }
            if (its == 29)
            {
// 				msg_info()<<"Warning: Mat.h :: svddcmp: no convergence in 30 svdcmp iterations"<<std::endl;
                return;
            }
            x=w[l];
            nm=k-1;
            y=w[nm];
            g=rv1[nm];
            h=rv1[k];
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g=pythag(f,1.0);
            f=((x-z)*(x+z)+h*((y/(f+S_SIGN(g,f)))-h))/x;
            c=s=1.0;
            for (j=l; j<=nm; j++)
            {
                i=j+1;
                g=rv1[i];
                y=w[i];
                h=s*g;
                g=c*g;
                z=pythag(f,h);
                rv1[j]=z;
                c=f/z;
                s=h/z;
                f=x*c+g*s;
                g=g*c-x*s;
                h=y*s;
                y *= c;
                for (jj=0; jj<n; jj++)
                {
                    x=v[jj][j];
                    z=v[jj][i];
                    v[jj][j]=x*c+z*s;
                    v[jj][i]=z*c-x*s;
                }
                z=pythag(f,h);
                w[j]=z;
                if (z)
                {
                    z=1.0/z;
                    c=f*z;
                    s=h*z;
                }
                f=c*g+s*y;
                x=c*y-s*g;
                for (jj=0; jj<m; jj++)
                {
                    y=a[jj][j];
                    z=a[jj][i];
                    a[jj][j]=y*c+z*s;
                    a[jj][i]=z*c-y*s;
                }
            }
            rv1[l]=0.0;
            rv1[k]=f;
            w[k]=x;
        }
    }
}

/// return the condition number of the matrix a following the euclidian norm (using the svd decomposition to find singular values)
template< int m, int n, typename Real>
Real cond(Mat<m,n,Real> &a)
{
    Vec<n,Real>w;
    Mat<n,m,Real> *v = new Mat<n,m,Real>();

    svddcmp( a, w, *v );

    delete v;

    return fabs(w[0]/w[n-1]);
}

} // namespace defaulttype

} // namespace sofa

#endif
