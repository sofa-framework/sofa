/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_POLARDECOMPOSE_H
#define SOFA_HELPER_POLARDECOMPOSE_H

#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{
using defaulttype::Mat;
using defaulttype::Vec;





/** QR decomposition
  Compute an orthonormal right-handed 3x3 basis based on two vectors using Gram-Schmidt orthogonalization.
  The basis vectors are the columns of the returned matrix. The matrix represents the rotation of the local frame with respect to the reference frame.
  The first basis vector is aligned to the first given vector, the second basis vector is in the plane of the two first given vectors, and the third basis vector is orthogonal to the two others.
  Undefined result if one of the vectors is null, or if the two vectors are parallel.
  */
template<class real>
void getRotation(Mat<3,3,real>& r, Vec<3,real>& edgex, Vec<3,real>& edgey  )
{
    edgex.normalize();
    Vec<3,real> edgez = cross( edgex, edgey );
    edgez.normalize();
    edgey = cross( edgez, edgex );

    r[0][0] = edgex[0]; r[0][1]= edgey[0]; r[0][2]= edgez[0];
    r[1][0] = edgex[1]; r[1][1]= edgey[1]; r[1][2]= edgez[1];
    r[2][0] = edgex[2]; r[2][1]= edgey[2]; r[2][2]= edgez[2];
}





//////////// Polar decomposition
////// FROM Decompose.c
////// Ken Shoemake, 1993

/******* Matrix Preliminaries *******/

/** Set MadjT to transpose of inverse of M times determinant of M **/
template<class Real>
void adjoint_transpose(const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& MadjT)
{
    MadjT[0] = cross(M[1],M[2]);
    MadjT[1] = cross(M[2],M[0]);
    MadjT[2] = cross(M[0],M[1]);
}

/** Compute the infinity norm of M **/
template<class Real>
Real norm_inf(const defaulttype::Mat<3,3,Real>& M)
{
    Real sum, max = 0;
    for (int i=0; i<3; i++)
    {
        sum = (Real)(fabs(M[i][0])+fabs(M[i][1])+fabs(M[i][2]));
        if (max<sum) max = sum;
    }
    return max;
}

/** Compute the 1 norm of M **/
template<class Real>
Real norm_one(const defaulttype::Mat<3,3,Real>& M)
{
    Real sum, max = 0;
    for (int i=0; i<3; i++)
    {
        sum = (Real)(fabs(M[0][i])+fabs(M[1][i])+fabs(M[2][i]));
        if (max<sum) max = sum;
    }
    return max;
}

/** Return index of column of M containing maximum abs entry, or -1 if M=0 **/
template<class Real>
int find_max_col(const defaulttype::Mat<3,3,Real>& M)
{
    Real abs, max = 0;
    int col = -1;
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
        {
            abs = M[i][j]; if (abs<0.0) abs = -abs;
            if (abs>max) {max = abs; col = j;}
        }
    return col;
}

/** Setup u for Household reflection to zero all v components but first **/
template<class Real>
void make_reflector(const defaulttype::Vec<3,Real>& v, defaulttype::Vec<3,Real>& u)
{
    Real s = (Real)sqrt(dot(v, v));
    u[0] = v[0]; u[1] = v[1];
    u[2] = v[2] + ((v[2]<0.0) ? -s : s);
    s = (Real)sqrt(2.0/dot(u, u));
    u[0] = u[0]*s; u[1] = u[1]*s; u[2] = u[2]*s;
}

/** Apply Householder reflection represented by u to column vectors of M **/
template<class Real>
void reflect_cols(defaulttype::Mat<3,3,Real>& M, const defaulttype::Vec<3,Real>& u)
{
    for (int i=0; i<3; i++)
    {
        Real s = u[0]*M[0][i] + u[1]*M[1][i] + u[2]*M[2][i];
        for (int j=0; j<3; j++)
            M[j][i] -= u[j]*s;
    }
}
/** Apply Householder reflection represented by u to row vectors of M **/
template<class Real>
void reflect_rows(defaulttype::Mat<3,3,Real>& M, const defaulttype::Vec<3,Real>& u)
{
    for (int i=0; i<3; i++)
    {
        Real s = dot(u, M[i]);
        for (int j=0; j<3; j++)
            M[i][j] -= u[j]*s;
    }
}

/** Find orthogonal factor Q of rank 1 (or less) M **/
template<class Real>
void do_rank1(defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q)
{
    defaulttype::Vec<3,Real> v1, v2;
    Real s;
    int col;
    Q.identity();
    /* If rank(M) is 1, we should find a non-zero column in M */
    col = find_max_col(M);
    if (col<0) return; /* Rank is 0 */
    v1[0] = M[0][col]; v1[1] = M[1][col]; v1[2] = M[2][col];
    make_reflector(v1, v1); reflect_cols(M, v1);
    v2[0] = M[2][0]; v2[1] = M[2][1]; v2[2] = M[2][2];
    make_reflector(v2, v2); reflect_rows(M, v2);
    s = M[2][2];
    if (s<0.0) Q[2][2] = -1.0;
    reflect_cols(Q, v1); reflect_rows(Q, v2);
}

/** Find orthogonal factor Q of rank 2 (or less) M using adjoint transpose **/
template<class Real>
void do_rank2(defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& MadjT, defaulttype::Mat<3,3,Real>& Q)
{
    defaulttype::Vec<3,Real> v1, v2;
    Real w, x, y, z, c, s, d;
    int col;
    /* If rank(M) is 2, we should find a non-zero column in MadjT */
    col = find_max_col(MadjT);
    if (col<0) {do_rank1(M, Q); return;} /* Rank<2 */
    v1[0] = MadjT[0][col]; v1[1] = MadjT[1][col]; v1[2] = MadjT[2][col];
    make_reflector(v1, v1); reflect_cols(M, v1);
    v2 = cross(M[0], M[1]);
    make_reflector(v2, v2); reflect_rows(M, v2);
    w = M[0][0]; x = M[0][1]; y = M[1][0]; z = M[1][1];
    if (w*z>x*y)
    {
        c = z+w; s = y-x; d = sqrt(c*c+s*s); c = c/d; s = s/d;
        Q[0][0] = Q[1][1] = c; Q[0][1] = -(Q[1][0] = s);
    }
    else
    {
        c = z-w; s = y+x; d = sqrt(c*c+s*s); c = c/d; s = s/d;
        Q[0][0] = -(Q[1][1] = c); Q[0][1] = Q[1][0] = s;
    }
    Q[0][2] = Q[2][0] = Q[1][2] = Q[2][1] = 0.0; Q[2][2] = 1.0;
    reflect_cols(Q, v1); reflect_rows(Q, v2);
}

/******* Polar Decomposition *******/

/** Polar Decomposition of 3x3 matrix,
 * M = QS.  See Nicholas Higham and Robert S. Schreiber,
 * Fast Polar Decomposition of An Arbitrary Matrix,
 * Technical Report 88-942, October 1988,
 * Department of Computer Science, Cornell University.
 */
template<class Real>
Real polarDecomposition(const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q, defaulttype::Mat<3,3,Real>& S)
{
    defaulttype::Mat<3,3,Real> Mk, MadjTk, Ek;
    Real det, M_one, M_inf, MadjT_one, MadjT_inf, E_one, gamma, g1, g2;
    Mk.transpose(M);
    M_one = norm_one(Mk);  M_inf = norm_inf(Mk);
    do
    {
        adjoint_transpose(Mk, MadjTk);
        det = dot(Mk[0], MadjTk[0]);
        if (det==0.0)
        {
            do_rank2(Mk, MadjTk, Mk);
            break;
        }
        MadjT_one = norm_one(MadjTk); MadjT_inf = norm_inf(MadjTk);
        gamma = (Real)sqrt(sqrt((MadjT_one*MadjT_inf)/(M_one*M_inf))/fabs(det));
        g1 = gamma*((Real)0.5);
        g2 = ((Real)0.5)/(gamma*det);
        Ek = Mk;
        Mk = Mk*g1 + MadjTk*g2;
        Ek -= Mk;
        E_one = norm_one(Ek);
        M_one = norm_one(Mk);  M_inf = norm_inf(Mk);
    }
    while (E_one>(M_one*1.0e-6));
    Q.transpose(Mk);
    S = Mk*M;
    for (int i=0; i<3; i++)
        for (int j=i+1; j<3; j++)
            S[i][j] = S[j][i] = ((Real)0.5)*(S[i][j]+S[j][i]);
    return (det);

}



/** The same than previous except we do not care about S
 */
template<class Real>
Real polarDecomposition( const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q )
{
    defaulttype::Mat<3,3,Real> Mk, MadjTk, Ek;
    Real det, M_one, M_inf, MadjT_one, MadjT_inf, E_one, gamma, g1, g2;
    Mk.transpose(M);
    M_one = norm_one(Mk);  M_inf = norm_inf(Mk);
    do
    {
        adjoint_transpose(Mk, MadjTk);
        det = dot(Mk[0], MadjTk[0]);
        if (det==0.0)
        {
            do_rank2(Mk, MadjTk, Mk);
            break;
        }
        MadjT_one = norm_one(MadjTk); MadjT_inf = norm_inf(MadjTk);
        gamma = (Real)sqrt(sqrt((MadjT_one*MadjT_inf)/(M_one*M_inf))/fabs(det));
        g1 = gamma*((Real)0.5);
        g2 = ((Real)0.5)/(gamma*det);
        Ek = Mk;
        Mk = Mk*g1 + MadjTk*g2;
        Ek -= Mk;
        E_one = norm_one(Ek);
        M_one = norm_one(Mk);  M_inf = norm_inf(Mk);
    }
    while (E_one>(M_one*1.0e-6));
    Q.transpose(Mk);
    return (det);
}



///////////////////////////////



/** Eigensystem decomposition: eigenvalues @param diag and eigenvectors (columns of @param V) of the 3x3 Real Matrix @param M
This is derived from the non-longer existing library magic-software.
Should be faster than using generic size algorithm from Eigen library.
*/
template<class Real>
void eigenDecomposition( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &V, defaulttype::Vec<3,Real> &diag )
{
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef defaulttype::Mat<3,3,Real> Mat33;


    Vec3 subDiag;

    //////////////////////
    ///// Tridiagonalize
    //////////////////////

    const Real &fM00 = M[0][0];
    Real fM01 = M[0][1];
    Real fM02 = M[0][2];
    const Real &fM11 = M[1][1];
    const Real &fM12 = M[1][2];
    const Real &fM22 = M[2][2];

    diag[0] = fM00;
    subDiag[2] = (Real)0.0;
    if ( fM02 != (Real)0.0 )
    {
        Real fLength = helper::rsqrt(fM01*fM01+fM02*fM02);
        Real fInvLength = ((Real)1.0)/fLength;
        fM01 *= fInvLength;
        fM02 *= fInvLength;
        Real fQ = ((Real)2.0)*fM01*fM12+fM02*(fM22-fM11);
        diag[1] = fM11+fM02*fQ;
        diag[2] = fM22-fM02*fQ;
        subDiag[0] = fLength;
        subDiag[1] = fM12-fM01*fQ;
        V[0][0] = (Real)1.0; V[0][1] = (Real)0.0; V[0][2] = (Real)0.0;
        V[1][0] = (Real)0.0; V[1][1] = fM01;      V[1][2] = fM02;
        V[2][0] = (Real)0.0; V[2][1] = fM02;      V[2][2] = -fM01;
    }
    else
    {
        diag[1] = fM11;
        diag[2] = fM22;
        subDiag[0] = fM01;
        subDiag[1] = fM12;
        V.identity();
    }



    /////////////////////
    ///// QLAlgorithm
    /////////////////////

    static const int iMaxIter = 32;
    static const int iSize = 3;

    for (int i0 = 0; i0 < iSize; ++i0)
    {
        int i1;
        for (i1 = 0; i1 < iMaxIter; ++i1)
        {
            int i2;
            for (i2 = i0; i2 <= iSize-2; ++i2)
            {
                Real fTmp = helper::rabs(diag[i2]) + helper::rabs(diag[i2+1]);
                if ( helper::rabs(subDiag[i2]) + fTmp == fTmp )
                    break;
            }
            if ( i2 == i0 )
                break;
            Real fG = (diag[i0+1] - diag[i0])/(((Real)2.0) *  subDiag[i0]);
            Real fR = helper::rsqrt(fG*fG+(Real)1.0);
            if ( fG < (Real)0.0 )
                fG = diag[i2]-diag[i0]+subDiag[i0]/(fG-fR);
            else
                fG = diag[i2]-diag[i0]+subDiag[i0]/(fG+fR);

            Real fSin = 1.0;
            Real fCos = 1.0;
            Real fP   = 0.0;

            for (int i3 = i2-1; i3 >= i0; --i3)
            {
                Real fF = fSin*subDiag[i3];
                Real fB = fCos*subDiag[i3];
                if ( helper::rabs(fF) >= helper::rabs(fG) )
                {
                    fCos = fG/fF;
                    fR = helper::rsqrt(fCos*fCos+(Real)1.0);
                    subDiag[i3+1] = fF*fR;
                    fSin = ((Real)1.0)/fR;
                    fCos *= fSin;
                }
                else
                {
                    fSin = fF/fG;
                    fR = helper::rsqrt(fSin*fSin+(Real)1.0);
                    subDiag[i3+1] = fG*fR;
                    fCos = ((Real)1.0)/fR;
                    fSin *= fCos;
                }
                fG = diag[i3+1]-fP;
                fR = (diag[i3]-fG)*fSin+((Real)2.0)*fB*fCos;
                fP = fSin*fR;
                diag[i3+1] = fG+fP;
                fG = fCos*fR-fB;
                for (int i4 = 0; i4 < iSize; ++i4)
                {
                    fF = V[i4][i3+1];
                    V[i4][i3+1] = fSin*V[i4][i3]+fCos*fF;
                    V[i4][i3]   = fCos*V[i4][i3]-fSin*fF;
                }
            }
            diag[i0] -= fP;
            subDiag[i0] = fG;
            subDiag[i2] = (Real)0.0;
        }
        if ( i1 == iMaxIter )
            return;
    }
}



} // namespace helper

} // namespace sofa

#endif
