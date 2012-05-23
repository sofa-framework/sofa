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
#ifndef SOFA_HELPER_POLARDECOMPOSE_INL
#define SOFA_HELPER_POLARDECOMPOSE_INL

#include "decompose.h"


namespace sofa
{

namespace helper
{

using defaulttype::Mat;
using defaulttype::Vec;


template<class Real>
void Decompose<Real>::getRotation( Mat<3,3,Real>& r, Vec<3,Real>& edgex, Vec<3,Real>& edgey )
{
    edgex.normalize();
    Vec<3,Real> edgez = cross( edgex, edgey );
    edgez.normalize();
    edgey = cross( edgez, edgex );

    r[0][0] = edgex[0]; r[0][1] = edgey[0]; r[0][2] = edgez[0];
    r[1][0] = edgex[1]; r[1][1] = edgey[1]; r[1][2] = edgez[1];
    r[2][0] = edgex[2]; r[2][1] = edgey[2]; r[2][2] = edgez[2];
}


template<class Real>
void Decompose<Real>::QRDecomposition( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &R )
{
    Vec<3,Real> edgex( M[0][0], M[1][0], M[2][0] );
    Vec<3,Real> edgey( M[0][1], M[1][1], M[2][1] );

    getRotation( R, edgex, edgey );
}



template<class Real>
void Decompose<Real>::QRDecomposition_stable( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &R )
{
    Vec<3,Real> edgex( M[0][0], M[1][0], M[2][0] );
    Vec<3,Real> edgey( M[0][1], M[1][1], M[2][1] );
    Vec<3,Real> edgez;

    Real nx = edgex.norm2();
    Real ny = edgey.norm2();

    if( nx < zeroTolerance() )
    {
        if( ny < zeroTolerance() )
        {
            edgez.set( M[0][2], M[1][2], M[2][2] );
            Real nz = edgez.norm2();

            if( nz < zeroTolerance() ) // edgex, edgey, edgez are null -> collapsed to a point
            {
                //std::cerr<<"helper::QRDecomposition_stable collapased to point "<<M<<std::endl;
                R.identity();
                return;
            }
            else // collapsed to edgez
            {
                //std::cerr<<"QRDecomposition_stable collapased to edgez "<<M<<std::endl;

                edgez.normalizeWithNorm( helper::rsqrt(nz) );

                // check the main direction of edgez to try to take a not too close arbritary vector
                Real abs0 = helper::rabs( edgez[0] );
                Real abs1 = helper::rabs( edgez[1] );
                Real abs2 = helper::rabs( edgez[2] );
                if( abs0 > abs1 )
                {
                    if( abs0 > abs2 )
                    {
                        edgex[0] = 0; edgex[1] = 1; edgex[2] = 0;
                    }
                    else
                    {
                        edgex[0] = 1; edgex[1] = 0; edgex[2] = 0;
                    }
                }
                else
                {
                    if( abs1 > abs2 )
                    {
                        edgex[0] = 0; edgex[1] = 0; edgex[2] = 1;
                    }
                    else
                    {
                        edgex[0] = 1; edgex[1] = 0; edgex[2] = 0;
                    }
                }

                edgey = cross( edgez, edgex );
                edgey.normalize();
                edgex = cross( edgey, edgez );
            }
        }
        else
        {
            edgey.normalizeWithNorm( helper::rsqrt(ny) );

            edgez.set( M[0][2], M[1][2], M[2][2] );
            Real nz = edgez.norm2();

            if( nz < zeroTolerance() ) // collapsed to edgey
            {
                //std::cerr<<"QRDecomposition_stable collapased to edgey "<<M<<std::endl;

                // check the main direction of edgey to try to take a not too close arbritary vector
                Real abs0 = helper::rabs( edgey[0] );
                Real abs1 = helper::rabs( edgey[1] );
                Real abs2 = helper::rabs( edgey[2] );
                if( abs0 > abs1 )
                {
                    if( abs0 > abs2 )
                    {
                        edgez[0] = 0; edgez[1] = 1; edgez[2] = 0;
                    }
                    else
                    {
                        edgez[0] = 1; edgez[1] = 0; edgez[2] = 0;
                    }
                }
                else
                {
                    if( abs1 > abs2 )
                    {
                        edgez[0] = 0; edgez[1] = 0; edgez[2] = 1;
                    }
                    else
                    {
                        edgez[0] = 1; edgez[1] = 0; edgez[2] = 0;
                    }
                }

                edgex = cross( edgey, edgez );
                edgex.normalize();
                edgez = cross( edgex, edgey );
            }
            else // collapsed to face (edgey, edgez)
            {
                //std::cerr<<"QRDecomposition_stable collapased to face (edgey, edgez) "<<M<<std::endl;

                edgex = cross( edgey, edgez );
                edgex.normalize();
                edgez = cross( edgex, edgey );
            }
        }
    }
    else
    {
        edgex.normalizeWithNorm( helper::rsqrt(nx) );

        if( ny < zeroTolerance() )
        {

            edgez.set( M[0][2], M[1][2], M[2][2] );
            Real nz = edgez.norm2();

            if( nz < zeroTolerance() ) // collapsed to edgex
            {
                //std::cerr<<"QRDecomposition_stable ollapased to edgex "<<M<<std::endl;

                // check the main direction of edgex to try to take a not too close arbritary vector
                Real abs0 = helper::rabs( edgex[0] );
                Real abs1 = helper::rabs( edgex[1] );
                Real abs2 = helper::rabs( edgex[2] );
                if( abs0 > abs1 )
                {
                    if( abs0 > abs2 )
                    {
                        edgey[0] = 0; edgey[1] = 1; edgey[2] = 0;
                    }
                    else
                    {
                        edgey[0] = 1; edgey[1] = 0; edgey[2] = 0;
                    }
                }
                else
                {
                    if( abs1 > abs2 )
                    {
                        edgey[0] = 0; edgey[1] = 0; edgey[2] = 1;
                    }
                    else
                    {
                        edgey[0] = 1; edgey[1] = 0; edgey[2] = 0;
                    }
                }

                edgez = cross( edgex, edgey );
                edgez.normalize();
                edgey = cross( edgez, edgex );
            }
            else // collapsed to face (edgez,edgex)
            {
                //std::cerr<<"QRDecomposition_stable collapased to face (edgez, edgex) "<<M<<std::endl;

                edgey = cross( edgez, edgex );
                edgey.normalize();
                edgez = cross( edgex, edgey );
            }
        }
        else // edgex & edgey are ok (either not collapsed or collapsed to face (edgex,edgey) )
        {
            edgez = cross( edgex, edgey );
            edgez.normalize();
            edgey = cross( edgez, edgex );
        }
    }

    R[0][0] = edgex[0]; R[0][1] = edgey[0]; R[0][2] = edgez[0];
    R[1][0] = edgex[1]; R[1][1] = edgey[1]; R[1][2] = edgez[1];
    R[2][0] = edgex[2]; R[2][1] = edgey[2]; R[2][2] = edgez[2];
}



//////////// Polar Decomposition
////// FROM Decompose.c
////// Ken Shoemake, 1993

/******* Matrix Preliminaries *******/

template<class Real>
void Decompose<Real>::adjoint_transpose(const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& MadjT)
{
    MadjT[0] = cross(M[1],M[2]);
    MadjT[1] = cross(M[2],M[0]);
    MadjT[2] = cross(M[0],M[1]);
}

template<class Real>
Real Decompose<Real>::norm_inf(const defaulttype::Mat<3,3,Real>& M)
{
    Real sum, max = 0;
    for (int i=0; i<3; i++)
    {
        sum = (Real)(fabs(M[i][0])+fabs(M[i][1])+fabs(M[i][2]));
        if (max<sum) max = sum;
    }
    return max;
}

template<class Real>
Real Decompose<Real>::norm_one(const defaulttype::Mat<3,3,Real>& M)
{
    Real sum, max = 0;
    for (int i=0; i<3; i++)
    {
        sum = (Real)(fabs(M[0][i])+fabs(M[1][i])+fabs(M[2][i]));
        if (max<sum) max = sum;
    }
    return max;
}


template<class Real>
int Decompose<Real>::find_max_col(const defaulttype::Mat<3,3,Real>& M)
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

template<class Real>
void Decompose<Real>::make_reflector(const defaulttype::Vec<3,Real>& v, defaulttype::Vec<3,Real>& u)
{
    Real s = (Real)sqrt(dot(v, v));
    u[0] = v[0]; u[1] = v[1];
    u[2] = v[2] + ((v[2]<0.0) ? -s : s);
    s = (Real)sqrt(2.0/dot(u, u));
    u[0] = u[0]*s; u[1] = u[1]*s; u[2] = u[2]*s;
}


template<class Real>
void Decompose<Real>::reflect_cols(defaulttype::Mat<3,3,Real>& M, const defaulttype::Vec<3,Real>& u)
{
    for (int i=0; i<3; i++)
    {
        Real s = u[0]*M[0][i] + u[1]*M[1][i] + u[2]*M[2][i];
        for (int j=0; j<3; j++)
            M[j][i] -= u[j]*s;
    }
}

template<class Real>
void Decompose<Real>::reflect_rows(defaulttype::Mat<3,3,Real>& M, const defaulttype::Vec<3,Real>& u)
{
    for (int i=0; i<3; i++)
    {
        Real s = dot(u, M[i]);
        for (int j=0; j<3; j++)
            M[i][j] -= u[j]*s;
    }
}

template<class Real>
void Decompose<Real>::do_rank1(defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q)
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


template<class Real>
void Decompose<Real>::do_rank2(defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& MadjT, defaulttype::Mat<3,3,Real>& Q)
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


template<class Real>
Real Decompose<Real>::polarDecomposition(const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q, defaulttype::Mat<3,3,Real>& S)
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
    while (E_one>(M_one*zeroTolerance()));
    Q.transpose(Mk);
    S = Mk*M;
    for (int i=0; i<3; i++)
        for (int j=i+1; j<3; j++)
            S[i][j] = S[j][i] = ((Real)0.5)*(S[i][j]+S[j][i]);
    return (det);

}



template<class Real>
Real Decompose<Real>::polarDecomposition( const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q )
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
    while (E_one>(M_one*zeroTolerance()));
    Q.transpose(Mk);
    return (det);
}


template<class Real>
void Decompose<Real>::polarDecomposition( const defaulttype::Mat<2,2,Real>& M, defaulttype::Mat<2,2,Real>& Q )
{
    Q[0][0] =  M[1][1];
    Q[0][1] = -M[1][0];
    Q[1][0] = -M[0][1];
    Q[1][1] =  M[0][0];
    Q = M + ( determinant( M ) < 0 ? -1 : 1 ) * Q;
}


///////////////////////////////




template<class Real>
Real Decompose<Real>::zeroTolerance()
{
    return (Real)1e-6;
}

template<>
float Decompose<float>::zeroTolerance()
{
    return 1e-6f;
}

template<>
double Decompose<double>::zeroTolerance()
{
    return 1e-8;
}


template <typename Real>
void Decompose<Real>::ComputeRoots(const Mat<3,3,Real>& A, double root[3])
{
    static const Real msInv3 = (Real)1.0/(Real)3.0;
    static const Real msRoot3 = helper::rsqrt((Real)3.0);

    // Convert the unique matrix entries to double precision.
    double a00 = (double)A[0][0];
    double a01 = (double)A[0][1];
    double a02 = (double)A[0][2];
    double a11 = (double)A[1][1];
    double a12 = (double)A[1][2];
    double a22 = (double)A[2][2];

    // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
    // eigenvalues are the roots to this equation, all guaranteed to be
    // real-valued, because the matrix is symmetric.
    double c0 = a00*a11*a22 + 2.0*a01*a02*a12 - a00*a12*a12 -
            a11*a02*a02 - a22*a01*a01;

    double c1 = a00*a11 - a01*a01 + a00*a22 - a02*a02 +
            a11*a22 - a12*a12;

    double c2 = a00 + a11 + a22;

    // Construct the parameters used in classifying the roots of the equation
    // and in solving the equation for the roots in closed form.
    double c2Div3 = c2*msInv3;
    double aDiv3 = (c1 - c2*c2Div3)*msInv3;
    if (aDiv3 > 0.0)
    {
        aDiv3 = 0.0;
    }

    double halfMB = 0.5*(c0 + c2Div3*(2.0*c2Div3*c2Div3 - c1));

    double q = halfMB*halfMB + aDiv3*aDiv3*aDiv3;
    if (q > 0.0)
    {
        q = 0.0;
    }

    // Compute the eigenvalues by solving for the roots of the polynomial.
    double magnitude = helper::rsqrt(-aDiv3);

    //double angle = ATan2(helper::rsqrt(-q), halfMB)*msInv3;
    // Mathematically, ATan2(0,0) is undefined, but ANSI standards
    // require the function to return 0.
    double angle;
    double tany = helper::rsqrt(-q);
    if( halfMB != 0 || tany != 0 )
    {
        angle = atan2( tany, halfMB );
    }
    else angle = 0;


    double cs = cos(angle);
    double sn = sin(angle);
    double root0 = c2Div3 + 2.0*magnitude*cs;
    double root1 = c2Div3 - magnitude*(cs + msRoot3*sn);
    double root2 = c2Div3 - magnitude*(cs - msRoot3*sn);

    // Sort in increasing order.
    if (root1 >= root0)
    {
        root[0] = root0;
        root[1] = root1;
    }
    else
    {
        root[0] = root1;
        root[1] = root0;
    }

    if (root2 >= root[1])
    {
        root[2] = root2;
    }
    else
    {
        root[2] = root[1];
        if (root2 >= root[0])
        {
            root[1] = root2;
        }
        else
        {
            root[1] = root[0];
            root[0] = root2;
        }
    }
}

template <typename Real>
bool Decompose<Real>::PositiveRank(Mat<3,3,Real>& M, Real& maxEntry, Vec<3,Real>& maxRow)
{
    // Locate the maximum-magnitude entry of the matrix.
    maxEntry = (Real)-1;
    int row, maxRowIndex = -1;
    for (row = 0; row < 3; ++row)
    {
        for (int col = row; col < 3; ++col)
        {
            Real absValue = helper::rabs(M[row][col]);
            if (absValue > maxEntry)
            {
                maxEntry = absValue;
                maxRowIndex = row;
            }
        }
    }

    // Return the row containing the maximum, to be used for eigenvector
    // construction.
    maxRow = M[maxRowIndex];

    return maxEntry >= zeroTolerance();
}


template<class Real>
void Decompose<Real>::GenerateComplementBasis(Vec<3,Real>& vec0, Vec<3,Real>& vec1, const Vec<3,Real>& vec2)
{
    float invLength;

    if (helper::rabs(vec2[0]) >= helper::rabs(vec2[1]))
    {
        // vec2.x or vec2.z is the largest magnitude component, swap them
        invLength = 1.0f/helper::rsqrt(vec2[0]*vec2[0] + vec2[2]*vec2[2]);
        vec0[0] = -vec2[2]*invLength;
        vec0[1] = 0.0f;
        vec0[2] = +vec2[0]*invLength;
        vec1[0] = vec2[1]*vec0[2];
        vec1[1] = vec2[2]*vec0[0] - vec2[0]*vec0[2];
        vec1[2] = -vec2[1]*vec0[0];
    }
    else
    {
        // vec2.y or vec2.z is the largest magnitude component, swap them
        invLength = 1.0f/helper::rsqrt(vec2[1]*vec2[1] + vec2[2]*vec2[2]);
        vec0[0] = 0.0f;
        vec0[1] = +vec2[2]*invLength;
        vec0[2] = -vec2[1]*invLength;
        vec1[0] = vec2[1]*vec0[2] - vec2[2]*vec0[1];
        vec1[1] = -vec2[0]*vec0[2];
        vec1[2] = vec2[0]*vec0[1];
    }
}

template <typename Real>
void Decompose<Real>::ComputeVectors(const Mat<3,3,Real>& A, Vec<3,Real>& U2, int i0, int i1, int i2, Mat<3,3,Real> &V, Vec<3,Real> &diag)
{
    Vec<3,Real> U0, U1;
    GenerateComplementBasis(U0, U1, U2);

    // V[i2] = c0*U0 + c1*U1,  c0^2 + c1^2=1
    // e2*V[i2] = c0*A*U0 + c1*A*U1
    // e2*c0 = c0*U0.Dot(A*U0) + c1*U0.Dot(A*U1) = d00*c0 + d01*c1
    // e2*c1 = c0*U1.Dot(A*U0) + c1*U1.Dot(A*U1) = d01*c0 + d11*c1
    Vec<3,Real> tmp = A*U0;
    Real p00 = diag[i2] - U0 * tmp;
    Real p01 = U1 * tmp;
    Real p11 = diag[i2] - U1 * (A*U1);
    Real invLength;
    Real maxValue = helper::rabs(p00);
    int row = 0;
    Real absValue = helper::rabs(p01);
    if (absValue > maxValue)
    {
        maxValue = absValue;
    }
    absValue = helper::rabs(p11);
    if (absValue > maxValue)
    {
        maxValue = absValue;
        row = 1;
    }

    if (maxValue >= zeroTolerance())
    {
        if (row == 0)
        {
            invLength = (Real)1 / helper::rsqrt(p00*p00 + p01*p01);
            p00 *= invLength;
            p01 *= invLength;
            V[i2] = U0*p01 + U1*p00;
        }
        else
        {
            invLength = (Real)1 / helper::rsqrt(p11*p11 + p01*p01);
            p11 *= invLength;
            p01 *= invLength;
            V[i2] = U0*p11 + U1*p01;
        }
    }
    else
    {
        if (row == 0)
        {
            V[i2] = U1;
        }
        else
        {
            V[i2] = U0;
        }
    }

    // V[i0] = c0*U2 + c1*Cross(U2,V[i2]) = c0*R + c1*S
    // e0*V[i0] = c0*A*R + c1*A*S
    // e0*c0 = c0*R.Dot(A*R) + c1*R.Dot(A*S) = d00*c0 + d01*c1
    // e0*c1 = c0*S.Dot(A*R) + c1*S.Dot(A*S) = d01*c0 + d11*c1
    Vec<3,Real> S = cross( U2, V[i2] );
    tmp = A*U2;
    p00 = diag[i0] - U2 * tmp;
    p01 = S * tmp;
    p11 = diag[i0] - S * (A*S);
    maxValue = helper::rabs(p00);
    row = 0;
    absValue = helper::rabs(p01);
    if (absValue > maxValue)
    {
        maxValue = absValue;
    }
    absValue = helper::rabs(p11);
    if (absValue > maxValue)
    {
        maxValue = absValue;
        row = 1;
    }

    if (maxValue >= zeroTolerance())
    {
        if (row == 0)
        {
            invLength = (Real)1 / helper::rsqrt(p00*p00 + p01*p01);
            p00 *= invLength;
            p01 *= invLength;
            V[i0] = p01*U2 + S*p00;
        }
        else
        {
            invLength = (Real)1 / helper::rsqrt(p11*p11 + p01*p01);
            p11 *= invLength;
            p01 *= invLength;
            V[i0] = U2*p11 + S*p01;
        }
    }
    else
    {
        if (row == 0)
        {
            V[i0] = S;
        }
        else
        {
            V[i0] = U2;
        }
    }

    V[i1] = cross( V[i2], V[i0] );
}



template <typename Real>
void Decompose<Real>::eigenDecomposition( const defaulttype::Mat<3,3,Real> &A, defaulttype::Mat<3,3,Real> &V, defaulttype::Vec<3,Real> &diag )
{
    // Scale the matrix so its entries are in [-1,1].  The scaling is applied
    // only when at least one matrix entry has magnitude larger than 1.
    Mat<3,3,Real> AScaled = A;
    Real* scaledEntry = (Real*)&AScaled[0];
    Real maxValue = helper::rabs(scaledEntry[0]);
    Real absValue = helper::rabs(scaledEntry[1]);
    if (absValue > maxValue)
    {
        maxValue = absValue;
    }
    absValue = helper::rabs(scaledEntry[2]);
    if (absValue > maxValue)
    {
        maxValue = absValue;
    }
    absValue = helper::rabs(scaledEntry[4]);
    if (absValue > maxValue)
    {
        maxValue = absValue;
    }
    absValue = helper::rabs(scaledEntry[5]);
    if (absValue > maxValue)
    {
        maxValue = absValue;
    }
    absValue = helper::rabs(scaledEntry[8]);
    if (absValue > maxValue)
    {
        maxValue = absValue;
    }

    int i;
    if (maxValue > (Real)1)
    {
        Real invMaxValue = ((Real)1)/maxValue;
        for (i = 0; i < 9; ++i)
        {
            scaledEntry[i] *= invMaxValue;
        }
    }

    // Compute the eigenvalues using double-precision arithmetic.
    double root[3];
    ComputeRoots(AScaled,root);
    diag[0] = (Real)root[0];
    diag[1] = (Real)root[1];
    diag[2] = (Real)root[2];

    Real maxEntry[3];
    Vec<3,Real> maxRow[3];
    for (i = 0; i < 3; ++i)
    {
        Mat<3,3,Real> M = AScaled;
        M[0][0] -= diag[i];
        M[1][1] -= diag[i];
        M[2][2] -= diag[i];
        if (!PositiveRank(M, maxEntry[i], maxRow[i]))
        {
            // Rescale back to the original size.
            if (maxValue > (Real)1)
            {
                for (int j = 0; j < 3; ++j)
                {
                    diag[j] *= maxValue;
                }
            }

            V.identity();
            return;
        }
    }

    Real totalMax = maxEntry[0];
    i = 0;
    if (maxEntry[1] > totalMax)
    {
        totalMax = maxEntry[1];
        i = 1;
    }
    if (maxEntry[2] > totalMax)
    {
        i = 2;
    }

    if (i == 0)
    {
        maxRow[0].normalize();
        ComputeVectors(AScaled, maxRow[0], 1, 2, 0, V, diag);
    }
    else if (i == 1)
    {
        maxRow[1].normalize();
        ComputeVectors(AScaled, maxRow[1], 2, 0, 1, V, diag);
    }
    else
    {
        maxRow[2].normalize();
        ComputeVectors(AScaled, maxRow[2], 0, 1, 2, V, diag);
    }

    // Rescale back to the original size.
    if (maxValue > (Real)1)
    {
        for (i = 0; i < 3; ++i)
        {
            diag[i] *= maxValue;
        }
    }

    V.transpose();
}


template <typename Real>
void Decompose<Real>::eigenDecomposition( const defaulttype::Mat<2,2,Real> &A, defaulttype::Mat<2,2,Real> &V, defaulttype::Vec<2,Real> &diag )
{
    Real inv2 = A[0][0] + A[1][1]; // trace(A)
    Real inv1 = inv2 * (Real)0.5; // trace(A) / 2
    inv2 = helper::rsqrt( inv2*inv2*(Real)0.25 - determinant( A ) ); // sqrt( tr(A)*tr(A) / 4 - det(A) )

    diag[0] = inv1 + inv2;
    diag[1] = inv1 - inv2;

    if( helper::rabs( A[1][0] ) < zeroTolerance() ) // c == 0
    {
        if( helper::rabs( A[0][1] ) < zeroTolerance() ) // b == 0
        {
            V.identity();
            return;
        }
        else
        {
            V[0].set( A[0][1], diag[0] - A[0][0] ); V[0].normalize();
            V[1].set( A[0][1], diag[1] - A[0][0] ); V[1].normalize();
        }
    }
    else
    {
        V[0][0] = diag[0] - A[1][1];
        V[0][1] = diag[1] - A[1][1];
        V[1][0] = V[1][1] = A[1][0];

        V[0].set( diag[0] - A[1][1], A[1][0] ); V[0].normalize();
        V[1].set( diag[1] - A[1][1], A[1][0] ); V[1].normalize();
    }

    V.transpose();
}




template <typename Real>
template <int iSize>
void Decompose<Real>::QLAlgorithm( defaulttype::Vec<iSize,Real> &diag, defaulttype::Vec<iSize,Real> &subDiag, defaulttype::Mat<iSize,iSize,Real> &V )
{
    static const int iMaxIter = 32;

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

/** Eigensystem decomposition: eigenvalues @param diag and eigenvectors (columns of @param V) of the 3x3 Real Matrix @param M
  * Derived from Wild Magic Library
  */
template<class Real>
void Decompose<Real>::eigenDecomposition_iterative( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &V, defaulttype::Vec<3,Real> &diag )
{
    Vec<3,Real> subDiag;

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

    ////////////

    QLAlgorithm( diag, subDiag, V );

}





template<class Real>
void Decompose<Real>::eigenDecomposition_iterative( const defaulttype::Mat<2,2,Real> &M, defaulttype::Mat<2,2,Real> &V, defaulttype::Vec<2,Real> &diag )
{
    typedef defaulttype::Vec<2,Real> Vec2;
    typedef defaulttype::Mat<2,2,Real> Mat22;

    Vec<2,Real> subDiag;

    // matrix is already tridiagonal
    diag[0] = M[0][0];
    diag[1] = M[1][1];
    subDiag[0] = M[0][1];
    subDiag[1] = 0.0;
    V.identity();

    QLAlgorithm( diag, subDiag, V );
}

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_POLARDECOMPOSE_INL
