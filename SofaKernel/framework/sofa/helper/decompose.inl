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
#ifndef SOFA_HELPER_DECOMPOSE_INL
#define SOFA_HELPER_DECOMPOSE_INL
#include "decompose.h"

#include <limits>
#include <sofa/helper/logging/Messaging.h>

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
void Decompose<Real>::QRDecomposition( const defaulttype::Mat<3,2,Real> &M, defaulttype::Mat<3,2,Real> &r )
{
    Vec<3,Real> edgex( M[0][0], M[1][0], M[2][0] );
    Vec<3,Real> edgey( M[0][1], M[1][1], M[2][1] );

    edgex.normalize();
    Vec<3,Real> edgez = cross( edgex, edgey );
    edgez.normalize();
    edgey = cross( edgez, edgex );

    r[0][0] = edgex[0]; r[0][1] = edgey[0];
    r[1][0] = edgex[1]; r[1][1] = edgey[1];
    r[2][0] = edgex[2]; r[2][1] = edgey[2];
}

template<class Real>
void Decompose<Real>::QRDecomposition( const defaulttype::Mat<2,2,Real> &M, defaulttype::Mat<2,2,Real> &r )
{
    Vec<2,Real> edgex( M[0][0], M[1][0] ); edgex.normalize();
    r[0][0] = edgex[0];  r[1][0] = edgex[1];
    r[0][1] = -edgex[1]; r[1][1] = edgex[0];
}

template<class Real>
bool Decompose<Real>::QRDecomposition_stable( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &R )
{
    bool degenerated;

    Vec<3,Real> edgex( M[0][0], M[1][0], M[2][0] );
    Vec<3,Real> edgey( M[0][1], M[1][1], M[2][1] );
    Vec<3,Real> edgez;

    Real nx = edgex.norm2();
    Real ny = edgey.norm2();

    if( nx < zeroTolerance() )
    {
        degenerated = true;
        if( ny < zeroTolerance() )
        {
            edgez.set( M[0][2], M[1][2], M[2][2] );
            Real nz = edgez.norm2();

            if( nz < zeroTolerance() ) // edgex, edgey, edgez are null -> collapsed to a point
            {
                //msg_info()<<"helper::QRDecomposition_stable collapased to point "<<M<<std::endl;
                R.identity();
                return degenerated;
            }
            else // collapsed to edgez
            {
                //msg_info()<<"QRDecomposition_stable collapased to edgez "<<M<<std::endl;

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
                //msg_info()<<"QRDecomposition_stable collapased to edgey "<<M<<std::endl;

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
                //msg_info()<<"QRDecomposition_stable collapased to face (edgey, edgez) "<<M<<std::endl;

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
            degenerated = true;

            edgez.set( M[0][2], M[1][2], M[2][2] );
            Real nz = edgez.norm2();

            if( nz < zeroTolerance() ) // collapsed to edgex
            {
                //msg_info()<<"QRDecomposition_stable ollapased to edgex "<<M<<std::endl;

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
                //msg_info()<<"QRDecomposition_stable collapased to face (edgez, edgex) "<<M<<std::endl;

                edgey = cross( edgez, edgex );
                edgey.normalize();
                edgez = cross( edgex, edgey );
            }
        }
        else // edgex & edgey are ok (either not collapsed or collapsed to face (edgex,edgey) )
        {
            degenerated = false;

            edgez = cross( edgex, edgey );
            edgez.normalize();
            edgey = cross( edgez, edgex );
        }
    }

    R[0][0] = edgex[0]; R[0][1] = edgey[0]; R[0][2] = edgez[0];
    R[1][0] = edgex[1]; R[1][1] = edgey[1]; R[1][2] = edgez[1];
    R[2][0] = edgex[2]; R[2][1] = edgey[2]; R[2][2] = edgez[2];

    return degenerated;
}


template<class Real>
bool Decompose<Real>::QRDecomposition_stable( const defaulttype::Mat<3,2,Real> &M, defaulttype::Mat<3,2,Real> &r )
{
    bool degenerated;

    Vec<3,Real> edgex( M[0][0], M[1][0], M[2][0] );
    Vec<3,Real> edgey( M[0][1], M[1][1], M[2][1] );

    Real nx = edgex.norm2();
    Real ny = edgey.norm2();

    if( nx < zeroTolerance() )
    {
        if( ny < zeroTolerance() ) // edgex, edgey are null -> collapsed to a point
        {
            r[0][0] = 1; r[0][1] = 0;
            r[1][0] = 0; r[1][1] = 1;
            r[2][0] = 0; r[2][1] = 0;
            return true;
        }
        else // collapsed to edgey
        {
            degenerated = true;
            edgey.normalizeWithNorm( helper::rsqrt(ny) );

            // check the main direction of edgex to try to take a not too close arbritary vector
            Real abs0 = helper::rabs( edgey[0] );
            Real abs1 = helper::rabs( edgey[1] );
            Real abs2 = helper::rabs( edgey[2] );
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

            Vec<3,Real> edgez = cross( edgex, edgey );
            edgez.normalize();
            edgex = cross( edgey, edgez );
        }
    }
    else // not collapsed
    {
        edgex.normalizeWithNorm( helper::rsqrt(nx) );

        if( ny < zeroTolerance() ) // collapsed to edgex
        {
            degenerated = true;

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

            Vec<3,Real> edgez = cross( edgex, edgey );
            edgez.normalize();
            edgey = cross( edgez, edgex );
        }
        else // edgex & edgey are ok (not collapsed)
        {
            edgey.normalizeWithNorm( helper::rsqrt(ny) );

            degenerated = false;

            Vec<3,Real> edgez = cross( edgex, edgey );
            edgez.normalize();
            edgey = cross( edgez, edgex );
        }
    }

    r[0][0] = edgex[0]; r[0][1] = edgey[0];
    r[1][0] = edgex[1]; r[1][1] = edgey[1];
    r[2][0] = edgex[2]; r[2][1] = edgey[2];

    return degenerated;
}


template<class Real>
bool Decompose<Real>::QRDecomposition_stable( const defaulttype::Mat<2,2,Real> &M, defaulttype::Mat<2,2,Real> &r )
{
    bool degenerated;

    Vec<2,Real> edge( M[0][0], M[1][0] );
    Real n = edge.norm2();

    if( n < zeroTolerance() )
    {
        edge.set( M[0][1], M[1][1] );
        n = edge.norm2();
        if( n < zeroTolerance() )
        {
            // colapsed to a point
            r[0][0] = 1; r[1][0] = 0;
            r[0][1] = 0; r[1][1] = 1;

            degenerated = true;
        }
        else
        {
            // edgex collapsed but edgey not colapsed
            edge /= helper::rsqrt(n);
            r[0][0] = edge[1]; r[1][0] = -edge[0];
            r[0][1] = edge[0]; r[1][1] =  edge[1];

            degenerated = true;
        }
    }
    else
    {
        // edgex not colapsed
        edge /= helper::rsqrt(n);
        r[0][0] =  edge[0]; r[1][0] = edge[1];
        r[0][1] = -edge[1]; r[1][1] = edge[0];

        degenerated = false; // todo test edgey?
    }

    return degenerated;
}


//////////// Polar Decomposition
////// FROM Decompose.c
////// Ken Shoemake, 1993

/******* Matrix Preliminaries *******/

//template<class Real>
//void Decompose<Real>::adjoint_transpose(const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& MadjT)
//{
//    MadjT[0] = cross(M[1],M[2]);
//    MadjT[1] = cross(M[2],M[0]);
//    MadjT[2] = cross(M[0],M[1]);
//}

//template<class Real>
//Real Decompose<Real>::norm_inf(const defaulttype::Mat<3,3,Real>& M)
//{
//    Real sum, max = 0;
//    for (int i=0; i<3; i++)
//    {
//        sum = (Real)(fabs(M[i][0])+fabs(M[i][1])+fabs(M[i][2]));
//        if (max<sum) max = sum;
//    }
//    return max;
//}

//template<class Real>
//Real Decompose<Real>::norm_one(const defaulttype::Mat<3,3,Real>& M)
//{
//    Real sum, max = 0;
//    for (int i=0; i<3; i++)
//    {
//        sum = (Real)(fabs(M[0][i])+fabs(M[1][i])+fabs(M[2][i]));
//        if (max<sum) max = sum;
//    }
//    return max;
//}


//template<class Real>
//int Decompose<Real>::find_max_col(const defaulttype::Mat<3,3,Real>& M)
//{
//    Real abs, max = 0;
//    int col = -1;
//    for (int i=0; i<3; i++)
//        for (int j=0; j<3; j++)
//        {
//            abs = M[i][j]; if (abs<0.0) abs = -abs;
//            if (abs>max) {max = abs; col = j;}
//        }
//    return col;
//}

//template<class Real>
//void Decompose<Real>::make_reflector(const defaulttype::Vec<3,Real>& v, defaulttype::Vec<3,Real>& u)
//{
//    Real s = (Real)sqrt(dot(v, v));
//    u[0] = v[0]; u[1] = v[1];
//    u[2] = v[2] + ((v[2]<0.0) ? -s : s);
//    s = (Real)sqrt(2.0/dot(u, u));
//    u[0] = u[0]*s; u[1] = u[1]*s; u[2] = u[2]*s;
//}


//template<class Real>
//void Decompose<Real>::reflect_cols(defaulttype::Mat<3,3,Real>& M, const defaulttype::Vec<3,Real>& u)
//{
//    for (int i=0; i<3; i++)
//    {
//        Real s = u[0]*M[0][i] + u[1]*M[1][i] + u[2]*M[2][i];
//        for (int j=0; j<3; j++)
//            M[j][i] -= u[j]*s;
//    }
//}

//template<class Real>
//void Decompose<Real>::reflect_rows(defaulttype::Mat<3,3,Real>& M, const defaulttype::Vec<3,Real>& u)
//{
//    for (int i=0; i<3; i++)
//    {
//        Real s = dot(u, M[i]);
//        for (int j=0; j<3; j++)
//            M[i][j] -= u[j]*s;
//    }
//}

//template<class Real>
//void Decompose<Real>::do_rank1(defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q)
//{
//    defaulttype::Vec<3,Real> v1, v2;
//    Real s;
//    int col;
//    Q.identity();
//    /* If rank(M) is 1, we should find a non-zero column in M */
//    col = find_max_col(M);
//    if (col<0) return; /* Rank is 0 */
//    v1[0] = M[0][col]; v1[1] = M[1][col]; v1[2] = M[2][col];
//    make_reflector(v1, v1); reflect_cols(M, v1);
//    v2[0] = M[2][0]; v2[1] = M[2][1]; v2[2] = M[2][2];
//    make_reflector(v2, v2); reflect_rows(M, v2);
//    s = M[2][2];
//    if (s<0.0) Q[2][2] = -1.0;
//    reflect_cols(Q, v1); reflect_rows(Q, v2);
//}


//template<class Real>
//void Decompose<Real>::do_rank2(defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& MadjT, defaulttype::Mat<3,3,Real>& Q)
//{
//    defaulttype::Vec<3,Real> v1, v2;
//    Real w, x, y, z, c, s, d;
//    int col;
//    /* If rank(M) is 2, we should find a non-zero column in MadjT */
//    col = find_max_col(MadjT);
//    if (col<0) {do_rank1(M, Q); return;} /* Rank<2 */
//    v1[0] = MadjT[0][col]; v1[1] = MadjT[1][col]; v1[2] = MadjT[2][col];
//    make_reflector(v1, v1); reflect_cols(M, v1);
//    v2 = cross(M[0], M[1]);
//    make_reflector(v2, v2); reflect_rows(M, v2);
//    w = M[0][0]; x = M[0][1]; y = M[1][0]; z = M[1][1];
//    if (w*z>x*y)
//    {
//        c = z+w; s = y-x; d = sqrt(c*c+s*s); c = c/d; s = s/d;
//        Q[0][0] = Q[1][1] = c; Q[0][1] = -(Q[1][0] = s);
//    }
//    else
//    {
//        c = z-w; s = y+x; d = sqrt(c*c+s*s); c = c/d; s = s/d;
//        Q[0][0] = -(Q[1][1] = c); Q[0][1] = Q[1][0] = s;
//    }
//    Q[0][2] = Q[2][0] = Q[1][2] = Q[2][1] = 0.0; Q[2][2] = 1.0;
//    reflect_cols(Q, v1); reflect_rows(Q, v2);
//}


//template<class Real>
//Real Decompose<Real>::polarDecomposition(const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q, defaulttype::Mat<3,3,Real>& S)
//{
//    defaulttype::Mat<3,3,Real> Mk, MadjTk, Ek;
//    Real det, M_one, M_inf, MadjT_one, MadjT_inf, E_one, gamma, g1, g2;
//    Mk.transpose(M);
//    M_one = norm_one(Mk);  M_inf = norm_inf(Mk);
//    do
//    {
//        adjoint_transpose(Mk, MadjTk);
//        det = dot(Mk[0], MadjTk[0]);
//        if (det==0.0)
//        {
//            do_rank2(Mk, MadjTk, Mk);
//            break;
//        }
//        MadjT_one = norm_one(MadjTk); MadjT_inf = norm_inf(MadjTk);
//        gamma = (Real)sqrt(sqrt((MadjT_one*MadjT_inf)/(M_one*M_inf))/fabs(det));
//        g1 = gamma*((Real)0.5);
//        g2 = ((Real)0.5)/(gamma*det);
//        Ek = Mk;
//        Mk = Mk*g1 + MadjTk*g2;
//        Ek -= Mk;
//        E_one = norm_one(Ek);
//        M_one = norm_one(Mk);  M_inf = norm_inf(Mk);
//    }
//    while (E_one>(M_one*zeroTolerance()));
//    Q.transpose(Mk);
//    S = Mk*M;
//    for (int i=0; i<3; i++)
//        for (int j=i+1; j<3; j++)
//            S[i][j] = S[j][i] = ((Real)0.5)*(S[i][j]+S[j][i]);
//    return (det);

//}


/// original code by Ken Shoemake, 1993
/// version simplified by Jernej Barbič imported from Vega
template<class Real>
Real Decompose<Real>::polarDecomposition( const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q, defaulttype::Mat<3,3,Real>& S )
{
  defaulttype::Mat<3,3,Real> Mk;
  defaulttype::Mat<3,3,Real> Ek;
  Real det, M_oneNorm, M_infNorm, E_oneNorm;

  // Mk = M^T
  Mk.transpose( M );

  M_oneNorm = oneNorm(Mk);
  M_infNorm = infNorm(Mk);

  do
  {

    defaulttype::Mat<3,3,Real> MadjTk;

    // row 2 x row 3
    MadjTk[0] = cross( Mk[1], Mk[2] );
    // row 3 x row 1
    MadjTk[1] = cross( Mk[2], Mk[0] );
    // row 1 x row 2
    MadjTk[2] = cross( Mk[0], Mk[1] );

    det = Mk(0,0) * MadjTk(0,0) + Mk(0,1) * MadjTk(0,1) + Mk(0,2) * MadjTk(0,2);
    if (det == 0.0)
    {
        msg_warning("Decompose")<<"polarDecomposition: zero determinant encountered.";
        break;
    }

    Real MadjT_one = oneNorm(MadjTk);
    Real MadjT_inf = infNorm(MadjTk);

    Real gamma = sqrt(sqrt((MadjT_one * MadjT_inf) / (M_oneNorm * M_infNorm)) / fabs(det));
    Real g1 = gamma * static_cast<Real>(0.5);
    Real g2 = static_cast<Real>(0.5) / (gamma * det);

    Ek = Mk;
    Mk = Mk * g1 + MadjTk * g2;
    Ek -= Mk;

    E_oneNorm = oneNorm(Ek);
    M_oneNorm = oneNorm(Mk);
    M_infNorm = infNorm(Mk);

  }
  while ( E_oneNorm > M_oneNorm * zeroTolerance() );


  // Q = Mk^T
  Q.transpose( Mk );
  S = Mk * M;

  // S must be symmetric; enforce the symmetry
  for (int i=0; i<3; i++)
    for (int j=i; j<3; j++)
        S(i,j) = S(j,i) = static_cast<Real>(0.5) * (S(i,j) + S(j,i));

  return det;
}


template<class Real>
Real Decompose<Real>::polarDecomposition( const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q )
{
  defaulttype::Mat<3,3,Real> Mk;
  defaulttype::Mat<3,3,Real> Ek;
  Real det, M_oneNorm, M_infNorm, E_oneNorm;

  // Mk = M^T
  Mk.transpose( M );

  M_oneNorm = oneNorm(Mk);
  M_infNorm = infNorm(Mk);

  do
  {
    defaulttype::Mat<3,3,Real> MadjTk;

    // row 2 x row 3
    MadjTk[0] = cross( Mk[1], Mk[2] );
    // row 3 x row 1
    MadjTk[1] = cross( Mk[2], Mk[0] );
    // row 1 x row 2
    MadjTk[2] = cross( Mk[0], Mk[1] );

    det = Mk(0,0) * MadjTk(0,0) + Mk(0,1) * MadjTk(0,1) + Mk(0,2) * MadjTk(0,2);
    if (det == 0.0)
    {
        msg_warning("Decompose")<<"polarDecomposition: zero determinant encountered.";
        break;
    }

    Real MadjT_one = oneNorm(MadjTk);
    Real MadjT_inf = infNorm(MadjTk);

    Real gamma = sqrt(sqrt((MadjT_one * MadjT_inf) / (M_oneNorm * M_infNorm)) / fabs(det));
    Real g1 = gamma * static_cast<Real>(0.5);
    Real g2 = static_cast<Real>(0.5) / (gamma * det);

    Ek = Mk;
    Mk = Mk * g1 + MadjTk * g2;
    Ek -= Mk;

    E_oneNorm = oneNorm(Ek);
    M_oneNorm = oneNorm(Mk);
    M_infNorm = infNorm(Mk);
  }
  while ( E_oneNorm > M_oneNorm * zeroTolerance() );

  // Q = Mk^T
  Q.transpose( Mk );

  return det;
}




//template<class Real>
//Real Decompose<Real>::polarDecomposition( const defaulttype::Mat<3,3,Real>& M, defaulttype::Mat<3,3,Real>& Q )
//{
//    defaulttype::Mat<3,3,Real> Mk, MadjTk, Ek;
//    Real det, M_one, M_inf, MadjT_one, MadjT_inf, E_one, gamma, g1, g2;
//    Mk.transpose(M);
//    M_one = norm_one(Mk);  M_inf = norm_inf(Mk);
//    do
//    {
//        adjoint_transpose(Mk, MadjTk);
//        det = dot(Mk[0], MadjTk[0]);
//        if (det==0.0)
//        {
//            do_rank2(Mk, MadjTk, Mk);
//            break;
//        }
//        MadjT_one = norm_one(MadjTk); MadjT_inf = norm_inf(MadjTk);
//        gamma = (Real)sqrt(sqrt((MadjT_one*MadjT_inf)/(M_one*M_inf))/fabs(det));
//        g1 = gamma*((Real)0.5);
//        g2 = ((Real)0.5)/(gamma*det);
//        Ek = Mk;
//        Mk = Mk*g1 + MadjTk*g2;
//        Ek -= Mk;
//        E_one = norm_one(Ek);
//        M_one = norm_one(Mk);  M_inf = norm_inf(Mk);
//    }
//    while (E_one>(M_one*zeroTolerance()));
//    Q.transpose(Mk);
//    return (det);
//}


template<class Real>
void Decompose<Real>::polarDecomposition( const defaulttype::Mat<2,2,Real>& M, defaulttype::Mat<2,2,Real>& Q )
{
    Q[0][0] =  M[1][1];
    Q[0][1] = -M[1][0];
    Q[1][0] = -M[0][1];
    Q[1][1] =  M[0][0];
    Q = M + ( determinant( M ) < 0 ? (Real)-1.0 : (Real)1.0 ) * Q;

    for (unsigned int i=0; i<2; i++)
    {
        Real normColi = sqrt(Q[0][i]*Q[0][i] +  Q[1][i] * Q[1][i] );
        Q[0][i] *= (1/normColi);
        Q[1][i] *= (1/normColi);
    }
}




template<class Real>
bool Decompose<Real>::polarDecomposition_stable( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &Q, defaulttype::Mat<3,3,Real> &S )
{
    bool degenerated = polarDecomposition_stable( M, Q );
    S = Q.multTranspose( M ); // S = Qt * M
    //S = V.multDiagonal( Sdiag ).multTransposed( V ); // S = V * Sdiag * Vt

    return degenerated;
}

template<class Real>
bool Decompose<Real>::polarDecomposition_stable( const defaulttype::Mat<3,3,Real> &M, defaulttype::Mat<3,3,Real> &Q )
{
    defaulttype::Mat<3,3,Real> U, V;
    defaulttype::Vec<3,Real> Sdiag;
    bool degenerated = helper::Decompose<Real>::SVD_stable( M, U, Sdiag, V );

    Q = U.multTransposed( V ); // Q = U * Vt

    return degenerated;
}

template<class Real>
bool Decompose<Real>::polarDecomposition_stable( const defaulttype::Mat<2,2,Real> &M, defaulttype::Mat<2,2,Real> &Q, defaulttype::Mat<2,2,Real> &S )
{
    bool degenerated = polarDecomposition_stable( M, Q );
    S = Q.multTranspose( M ); // S = Qt * M
    //S = V.multDiagonal( Sdiag ).multTransposed( V ); // S = V * Sdiag * Vt

    return degenerated;
}

template<class Real>
bool Decompose<Real>::polarDecomposition_stable( const defaulttype::Mat<2,2,Real> &M, defaulttype::Mat<2,2,Real> &Q )
{
    defaulttype::Mat<2,2,Real> U, V;
    defaulttype::Vec<2,Real> Sdiag;
    bool degenerated = helper::Decompose<Real>::SVD_stable( M, U, Sdiag, V );

    Q = U.multTransposed( V ); // Q = U * Vt

    return degenerated;
}


template<class Real>
void Decompose<Real>::polarDecomposition( const defaulttype::Mat<3,2,Real> &M, defaulttype::Mat<3,2,Real> &Q, defaulttype::Mat<2,2,Real> &S )
{
    defaulttype::Mat<3,2,Real> U;
    defaulttype::Mat<2,2,Real> V;
    defaulttype::Vec<2,Real> Sdiag;
    helper::Decompose<Real>::SVD_stable( M, U, Sdiag, V );

    Q = U.multTransposed( V );
    S = Q.multTranspose( M );
}



template<class Real>
defaulttype::Mat<3,3,Real> Decompose<Real>::skewMat( const defaulttype::Vec<3,Real>& v )
{
    defaulttype::Mat<3,3,Real> M;
    M[0][1] = -v[2]; M[1][0] = -M[0][1];
    M[0][2] =  v[1]; M[2][0] = -M[0][2];
    M[1][2] = -v[0]; M[2][1] = -M[1][2];
    return M;
}

template<class Real>
defaulttype::Vec<3,Real> Decompose<Real>::skewVec( const defaulttype::Mat<3,3,Real>& M )
{
    defaulttype::Vec<3,Real> v;
    v[0] = (Real)0.5 * ( M[2][1] - M[1][2] );
    v[1] = (Real)0.5 * ( M[0][2] - M[2][0] );
    v[2] = (Real)0.5 * ( M[1][0] - M[0][1] );
    return v;
}

template<class Real>
void Decompose<Real>::polarDecompositionGradient_G( const defaulttype::Mat<3,3,Real>& Q, const defaulttype::Mat<3,3,Real>& S, defaulttype::Mat<3,3,Real>& invG )
{
    // invG = ((tr(S)*I-S)*Qt)^-1

    defaulttype::Mat<3,3,Real> G = -S;
    Real trace = defaulttype::trace( S );
    G[0][0] += trace;
    G[1][1] += trace;
    G[2][2] += trace;

    G = G.multTransposed( Q );

    invG.invert( G );
}


template<class Real>
void Decompose<Real>::polarDecompositionGradient_dQ( const defaulttype::Mat<3,3,Real>& invG, const defaulttype::Mat<3,3,Real>& Q, const defaulttype::Mat<3,3,Real>& dM, defaulttype::Mat<3,3,Real>& dQ )
{
    // omega = invG * (2 * skew(Q^T * dM))
    defaulttype::Vec<3,Real> omega = invG * skewVec( Q.multTranspose( dM ) ) * 2;

    dQ = skewMat( omega ) * Q;
}

template<class Real>
void Decompose<Real>::polarDecompositionGradient_dQOverdM(const defaulttype::Mat<3,3,Real> &Q, const defaulttype::Mat<3,3,Real> &invG,  defaulttype::Mat<9,9,Real>& J)
{
J[0][0]=(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*Q[1][0]+(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*Q[2][0]; J[0][1]=(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*Q[1][1]+(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*Q[2][1]; J[0][2]=(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*Q[1][2]+(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*Q[2][2]; J[0][3]=(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*Q[0][0]+(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*Q[2][0]; J[0][4]=(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*Q[0][1]+(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*Q[2][1]; J[0][5]=(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*Q[0][2]+(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*Q[2][2]; J[0][6]=(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*Q[0][0]+(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*Q[1][0]; J[0][7]=(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*Q[0][1]+(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*Q[1][1]; J[0][8]=(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*Q[0][2]+(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*Q[1][2];
J[1][0]=(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*Q[1][0]+(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*Q[2][0]; J[1][1]=(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*Q[1][1]+(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*Q[2][1]; J[1][2]=(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*Q[1][2]+(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*Q[2][2]; J[1][3]=(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*Q[0][0]+(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*Q[2][0]; J[1][4]=(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*Q[0][1]+(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*Q[2][1]; J[1][5]=(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*Q[0][2]+(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*Q[2][2]; J[1][6]=(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*Q[0][0]+(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*Q[1][0]; J[1][7]=(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*Q[0][1]+(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*Q[1][1]; J[1][8]=(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*Q[0][2]+(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*Q[1][2];
J[2][0]=(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*Q[1][0]+(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*Q[2][0]; J[2][1]=(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*Q[1][1]+(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*Q[2][1]; J[2][2]=(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*Q[1][2]+(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*Q[2][2]; J[2][3]=(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*Q[0][0]+(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*Q[2][0]; J[2][4]=(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*Q[0][1]+(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*Q[2][1]; J[2][5]=(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*Q[0][2]+(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*Q[2][2]; J[2][6]=(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*Q[0][0]+(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*Q[1][0]; J[2][7]=(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*Q[0][1]+(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*Q[1][1]; J[2][8]=(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*Q[0][2]+(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*Q[1][2];
J[3][0]=(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*Q[1][0]+(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*Q[2][0]; J[3][1]=(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*Q[1][1]+(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*Q[2][1]; J[3][2]=(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*Q[1][2]+(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*Q[2][2]; J[3][3]=(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*Q[0][0]+(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*Q[2][0]; J[3][4]=(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*Q[0][1]+(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*Q[2][1]; J[3][5]=(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*Q[0][2]+(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*Q[2][2]; J[3][6]=(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*Q[0][0]+(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*Q[1][0]; J[3][7]=(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*Q[0][1]+(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*Q[1][1]; J[3][8]=(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*Q[0][2]+(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*Q[1][2];
J[4][0]=(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*Q[1][0]+(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*Q[2][0]; J[4][1]=(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*Q[1][1]+(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*Q[2][1]; J[4][2]=(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*Q[1][2]+(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*Q[2][2]; J[4][3]=(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*Q[0][0]+(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*Q[2][0]; J[4][4]=(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*Q[0][1]+(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*Q[2][1]; J[4][5]=(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*Q[0][2]+(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*Q[2][2]; J[4][6]=(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*Q[0][0]+(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*Q[1][0]; J[4][7]=(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*Q[0][1]+(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*Q[1][1]; J[4][8]=(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*Q[0][2]+(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*Q[1][2];
J[5][0]=(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*Q[1][0]+(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*Q[2][0]; J[5][1]=(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*Q[1][1]+(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*Q[2][1]; J[5][2]=(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*Q[1][2]+(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*Q[2][2]; J[5][3]=(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*Q[0][0]+(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*Q[2][0]; J[5][4]=(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*Q[0][1]+(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*Q[2][1]; J[5][5]=(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*Q[0][2]+(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*Q[2][2]; J[5][6]=(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*Q[0][0]+(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*Q[1][0]; J[5][7]=(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*Q[0][1]+(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*Q[1][1]; J[5][8]=(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*Q[0][2]+(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*Q[1][2];
J[6][0]=(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*Q[1][0]+(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*Q[2][0]; J[6][1]=(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*Q[1][1]+(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*Q[2][1]; J[6][2]=(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*Q[1][2]+(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*Q[2][2]; J[6][3]=(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*Q[0][0]+(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*Q[2][0]; J[6][4]=(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*Q[0][1]+(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*Q[2][1]; J[6][5]=(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*Q[0][2]+(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*Q[2][2]; J[6][6]=(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*Q[0][0]+(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*Q[1][0]; J[6][7]=(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*Q[0][1]+(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*Q[1][1]; J[6][8]=(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*Q[0][2]+(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*Q[1][2];
J[7][0]=(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*Q[1][0]+(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*Q[2][0]; J[7][1]=(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*Q[1][1]+(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*Q[2][1]; J[7][2]=(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*Q[1][2]+(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*Q[2][2]; J[7][3]=(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*Q[0][0]+(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*Q[2][0]; J[7][4]=(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*Q[0][1]+(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*Q[2][1]; J[7][5]=(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*Q[0][2]+(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*Q[2][2]; J[7][6]=(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*Q[0][0]+(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*Q[1][0]; J[7][7]=(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*Q[0][1]+(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*Q[1][1]; J[7][8]=(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*Q[0][2]+(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*Q[1][2];
J[8][0]=(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*Q[1][0]+(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*Q[2][0]; J[8][1]=(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*Q[1][1]+(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*Q[2][1]; J[8][2]=(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*Q[1][2]+(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*Q[2][2]; J[8][3]=(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*Q[0][0]+(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*Q[2][0]; J[8][4]=(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*Q[0][1]+(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*Q[2][1]; J[8][5]=(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*Q[0][2]+(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*Q[2][2]; J[8][6]=(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*Q[0][0]+(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*Q[1][0]; J[8][7]=(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*Q[0][1]+(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*Q[1][1]; J[8][8]=(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*Q[0][2]+(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*Q[1][2];
}

// another method based on the relation : M=QS -> dQ = (dM - Q dS)S^-1  ->  dQ = (dM - dSOverdM.dM)S^-1  -> dQ = JdM
// coeff between dMij and dQkl = sum_m ( (1) - sum_n ( dSOverdM(n,m,i,j).Q(k,n) ) .Sinv(l,m)
template<class Real>
void Decompose<Real>::polarDecompositionGradient_dQOverdM(const defaulttype::Mat<3,3,Real> &Q, const defaulttype::Mat<3,3,Real> &Sinv, const defaulttype::Mat<9,9,Real>& dSOverdM, defaulttype::Mat<9,9,Real>& J)
{
    for(unsigned i=0;i<3;i++)
        for(unsigned j=0;j<3;j++)
            for(unsigned k=0;k<3;k++)
                for(unsigned l=0;l<3;l++)
                {
                    J(l+3*k,j+3*i)=0;
                    for(unsigned m=0;m<3;m++)
                    {
                        Real c;
                        if(k==i && j==m) c=1.; else c=0;
                        for(unsigned n=0;n<3;n++) c -= Q(k,n)*dSOverdM(m+3*n,j+3*i);
                        J(l+3*k,j+3*i) += Sinv(l,m)*c;
                    }
                }
}

template<class Real>
void Decompose<Real>::polarDecompositionGradient_dS( const defaulttype::Mat<3,3,Real>& Q, const defaulttype::Mat<3,3,Real>& S, const defaulttype::Mat<3,3,Real>& dQ, const defaulttype::Mat<3,3,Real>& dM, defaulttype::Mat<3,3,Real>& dS )
{
    dS = Q.multTranspose( dM - dQ * S ); // dS = Qt * ( dM - dQ * S )
}

template<class Real>
void Decompose<Real>::polarDecompositionGradient_dSOverdM(const defaulttype::Mat<3,3,Real> &Q, const defaulttype::Mat<3,3,Real> &M, const  defaulttype::Mat<3,3,Real>& invG,  defaulttype::Mat<9,9,Real>& J)
{
J[0][0]=Q[0][0]*(1-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][0]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][0])+Q[1][0]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][0]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][0])+Q[2][0]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][0]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][0]); J[0][1]=Q[0][0]*(-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][0]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][0])+Q[1][0]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][0]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][0])+Q[2][0]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][0]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][0]); J[0][2]=Q[0][0]*(-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][0]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][0])+Q[1][0]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][0]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][0])+Q[2][0]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][0]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][0]); J[0][3]=Q[0][0]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][0]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][0])+Q[1][0]*(1-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][0]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][0])+Q[2][0]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][0]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][0]); J[0][4]=Q[0][0]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][0]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][0])+Q[1][0]*(-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][0]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][0])+Q[2][0]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][0]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][0]); J[0][5]=Q[0][0]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][0]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][0])+Q[1][0]*(-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][0]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][0])+Q[2][0]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][0]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][0]); J[0][6]=Q[0][0]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][0]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][0])+Q[1][0]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][0]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][0])+Q[2][0]*(1-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][0]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][0]); J[0][7]=Q[0][0]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][0]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][0])+Q[1][0]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][0]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][0])+Q[2][0]*(-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][0]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][0]); J[0][8]=Q[0][0]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][0]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][0])+Q[1][0]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][0]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][0])+Q[2][0]*(-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][0]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][0]);
J[1][0]=Q[0][0]*(-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][1]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][1])+Q[1][0]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][1]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][1])+Q[2][0]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][1]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][1]); J[1][1]=Q[0][0]*(1-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][1]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][1])+Q[1][0]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][1]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][1])+Q[2][0]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][1]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][1]); J[1][2]=Q[0][0]*(-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][1]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][1])+Q[1][0]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][1]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][1])+Q[2][0]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][1]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][1]); J[1][3]=Q[0][0]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][1]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][1])+Q[1][0]*(-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][1]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][1])+Q[2][0]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][1]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][1]); J[1][4]=Q[0][0]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][1]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][1])+Q[1][0]*(1-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][1]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][1])+Q[2][0]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][1]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][1]); J[1][5]=Q[0][0]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][1]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][1])+Q[1][0]*(-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][1]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][1])+Q[2][0]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][1]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][1]); J[1][6]=Q[0][0]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][1]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][1])+Q[1][0]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][1]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][1])+Q[2][0]*(-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][1]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][1]); J[1][7]=Q[0][0]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][1]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][1])+Q[1][0]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][1]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][1])+Q[2][0]*(1-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][1]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][1]); J[1][8]=Q[0][0]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][1]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][1])+Q[1][0]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][1]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][1])+Q[2][0]*(-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][1]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][1]);
J[2][0]=Q[0][0]*(-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][2]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][2])+Q[1][0]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][2]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][2])+Q[2][0]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][2]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][2]); J[2][1]=Q[0][0]*(-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][2]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][2])+Q[1][0]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][2]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][2])+Q[2][0]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][2]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][2]); J[2][2]=Q[0][0]*(1-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][2]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][2])+Q[1][0]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][2]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][2])+Q[2][0]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][2]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][2]); J[2][3]=Q[0][0]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][2]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][2])+Q[1][0]*(-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][2]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][2])+Q[2][0]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][2]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][2]); J[2][4]=Q[0][0]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][2]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][2])+Q[1][0]*(-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][2]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][2])+Q[2][0]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][2]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][2]); J[2][5]=Q[0][0]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][2]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][2])+Q[1][0]*(1-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][2]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][2])+Q[2][0]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][2]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][2]); J[2][6]=Q[0][0]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][2]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][2])+Q[1][0]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][2]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][2])+Q[2][0]*(-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][2]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][2]); J[2][7]=Q[0][0]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][2]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][2])+Q[1][0]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][2]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][2])+Q[2][0]*(-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][2]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][2]); J[2][8]=Q[0][0]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][2]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][2])+Q[1][0]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][2]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][2])+Q[2][0]*(1-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][2]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][2]);
J(3)=J(1); //J[3][0]=Q[0][1]*(1-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][0]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][0])+Q[1][1]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][0]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][0])+Q[2][1]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][0]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][0]); J[3][1]=Q[0][1]*(-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][0]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][0])+Q[1][1]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][0]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][0])+Q[2][1]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][0]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][0]); J[3][2]=Q[0][1]*(-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][0]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][0])+Q[1][1]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][0]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][0])+Q[2][1]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][0]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][0]); J[3][3]=Q[0][1]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][0]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][0])+Q[1][1]*(1-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][0]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][0])+Q[2][1]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][0]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][0]); J[3][4]=Q[0][1]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][0]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][0])+Q[1][1]*(-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][0]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][0])+Q[2][1]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][0]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][0]); J[3][5]=Q[0][1]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][0]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][0])+Q[1][1]*(-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][0]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][0])+Q[2][1]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][0]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][0]); J[3][6]=Q[0][1]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][0]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][0])+Q[1][1]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][0]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][0])+Q[2][1]*(1-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][0]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][0]); J[3][7]=Q[0][1]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][0]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][0])+Q[1][1]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][0]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][0])+Q[2][1]*(-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][0]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][0]); J[3][8]=Q[0][1]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][0]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][0])+Q[1][1]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][0]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][0])+Q[2][1]*(-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][0]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][0]);
J[4][0]=Q[0][1]*(-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][1]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][1])+Q[1][1]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][1]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][1])+Q[2][1]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][1]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][1]); J[4][1]=Q[0][1]*(1-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][1]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][1])+Q[1][1]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][1]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][1])+Q[2][1]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][1]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][1]); J[4][2]=Q[0][1]*(-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][1]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][1])+Q[1][1]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][1]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][1])+Q[2][1]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][1]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][1]); J[4][3]=Q[0][1]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][1]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][1])+Q[1][1]*(-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][1]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][1])+Q[2][1]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][1]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][1]); J[4][4]=Q[0][1]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][1]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][1])+Q[1][1]*(1-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][1]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][1])+Q[2][1]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][1]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][1]); J[4][5]=Q[0][1]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][1]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][1])+Q[1][1]*(-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][1]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][1])+Q[2][1]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][1]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][1]); J[4][6]=Q[0][1]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][1]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][1])+Q[1][1]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][1]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][1])+Q[2][1]*(-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][1]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][1]); J[4][7]=Q[0][1]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][1]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][1])+Q[1][1]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][1]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][1])+Q[2][1]*(1-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][1]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][1]); J[4][8]=Q[0][1]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][1]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][1])+Q[1][1]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][1]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][1])+Q[2][1]*(-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][1]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][1]);
J[5][0]=Q[0][1]*(-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][2]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][2])+Q[1][1]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][2]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][2])+Q[2][1]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][2]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][2]); J[5][1]=Q[0][1]*(-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][2]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][2])+Q[1][1]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][2]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][2])+Q[2][1]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][2]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][2]); J[5][2]=Q[0][1]*(1-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][2]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][2])+Q[1][1]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][2]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][2])+Q[2][1]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][2]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][2]); J[5][3]=Q[0][1]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][2]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][2])+Q[1][1]*(-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][2]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][2])+Q[2][1]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][2]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][2]); J[5][4]=Q[0][1]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][2]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][2])+Q[1][1]*(-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][2]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][2])+Q[2][1]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][2]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][2]); J[5][5]=Q[0][1]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][2]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][2])+Q[1][1]*(1-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][2]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][2])+Q[2][1]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][2]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][2]); J[5][6]=Q[0][1]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][2]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][2])+Q[1][1]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][2]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][2])+Q[2][1]*(-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][2]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][2]); J[5][7]=Q[0][1]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][2]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][2])+Q[1][1]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][2]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][2])+Q[2][1]*(-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][2]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][2]); J[5][8]=Q[0][1]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][2]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][2])+Q[1][1]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][2]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][2])+Q[2][1]*(1-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][2]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][2]);
J(6)=J(2); //J[6][0]=Q[0][2]*(1-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][0]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][0])+Q[1][2]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][0]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][0])+Q[2][2]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][0]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][0]); J[6][1]=Q[0][2]*(-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][0]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][0])+Q[1][2]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][0]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][0])+Q[2][2]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][0]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][0]); J[6][2]=Q[0][2]*(-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][0]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][0])+Q[1][2]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][0]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][0])+Q[2][2]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][0]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][0]); J[6][3]=Q[0][2]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][0]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][0])+Q[1][2]*(1-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][0]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][0])+Q[2][2]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][0]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][0]); J[6][4]=Q[0][2]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][0]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][0])+Q[1][2]*(-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][0]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][0])+Q[2][2]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][0]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][0]); J[6][5]=Q[0][2]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][0]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][0])+Q[1][2]*(-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][0]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][0])+Q[2][2]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][0]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][0]); J[6][6]=Q[0][2]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][0]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][0])+Q[1][2]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][0]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][0])+Q[2][2]*(1-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][0]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][0]); J[6][7]=Q[0][2]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][0]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][0])+Q[1][2]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][0]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][0])+Q[2][2]*(-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][0]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][0]); J[6][8]=Q[0][2]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][0]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][0])+Q[1][2]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][0]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][0])+Q[2][2]*(-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][0]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][0]);
J(7)=J(5); //J[7][0]=Q[0][2]*(-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][1]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][1])+Q[1][2]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][1]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][1])+Q[2][2]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][1]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][1]); J[7][1]=Q[0][2]*(1-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][1]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][1])+Q[1][2]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][1]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][1])+Q[2][2]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][1]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][1]); J[7][2]=Q[0][2]*(-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][1]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][1])+Q[1][2]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][1]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][1])+Q[2][2]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][1]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][1]); J[7][3]=Q[0][2]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][1]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][1])+Q[1][2]*(-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][1]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][1])+Q[2][2]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][1]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][1]); J[7][4]=Q[0][2]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][1]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][1])+Q[1][2]*(1-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][1]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][1])+Q[2][2]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][1]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][1]); J[7][5]=Q[0][2]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][1]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][1])+Q[1][2]*(-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][1]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][1])+Q[2][2]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][1]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][1]); J[7][6]=Q[0][2]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][1]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][1])+Q[1][2]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][1]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][1])+Q[2][2]*(-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][1]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][1]); J[7][7]=Q[0][2]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][1]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][1])+Q[1][2]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][1]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][1])+Q[2][2]*(1-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][1]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][1]); J[7][8]=Q[0][2]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][1]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][1])+Q[1][2]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][1]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][1])+Q[2][2]*(-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][1]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][1]);
J[8][0]=Q[0][2]*(-(invG[2][1]*Q[0][2]-invG[2][2]*Q[0][1])*M[1][2]-(-invG[1][1]*Q[0][2]+invG[1][2]*Q[0][1])*M[2][2])+Q[1][2]*(-(-invG[2][1]*Q[0][2]+invG[2][2]*Q[0][1])*M[0][2]-(invG[0][1]*Q[0][2]-invG[0][2]*Q[0][1])*M[2][2])+Q[2][2]*(-(invG[1][1]*Q[0][2]-invG[1][2]*Q[0][1])*M[0][2]-(-invG[0][1]*Q[0][2]+invG[0][2]*Q[0][1])*M[1][2]); J[8][1]=Q[0][2]*(-(-invG[2][0]*Q[0][2]+invG[2][2]*Q[0][0])*M[1][2]-(invG[1][0]*Q[0][2]-invG[1][2]*Q[0][0])*M[2][2])+Q[1][2]*(-(invG[2][0]*Q[0][2]-invG[2][2]*Q[0][0])*M[0][2]-(-invG[0][0]*Q[0][2]+invG[0][2]*Q[0][0])*M[2][2])+Q[2][2]*(-(-invG[1][0]*Q[0][2]+invG[1][2]*Q[0][0])*M[0][2]-(invG[0][0]*Q[0][2]-invG[0][2]*Q[0][0])*M[1][2]); J[8][2]=Q[0][2]*(1-(invG[2][0]*Q[0][1]-invG[2][1]*Q[0][0])*M[1][2]-(-invG[1][0]*Q[0][1]+invG[1][1]*Q[0][0])*M[2][2])+Q[1][2]*(-(-invG[2][0]*Q[0][1]+invG[2][1]*Q[0][0])*M[0][2]-(invG[0][0]*Q[0][1]-invG[0][1]*Q[0][0])*M[2][2])+Q[2][2]*(-(invG[1][0]*Q[0][1]-invG[1][1]*Q[0][0])*M[0][2]-(-invG[0][0]*Q[0][1]+invG[0][1]*Q[0][0])*M[1][2]); J[8][3]=Q[0][2]*(-(invG[2][1]*Q[1][2]-invG[2][2]*Q[1][1])*M[1][2]-(-invG[1][1]*Q[1][2]+invG[1][2]*Q[1][1])*M[2][2])+Q[1][2]*(-(-invG[2][1]*Q[1][2]+invG[2][2]*Q[1][1])*M[0][2]-(invG[0][1]*Q[1][2]-invG[0][2]*Q[1][1])*M[2][2])+Q[2][2]*(-(invG[1][1]*Q[1][2]-invG[1][2]*Q[1][1])*M[0][2]-(-invG[0][1]*Q[1][2]+invG[0][2]*Q[1][1])*M[1][2]); J[8][4]=Q[0][2]*(-(-invG[2][0]*Q[1][2]+invG[2][2]*Q[1][0])*M[1][2]-(invG[1][0]*Q[1][2]-invG[1][2]*Q[1][0])*M[2][2])+Q[1][2]*(-(invG[2][0]*Q[1][2]-invG[2][2]*Q[1][0])*M[0][2]-(-invG[0][0]*Q[1][2]+invG[0][2]*Q[1][0])*M[2][2])+Q[2][2]*(-(-invG[1][0]*Q[1][2]+invG[1][2]*Q[1][0])*M[0][2]-(invG[0][0]*Q[1][2]-invG[0][2]*Q[1][0])*M[1][2]); J[8][5]=Q[0][2]*(-(invG[2][0]*Q[1][1]-invG[2][1]*Q[1][0])*M[1][2]-(-invG[1][0]*Q[1][1]+invG[1][1]*Q[1][0])*M[2][2])+Q[1][2]*(1-(-invG[2][0]*Q[1][1]+invG[2][1]*Q[1][0])*M[0][2]-(invG[0][0]*Q[1][1]-invG[0][1]*Q[1][0])*M[2][2])+Q[2][2]*(-(invG[1][0]*Q[1][1]-invG[1][1]*Q[1][0])*M[0][2]-(-invG[0][0]*Q[1][1]+invG[0][1]*Q[1][0])*M[1][2]); J[8][6]=Q[0][2]*(-(invG[2][1]*Q[2][2]-invG[2][2]*Q[2][1])*M[1][2]-(-invG[1][1]*Q[2][2]+invG[1][2]*Q[2][1])*M[2][2])+Q[1][2]*(-(-invG[2][1]*Q[2][2]+invG[2][2]*Q[2][1])*M[0][2]-(invG[0][1]*Q[2][2]-invG[0][2]*Q[2][1])*M[2][2])+Q[2][2]*(-(invG[1][1]*Q[2][2]-invG[1][2]*Q[2][1])*M[0][2]-(-invG[0][1]*Q[2][2]+invG[0][2]*Q[2][1])*M[1][2]); J[8][7]=Q[0][2]*(-(-invG[2][0]*Q[2][2]+invG[2][2]*Q[2][0])*M[1][2]-(invG[1][0]*Q[2][2]-invG[1][2]*Q[2][0])*M[2][2])+Q[1][2]*(-(invG[2][0]*Q[2][2]-invG[2][2]*Q[2][0])*M[0][2]-(-invG[0][0]*Q[2][2]+invG[0][2]*Q[2][0])*M[2][2])+Q[2][2]*(-(-invG[1][0]*Q[2][2]+invG[1][2]*Q[2][0])*M[0][2]-(invG[0][0]*Q[2][2]-invG[0][2]*Q[2][0])*M[1][2]); J[8][8]=Q[0][2]*(-(invG[2][0]*Q[2][1]-invG[2][1]*Q[2][0])*M[1][2]-(-invG[1][0]*Q[2][1]+invG[1][1]*Q[2][0])*M[2][2])+Q[1][2]*(-(-invG[2][0]*Q[2][1]+invG[2][1]*Q[2][0])*M[0][2]-(invG[0][0]*Q[2][1]-invG[0][1]*Q[2][0])*M[2][2])+Q[2][2]*(1-(invG[1][0]*Q[2][1]-invG[1][1]*Q[2][0])*M[0][2]-(-invG[0][0]*Q[2][1]+invG[0][1]*Q[2][0])*M[1][2]);
}


// another method based on the relation :  M^TM = S.S -> M^TdM +dM^TM = dS.S + S.dS  -> J1.dM = J2.dS  -> J.dM = dS
// requires the inversion of a 6x6 matrix
template<class Real>
void Decompose<Real>::polarDecompositionGradient_dSOverdM(const defaulttype::Mat<3,3,Real> &M, const defaulttype::Mat<3,3,Real> &S,  defaulttype::Mat<9,9,Real>& J)
{
    defaulttype::Mat<6,9,Real> J1;
    J1(0,0)=2*M(0,0);   J1(0,3)=2*M(1,0);   J1(0,6)=2*M(2,0);
    J1(1,0)=M(0,1);     J1(1,1)=M(0,0);     J1(1,3)=M(1,1);     J1(1,4)=M(1,0); J1(1,6)=M(2,1); J1(1,7)=M(2,0);
    J1(2,0)=M(0,2);     J1(2,2)=M(0,0);     J1(2,3)=M(1,2);     J1(2,5)=M(1,0); J1(2,6)=M(2,2); J1(2,8)=M(2,0);;
    J1(3,1)=2*M(0,1);   J1(3,4)=2*M(1,1);   J1(3,7)=2*M(2,1);
    J1(4,1)=M(0,2);     J1(4,2)=M(0,1);     J1(4,4)=M(1,2);     J1(4,5)=M(1,1); J1(4,7)=M(2,2); J1(4,8)=M(2,1);
    J1(5,2)=2*M(0,2);   J1(5,5)=2*M(1,2);   J1(5,8)=2*M(2,2);

    defaulttype::Mat<6,6,Real> J2;
    J2(0,0)=2*S(0,0); J2(0,1)=2*S(1,0); J2(0,2)=2*S(2,0);
    J2(1,0)=S(1,0); J2(1,1)=S(0,0)+S(1,1); J2(1,2)=S(2,1); J2(1,3)=S(1,0); J2(1,4)=S(2,0);
    J2(2,0)=S(2,0); J2(2,1)=S(2,1); J2(2,2)=S(2,2)+S(0,0); J2(2,4)=S(1,0); J2(2,5)=S(2,0);
    J2(3,1)=2*S(1,0); J2(3,3)=2*S(1,1); J2(3,4)=2*S(2,1);
    J2(4,1)=S(2,0); J2(4,2)=S(1,0); J2(4,3)=S(2,1); J2(4,4)=S(2,2)+S(1,1); J2(4,5)=S(2,1);
    J2(5,2)=2*S(2,0); J2(5,4)=2*S(2,1); J2(5,5)=2*S(2,2);
    defaulttype::Mat<6,6,Real> J2inv; J2inv.invert(J2);

    defaulttype::Mat<6,9,Real> J69 = J2inv*J1;

    J(0)=J69(0);
    J(1)=J(3)=J69(1);
    J(2)=J(6)=J69(2);
    J(4)=J69(3);
    J(5)=J(7)=J69(4);
    J(8)=J69(5);
}


template<class Real>
bool Decompose<Real>::polarDecomposition_stable_Gradient_dQ( const defaulttype::Mat<3,3,Real>& U, const defaulttype::Vec<3,Real>& Sdiag, const defaulttype::Mat<3,3,Real>& V, const defaulttype::Mat<3,3,Real>& dM, defaulttype::Mat<3,3,Real>& dQ )
{
    defaulttype::Mat<3,3,Real> UtdMV = U.multTranspose( dM ).multTransposed( V );
    defaulttype::Mat<3,3,Real> omega;

    for( int i=0 ; i<3 ; ++i )
    {
        int j = (i+1)%3;

        Real A = Sdiag[i] + Sdiag[j];
        if( /*helper::rabs*/( A ) < zeroTolerance() ) // only the smallest eigen-value should be negative so abs should not be necessary
        {
            //omega[i][j] = (Real)0;
            return false;
        }
        else
        {
            omega[i][j] = ( UtdMV[i][j] - UtdMV[j][i] ) / A;
        }
        omega[j][i] = -omega[i][j];
    }

    dQ = U * omega * V;

    return true;
}

template<class Real>
bool Decompose<Real>::polarDecompositionGradient_dQ( const defaulttype::Mat<3,2,Real>& U, const defaulttype::Vec<2,Real>& Sdiag, const defaulttype::Mat<2,2,Real>& V, const defaulttype::Mat<3,2,Real>& dM, defaulttype::Mat<3,2,Real>& dQ )
{
    defaulttype::Mat<2,2,Real> UtdMV = U.multTranspose( dM ).multTransposed( V );
    defaulttype::Mat<2,2,Real> omega;

    Real A = Sdiag[0] + Sdiag[1];

    if( /*helper::rabs*/( A ) < zeroTolerance() ) return false;

    omega[0][1] = ( UtdMV[0][1] - UtdMV[1][0] ) / A;
    omega[1][0] = -omega[0][1];

    dQ = U * omega * V;

    return true;
}



template<class Real>
bool Decompose<Real>::polarDecomposition_stable_Gradient_dQOverdM( const defaulttype::Mat<3,3,Real> &U, const defaulttype::Vec<3,Real> &Sdiag, const defaulttype::Mat<3,3,Real> &V, defaulttype::Mat<9,9,Real>& dQOverdM )
{

    Mat< 3,3, Mat<3,3,Real> > omega;

    for( int i=0 ; i<3 ; ++i ) // line of dM
        for( int j=0 ; j<3 ; ++j ) // col of dM
        {
            for( int k=0 ; k<3 ; ++k ) // resolve 3 2x2 systems to find omegaU[i][j] & omegaV[i][j]
            {
                int l=(k+1)%3;

                Real A = Sdiag[k] + Sdiag[l];

                if( /*helper::rabs*/( A ) < zeroTolerance() ) // only the smallest eigen-value should be negative so abs should not be necessary
                {
                    return false;
                }
                else
                {
                    omega[i][j][k][l] = ( U[i][k]*V[l][j] - U[i][l]*V[k][j] ) / A;
                }

                omega[i][j][l][k] = -omega[i][j][k][l]; // skew-symmetric (antisymmetric)
            }
            omega[i][j] = U * omega[i][j] * V;
        }


    // transposed and reformated in 9x9 matrice
    for( int i=0 ; i<3 ; ++i )
        for( int j=0 ; j<3 ; ++j )
            for( int k=0 ; k<3 ; ++k )
                for( int l=0 ; l<3 ; ++l )
                {
                    dQOverdM[i*3+j][k*3+l] = omega[k][l][i][j];
                }

    return true;
}


template<class Real>
bool Decompose<Real>::polarDecompositionGradient_dQOverdM( const defaulttype::Mat<3,2,Real>& U, const defaulttype::Vec<2,Real>& Sdiag, const defaulttype::Mat<2,2,Real>& V, defaulttype::Mat<6,6,Real>& dQOverdM )
{
    Mat< 3,2, Mat<3,2,Real> > dQdMij;

    for( int i=0 ; i<3 ; ++i ) // line of dM
        for( int j=0 ; j<2 ; ++j ) // col of dM
        {
            Real A = Sdiag[0] + Sdiag[1];

            if( /*helper::rabs*/( A ) < zeroTolerance() ) return false;

            Mat<2,2,Real> omega;

            omega[0][1] = ( U[i][0]*V[1][j] - U[i][1]*V[0][j] ) / A;
            omega[1][0] = -omega[0][1]; // skew-symmetric (antisymmetric)

            dQdMij[i][j] = U * omega * V;
        }

    // transposed and reformated in plain matrice
    for( int k=0 ; k<3 ; ++k )
        for( int l=0 ; l<2 ; ++l )
            for( int j=0 ; j<2 ; ++j )
                for( int i=0 ; i<3 ; ++i )
                    dQOverdM[i*2+j][k*2+l] = dQdMij[k][l][i][j];

    return true;
}


///////////////////////////////


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
    Real invLength;

    if (helper::rabs(vec2[0]) >= helper::rabs(vec2[1]))
    {
        // vec2.x or vec2.z is the largest magnitude component, swap them
        invLength = (Real)(1.0/helper::rsqrt(vec2[0]*vec2[0] + vec2[2]*vec2[2]));
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
        invLength = (Real)(1.0/helper::rsqrt(vec2[1]*vec2[1] + vec2[2]*vec2[2]));
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
    Real maxValue = helper::rabs( AScaled[0][0] );
    Real absValue = helper::rabs( AScaled[0][1] );
    if( absValue > maxValue )
    {
        maxValue = absValue;
    }
    absValue = helper::rabs( AScaled[0][2] );
    if( absValue > maxValue )
    {
        maxValue = absValue;
    }
    absValue = helper::rabs( AScaled[1][1] );
    if( absValue > maxValue )
    {
        maxValue = absValue;
    }
    absValue = helper::rabs( AScaled[1][2] );
    if( absValue > maxValue )
    {
        maxValue = absValue;
    }
    absValue = helper::rabs(AScaled[2][2]);
    if( absValue > maxValue )
    {
        maxValue = absValue;
    }

    int i, j;
    if( maxValue > (Real)1 )
    {
        Real invMaxValue = ((Real)1)/maxValue;
        for( i = 0; i < 3; ++i )
            for( j = 0; j < 3; ++j )
            {
                AScaled[i][j] *= invMaxValue;
            }
    }

    // Compute the eigenvalues using double-precision arithmetic.
    double root[3];
    ComputeRoots (AScaled, root );
    diag[0] = (Real)root[0];
    diag[1] = (Real)root[1];
    diag[2] = (Real)root[2];

    Real maxEntry[3];
    Vec<3,Real> maxRow[3];
    for( i = 0; i < 3; ++i )
    {
        Mat<3,3,Real> M = AScaled;
        M[0][0] -= diag[i];
        M[1][1] -= diag[i];
        M[2][2] -= diag[i];
        if( !PositiveRank( M, maxEntry[i], maxRow[i] ) )
        {
            // Rescale back to the original size.
            if( maxValue > (Real)1 )
            {
                for( j = 0; j < 3; ++j )
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
    if( maxEntry[1] > totalMax )
    {
        totalMax = maxEntry[1];
        i = 1;
    }
    if( maxEntry[2] > totalMax )
    {
        i = 2;
    }

    if( i == 0 )
    {
        maxRow[0].normalize();
        ComputeVectors(AScaled, maxRow[0], 1, 2, 0, V, diag);
    }
    else if( i == 1 )
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
    if( maxValue > (Real)1 )
    {
        for( i = 0; i < 3; ++i )
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
//    typedef defaulttype::Vec<2,Real> Vec2;
//    typedef defaulttype::Mat<2,2,Real> Mat22;

    Vec<2,Real> subDiag;

    // matrix is already tridiagonal
    diag[0] = M[0][0];
    diag[1] = M[1][1];
    subDiag[0] = M[0][1];
    subDiag[1] = 0.0;
    V.identity();

    QLAlgorithm( diag, subDiag, V );
}



//////////////////////////////////



template<class Real>
void Decompose<Real>::SVD( const defaulttype::Mat<3,3,Real> &F, defaulttype::Mat<3,3,Real> &U, defaulttype::Vec<3,Real> &S, defaulttype::Mat<3,3,Real> &V )
{
    defaulttype::Mat<3,3,Real> FtF = F.multTranspose( F );

    helper::Decompose<Real>::eigenDecomposition_iterative( FtF, V, S ); // eigen problem to obtain an orthogonal matrix V and diagonal S
    // at that point S = S^2

    defaulttype::Vec<3,Real> S_1;
    for( int i = 0 ; i<3; ++i )
    {
        if( S[i] < zeroTolerance() ) // numerical issues
        {
            S[i] = (Real)0;
            S_1[i] = (Real)1;
        }
        else
        {
            S[i] = helper::rsqrt( S[i] );
            S_1[i] = (Real)1. / S[i];
        }
    }

    U = F * V.multDiagonal( S_1 );
}


template<class Real>
bool Decompose<Real>::SVD_stable( const defaulttype::Mat<3,3,Real> &F, defaulttype::Mat<3,3,Real> &U, defaulttype::Vec<3,Real> &S, defaulttype::Mat<3,3,Real> &V )
{
    defaulttype::Mat<3,3,Real> FtF = F.multTranspose( F );

    helper::Decompose<Real>::eigenDecomposition_iterative( FtF, V, S ); // eigen problem to obtain an orthogonal matrix V and diagonal S
    // at that point S = S^2

    // if V is a reflexion -> made it a rotation by negating a column
    if( determinant(V) < (Real)0 )
        for( int i=0 ; i<3; ++i )
            V[i][0] = -V[i][0];

    // the numbers of strain values too close to 0 indicates the kind of degenerescence
    int degenerated = 0;

    defaulttype::Vec<3,Real> S_1;

    for( int i = 0 ; i<3; ++i )
    {
        if( S[i] < zeroTolerance() ) // numerical issues
        {
            degenerated++;
            S[i] = (Real)0;
            S_1[i] = (Real)1;
        }
        else
        {
            S[i] = helper::rsqrt( S[i] );
            S_1[i] = (Real)1. / S[i];
        }
    }


    // sort eigenvalues from small to large
    Vec<3,unsigned> Sorder;
    if( S[0]<S[1] )
    {
        if( S[0]<S[2] )
        {
            Sorder[0] = 0;
            if( S[1]<S[2] )
            {
                Sorder[1] = 1;
                Sorder[2] = 2;
            }
            else
            {
                Sorder[1] = 2;
                Sorder[2] = 1;
            }
        }
        else
        {
            Sorder[0] = 2;
            Sorder[1] = 0;
            Sorder[2] = 1;
        }
    }
    else
    {
        if( S[1]<S[2] )
        {
            Sorder[0] = 1;
            if( S[0]<S[2] )
            {
                Sorder[1] = 0;
                Sorder[2] = 2;
            }
            else
            {
                Sorder[1] = 2;
                Sorder[2] = 0;
            }
        }
        else
        {
            Sorder[0] = 2;
            Sorder[1] = 1;
            Sorder[2] = 0;
        }
    }


    switch( degenerated )
    {
    case 0: // no null value -> eventually inverted but not degenerate
        U = F * V.multDiagonal( S_1 );
        break;
    case 1: // 1 null value -> collapsed to a plane -> keeps the 2 valid edges and construct the third
    {
        U = F * V.multDiagonal( S_1 );

        Vec<3,Real> c = cross( Vec<3,Real>(U[0][Sorder[1]],U[1][Sorder[1]],U[2][Sorder[1]]), Vec<3,Real>(U[0][Sorder[2]],U[1][Sorder[2]],U[2][Sorder[2]]) );
        U[0][Sorder[0]] = c[0];
        U[1][Sorder[0]] = c[1];
        U[2][Sorder[0]] = c[2];
        break;
    }
    case 2: // 2 null values -> collapsed to an edge -> keeps the valid edge and build 2 orthogonal vectors
    {
        U = F * V.multDiagonal( S_1 );

        // TODO: check if there is a more efficient way to do this

        Vec<3,Real> edge0, edge1, edge2( U[0][Sorder[2]], U[1][Sorder[2]], U[2][Sorder[2]] );

        // check the main direction of edge2 to try to take a not too close arbritary vector
        Real abs0 = helper::rabs( edge2[0] );
        Real abs1 = helper::rabs( edge2[1] );
        Real abs2 = helper::rabs( edge2[2] );
        if( abs0 > abs1 )
        {
            if( abs0 > abs2 )
            {
                edge0[0] = 0; edge0[1] = 1; edge0[2] = 0;
            }
            else
            {
                edge0[0] = 1; edge0[1] = 0; edge0[2] = 0;
            }
        }
        else
        {
            if( abs1 > abs2 )
            {
                edge0[0] = 0; edge0[1] = 0; edge0[2] = 1;
            }
            else
            {
                edge0[0] = 1; edge0[1] = 0; edge0[2] = 0;
            }
        }

        edge1 = cross( edge2, edge0 );
        edge1.normalize();
        edge0 = cross( edge1, edge2 );

        U[0][Sorder[0]] = edge0[0];
        U[1][Sorder[0]] = edge0[1];
        U[2][Sorder[0]] = edge0[2];

        U[0][Sorder[1]] = edge1[0];
        U[1][Sorder[1]] = edge1[1];
        U[2][Sorder[1]] = edge1[2];

        break;
    }
    case 3: // 3 null values -> collapsed to a point -> build any orthogonal frame
        U.identity();
        break;
    }

    bool inverted = ( determinant(U) < (Real)0 );

    // un-inverting the element -> made U a rotation by negating a column
    if( inverted )
    {
        U[0][Sorder[0]] *= (Real)-1;
        U[1][Sorder[0]] *= (Real)-1;
        U[2][Sorder[0]] *= (Real)-1;

        S[Sorder[0]] *= (Real)-1;
    }

    return degenerated || inverted;
}


template<class Real>
bool Decompose<Real>::SVD_stable( const defaulttype::Mat<2,2,Real> &F, defaulttype::Mat<2,2,Real> &U, defaulttype::Vec<2,Real> &S, defaulttype::Mat<2,2,Real> &V )
{
    defaulttype::Mat<2,2,Real> FtF = F.multTranspose( F ); // transformation from actual pos to rest pos

    helper::Decompose<Real>::eigenDecomposition_iterative( FtF, V, S );

    // if V is a reflexion -> made it a rotation by negating a column
    if( determinant(V) < (Real)0 )
        for( int i=0 ; i<2; ++i )
            V[i][0] = -V[i][0];

    // the numbers of strain values too close to 0 indicates the kind of degenerescence
    int degenerated = 0;

    // compute the diagonalized strain and take the inverse
    defaulttype::Vec<2,Real> S_1;
    for( int i = 0 ; i<2; ++i )
    {
        if( S[i] < zeroTolerance() ) // numerical issues
        {
            degenerated++;
            S[i] = (Real)0;
            S_1[i] = (Real)1;
        }
        else
        {
            S[i] = helper::rsqrt( S[i] );
            S_1[i] = (Real)1.0 / S[i];
        }
    }

    switch( degenerated )
    {
    case 0: // no null value -> eventually inverted but not degenerate
        U = F * V.multDiagonal( S_1 );
        break;
    case 1: // 1 null value -> collapsed to an edge -> keeps the valid edge and build 2 orthogonal vectors
    {
        U = F * V.multDiagonal( S_1 );
        int min, max; if( S[0] > S[1] ) { min=1; max=0; }
        else { min=0; max=1; }   // eigen values order

        Vec<3,Real> edge0, edge1( U[0][max], U[1][max], U[2][max] ), edge2;

        // check the main direction of edge2 to try to take a not too close arbritary vector
        Real abs0 = helper::rabs( edge1[0] );
        Real abs1 = helper::rabs( edge1[1] );
        Real abs2 = helper::rabs( edge1[2] );
        if( abs0 > abs1 )
        {
            if( abs0 > abs2 )
            {
                edge0[0] = 0; edge0[1] = 1; edge0[2] = 0;
            }
            else
            {
                edge0[0] = 1; edge0[1] = 0; edge0[2] = 0;
            }
        }
        else
        {
            if( abs1 > abs2 )
            {
                edge0[0] = 0; edge0[1] = 0; edge0[2] = 1;
            }
            else
            {
                edge0[0] = 1; edge0[1] = 0; edge0[2] = 0;
            }
        }

        edge2 = cross( edge0, edge1 );
        edge2.normalize();
        edge0 = cross( edge1, edge2 );

        U[0][min] = edge0[0];
        U[1][min] = edge0[1];
        U[2][min] = edge0[2];

        break;
    }
    case 2: // 2 null values -> collapsed to a point -> build any orthogonal frame
    {
        int min, max; if( S[0] > S[1] ) { min=1; max=0; }
        else { min=0; max=1; }   // eigen values order
        U[0][min] = 1;
        U[1][min] = 0;
        U[0][max] = 0;
        U[1][max] = 1;
        break;
    }
    }

    bool inverted = ( determinant(U) < (Real)0 );

    // un-inverting the element -> made U a rotation by negating a column
    if( inverted )
    {
        int min = S[0] > S[1] ? 1 : 0;   // min eigen value index

        U[0][min] *= (Real)-1;
        U[1][min] *= (Real)-1;

        S[min] *= (Real)-1;
    }

    return degenerated || inverted;
}


template<class Real>
void Decompose<Real>::SVD( const defaulttype::Mat<3,2,Real> &F, defaulttype::Mat<3,2,Real> &U, defaulttype::Vec<2,Real> &S, defaulttype::Mat<2,2,Real> &V )
{
    defaulttype::Mat<2,2,Real> FtF = F.multTranspose( F ); // transformation from actual pos to rest pos

    helper::Decompose<Real>::eigenDecomposition_iterative( FtF, V, S );

    // compute the diagonalized strain and take the inverse
    defaulttype::Vec<2,Real> S_1;
    for( int i = 0 ; i<2; ++i )
    {
        if( S[i] < zeroTolerance() ) // numerical issues
        {
            S[i] = (Real)0;
            S_1[i] = (Real)1;
        }
        else
        {
            S[i] = helper::rsqrt( S[i] );
            S_1[i] = (Real)1.0 / S[i];
        }
    }

    U = F * V.multDiagonal( S_1 );
}


template<class Real>
bool Decompose<Real>::SVD_stable( const defaulttype::Mat<3,2,Real> &F, defaulttype::Mat<3,2,Real> &U, defaulttype::Vec<2,Real> &S, defaulttype::Mat<2,2,Real> &V )
{
    defaulttype::Mat<2,2,Real> FtF = F.multTranspose( F ); // transformation from actual pos to rest pos

    helper::Decompose<Real>::eigenDecomposition_iterative( FtF, V, S );

    // if V is a reflexion -> made it a rotation by negating a column
    if( determinant(V) < (Real)0 )
        for( int i=0 ; i<2; ++i )
            V[i][0] = -V[i][0];

    // the numbers of strain values too close to 0 indicates the kind of degenerescence
    int degenerated = 0;

    // compute the diagonalized strain and take the inverse
    defaulttype::Vec<2,Real> S_1;
    for( int i = 0 ; i<2; ++i )
    {
        if( S[i] < zeroTolerance() ) // numerical issues
        {
            degenerated++;
            S[i] = (Real)0;
            S_1[i] = (Real)1;
        }
        else
        {
            S[i] = helper::rsqrt( S[i] );
            S_1[i] = (Real)1.0 / S[i];
        }
    }

    // TODO check for degenerate cases (collapsed to a point, to an edge)
    // note that inversion is not defined for a 2d element in a 3d world
    switch( degenerated )
    {
    case 0: // no null value -> eventually inverted but not degenerate
        U = F * V.multDiagonal( S_1 );
        break;
    case 1: // 1 null value -> collapsed to an edge -> keeps the valid edge and build 2 orthogonal vectors
    {
        U = F * V.multDiagonal( S_1 );
        int min, max; if( S[0] > S[1] ) { min=1; max=0; }
        else { min=0; max=1; }   // eigen values order

        Vec<3,Real> edge0, edge1( U[0][max], U[1][max], U[2][max] ), edge2;

        // check the main direction of edge2 to try to take a not too close arbritary vector
        Real abs0 = helper::rabs( edge1[0] );
        Real abs1 = helper::rabs( edge1[1] );
        Real abs2 = helper::rabs( edge1[2] );
        if( abs0 > abs1 )
        {
            if( abs0 > abs2 )
            {
                edge0[0] = 0; edge0[1] = 1; edge0[2] = 0;
            }
            else
            {
                edge0[0] = 1; edge0[1] = 0; edge0[2] = 0;
            }
        }
        else
        {
            if( abs1 > abs2 )
            {
                edge0[0] = 0; edge0[1] = 0; edge0[2] = 1;
            }
            else
            {
                edge0[0] = 1; edge0[1] = 0; edge0[2] = 0;
            }
        }

        edge2 = cross( edge0, edge1 );
        edge2.normalize();
        edge0 = cross( edge1, edge2 );

        U[0][min] = edge0[0];
        U[1][min] = edge0[1];
        U[2][min] = edge0[2];

        break;
    }
    case 2: // 2 null values -> collapsed to a point -> build any orthogonal frame
    {
        int min, max; if( S[0] > S[1] ) { min=1; max=0; }
        else { min=0; max=1; }   // eigen values order
        U[0][min] = 1;
        U[1][min] = 0;
        U[2][min] = 0;
        U[0][max] = 0;
        U[1][max] = 1;
        U[2][max] = 0;
        break;
    }
    }

    return (degenerated!=0);
}


#define TIKHONOV_REGULARIZATION

template<class Real>
bool Decompose<Real>::SVDGradient_dUdVOverdM( const defaulttype::Mat<3,3,Real> &U, const defaulttype::Vec<3,Real> &S, const defaulttype::Mat<3,3,Real> &V, defaulttype::Mat<9,9,Real>& dUOverdM, defaulttype::Mat<9,9,Real>& dVOverdM )
{
    Mat< 3,3, Mat<3,3,Real> > omegaU, omegaV;

    for( int i=0 ; i<3 ; ++i ) // line of dM
        for( int j=0 ; j<3 ; ++j ) // col of dM
        {
            for( int k=0 ; k<3 ; ++k ) // resolve 3 2x2 systems to find omegaU[i][j] & omegaV[i][j]
            {
                int l=(k+1)%3;
                defaulttype::Mat<2,2,Real> A, invA;
                A[0][0] = A[1][1] = S[l];
                A[0][1] = A[1][0] = S[k];
                defaulttype::Vec<2,Real> v( U[i][k]*V[l][j], -U[i][l]*V[k][j] ), w;

                if( helper::rabs( S[k]-S[l] ) > zeroTolerance() )
                {
                    invA.invert( A );
                    w = invA * v;
                }
                else
                {
#ifndef TIKHONOV_REGULARIZATION
                    return false;
#else
                    // Tikhonov regularization w = (AtA + I)^-1 At v (suggested in "Invertible Isotropic Hyperelasticity using SVD Gradients", F Sin, Y Zhu, Y Li, D Schroeder, J Barbič, Poster SCA 2011)
                    defaulttype::Mat<2,2,Real> AtA = A.multTranspose( A );
                    AtA[0][0] += (Real)1;
                    AtA[1][1] += (Real)1;
                    invA.invert( AtA );
                    w = invA.multTransposed( A ) * v;
#endif
                }

                //dU[k*3+l][i*3+j] = w[0]; dU[l*3+k][i*3+j] = -w[0];
                //dV[k*3+l][i*3+j] = w[1]; dV[l*3+k][i*3+j] = -w[1];

                omegaU[i][j][k][l] = w[0]; omegaU[i][j][l][k] = -w[0];
                omegaV[i][j][k][l] = w[1]; omegaV[i][j][l][k] = -w[1];
            }
            omegaU[i][j] = U * omegaU[i][j];
            omegaV[i][j] = omegaV[i][j] * V;
        }


//    for( int i=0 ; i<3 ; ++i )
//    for( int j=0 ; j<3 ; ++j )
//        for( int k=0 ; k<3 ; ++k )
//        for( int l=0 ; l<3 ; ++l )
//    {
//        dU[i][j] += omegaU[i*3+j][k*3+l] * dM[k][l];
//        dV[i][j] += omegaV[i*3+j][k*3+l] * dM[k][l];
//    }

    // transposed and reformated in 9x9 matrices
    for( int i=0 ; i<3 ; ++i )
        for( int j=0 ; j<3 ; ++j )
            for( int k=0 ; k<3 ; ++k )
                for( int l=0 ; l<3 ; ++l )
                {
                    //dU[i][j] += omegaU[k][l][i][j] * dM[k][l];
                    //dV[i][j] += omegaV[k][l][i][j] * dM[k][l];

                    //dUOverdM[i][j][k][l] = omegaU[k][l][i][j];

                    dUOverdM[i*3+j][k*3+l] = omegaU[k][l][i][j];
                    dVOverdM[i*3+j][k*3+l] = omegaV[k][l][i][j];
                }

//            for( int i=0 ; i<3 ; ++i )
//            for( int j=0 ; j<3 ; ++j )
//                for( int k=0 ; k<3 ; ++k )
//                for( int l=0 ; l<3 ; ++l )
//            {
//                //dU[i][j] += dUOverdM[i][j][k][l] * dM[k][l];
//                    dU[i][j] += dUOverdM[i*3+j][k*3+l] * dM[k][l];
//            }

    return true;

}



template<class Real>
bool Decompose<Real>::SVDGradient_dUdVOverdM( const defaulttype::Mat<3,2,Real> &U, const defaulttype::Vec<2,Real> &S, const defaulttype::Mat<2,2,Real> &V, defaulttype::Mat<6,6,Real>& dUOverdM, defaulttype::Mat<4,6,Real>& dVOverdM )
{
    Mat< 3,2, Mat<3,2,Real> > dUdMij;
    Mat< 3,2, Mat<2,2,Real> > dVdMij;

    for( int i=0 ; i<3 ; ++i ) // line of dM
        for( int j=0 ; j<2 ; ++j ) // col of dM
        {
            Mat<2,2,Real> omegaU, omegaV;
            defaulttype::Mat<2,2,Real> A, invA;
            A[0][0] = A[1][1] = S[1];
            A[0][1] = A[1][0] = S[0];
            defaulttype::Vec<2,Real> v( U[i][0]*V[1][j], -U[i][1]*V[0][j] ), w;

            if( helper::rabs( S[0]-S[1] ) > zeroTolerance() )
            {
                invA.invert( A );
                w = invA * v;
            }
            else
            {
#ifndef TIKHONOV_REGULARIZATION
                    return false;
#else
                // Tikhonov regularization w = (AtA + I)^-1 At v (suggested in "Invertible Isotropic Hyperelasticity using SVD Gradients", F Sin, Y Zhu, Y Li, D Schroeder, J Barbič, Poster SCA 2011)
                defaulttype::Mat<2,2,Real> AtA = A.multTranspose( A );
                AtA[0][0] += (Real)1;
                AtA[1][1] += (Real)1;
                invA.invert( AtA );
                w = invA.multTransposed( A ) * v;
#endif
            }

            omegaU[0][1] = w[0]; omegaU[1][0] = -w[0];
            omegaV[0][1] = w[1]; omegaV[1][0] = -w[1];

            dUdMij[i][j] = U * omegaU;
            dVdMij[i][j] = omegaV * V;
        }

    // transposed and reformated in plain matrices
    for( int k=0 ; k<3 ; ++k )
        for( int l=0 ; l<2 ; ++l )
            for( int j=0 ; j<2 ; ++j )
            {
                for( int i=0 ; i<3 ; ++i )
                    dUOverdM[i*2+j][k*2+l] = dUdMij[k][l][i][j];

                for( int i=0 ; i<2 ; ++i )
                    dVOverdM[i*2+j][k*2+l] = dVdMij[k][l][i][j];
            }

    return true;
}


template<class Real>
bool Decompose<Real>::SVDGradient_dUdV( const defaulttype::Mat<3,3,Real> &U, const defaulttype::Vec<3,Real> &S, const defaulttype::Mat<3,3,Real> &V, const defaulttype::Mat<3,3,Real>& dM, defaulttype::Mat<3,3,Real>& dU, defaulttype::Mat<3,3,Real>& dV )
{
    defaulttype::Mat<3,3,Real> UtdMV = U.multTranspose( dM ).multTransposed( V );
    defaulttype::Mat<3,3,Real> omegaU, omegaV;

    for( int i=0 ; i<3 ; ++i )
    {
        int j=(i+1)%3;
        defaulttype::Mat<2,2,Real> A, invA;
        A[0][0] = A[1][1] = S[j];
        A[0][1] = A[1][0] = S[i];
        defaulttype::Vec<2,Real> v( UtdMV[i][j], -UtdMV[j][i] ), w;

        if( helper::rabs( S[i]-S[j] ) > zeroTolerance() )
        {
            invA.invert( A );
            w = invA * v;
        }
        else
        {
#ifndef TIKHONOV_REGULARIZATION
                    return false;
#else
            // Tikhonov regularization w = (AtA + I)^-1 At v (suggested in "Invertible Isotropic Hyperelasticity using SVD Gradients", F Sin, Y Zhu, Y Li, D Schroeder, J Barbič, Poster SCA 2011)
            defaulttype::Mat<2,2,Real> AtA = A.multTranspose( A );
            AtA[0][0] += (Real)1;
            AtA[1][1] += (Real)1;
            invA.invert( AtA );
            w = invA.multTransposed( A ) * v;
#endif
        }

        omegaU[i][j] = w[0]; omegaU[j][i] = -w[0];
        omegaV[i][j] = w[1]; omegaV[j][i] = -w[1];
    }

    dU = U * omegaU;
    dV = omegaV * V;

    return true;
}


template<class Real>
bool Decompose<Real>::SVDGradient_dUdV( const defaulttype::Mat<3,2,Real> &U, const defaulttype::Vec<2,Real> &S, const defaulttype::Mat<2,2,Real> &V, const defaulttype::Mat<3,2,Real>& dM, defaulttype::Mat<3,2,Real>& dU, defaulttype::Mat<2,2,Real>& dV )
{
    defaulttype::Mat<2,2,Real> UtdMV = U.multTranspose( dM ).multTransposed( V );
    defaulttype::Mat<2,2,Real> omegaU;
    defaulttype::Mat<2,2,Real> omegaV;

    defaulttype::Mat<2,2,Real> A, invA;
    A[0][0] = A[1][1] = S[1];
    A[0][1] = A[1][0] = S[0];
    defaulttype::Vec<2,Real> v( UtdMV[0][1], -UtdMV[1][0] ), w;

    if( helper::rabs( S[0]-S[1] ) > zeroTolerance() )
    {
        invA.invert( A );
        w = invA * v;
    }
    else
    {
#ifndef TIKHONOV_REGULARIZATION
                    return false;
#else
        // Tikhonov regularization w = (AtA + I)^-1 At v (suggested in "Invertible Isotropic Hyperelasticity using SVD Gradients", F Sin, Y Zhu, Y Li, D Schroeder, J Barbič, Poster SCA 2011)
        defaulttype::Mat<2,2,Real> AtA = A.multTranspose( A );
        AtA[0][0] += (Real)1;
        AtA[1][1] += (Real)1;
        invA.invert( AtA );
        w = invA.multTransposed( A ) * v;
#endif
    }

    omegaU[0][1] = w[0]; omegaU[1][0] = -w[0];
    omegaV[0][1] = w[1]; omegaV[1][0] = -w[1];

    dU = U * omegaU;
    dV = omegaV * V;

    return true;
}




/////////////////////////////////////////////////////////////////


// Numerical diagonalization of 3x3 matrcies
// Copyright (C) 2006  Joachim Kopp
// http://www.mpi-hd.mpg.de/personalhomes/globes/3x3/

static const double M_SQRT3 = 1.73205080756887729352744634151;   // sqrt(3)

template <class Real>
int dsyevc3( const defaulttype::Mat<3,3,Real> &A, defaulttype::Vec<3,Real> &w)
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
  Real m, c1, c0;

  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
  Real de = A[0][1] * A[1][2];                                    // d * e
  Real dd = helper::SQR(A[0][1]);                                         // d^2
  Real ee = helper::SQR(A[1][2]);                                         // e^2
  Real ff = helper::SQR(A[0][2]);                                         // f^2
  m  = A[0][0] + A[1][1] + A[2][2];
  c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
          - (dd + ee + ff);
  c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2]
            - static_cast<Real>(2.0) * A[0][2]*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

  Real p, sqrt_p, q, c, s, phi;
  p = helper::SQR(m) - static_cast<Real>(3.0) * c1;
  q = m*(p - static_cast<Real>(3.0/2.0)*c1) - static_cast<Real>(27.0/2.0)*c0;
  sqrt_p = sqrt(fabs(p));

  phi = static_cast<Real>(27.0 * ( 0.25*helper::SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0)));
  phi = static_cast<Real>(1.0/3.0) * atan2(sqrt(fabs(phi)), q);

  c = sqrt_p*cos(phi);
  s = static_cast<Real>(1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = static_cast<Real>(1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;

  return 0;
}





template <class Real>
inline void dsytrd3(const defaulttype::Mat<3,3,Real> &A, defaulttype::Mat<3,3,Real> &Q, defaulttype::Vec<3,Real> &d, defaulttype::Vec<3,Real> &e)
// ----------------------------------------------------------------------------
// Reduces a symmetric 3x3 matrix to tridiagonal form by applying
// (unitary) Householder transformations:
//            [ d[0]  e[0]       ]
//    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
//            [       e[1]  d[2] ]
// The function accesses only the diagonal and upper triangular parts of
// A. The access is read-only.
// ---------------------------------------------------------------------------
{
  const int n = 3;
  Real u[n], q[n];
  Real omega, f;
  Real K, h, g;

  // Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
  for (int i=0; i < n; i++)
  {
    Q[i][i] = 1.0;
    for (int j=0; j < i; j++)
      Q[i][j] = Q[j][i] = 0.0;
  }
#endif

  // Bring first row and column to the desired form
  h = helper::SQR(A[0][1]) + helper::SQR(A[0][2]);
  if (A[0][1] > 0)
    g = -sqrt(h);
  else
    g = sqrt(h);
  e[0] = g;
  f    = g * A[0][1];
  u[1] = A[0][1] - g;
  u[2] = A[0][2];

  omega = h - f;
  if (omega > 0.0)
  {
    omega = static_cast<Real>(1.0) / omega;
    K     = 0.0;
    for (int i=1; i < n; i++)
    {
      f    = A[1][i] * u[1] + A[i][2] * u[2];
      q[i] = omega * f;                  // p
      K   += u[i] * f;                   // u* A u
    }
    K *= static_cast<Real>(0.5) * helper::SQR(omega);

    for (int i=1; i < n; i++)
      q[i] = q[i] - K * u[i];

    d[0] = A[0][0];
    d[1] = A[1][1] - static_cast<Real>(2.0)*q[1]*u[1];
    d[2] = A[2][2] - static_cast<Real>(2.0)*q[2]*u[2];

    // Store inverse Householder transformation in Q
#ifndef EVALS_ONLY
    for (int j=1; j < n; j++)
    {
      f = omega * u[j];
      for (int i=1; i < n; i++)
        Q[i][j] = Q[i][j] - f*u[i];
    }
#endif

    // Calculate updated A[1][2] and store it in e[1]
    e[1] = A[1][2] - q[1]*u[2] - u[1]*q[2];
  }
  else
  {
    for (int i=0; i < n; i++)
      d[i] = A[i][i];
    e[1] = A[1][2];
  }
}


template <class Real>
int dsyevq3(const defaulttype::Mat<3,3,Real> &A, defaulttype::Mat<3,3,Real> &Q, defaulttype::Vec<3,Real> &w)
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using the QL algorithm with implicit shifts, preceded by a
// Householder reduction to tridiagonal form.
// The function accesses only the diagonal and upper triangular parts of A.
// The access is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error (no convergence)
{
  const int n = 3;
  defaulttype::Vec<3,Real> e;                   // The third element is used only as temporary workspace
  Real g, r, p, f, b, s, c, t; // Intermediate storage
  int m;

  // Transform A to real tridiagonal form by the Householder method
  dsytrd3(A, Q, w, e);

  // Calculate eigensystem of the remaining real symmetric tridiagonal matrix
  // with the QL method
  //
  // Loop over all off-diagonal elements
  for (int l=0; l < n-1; l++)
  {
    int nIter = 0;
    while (1)
    {
      // Check for convergence and exit iteration loop if off-diagonal
      // element e(l) is zero
      for (m=l; m <= n-2; m++)
      {
        g = fabs(w[m])+fabs(w[m+1]);
        if (fabs(e[m]) + g == g)
          break;
      }
      if (m == l)
        break;

      if (nIter++ >= 30)
        return -1;

      // Calculate g = d_m - k
      g = (w[l+1] - w[l]) / (e[l] + e[l]);
      r = sqrt(g*g + static_cast<Real>(1.0));
      if (g > 0)
        g = w[m] - w[l] + e[l]/(g + r);
      else
        g = w[m] - w[l] + e[l]/(g - r);

      s = c = 1.0;
      p = 0.0;
      for (int i=m-1; i >= l; i--)
      {
        f = s * e[i];
        b = c * e[i];
        if (fabs(f) > fabs(g))
        {
          c      = g / f;
          r      = sqrt(c*c + static_cast<Real>(1.0));
          e[i+1] = f * r;
          c     *= (s = static_cast<Real>(1.0)/r);
        }
        else
        {
          s      = f / g;
          r      = sqrt(s*s + static_cast<Real>(1.0));
          e[i+1] = g * r;
          s     *= (c = static_cast<Real>(1.0)/r);
        }

        g = w[i+1] - p;
        r = (w[i] - g)*s + static_cast<Real>(2.0)*c*b;
        p = s * r;
        w[i+1] = g + p;
        g = c*r - b;

        // Form eigenvectors
#ifndef EVALS_ONLY
        for (int k=0; k < n; k++)
        {
          t = Q[k][i+1];
          Q[k][i+1] = s*Q[k][i] + c*t;
          Q[k][i]   = c*Q[k][i] - s*t;
        }
#endif
      }
      w[l] -= p;
      e[l]  = g;
      e[m]  = 0.0;
    }
  }

  return 0;
}





template<class Real>
int Decompose<Real>::symmetricDiagonalization( const defaulttype::Mat<3,3,Real> &A, defaulttype::Mat<3,3,Real> &Q, defaulttype::Vec<3,Real> &w )
// ----------------------------------------------------------------------------
// originally named dsyevh3
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors. However,
// if conditions are such that a large error in the results is to be
// expected, the routine falls back to using the slower, but more
// accurate QL algorithm. Only the diagonal and upper triangular parts of A need
// to contain meaningful values. Access to A is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3(), dsytrd3(), dsyevq3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1: Simplified fallback condition --> speed-up
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
  Real norm;          // Squared norm or inverse norm of current eigenvector
//  Real n0, n1;        // Norm of first and second columns of A
  Real error;         // Estimated maximum roundoff error
  Real t, u;          // Intermediate storage
  int j;                // Loop counter
#endif

  // Calculate eigenvalues
  dsyevc3(A, w);

#ifndef EVALS_ONLY
//  n0 = SQR(A[0][0]) + SQR(A[0][1]) + SQR(A[0][2]);
//  n1 = SQR(A[0][1]) + SQR(A[1][1]) + SQR(A[1][2]);

  t = fabs(w[0]);
  if ((u=fabs(w[1])) > t)
    t = u;
  if ((u=fabs(w[2])) > t)
    t = u;
  if (t < 1.0)
    u = t;
  else
    u = helper::SQR(t);
  error = static_cast<Real>(256.0) * std::numeric_limits<Real>::epsilon() * helper::SQR(u);
//  error = 256.0 * std::numeric_limits<Real>::epsilon() * (n0 + u) * (n1 + u);

  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
  Q[2][1] = SQR(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = (A - w[0]).e1 x (A - w[0]).e2
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = (A[0][0] - w[0]) * (A[1][1] - w[0]) - Q[2][1];
  norm    = helper::SQR(Q[0][0]) + helper::SQR(Q[1][0]) + helper::SQR(Q[2][0]);

  // If vectors are nearly linearly dependent, or if there might have
  // been large cancellations in the calculation of A[i][i] - w[0], fall
  // back to QL algorithm
  // Note that this simultaneously ensures that multiple eigenvalues do
  // not cause problems: If w[0] = w[1], then A - w[0] * I has rank 1,
  // i.e. all columns of A - w[0] * I are linearly dependent.
  if (norm <= error)
    return dsyevq3(A, Q, w);
  else                      // This is the standard branch
  {
    norm = sqrt(static_cast<Real>(1.0) / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }

  // Calculate second eigenvector by the formula
  //   v[1] = (A - w[1]).e1 x (A - w[1]).e2
  Q[0][1]  = Q[0][1] + A[0][2]*w[1];
  Q[1][1]  = Q[1][1] + A[1][2]*w[1];
  Q[2][1]  = (A[0][0] - w[1]) * (A[1][1] - w[1]) - Q[2][1];
  norm     = helper::SQR(Q[0][1]) + helper::SQR(Q[1][1]) + helper::SQR(Q[2][1]);
  if (norm <= error)
    return dsyevq3(A, Q, w);
  else
  {
    norm = sqrt(static_cast<Real>(1.0) / norm);
    for (j=0; j < 3; j++)
      Q[j][1] = Q[j][1] * norm;
  }

  // Calculate third eigenvector according to
  //   v[2] = v[0] x v[1]
  Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
  Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
  Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];
#endif

  return 0;
}



template<class Real>
void Decompose<Real>::PSDProjection( defaulttype::Mat<3,3,Real> &A )
{
    defaulttype::Mat<3,3,Real> Q;
    defaulttype::Vec<3,Real> w;
    if( !symmetricDiagonalization( A, Q, w ) )
    {
        bool modified = false;
        for( int i=0 ; i<3 ; ++i )
            if( w[i] < 0 ){ w[i] = 0; modified = true;  }

        if( modified )
        {
//            defaulttype::Mat<3,3,Real> invQ; invQ.invert( Q );
//            A = Q.multDiagonal( w )*invQ;
            A = (Q.multDiagonal( w )).multTransposed( Q ); // A = Q*wId*Q^T  // only valid for symmetric matrices
        }
    }
}


template<class Real>
inline void dsyev2(Real A, Real B, Real C, Real &rt1, Real &rt2,
                   Real &cs, Real &sn)
// ----------------------------------------------------------------------------
// Calculates the eigensystem of a real symmetric 2x2 matrix
//    [ A  B ]
//    [ B  C ]
// in the form
//    [ A  B ]  =  [ cs  -sn ] [ rt1   0  ] [  cs  sn ]
//    [ B  C ]     [ sn   cs ] [  0   rt2 ] [ -sn  cs ]
// where rt1 >= rt2. Note that this convention is different from the one used
// in the LAPACK routine DLAEV2, where |rt1| >= |rt2|.
// ----------------------------------------------------------------------------
{
  Real sm = A + C;
  Real df = A - C;
  Real rt = sqrt(helper::SQR(df) + static_cast<Real>(4.0)*B*B);
  Real t;

  if (sm > 0.0)
  {
    rt1 = static_cast<Real>(0.5) * (sm + rt);
    t = static_cast<Real>(1.0)/rt1;
    rt2 = (A*t)*C - (B*t)*B;
  }
  else if (sm < 0.0)
  {
    rt2 = static_cast<Real>(0.5) * (sm - rt);
    t = static_cast<Real>(1.0)/rt2;
    rt1 = (A*t)*C - (B*t)*B;
  }
  else       // This case needs to be treated separately to avoid div by 0
  {
    rt1 = static_cast<Real>(0.5) * rt;
    rt2 = static_cast<Real>(-0.5) * rt;
  }

  // Calculate eigenvectors
  if (df > 0.0)
    cs = df + rt;
  else
    cs = df - rt;

  if (fabs(cs) > 2.0*fabs(B))
  {
    t   = static_cast<Real>(-2.0) * B / cs;
    sn = static_cast<Real>(1.0 / sqrt(1.0 + helper::SQR(t)));
    cs = t * sn;
  }
  else if (fabs(B) == 0.0)
  {
    cs = 1.0;
    sn = 0.0;
  }
  else
  {
    t   = static_cast<Real>(-0.5) * cs / B;
    cs = static_cast<Real>(1.0 / sqrt(1.0 + helper::SQR(t)));
    sn = t * cs;
  }

  if (df > 0.0)
  {
    t   = cs;
    cs = -sn;
    sn = t;
  }
}




template<class Real>
void Decompose<Real>::PSDProjection( defaulttype::Mat<2,2,Real> &A )
{
    defaulttype::Mat<2,2,Real> Q;
    defaulttype::Vec<2,Real> w;
    dsyev2( (Real)A[0][0], (Real)A[0][1], (Real)A[1][1], w[0], w[1], Q[0][0], Q[1][0] );

    bool modified = false;
    for( int i=0 ; i<2 ; ++i )
        if( w[i] < 0 ){ w[i] = 0; modified = true; }

    if( modified )
    {
        Q[1][1] = Q[0][0];
        Q[0][1] = -Q[1][0];

        A = Q.multDiagonal( w ).multTransposed( Q ); // A = Q*wId*Q^T
    }
}



template<class Real>
void Decompose<Real>::PSDProjection( Real& A00, Real& A01, Real& A10, Real& A11 )
{
    defaulttype::Mat<2,2,Real> Q;
    defaulttype::Vec<2,Real> w;
    dsyev2( A00, A01, A11, w[0], w[1], Q[0][0], Q[1][0] );

    bool modified = false;
    for( int i=0 ; i<2 ; ++i )
        if( w[i] < 0 ){ w[i] = 0; modified = true; }

    if( modified )
    {
        Q[1][1] = Q[0][0];
        Q[0][1] = -Q[1][0];

        defaulttype::Mat<2,2,Real> tmp = Q.multDiagonal( w );

        // A = Q*wId*Q^T

        A00 = Q[0][0]*tmp[0][0] + Q[0][1]*tmp[0][1];
        A01 = A10 = Q[1][0]*tmp[0][0] + Q[1][1]*tmp[1][1];
        A11 = Q[1][0]*tmp[1][0] + Q[1][1]*tmp[1][1];
    }
}





template<class Real>
void Decompose<Real>::NSDProjection( defaulttype::Mat<3,3,Real> &A )
{
    defaulttype::Mat<3,3,Real> Q;
    defaulttype::Vec<3,Real> w;
    if( !symmetricDiagonalization( A, Q, w ) )
    {
        bool modified = false;
        for( int i=0 ; i<3 ; ++i )
            if( w[i] > 0 ){ w[i] = 0; modified = true;  }

        if( modified )
        {
//            defaulttype::Mat<3,3,Real> invQ; invQ.invert( Q );
//            A = Q.multDiagonal( w )*invQ;
            A = (Q.multDiagonal( w )).multTransposed( Q ); // A = Q*wId*Q^T  // only valid for symmetric matrices
        }
    }
}




template<class Real>
void Decompose<Real>::NSDProjection( defaulttype::Mat<2,2,Real> &A )
{
    defaulttype::Mat<2,2,Real> Q;
    defaulttype::Vec<2,Real> w;
    dsyev2( (Real)A[0][0], (Real)A[0][1], (Real)A[1][1], w[0], w[1], Q[0][0], Q[1][0] );

    bool modified = false;
    for( int i=0 ; i<2 ; ++i )
        if( w[i] > 0 ){ w[i] = 0; modified = true; }

    if( modified )
    {
        Q[1][1] = Q[0][0];
        Q[0][1] = -Q[1][0];

        A = Q.multDiagonal( w ).multTransposed( Q ); // A = Q*wId*Q^T
    }
}



template<class Real>
void Decompose<Real>::NSDProjection( Real& A00, Real& A01, Real& A10, Real& A11 )
{
    defaulttype::Mat<2,2,Real> Q;
    defaulttype::Vec<2,Real> w;
    dsyev2( A00, A01, A11, w[0], w[1], Q[0][0], Q[1][0] );

    bool modified = false;
    for( int i=0 ; i<2 ; ++i )
        if( w[i] > 0 ){ w[i] = 0; modified = true; }

    if( modified )
    {
        Q[1][1] = Q[0][0];
        Q[0][1] = -Q[1][0];

        defaulttype::Mat<2,2,Real> tmp = Q.multDiagonal( w );

        // A = Q*wId*Q^T

        A00 = Q[0][0]*tmp[0][0] + Q[0][1]*tmp[0][1];
        A01 = A10 = Q[1][0]*tmp[0][0] + Q[1][1]*tmp[1][1];
        A11 = Q[1][0]*tmp[1][0] + Q[1][1]*tmp[1][1];
    }
}

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_DECOMPOSE_INL
