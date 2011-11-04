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
#ifndef SOFA_HELPER_DUALQUAT_INL
#define SOFA_HELPER_DUALQUAT_INL

#include "DualQuat.h"

namespace sofa
{

namespace helper
{

// Constructor
template<class Real>
DualQuat<Real>::DualQuat ()
{
    for ( int i = 0; i < 2; i++ )
        for ( int j = 0; j < 4; j++ )
            _q[i][j] = 0;
}

// Constructor
template<class Real>
DualQuat<Real>::DualQuat ( const DualQuat<Real>& q )
{
    _q[0] = q[0];
    _q[1] = q[1];
}

// Constructor
template<class Real>
DualQuat<Real>::DualQuat ( const Quat& qRe, const Quat& qIm )
{
    _q[0] = qRe;
    _q[1] = qIm;
}

// Constructor
template<class Real>
DualQuat<Real>::DualQuat ( const Vec& tr, const Quat& q )
{
    fromTransQuat ( tr, q );
}

// Constructor
template<class Real>
DualQuat<Real>::DualQuat ( const Vec& tr1, const Quat& q1, const Vec& tr2, const Quat& q2 )
{
    //TODO// refaire la fonction en calculant directement le vecteur. Histoire d'aller plus vite...
    defaulttype::Mat<4,4,Real> minusTr1M4, tr2M4, rigidM4, relRot1M4;
    defaulttype::Mat<3,3,Real> relRot1M3;

    minusTr1M4.identity();
    tr2M4.identity();
    relRot1M4.identity();

    minusTr1M4 ( 0, 3 ) = -tr1[0];
    minusTr1M4 ( 1, 3 ) = -tr1[1];
    minusTr1M4 ( 2, 3 ) = -tr1[2];
    tr2M4 ( 0, 3 ) = tr2[0];
    tr2M4 ( 1, 3 ) = tr2[1];
    tr2M4 ( 2, 3 ) = tr2[2];

    // The rotational part of the matrix rigidM4 is the same as 'q2 * q1.inverse()'
    Quat rotQ = ( q2 * q1.inverse());
    rotQ.toMatrix ( relRot1M3 );

    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
            relRot1M4 ( i, j ) = relRot1M3 ( i, j );

    // We get the rigid transformation from state 1 to state 2.
    rigidM4 = tr2M4 * relRot1M4 * minusTr1M4;

    Vec tr = Vec ( rigidM4 ( 0, 3 ), rigidM4 ( 1, 3 ), rigidM4 ( 2, 3 ) );

    fromTransQuat ( tr, rotQ );
}

// Destructor
template<class Real>
DualQuat<Real>::~DualQuat()
{
}

template<class Real>
DualQuat<Real> DualQuat<Real>::identity()
{
    return DualQuat( Quat::identity(), Quat::identity());
}

template<class Real>
void DualQuat<Real>::normalize()
{
    Real mag = (Real) sqrt ( _q[0][0]*_q[0][0] + _q[0][1]*_q[0][1] + _q[0][2]*_q[0][2] + _q[0][3]*_q[0][3] );
    assert ( mag != 0 ); // We can't normalize a null dual quaternion.
    for ( int i = 0; i < 2; i++ )
        for ( int j = 0; j < 4; j++ )
            _q[i][j] /= mag;
}

template<class Real>
void DualQuat<Real>::fromTransQuat ( const Vec& tr, const Quat& q )
{
    // non-dual part (just copy quat q):
    _q[0] = q;

    // dual part:
    _q[1][3] = -(Real)0.5* ( tr[0]*q[0] + tr[1]*q[1] + tr[2]*q[2] ); // q[w]
    _q[1][0] =  (Real)0.5* ( tr[0]*q[3] + tr[1]*q[2] - tr[2]*q[1] ); // q[x]
    _q[1][1] =  (Real)0.5* ( -tr[0]*q[2] + tr[1]*q[3] + tr[2]*q[0] ); // q[y]
    _q[1][2] =  (Real)0.5* ( tr[0]*q[1] - tr[1]*q[0] + tr[2]*q[3] ); // q[z]

    // Do it an unit dual quat
    normalize();
}

template<class Real>
void DualQuat<Real>::toTransQuat ( Vec& vec, Quat& quat ) const
{
    // non-dual part (just copy quat q):
    quat = _q[0];

    // dual part:
    vec[0] = 2* ( -_q[1][3]*_q[0][0] + _q[1][0]*_q[0][3] - _q[1][1]*_q[0][2] + _q[1][2]*_q[0][1] );
    vec[1] = 2* ( -_q[1][3]*_q[0][1] + _q[1][0]*_q[0][2] + _q[1][1]*_q[0][3] - _q[1][2]*_q[0][0] );
    vec[2] = 2* ( -_q[1][3]*_q[0][2] - _q[1][0]*_q[0][1] + _q[1][1]*_q[0][0] + _q[1][2]*_q[0][3] );
}

template<class Real>
void DualQuat<Real>::toMatrix ( defaulttype::Matrix4& M ) const
{
    Real t0, t1, t2;
    t0 = 2 * ( -_q[1][3]*_q[0][0] + _q[1][0]*_q[0][3] - _q[1][1]*_q[0][2] + _q[1][2]*_q[0][1] );
    t1 = 2 * ( -_q[1][3]*_q[0][1] + _q[1][0]*_q[0][2] + _q[1][1]*_q[0][3] - _q[1][2]*_q[0][0] );
    t2 = 2 * ( -_q[1][3]*_q[0][2] - _q[1][0]*_q[0][1] + _q[1][1]*_q[0][0] + _q[1][2]*_q[0][3] );

    M ( 0, 0 ) = 1 - 2* ( _q[0][1]*_q[0][1] + _q[0][2]*_q[0][2] ); // 1 - 2(y0*y0) - 2(z0*z0)
    M ( 0, 1 ) =     2* ( _q[0][0]*_q[0][1] - _q[0][3]*_q[0][2] ); // 2(x0y0 - w0z0)
    M ( 0, 2 ) =     2* ( _q[0][0]*_q[0][2] + _q[0][3]*_q[0][1] ); // 2(x0z0 + w0y0)
    M ( 0, 3 ) = t0;
    M ( 1, 0 ) =     2* ( _q[0][0]*_q[0][1] + _q[0][3]*_q[0][2] ); // 2(x0y0 + w0z0)
    M ( 1, 1 ) = 1 - 2* ( _q[0][0]*_q[0][0] + _q[0][2]*_q[0][2] ); // 1 - 2(x0*x0) - 2(z0*z0)
    M ( 1, 2 ) =     2* ( _q[0][1]*_q[0][2] - _q[0][3]*_q[0][0] ); // 2(y0z0 - w0x0)
    M ( 1, 3 ) = t1;
    M ( 2, 0 ) =     2* ( _q[0][0]*_q[0][2] - _q[0][3]*_q[0][1] ); // 2(x0z0 - w0y0)
    M ( 2, 1 ) =     2* ( _q[0][1]*_q[0][2] + _q[0][3]*_q[0][0] ); // 2(y0z0 + w0x0)
    M ( 2, 2 ) = 1 - 2* ( _q[0][0]*_q[0][0] + _q[0][1]*_q[0][1] ); // 1 - 2(x0*x0) - 2(y0*y0)
    M ( 2, 3 ) = t2;
    M ( 3, 0 ) = M ( 3, 1 ) = M ( 3, 2 ) = 0;
    M ( 3, 3 ) = 1;
}

template<class Real>
void DualQuat<Real>::toGlMatrix ( double M[16] ) const
{
    // GL matrices are transposed.
    Real t0, t1, t2;
    t0 = 2 * ( -_q[1][3]*_q[0][0] + _q[1][0]*_q[0][3] - _q[1][1]*_q[0][2] + _q[1][2]*_q[0][1] );
    t1 = 2 * ( -_q[1][3]*_q[0][1] + _q[1][0]*_q[0][2] + _q[1][1]*_q[0][3] - _q[1][2]*_q[0][0] );
    t2 = 2 * ( -_q[1][3]*_q[0][2] - _q[1][0]*_q[0][1] + _q[1][1]*_q[0][0] + _q[1][2]*_q[0][3] );

    M[0] = 1 - 2* ( _q[0][1]*_q[0][1] + _q[0][2]*_q[0][2] ); // 1 - 2(y0*y0) - 2(z0*z0)
    M[1] =     2* ( _q[0][0]*_q[0][1] + _q[0][3]*_q[0][2] ); // 2(x0y0 + w0z0)
    M[2] =     2* ( _q[0][0]*_q[0][2] - _q[0][3]*_q[0][1] ); // 2(x0z0 - w0y0)
    M[3] = 0;
    M[4] =     2* ( _q[0][0]*_q[0][1] - _q[0][3]*_q[0][2] ); // 2(x0y0 - w0z0)
    M[5] = 1 - 2* ( _q[0][0]*_q[0][0] + _q[0][2]*_q[0][2] ); // 1 - 2(x0*x0) - 2(z0*z0)
    M[6] =     2* ( _q[0][1]*_q[0][2] + _q[0][3]*_q[0][0] ); // 2(y0z0 + w0x0)
    M[7] = 0;
    M[8] =     2* ( _q[0][0]*_q[0][2] + _q[0][3]*_q[0][1] ); // 2(x0z0 + w0y0)
    M[9] =     2* ( _q[0][1]*_q[0][2] - _q[0][3]*_q[0][0] ); // 2(y0z0 - w0x0)
    M[10] = 1 - 2* ( _q[0][0]*_q[0][0] + _q[0][1]*_q[0][1] ); // 1 - 2(x0*x0) - 2(y0*y0)
    M[11] = 0;
    M[12] = t0;
    M[13] = t1;
    M[14] = t2;
    M[15] = 1;
}

template<class Real>
void DualQuat<Real>::fromMatrix ( const defaulttype::Matrix4 M )
{
    defaulttype::Matrix3 rotPartM3;
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
            rotPartM3 ( i, j ) = M ( i, j );

    // non-dual part (just copy quat q):
    _q[0].fromMatrix ( rotPartM3 );

    // dual part:
    _q[1][3] = -(Real)(0.5* ( M( 0, 3)*_q[0][0] + M( 1, 3)*_q[0][1] + M( 2, 3)*_q[0][2] ));
    _q[1][0] =  (Real)(0.5* ( M( 0, 3)*_q[0][3] + M( 1, 3)*_q[0][2] - M( 2, 3)*_q[0][1] ));
    _q[1][1] =  (Real)(0.5* (-M( 0, 3)*_q[0][2] + M( 1, 3)*_q[0][3] + M( 2, 3)*_q[0][0] ));
    _q[1][2] =  (Real)(0.5* ( M( 0, 3)*_q[0][1] - M( 1, 3)*_q[0][0] + M( 2, 3)*_q[0][3] ));

    normalize();
}

template<class Real>
typename DualQuat<Real>::Vec DualQuat<Real>::transform ( const typename DualQuat<Real>::Vec& vec )
{
    defaulttype::Matrix4 M;
    toMatrix ( M );

    sofa::defaulttype::Vec<4, Real>res = M * sofa::defaulttype::Vec<4, Real> ( vec[0], vec[1], vec[2], 1 );
    return Vec ( res[0], res[1], res[2] );
}

template<class Real>
DualQuat<Real> DualQuat<Real>::operator+ ( const DualQuat<Real>& dq ) const
{
    DualQuat<Real>res;

    for ( int i = 0; i < 2; i++ )
        for ( int j = 0; j < 4; j++ )
            res[i][j] = _q[i][j] + dq[i][j];

    return res;
}

template<class Real>
DualQuat<Real> DualQuat<Real>::operator* ( const Real r ) const
{
    DualQuat<Real> res;
    for ( int i = 0; i < 2; i++ )
        for ( int j = 0; j < 4; j++ )
            res[i][j] = _q[i][j] * r;
    return res;
}

template<class Real>
DualQuat<Real> DualQuat<Real>::operator/ ( const Real r ) const
{
    DualQuat<Real> res;
    for ( int i = 0; i < 2; i++ )
        for ( int j = 0; j < 4; j++ )
            res[i][j] = _q[i][j] / r;
    return res;
}

template<class Real>
DualQuat<Real> DualQuat<Real>::operator= ( const DualQuat<Real>& dq )
{
    _q[0] = dq[0];
    _q[1] = dq[1];
    return ( *this );
}

template<class Real>
DualQuat<Real>& DualQuat<Real>::operator+= ( const DualQuat<Real>& dq )
{
    for ( int i = 0; i < 2; i++ )
        for ( int j = 0; j < 4; j++ )
            _q[i][j] += dq[i][j];

    return ( *this );
}

template<class Real>
const Quater<Real>& DualQuat<Real>::operator[] ( unsigned int i ) const
{
    return _q[i];
}

template<class Real>
Quater<Real>& DualQuat<Real>::operator[] ( unsigned int i )
{
    return _q[i];
}

} // namespace helper

} // namespace sofa

#endif
