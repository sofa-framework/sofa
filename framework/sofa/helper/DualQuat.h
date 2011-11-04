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
#ifndef SOFA_HELPER_DUALQUAT_H
#define SOFA_HELPER_DUALQUAT_H

#include <sofa/helper/Quater.h>
#include <sofa/defaulttype/Vec.h>
#include <iostream>
#include <newmat/newmat.h>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

template<class Real>
class SOFA_HELPER_API DualQuat
{
    typedef typename sofa::defaulttype::Vec<3,Real> Vec;
    typedef typename sofa::helper::Quater<Real> Quat;

private:
    Quat _q[2]; // Store the real and imaginary parts

public:
    DualQuat ();
    DualQuat ( const DualQuat<Real>& q );
    DualQuat ( const Quat& qRe, const Quat& qIm );
    // This constructor define a rigid transformation from the origin to the reference frame
    DualQuat ( const Vec& tr, const Quat& q );
    // This constructor define a rigid transformation from a reference frame to another
    DualQuat ( const Vec& tr1,   // The reference frame center at the beginning
            const Quat& q1,   // The reference frame orientation at the beginning
            const Vec& tr2,   // The reference frame center at the end
            const Quat& q2);  // The reference frame orientation at the end
    ~DualQuat();

    static DualQuat identity();

    void normalize(); // Normalize the dual quaternion
    void fromTransQuat ( const Vec& vec, const Quat& q);
    void toTransQuat ( Vec& vec, Quat& quat ) const;
    void toMatrix( defaulttype::Matrix4& M) const;
    void fromMatrix( const defaulttype::Matrix4 M);
    void toGlMatrix( double M[16]) const;
    Vec transform( const Vec& vec); // Apply the QD transformation to a point

    DualQuat<Real> operator+ ( const DualQuat<Real>& dq ) const;
    DualQuat<Real> operator* ( const Real r ) const;
    DualQuat<Real> operator/ ( const Real r ) const;
    DualQuat<Real> operator= ( const DualQuat<Real>& quat );
    DualQuat<Real>& operator+= ( const DualQuat<Real>& quat );
    const Quat& operator[] ( unsigned int i ) const;
    Quat& operator[] ( unsigned int i );

    inline friend std::ostream& operator<< ( std::ostream& os, const DualQuat<Real>& dq )
    { os << dq[0] << " " << dq[1]; return os;}

    inline friend std::istream& operator>> ( std::istream& is, DualQuat<Real>& dq )
    { is >> dq[0] >> dq[1]; return is;}
};

typedef DualQuat<double> DualQuatd; ///< alias
typedef DualQuat<float> DualQuatf; ///< alias

} // namespace helper

} // namespace sofa

#endif

