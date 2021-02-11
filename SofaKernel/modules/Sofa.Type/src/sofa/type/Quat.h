/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/type/config.h>

#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <cmath>
#include <cassert>
#include <iostream>

namespace sofa::type
{

template<class Real>
class SOFA_TYPE_API Quat
{
private:
    Real _q[4];

public:

    typedef Real value_type;
    typedef sofa::Size Size;

    Quat();
    ~Quat();
    Quat(Real x, Real y, Real z, Real w);
    template<class Real2>
    Quat(const Real2 q[]) { for (int i=0; i<4; i++) _q[i] = Real(q[i]); }
    template<class Real2>
    Quat(const Quat<Real2>& q) { for (int i=0; i<4; i++) _q[i] = Real(q[i]); }
    Quat( const type::Vec<3,Real>& axis, Real angle );

    /** Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo.        
        vFrom and vTo are assumed to be normalized.
    */
    Quat(const type::Vec<3, Real>& vFrom, const type::Vec<3, Real>& vTo);

    static Quat identity()
    {
        return Quat(0,0,0,1);
    }


    void set(Real x, Real y, Real z, Real w)
    {
        _q[0] = x;
        _q[1] = y;
        _q[2] = z;
        _q[3] = w;
    }


    /// Cast into a standard C array of elements.
    const Real* ptr() const
    {
        return this->_q;
    }

    /// Cast into a standard C array of elements.
    Real* ptr()
    {
        return this->_q;
    }

    /// Returns true if norm of Quatnion is one, false otherwise.
    bool isNormalized();

    /// Normalize a quaternion
    void normalize();

    void clear()
    {
        _q[0]=0.0;
        _q[1]=0.0;
        _q[2]=0.0;
        _q[3]=1.0;
    }

    void fromFrame(type::Vec<3,Real>& x, type::Vec<3,Real>&y, type::Vec<3,Real>&z);


    void fromMatrix(const type::Matrix3 &m);

    template<class Mat33>
    void toMatrix(Mat33 &m) const
    {
        m[0][0] = typename Mat33::Real (1 - 2 * (_q[1] * _q[1] + _q[2] * _q[2]));
        m[0][1] = typename Mat33::Real (2 * (_q[0] * _q[1] - _q[2] * _q[3]));
        m[0][2] = typename Mat33::Real (2 * (_q[2] * _q[0] + _q[1] * _q[3]));

        m[1][0] = typename Mat33::Real (2 * (_q[0] * _q[1] + _q[2] * _q[3]));
        m[1][1] = typename Mat33::Real (1 - 2 * (_q[2] * _q[2] + _q[0] * _q[0]));
        m[1][2] = typename Mat33::Real (2 * (_q[1] * _q[2] - _q[0] * _q[3]));

        m[2][0] = typename Mat33::Real (2 * (_q[2] * _q[0] - _q[1] * _q[3]));
        m[2][1] = typename Mat33::Real (2 * (_q[1] * _q[2] + _q[0] * _q[3]));
        m[2][2] = typename Mat33::Real (1 - 2 * (_q[1] * _q[1] + _q[0] * _q[0]));
    }

    /// Apply the rotation to a given vector
    template<class Vec>
    Vec rotate( const Vec& v ) const
    {
        return Vec(
                typename Vec::value_type((1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]))*v[0] + (2.0f * (_q[0] * _q[1] - _q[2] * _q[3])) * v[1] + (2.0f * (_q[2] * _q[0] + _q[1] * _q[3])) * v[2]),
                typename Vec::value_type((2.0f * (_q[0] * _q[1] + _q[2] * _q[3]))*v[0] + (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]))*v[1] + (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]))*v[2]),
                typename Vec::value_type((2.0f * (_q[2] * _q[0] - _q[1] * _q[3]))*v[0] + (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]))*v[1] + (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]))*v[2])
                );

    }

    /// Apply the inverse rotation to a given vector
    template<class Vec>
    Vec inverseRotate( const Vec& v ) const
    {
        return Vec(
                typename Vec::value_type((1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]))*v[0] + (2.0f * (_q[0] * _q[1] + _q[2] * _q[3])) * v[1] + (2.0f * (_q[2] * _q[0] - _q[1] * _q[3])) * v[2]),
                typename Vec::value_type((2.0f * (_q[0] * _q[1] - _q[2] * _q[3]))*v[0] + (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]))*v[1] + (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]))*v[2]),
                typename Vec::value_type((2.0f * (_q[2] * _q[0] + _q[1] * _q[3]))*v[0] + (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]))*v[1] + (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]))*v[2])
                );

    }

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analagous to adding
    /// translations to get a compound translation.
    Quat<Real> operator+(const Quat<Real> &q1) const;

    Quat<Real> operator*(const Quat<Real> &q1) const;

    Quat<Real> operator*(const Real &r) const;
    Quat<Real> operator/(const Real &r) const;
    void operator*=(const Real &r);
    void operator/=(const Real &r);

    /// Given two Quats, multiply them together to get a third quaternion.

    Quat quatVectMult(const type::Vec<3,Real>& vect);

    Quat vectQuatMult(const type::Vec<3,Real>& vect);

    Real& operator[](Size index)
    {
        assert(index < 4);
        return _q[index];
    }

    const Real& operator[](Size index) const
    {
        assert(index < 4);
        return _q[index];
    }

    Quat inverse() const;


    type::Vec<3,Real> quatToRotationVector() const;

    type::Vec<3,Real> toEulerVector() const;


    /*! Returns the slerp interpolation of Quatnions \p a and \p b, at time \p t.

     \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.

     When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
     between the Quatnions' orientations, by "flipping" the source Quatnion if needed (see
     negate()). */
    void slerp(const Quat& a, const Quat& b, Real t, bool allowFlip=true);

    // A useful function, builds a rotation matrix in Matrix based on
    // given quaternion.

    void buildRotationMatrix(Real m[4][4]) const;
    void writeOpenGlMatrix( double* m ) const;
    void writeOpenGlMatrix( float* m ) const;

    // This function computes a quaternion based on an axis (defined by
    // the given vector) and an angle about which to rotate.  The angle is
    // expressed in radians.
    Quat axisToQuat(type::Vec<3,Real> a, Real phi);
    void quatToAxis(type::Vec<3,Real> & a, Real &phi) const;


    static Quat createQuaterFromFrame(const type::Vec<3, Real> &lox, const type::Vec<3, Real> &loy,const type::Vec<3, Real> &loz);

    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quat createFromRotationVector(const V& a)
    {
        Real phi = Real(sqrt(a*a));
        if( phi < 1.0e-5 )
            return Quat(0,0,0,1);
        else
        {
            Real nor = 1/phi;
            Real s = Real(sin(phi/2));
            return Quat( a[0]*s*nor, a[1]*s*nor,a[2]*s*nor, Real(cos(phi/2)));
        }
    }

    /// Create a quaternion from Euler angles
    /// Thanks to https://github.com/mrdoob/three.js/blob/dev/src/math/Quatnion.js#L199
    enum class EulerOrder
    {
        XYZ, YXZ, ZXY, ZYX, YZX, XZY
    };

    static Quat createQuaterFromEuler( type::Vec<3,Real> v, EulerOrder order = EulerOrder::ZYX)
    {
        Real quat[4];

        Real c1 = cos( v.elems[0] / 2 );
        Real c2 = cos( v.elems[1] / 2 );
        Real c3 = cos( v.elems[2] / 2 );

        Real s1 = sin( v.elems[0] / 2 );
        Real s2 = sin( v.elems[1] / 2 );
        Real s3 = sin( v.elems[2] / 2 );

        switch(order)
        {
        case EulerOrder::XYZ:
            quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
            break;
        case EulerOrder::YXZ:
            quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
            break;
        case EulerOrder::ZXY:
            quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
            break;
        case EulerOrder::YZX:
            quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
            break;
        case EulerOrder::XZY:
            quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
            break;
        default:
        case EulerOrder::ZYX:
            quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
            quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
            quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
            quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
            break;
        }

        Quat quatResult{ quat[0], quat[1], quat[2], quat[3] };
        return quatResult;
    }


    /// Create a quaternion from Euler angles
    static Quat fromEuler( Real alpha, Real beta, Real gamma, EulerOrder order = EulerOrder::ZYX ){
        return createQuaterFromEuler( {alpha, beta, gamma }, order );
    }

    /// Create using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quat createFromRotationVector(T a0, T a1, T a2 )
    {
        Real phi = Real(sqrt(a0*a0+a1*a1+a2*a2));
        if( phi < 1.0e-5 )
            return Quat(0,0,0,1);
        else
        {
            Real nor = 1/phi;
            Real s = sin(phi/2.0);
            return Quat( a0*s*nor, a1*s*nor,a2*s*nor, cos(phi/2.0) );
        }
    }
    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quat set(const V& a) { return createFromRotationVector(a); }

    /// Create using using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quat set(T a0, T a1, T a2) { return createFromRotationVector(a0,a1,a2); }

    /// Return the quaternion resulting of the movement between 2 quaternions
    Quat quatDiff( Quat a, const Quat& b)
    {
        // If the axes are not oriented in the same direction, flip the axis and angle of a to get the same convention than b
        if (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]<0)
        {
            a[0] = -a[0];
            a[1] = -a[1];
            a[2] = -a[2];
            a[3] = -a[3];
        }

        Quat q = b.inverse() * a;
        return q;
    }

    /// Return the eulerian vector resulting of the movement between 2 quaternions
    type::Vec<3,Real> angularDisplacement( Quat a, const Quat& b)
    {
        return quatDiff(a,b).quatToRotationVector();    // Use of quatToRotationVector instead of toEulerVector:
                                                        // this is done to keep the old behavior (before the
                                                        // correction of the toEulerVector function).
    }

    /// Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo. vFrom and vTo are assumed to be normalized.
    void setFromUnitVectors(const type::Vec<3, Real>& vFrom, const type::Vec<3, Real>& vTo);


    // Print the quaternion (C style)
    void print();
    Quat<Real> slerp(Quat<Real> &q1, Real t);
    Quat<Real> slerp2(Quat<Real> &q1, Real t);

    void operator+=(const Quat& q2);
    void operator*=(const Quat& q2);

    bool operator==(const Quat& q) const
    {
        for (int i=0; i<4; i++)
            if ( std::abs( _q[i] - q._q[i] ) > EQUALITY_THRESHOLD ) return false;
        return true;
    }

    bool operator!=(const Quat& q) const
    {
        for (int i=0; i<4; i++)
            if ( std::abs( _q[i] - q._q[i] ) > EQUALITY_THRESHOLD ) return true;
        return false;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const Quat& v )
    {
        out<<v._q[0]<<" "<<v._q[1]<<" "<<v._q[2]<<" "<<v._q[3];
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, Quat& v )
    {
        in>>v._q[0]>>v._q[1]>>v._q[2]>>v._q[3];
        return in;
    }

    enum { static_size = 4 };
    static unsigned int size() {return 4;}

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 4 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for quaternions)
    enum { spatial_dimensions = 3 };
};

using Quatd = type::Quat<double>;
using Quatf = type::Quat<float>;
using Quaternion = type::Quat<SReal>;

#if !defined(SOFA_TYPE_QUAT_CPP)
extern template class SOFA_TYPE_API Quat<double>;
extern template class SOFA_TYPE_API Quat<float>;
#endif

} // namespace sofa::type
