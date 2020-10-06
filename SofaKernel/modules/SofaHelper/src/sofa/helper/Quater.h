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

#include <sofa/helper/config.h>

#include <sofa/helper/fixed_array.h>

#include <cmath>
#include <cassert>
#include <iostream>

namespace sofa::helper
{

template<class Real>
class SOFA_HELPER_API Quater
{
private:
    helper::fixed_array<Real, 4> _q;

public:
    typedef Real value_type;
    typedef std::size_t size_type;
    using Vector3 = helper::fixed_array<Real, 3>;

    Quater();
    ~Quater();
    Quater(Real x, Real y, Real z, Real w);
    template<class Real2>
    Quater(const Real2 q[]) { for (int i=0; i<4; i++) _q[i] = Real(q[i]); }
    template<class Real2>
    Quater(const Quater<Real2>& q) { for (int i=0; i<4; i++) _q[i] = Real(q[i]); }
    Quater( const Vector3& axis, Real angle );

    /** Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo.        
        vFrom and vTo are assumed to be normalized.
    */
    Quater(const Vector3& vFrom, const Vector3& vTo);

    static Quater identity();

    void set(Real x, Real y, Real z, Real w);

    /// Cast into a standard C array of elements.
    const Real* ptr() const;

    /// Cast into a standard C array of elements.
    Real* ptr();

    /// Returns true if norm of Quaternion is one, false otherwise.
    bool isNormalized();

    /// Normalize a quaternion
    void normalize();

    void clear();

    void fromFrame(Vector3& x, Vector3&y, Vector3&z);

    template <typename Mat33>
    void fromMatrix(const Mat33& m)
    {
        Real tr, s;

        tr = (Real)(m[0][0] + m[1][1] + m[2][2]);

        // check the diagonal
        if (tr > 0)
        {
            s = (float)sqrt(tr + 1);
            _q[3] = s * 0.5f; // w OK
            s = 0.5f / s;
            _q[0] = (Real)((m[2][1] - m[1][2]) * s); // x OK
            _q[1] = (Real)((m[0][2] - m[2][0]) * s); // y OK
            _q[2] = (Real)((m[1][0] - m[0][1]) * s); // z OK
        }
        else
        {
            if (m[1][1] > m[0][0] && m[2][2] <= m[1][1])
            {
                s = (Real)sqrt((m[1][1] - (m[2][2] + m[0][0])) + 1.0f);

                _q[1] = s * 0.5f; // y OK

                if (s != 0.0f)
                    s = 0.5f / s;

                _q[2] = (Real)((m[1][2] + m[2][1]) * s); // z OK
                _q[0] = (Real)((m[0][1] + m[1][0]) * s); // x OK
                _q[3] = (Real)((m[0][2] - m[2][0]) * s); // w OK
            }
            else if ((m[1][1] <= m[0][0] && m[2][2] > m[0][0]) || (m[2][2] > m[1][1]))
            {
                s = (Real)sqrt((m[2][2] - (m[0][0] + m[1][1])) + 1.0f);

                _q[2] = s * 0.5f; // z OK

                if (s != 0.0f)
                    s = 0.5f / s;

                _q[0] = (Real)((m[2][0] + m[0][2]) * s); // x OK
                _q[1] = (Real)((m[1][2] + m[2][1]) * s); // y OK
                _q[3] = (Real)((m[1][0] - m[0][1]) * s); // w OK
            }
            else
            {
                s = (Real)sqrt((m[0][0] - (m[1][1] + m[2][2])) + 1.0f);

                _q[0] = s * 0.5f; // x OK

                if (s != 0.0f)
                    s = 0.5f / s;

                _q[1] = (Real)((m[0][1] + m[1][0]) * s); // y OK
                _q[2] = (Real)((m[2][0] + m[0][2]) * s); // z OK
                _q[3] = (Real)((m[2][1] - m[1][2]) * s); // w OK
            }
        }
    }

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
    Quater<Real> operator+(const Quater<Real> &q1) const;

    Quater<Real> operator*(const Quater<Real> &q1) const;

    Quater<Real> operator*(const Real &r) const;
    Quater<Real> operator/(const Real &r) const;
    void operator*=(const Real &r);
    void operator/=(const Real &r);

    /// Given two Quaters, multiply them together to get a third quaternion.

    Quater quatVectMult(const Vector3& vect);

    Quater vectQuatMult(const Vector3& vect);

    Real& operator[](size_type index);
    const Real& operator[](size_type index) const;

    Quater inverse() const;

    Vector3 quatToRotationVector() const;

    Vector3 toEulerVector() const;


    /*! Returns the slerp interpolation of Quaternions \p a and \p b, at time \p t.

     \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.

     When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
     between the Quaternions' orientations, by "flipping" the source Quaternion if needed (see
     negate()). */
    void slerp(const Quater& a, const Quater& b, Real t, bool allowFlip=true);

    // A useful function, builds a rotation matrix in Matrix based on
    // given quaternion.

    void buildRotationMatrix(Real m[4][4]) const;
    void writeOpenGlMatrix( double* m ) const;
    void writeOpenGlMatrix( float* m ) const;

    // This function computes a quaternion based on an axis (defined by
    // the given vector) and an angle about which to rotate.  The angle is
    // expressed in radians.
    Quater axisToQuat(Vector3 a, Real phi);
    void quatToAxis(Vector3 & a, Real &phi) const;

    static Quater createQuaterFromFrame(const Vector3& lox, const Vector3& loy, const Vector3& loz);

    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quater createFromRotationVector(const V& a)
    {
        Real phi = Real(sqrt(a*a));
        if( phi < 1.0e-5 )
            return Quater(0,0,0,1);
        else
        {
            Real nor = 1/phi;
            Real s = Real(sin(phi/2));
            return Quater( a[0]*s*nor, a[1]*s*nor,a[2]*s*nor, Real(cos(phi/2)));
        }
    }

    /// Create a quaternion from Euler angles
    /// Thanks to https://github.com/mrdoob/three.js/blob/dev/src/math/Quaternion.js#L199
    enum class EulerOrder
    {
        XYZ, YXZ, ZXY, ZYX, YZX, XZY, NONE
    };

    static Quater createQuaterFromEuler(const Vector3& v, EulerOrder order = EulerOrder::ZYX);

    /// Create a quaternion from Euler angles
    static Quater fromEuler(Real alpha, Real beta, Real gamma, EulerOrder order = EulerOrder::ZYX);

    /// Create using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quater createFromRotationVector(T a0, T a1, T a2 )
    {
        Real phi = Real(sqrt(a0*a0+a1*a1+a2*a2));
        if( phi < 1.0e-5 )
            return Quater(0,0,0,1);
        else
        {
            Real nor = 1/phi;
            Real s = sin(phi/2.0);
            return Quater( a0*s*nor, a1*s*nor,a2*s*nor, cos(phi/2.0) );
        }
    }
    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quater set(const V& a) { return createFromRotationVector(a); }

    /// Create using using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quater set(T a0, T a1, T a2) { return createFromRotationVector(a0,a1,a2); }

    /// Return the quaternion resulting of the movement between 2 quaternions
    Quater quatDiff(Quater a, const Quater& b);

    /// Return the eulerian vector resulting of the movement between 2 quaternions
    Vector3 angularDisplacement(Quater a, const Quater& b);

    /// Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo. vFrom and vTo are assumed to be normalized.
    void setFromUnitVectors(const Vector3& vFrom, const Vector3& vTo);


    // Print the quaternion (C style)
    void print();
    Quater<Real> slerp(Quater<Real> &q1, Real t);
    Quater<Real> slerp2(Quater<Real> &q1, Real t);

    void operator+=(const Quater& q2);
    void operator*=(const Quater& q2);

    bool operator==(const Quater& q) const;

    bool operator!=(const Quater& q) const;

    /// write to an output stream
    inline friend std::ostream& operator << (std::ostream& out, const Quater& v)
    {
        out << v._q[0] << " " << v._q[1] << " " << v._q[2] << " " << v._q[3];
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> (std::istream& in, Quater& v)
    {
        in >> v._q[0] >> v._q[1] >> v._q[2] >> v._q[3];
        return in;
    }

    static constexpr unsigned int static_size = 4;
    static unsigned int size() {return 4;}

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr unsigned int total_size = 4;
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for quaternions)
    static constexpr unsigned int spatial_dimensions = 3;
};

#if  !defined(SOFA_HELPER_QUATER_CPP)
extern template class SOFA_HELPER_API Quater<double>;
extern template class SOFA_HELPER_API Quater<float>;
#endif

} // namespace sofa::helper



