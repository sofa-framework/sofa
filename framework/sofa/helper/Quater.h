/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_HELPER_QUATER_H
#define SOFA_HELPER_QUATER_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <math.h>
#include <assert.h>
#include <iostream>

namespace sofa
{

namespace helper
{

template<class Real>
class Quater
{
private:
    Real _q[4];

public:
    Quater();
    virtual ~Quater();
    Quater(Real x, Real y, Real z, Real w);
    template<class Real2>
    Quater(const Real2 q[]) { for (int i=0; i<4; i++) _q[i] = (Real)q[i]; }
    template<class Real2>
    Quater(const Quater<Real2>& q) { for (int i=0; i<4; i++) _q[i] = (Real)q[i]; }
    Quater( const defaulttype::Vec<3,Real>& axis, Real angle );

    static Quater identity()
    {
        return Quater(0,0,0,1);
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

    /// Normalize a quaternion
    void normalize();

    void clear()
    {
        _q[0]=0.0;
        _q[1]=0.0;
        _q[2]=0.0;
        _q[3]=1.0;
    }

    void fromMatrix(const defaulttype::Mat3x3d &m);

    template<class Mat33>
    void toMatrix(Mat33 &m) const
    {
        m[0][0] = (typename Mat33::Real) (1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
        m[0][1] = (typename Mat33::Real) (2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
        m[0][2] = (typename Mat33::Real) (2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));

        m[1][0] = (typename Mat33::Real) (2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
        m[1][1] = (typename Mat33::Real) (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
        m[1][2] = (typename Mat33::Real) (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]));

        m[2][0] = (typename Mat33::Real) (2.0f * (_q[2] * _q[0] - _q[1] * _q[3]));
        m[2][1] = (typename Mat33::Real) (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
        m[2][2] = (typename Mat33::Real) (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
    }

    /// Apply the rotation to a given vector
    template<class Vec>
    Vec rotate( const Vec& v ) const
    {
        return Vec(
                (typename Vec::value_type)((1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]))*v[0] + (2.0f * (_q[0] * _q[1] - _q[2] * _q[3])) * v[1] + (2.0f * (_q[2] * _q[0] + _q[1] * _q[3])) * v[2]),
                (typename Vec::value_type)((2.0f * (_q[0] * _q[1] + _q[2] * _q[3]))*v[0] + (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]))*v[1] + (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]))*v[2]),
                (typename Vec::value_type)((2.0f * (_q[2] * _q[0] - _q[1] * _q[3]))*v[0] + (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]))*v[1] + (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]))*v[2])
                );

    }

    /// Apply the inverse rotation to a given vector
    template<class Vec>
    Vec inverseRotate( const Vec& v ) const
    {
        return Vec(
                (typename Vec::value_type)((1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]))*v[0] + (2.0f * (_q[0] * _q[1] + _q[2] * _q[3])) * v[1] + (2.0f * (_q[2] * _q[0] - _q[1] * _q[3])) * v[2]),
                (typename Vec::value_type)((2.0f * (_q[0] * _q[1] - _q[2] * _q[3]))*v[0] + (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]))*v[1] + (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]))*v[2]),
                (typename Vec::value_type)((2.0f * (_q[2] * _q[0] + _q[1] * _q[3]))*v[0] + (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]))*v[1] + (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]))*v[2])
                );

    }

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analagous to adding
    /// translations to get a compound translation.
    //template <class T>
    //friend Quater<T> operator+(Quater<T> q1, Quater<T> q2);
    Quater<Real> operator+(const Quater<Real> &q1) const;

    Quater<Real> operator*(const Quater<Real> &q1) const;
    /// Given two Quaters, multiply them together to get a third quaternion.
    //template <class T>
    //friend Quater<T> operator*(const Quater<T>& q1, const Quater<T>& q2);

    Quater quatVectMult(const defaulttype::Vec3d& vect);

    Quater vectQuatMult(const defaulttype::Vec3d& vect);

    Real& operator[](int index)
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    const Real& operator[](int index) const
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    Quater inverse() const;

    defaulttype::Vec<3,Real> toEulerVector() const;

    // A useful function, builds a rotation matrix in Matrix based on
    // given quaternion.

    void buildRotationMatrix(Real m[4][4]) const;
    void writeOpenGlMatrix( double* m ) const;
    void writeOpenGlMatrix( float* m ) const;

    //void buildRotationMatrix(MATRIX4x4 m);

    //void buildRotationMatrix(Matrix &m);

    // This function computes a quaternion based on an axis (defined by
    // the given vector) and an angle about which to rotate.  The angle is
    // expressed in radians.
    Quater axisToQuat(defaulttype::Vec3d a, Real phi);

    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quater createFromRotationVector(const V& a)
    {
        Real phi = (Real)sqrt(a*a);
        if( phi < 1.0e-5 )
            return Quater(0,0,0,1);
        else
        {
            Real nor = 1/phi;
            Real s = (Real)sin(phi/2);
            return Quater( a[0]*s*nor, a[1]*s*nor,a[2]*s*nor, (Real)cos(phi/2) );
        }
    }

    /// Create using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quater createFromRotationVector(T a0, T a1, T a2 )
    {
        Real phi = (Real)sqrt((Real)(a0*a0+a1*a1+a2*a2));
        if( phi < 1.0e-5 )
            return Quater(0,0,0,1);
        else
        {
            Real nor = 1/phi;
            Real s = (Real)sin(phi/2);
            return Quater( a0*s*nor, a1*s*nor,a2*s*nor, (Real)cos(phi/2) );
        }
    }
    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quater set(const V& a) { return createFromRotationVector(a); }

    /// Create using using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quater set(T a0, T a1, T a2) { return createFromRotationVector(a0,a1,a2); }


    // Print the quaternion
//         inline friend std::ostream& operator<<(std::ostream& out, Quater Q)
// 		{
// 			return (out << "(" << Q._q[0] << "," << Q._q[1] << "," << Q._q[2] << ","
// 				<< Q._q[3] << ")");
// 		}

    // Print the quaternion (C style)
    void print();

    void operator+=(const Quater& q2);
    void operator*=(const Quater& q2);

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const Quater& v )
    {
        out<<v._q[0]<<" "<<v._q[1]<<" "<<v._q[2]<<" "<<v._q[3];
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, Quater& v )
    {
        in>>v._q[0]>>v._q[1]>>v._q[2]>>v._q[3];
        return in;
    }

    static unsigned int size() {return 4;};
};

//typedef Quater<double> Quat; ///< alias
//typedef Quater<float> Quatf; ///< alias
//typedef Quater<double> Quaternion; ///< alias

} // namespace helper

} // namespace sofa

#endif

