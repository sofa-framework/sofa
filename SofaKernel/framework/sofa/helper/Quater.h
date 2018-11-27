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
#ifndef SOFA_HELPER_QUATER_H
#define SOFA_HELPER_QUATER_H

#include <iostream>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

template<class TReal>
class SOFA_HELPER_API Quater
{
private:
    TReal _q[4];

public:
    typedef Quater<TReal> Self;

    /// implement the real-numeric protocol
    typedef TReal Real;

    /// implements the container protocol.
    typedef TReal value_type;
    typedef int size_type;

    typedef defaulttype::Vec<3, TReal> Vec3;

    Quater();
    ~Quater();
    Quater(TReal x, TReal y, TReal z, TReal w);

    Quater(const TReal q[]) { for (int i=0; i<4; i++) _q[i] = (TReal)q[i]; }

    template<class TReal2>
    Quater(const Quater<TReal2>& q) { for (int i=0; i<4; i++) _q[i] = (TReal)q[i]; }
    Quater(const Self::Vec3& axis, TReal angle);

    /** Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo.        
        vFrom and vTo are assumed to be normalized.
    */
    Quater(const Self::Vec3& vFrom, const Self::Vec3& vTo);

    static Quater identity() { return Quater(0,0,0,1); }

    void set(TReal x, TReal y, TReal z, TReal w);

    /// Cast into a standard C array of elements.
    const TReal* ptr() const { return this->_q; }

    /// Cast into a standard C array of elements.
    TReal* ptr() { return this->_q; }

    /// Returns true if norm of Quaternion is one, false otherwise.
    bool isNormalized();

    /// Normalize a quaternion
    void normalize();

    void clear();

    void fromFrame(Self::Vec3& x, Self::Vec3&y, Self::Vec3&z);
    void fromMatrix(const defaulttype::Matrix3& m);

    /// Apply the rotation to a given vector
    Self::Vec3 rotate( const Self::Vec3& v ) const ;

    /// Apply the inverse rotation to a given vector
    Self::Vec3 inverseRotate( const Self::Vec3& v ) const ;

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analagous to adding
    /// translations to get a compound translation.
    Quater<TReal> operator+(const Quater<TReal> &q1) const;
    Quater<TReal> operator*(const Quater<TReal> &q1) const;
    Quater<TReal> operator*(const TReal &r) const;
    Quater<TReal> operator/(const TReal &r) const;
    void operator*=(const TReal &r);
    void operator/=(const TReal &r);

    /// Given two Quaters, multiply them together to get a third quaternion.
    Quater quatVectMult(const Self::Vec3& vect);
    Quater vectQuatMult(const Self::Vec3& vect);

    TReal& operator[](const int index)
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    const TReal& operator[](const int index) const
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    Quater inverse() const;

    Self::Vec3 quatToRotationVector() const;
    Self::Vec3 toEulerVector() const;

    /*! Returns the slerp interpolation of Quaternions \p a and \p b, at time \p t.

     \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.

     When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
     between the Quaternions' orientations, by "flipping" the source Quaternion if needed (see
     negate()). */
    void slerp(const Quater& a, const Quater& b, TReal t, bool allowFlip=true);

    /// A useful function, builds a rotation matrix in Matrix based on
    /// given quaternion.
    void buildRotationMatrix(TReal m[4][4]) const;
    void writeOpenGlMatrix( double* m ) const;
    void writeOpenGlMatrix( float* m ) const;

    /// This function computes a quaternion based on an axis (defined by
    /// the given vector) and an angle about which to rotate.  The angle is
    /// expressed in radians.
    Quater axisToQuat(Self::Vec3 a, TReal phi);
    void quatToAxis(Self::Vec3 & a, TReal &phi) const;

    static Quater createQuaterFromFrame(const Self::Vec3 &lox,
                                        const Self::Vec3 &loy,
                                        const Self::Vec3 &loz);

    /// Create using rotation vector (axis*angle) given in parent coordinates
    static Quater<TReal> createFromRotationVector(const Self::Vec3& a);

    /// Create using the entries of a rotation vector (axis*angle) given in parent coordinates
    static Quater<TReal> createFromRotationVector(const TReal a0, const TReal a1, const TReal a2);

    /// Create a quaternion from Euler angles
    static Quater<TReal> createQuaterFromEuler(const Self::Vec3& v);

    /// Create a quaternion from Euler angles
    static Quater fromEuler( TReal alpha, TReal beta, TReal gamma ){
            return createQuaterFromEuler( Self::Vec3(alpha, beta, gamma) );
    }

    /// Create using rotation vector (axis*angle) given in parent coordinates
    template<class V>
    static Quater set(const V& a) { return createFromRotationVector(a); }

    /// Create using using the entries of a rotation vector (axis*angle) given in parent coordinates
    template<class T>
    static Quater set(T a0, T a1, T a2) { return createFromRotationVector(a0,a1,a2); }

    /// Return the quaternion resulting of the movement between 2 quaternions
    Quater quatDiff( Quater a, const Quater& b);

    /// Return the eulerian vector resulting of the movement between 2 quaternions
    Self::Vec3 angularDisplacement( Quater a, const Quater& b);

    /// Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo. vFrom and vTo are assumed to be normalized.
    void setFromUnitVectors(const Self::Vec3& vFrom, const Self::Vec3& vTo);

    /// Print the quaternion (C style)
    void print();
    Quater<TReal> slerp(Quater<TReal> &q1, TReal t);
    Quater<TReal> slerp2(Quater<TReal> &q1, TReal t);

    void operator+=(const Quater& q2);
    void operator*=(const Quater& q2);
    bool operator==(const Quater& q) const;
    bool operator!=(const Quater& q) const;

    /// write to an output stream
    friend std::ostream& operator << ( std::ostream& out, const Quater<TReal>& v )
    {
        out<<v._q[0]<<" "<<v._q[1]<<" "<<v._q[2]<<" "<<v._q[3];
        return out;
    }

    /// read from an input stream
    friend std::istream& operator >> ( std::istream& in, Quater<TReal>& v )
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

#if !defined(SOFA_HELPER_QUATER_CPP)
extern template class SOFA_HELPER_API Quater<double>;
extern template class SOFA_HELPER_API Quater<float>;
#endif

} // namespace helper

} // namespace sofa

#endif

