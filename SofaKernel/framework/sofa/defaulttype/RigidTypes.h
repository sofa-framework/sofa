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
#ifndef SOFA_DEFAULTTYPE_RIGIDTYPES_H
#define SOFA_DEFAULTTYPE_RIGIDTYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/random.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

namespace sofa
{

namespace defaulttype
{

template<int N, typename real>
class RigidDeriv;

template<int N, typename real>
class RigidCoord;

template<int N, typename real>
class RigidMass;

template<int N, typename real>
class StdRigidTypes;

//=============================================================================
// 3D Rigids
//=============================================================================

/** Degrees of freedom of 3D rigid bodies. Orientations are modeled using quaternions.
 */
template<typename real>
class RigidDeriv<3, real>
{
public:
    typedef real value_type;
    typedef int size_type;
    typedef real Real;
    typedef Vec<3,Real> Pos;
    typedef Vec<3,Real> Rot;
    typedef Vec<3,Real> Vec3;
    typedef Vec<6,Real> VecAll;
    typedef helper::Quater<Real> Quat;

protected:
    Vec3 vCenter;
    Vec3 vOrientation;

public:
    friend class RigidCoord<3,real>;

    RigidDeriv()
    {
        clear();
    }

    RigidDeriv(const Vec3 &velCenter, const Vec3 &velOrient)
        : vCenter(velCenter), vOrientation(velOrient)
    {}

    template<typename real2>
    RigidDeriv(const RigidDeriv<3,real2>& c)
        : vCenter(c.getVCenter()), vOrientation(c.getVOrientation())
    {}

    template<typename real2>
    RigidDeriv(const Vec<6,real2> &v)
        : vCenter(Vec<3,real2>(v.data())), vOrientation(Vec<3,real2>(v.data()+3))
    {}

    template<typename real2>
    RigidDeriv(const real2* ptr)
        :vCenter(ptr),vOrientation(ptr+3)
    {
    }

    void clear()
    {
        vCenter.clear();
        vOrientation.clear();
    }

    template<typename real2>
    void operator=(const RigidDeriv<3,real2>& c)
    {
        vCenter = c.getVCenter();
        vOrientation = c.getVOrientation();
    }

    template<typename real2>
    void operator=(const Vec<3,real2>& v)
    {
        vCenter = v;
    }

    template<typename real2>
    void operator=(const Vec<6,real2>& v)
    {
        vCenter = v;
        vOrientation = Vec<3,real2>(v.data()+3);
    }

    void operator+=(const RigidDeriv& a)
    {
        vCenter += a.vCenter;
        vOrientation += a.vOrientation;
    }

    void operator-=(const RigidDeriv& a)
    {
        vCenter -= a.vCenter;
        vOrientation -= a.vOrientation;
    }

    RigidDeriv<3,real> operator+(const RigidDeriv<3,real>& a) const
    {
        RigidDeriv d;
        d.vCenter = vCenter + a.vCenter;
        d.vOrientation = vOrientation + a.vOrientation;
        return d;
    }

    template<typename real2>
    void operator*=(real2 a)
    {
        vCenter *= a;
        vOrientation *= a;
    }

    template<typename real2>
    void operator/=(real2 a)
    {
        vCenter /= a;
        vOrientation /= a;
    }



    RigidDeriv<3,real> operator-() const
    {
        return RigidDeriv(-vCenter, -vOrientation);
    }

    RigidDeriv<3,real> operator-(const RigidDeriv<3,real>& a) const
    {
        return RigidDeriv<3,real>(this->vCenter - a.vCenter, this->vOrientation-a.vOrientation);
    }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const RigidDeriv<3,real>& a) const
    {
        return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]+vCenter[2]*a.vCenter[2]
                +vOrientation[0]*a.vOrientation[0]+vOrientation[1]*a.vOrientation[1]
                +vOrientation[2]*a.vOrientation[2];
    }


    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt( vCenter*vCenter + vOrientation*vOrientation);
    }


    Vec3& getVCenter() { return vCenter; }
    Vec3& getVOrientation() { return vOrientation; }
    const Vec3& getVCenter() const { return vCenter; }
    const Vec3& getVOrientation() const { return vOrientation; }

    Vec3& getLinear() { return vCenter; }
    const Vec3& getLinear() const { return vCenter; }
    Vec3& getAngular() { return vOrientation; }
    const Vec3& getAngular() const { return vOrientation; }

    VecAll getVAll() const
    {
        return VecAll(vCenter, vOrientation);
    }

    /// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
    Vec3 velocityAtRotatedPoint(const Vec3& p) const
    {
        return vCenter - cross(p, vOrientation);
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const RigidDeriv<3,real>& v )
    {
        out<<v.vCenter<<" "<<v.vOrientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, RigidDeriv<3,real>& v )
    {
        in>>v.vCenter>>v.vOrientation;
        return in;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 6 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 3 };

    real* ptr() { return vCenter.ptr(); }
    const real* ptr() const { return vCenter.ptr(); }

    static unsigned int size() {return 6;}

    /// Access to i-th element.
    real& operator[](int i)
    {
        if (i<3)
            return this->vCenter(i);
        else
            return this->vOrientation(i-3);
    }

    /// Const access to i-th element.
    const real& operator[](int i) const
    {
        if (i<3)
            return this->vCenter(i);
        else
            return this->vOrientation(i-3);
    }

    /// @name Tests operators
    /// @{

    bool operator==(const RigidDeriv<3,real>& b) const
    {
        return vCenter == b.vCenter && vOrientation == b.vOrientation;
    }

    bool operator!=(const RigidDeriv<3,real>& b) const
    {
        return vCenter != b.vCenter || vOrientation != b.vOrientation;
    }

    /// @}

};

template<typename real, typename real2>
inline  RigidDeriv<3,real> operator*(RigidDeriv<3, real> r, real2 a)
{
    r*=a;
    return r;
}

template<typename real, typename real2>
inline RigidDeriv<3,real> operator/(RigidDeriv<3, real> r,real2 a)
{
    r/=a;
    return r;
}

template<int N,typename T>
typename RigidDeriv<N,T>::Pos& getLinear(RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<int N, typename T>
const typename RigidDeriv<N,T>::Pos& getLinear(const RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<int N, typename T>
typename RigidDeriv<N,T>::Rot& getAngular(RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<int N, typename T>
const typename RigidDeriv<N,T>::Rot& getAngular(const RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<int N,typename T>
typename RigidDeriv<N,T>::Pos& getVCenter(RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<int N, typename T>
const typename RigidDeriv<N,T>::Pos& getVCenter(const RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<int N, typename T>
typename RigidDeriv<N,T>::Rot& getVOrientation(RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<int N, typename T>
const typename RigidDeriv<N,T>::Rot& getVOrientation(const RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

/// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
template<typename T, typename R>
Vec<3,T> velocityAtRotatedPoint(const RigidDeriv<3,R>& v, const Vec<3,T>& p)
{
    return getLinear(v) + cross( getAngular(v),p );
}

template<typename real>
class RigidCoord<3,real>
{
public:
    typedef real value_type;
    typedef int size_type;
    typedef real Real;
    typedef Vec<3,Real> Pos;
    typedef helper::Quater<Real> Rot;
    typedef Vec<3,Real> Vec3;
    typedef helper::Quater<Real> Quat;
    typedef RigidDeriv<3,Real> Deriv;
    typedef Mat<4,4,Real> HomogeneousMat;
    typedef Vec<4,Real> HomogeneousVec;

protected:
    Vec3 center;
    Quat orientation;
public:
    RigidCoord (const Vec3 &posCenter, const Quat &orient)
        : center(posCenter), orientation(orient) {}
    RigidCoord () { clear(); }

    template<typename real2>
    RigidCoord(const RigidCoord<3,real2>& c)
        : center(c.getCenter()), orientation(c.getOrientation())
    {
    }


    void clear() { center.clear(); orientation.clear(); }

    /**
     * @brief Random rigid transform composed of 3 random translations and 3 random Euler angles
     * @param a Range of each random value: (-a,+a)
     * @return random rigid transform
     */
    static RigidCoord rand(SReal a)
    {
        RigidCoord t;
        t.center = Pos(  SReal(helper::drand(a)), SReal(helper::drand(a)), SReal(helper::drand(a)) );
        t.orientation = Quat::fromEuler( SReal(helper::drand(a)),SReal(helper::drand(a)), SReal(helper::drand(a)));
        return t;
    }

    template<typename real2>
    void operator=(const RigidCoord<3,real2>& c)
    {
        center = c.getCenter();
        orientation = c.getOrientation();
    }

    void operator =(const Vec3& p)
    {
        center = p;
    }

    //template<typename real2>
    //void operator =(const RigidCoord<3,real2>& c)
    //{
    //    center = c.getCenter();
    //    orientation = c.getOrientation();
    //}

    void operator +=(const Deriv& dg) {
        // R3 x SO(3) exponential integration
        center += dg.getVCenter();

        const Vec3 omega = dg.getVOrientation() / 2;
        const real theta = omega.norm();

        static const real epsilon = std::numeric_limits<real>::epsilon();

        if( theta < epsilon ) {
            // fallback to gnomonic projection
            Quat exp(omega[0], omega[1], omega[2], 1);
            exp.normalize();
            orientation = exp * orientation;
        } else {
            // expontential
            const real sinc = std::sin(theta) / theta;
            const Quat exp(sinc * omega[0],
                           sinc * omega[1],
                           sinc * omega[2],
                           std::cos(theta));
            orientation = exp * orientation;
        }

    }

    RigidCoord<3,real> operator+(const Deriv& dg) const {
        RigidCoord c = *this;
        c += dg;
        return c;
    }


    RigidCoord<3,real> operator-(const RigidCoord<3,real>& a) const
    {
        return RigidCoord<3,real>(this->center - a.getCenter(), a.orientation.inverse() * this->orientation);
    }

    RigidCoord<3,real> operator+(const RigidCoord<3,real>& a) const
    {
        return RigidCoord<3,real>(this->center + a.getCenter(), a.orientation * this->orientation);
    }

    RigidCoord<3,real> operator-() const
    {
        return RigidCoord<3,real>( -this->center, this->orientation.inverse() );
    }

    void operator +=(const RigidCoord<3,real>& a)
    {
        center += a.getCenter();
        orientation *= a.getOrientation();
    }

    template<typename real2>
    void operator*=(real2 a)
    {
        center *= a;
    }

    template<typename real2>
    void operator/=(real2 a)
    {
        center /= a;
    }

    template<typename real2>
    RigidCoord<3,real> operator*(real2 a) const
    {
        RigidCoord r = *this;
        r*=a;
        return r;
    }

    template<typename real2>
    RigidCoord<3,real> operator/(real2 a) const
    {
        RigidCoord r = *this;
        r/=a;
        return r;
    }



    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const RigidCoord<3,real>& a) const
    {
        return center[0]*a.center[0]+center[1]*a.center[1]+center[2]*a.center[2]
                +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
                +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
    }

    /** Squared norm. For the rotation we use the xyz components of the quaternion.
    Note that this is not equivalent to the angle, so a 2d rotation and the equivalent 3d rotation have different norms.
      */
    real norm2() const
    {
        return center*center
                + orientation[0]*orientation[0]
                + orientation[1]*orientation[1]
                + orientation[2]*orientation[2]; // xyzw quaternion has null x,y,z if rotation is null
    }

    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt(norm2());
    }


    Vec3& getCenter () { return center; }
    Quat& getOrientation () { return orientation; }
    const Vec3& getCenter () const { return center; }
    const Quat& getOrientation () const { return orientation; }

    static RigidCoord<3,real> identity()
    {
        RigidCoord c;
        return c;
    }

    Vec3 rotate(const Vec3& v) const
    {
        return orientation.rotate(v);
    }
    Vec3 inverseRotate(const Vec3& v) const
    {
        return orientation.inverseRotate(v);
    }

    Vec3 translate(const Vec3& v) const
    {
        return v + center;
    }

    /// Apply a transformation with respect to itself
    void multRight( const RigidCoord<3,real>& c )
    {
        center += orientation.rotate(c.getCenter());
        orientation = orientation * c.getOrientation();
    }

    /// compute the product with another frame on the right
    RigidCoord<3,real> mult( const RigidCoord<3,real>& c ) const
    {
        RigidCoord r;
        r.center = center + orientation.rotate( c.center );
        r.orientation = orientation * c.getOrientation();
        return r;
    }

    /// Set from the given matrix
    template<class Mat>
    void fromMatrix(const Mat& m)
    {
        center[0] = m[0][3];
        center[1] = m[1][3];
        center[2] = m[2][3];
        sofa::defaulttype::Mat<3,3,Real> rot; rot = m;
        orientation.fromMatrix(rot);
    }

    /// Write to the given matrix
    template<class Mat>
    void toMatrix( Mat& m) const
    {
        m.identity();
        orientation.toMatrix(m);
        m[0][3] = (typename Mat::Real)center[0];
        m[1][3] = (typename Mat::Real)center[1];
        m[2][3] = (typename Mat::Real)center[2];
    }

    /// create a homogeneous vector from a 3d vector
    template <typename V>
    static HomogeneousVec toHomogeneous( V v, real r=1. ){
        return HomogeneousVec( v[0], v[1], v[2], r );
    }
    /// create a 3d vector from a homogeneous vector
    template <typename V>
    static Vec3 fromHomogeneous( V v ){
        return Vec3( v[0], v[1], v[2] );
    }


    template<class Mat>
    void writeRotationMatrix( Mat& m) const
    {
        orientation.toMatrix(m);
    }

    /// Write the OpenGL transformation matrix
    void writeOpenGlMatrix( float m[16] ) const
    {
        orientation.writeOpenGlMatrix(m);
        m[12] = (float)center[0];
        m[13] = (float)center[1];
        m[14] = (float)center[2];
    }

    /// Apply the transform to a point, i.e. project a point from the child frame to the parent frame (translation and rotation)
    Vec3 projectPoint( const Vec3& v ) const
    {
        return orientation.rotate(v)+center;
    }

    /// Apply the transform to a vector, i.e. project a vector from the child frame to the parent frame (rotation only, no translation added)
    Vec3 projectVector( const Vec3& v ) const
    {
        return orientation.rotate(v);
    }

    /// Apply the inverse transform to a point, i.e. project a point from the parent frame to the child frame (translation and rotation)
    Vec3 unprojectPoint( const Vec3& v ) const
    {
        return orientation.inverseRotate(v-center);
    }

    ///  Apply the inverse transform to a vector, i.e. project a vector from the parent frame to the child frame (rotation only, no translation)
    Vec3 unprojectVector( const Vec3& v ) const
    {
        return orientation.inverseRotate(v);
    }

    /// obsolete. Use projectPoint.
    Vec3 pointToParent( const Vec3& v ) const { return projectPoint(v); }
    /// obsolete. Use unprojectPoint.
    Vec3 pointToChild( const Vec3& v ) const { return unprojectPoint(v); }


    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const RigidCoord<3,real>& v )
    {
        out<<v.center<<" "<<v.orientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, RigidCoord<3,real>& v )
    {
        in>>v.center>>v.orientation;
        return in;
    }
    static int max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 7 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 3 };

    real* ptr() { return center.ptr(); }
    const real* ptr() const { return center.ptr(); }

    static unsigned int size() {return 7;}

    /// Access to i-th element.
    real& operator[](int i)
    {
        if (i<3)
            return this->center(i);
        else
            return this->orientation[i-3];
    }

    /// Const access to i-th element.
    const real& operator[](int i) const
    {
        if (i<3)
            return this->center(i);
        else
            return this->orientation[i-3];
    }

    /// @name Tests operators
    /// @{

    bool operator==(const RigidCoord<3,real>& b) const
    {
        return center == b.center && orientation == b.orientation;
    }

    bool operator!=(const RigidCoord<3,real>& b) const
    {
        return center != b.center || orientation != b.orientation;
    }

    /// @}

};

template<typename real>
class RigidMass<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Mat<3,3,Real> Mat3x3;
    Real mass,volume;
    Mat3x3 inertiaMatrix;	      // Inertia matrix of the object
    Mat3x3 inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat3x3 invInertiaMatrix;	  // inverse of inertiaMatrix
    Mat3x3 invInertiaMassMatrix; // inverse of inertiaMassMatrix
    RigidMass(Real m=1)
    {
        mass = m;
        volume = 1;
        inertiaMatrix.identity();
        recalc();
    }
    void operator=(Real m)
    {
        mass = m;
        recalc();
    }
    void operator+=(Real m)
    {
        mass += m;
        recalc();
    }
    void operator-=(Real m)
    {
        mass -= m;
        recalc();
    }
    // operator to cast to const Real
    operator const Real() const
    {
        return mass;
    }
    void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        invInertiaMatrix.invert(inertiaMatrix);
        invInertiaMassMatrix.invert(inertiaMassMatrix);
    }

    inline friend std::ostream& operator << (std::ostream& out, const RigidMass<3, real>& m )
    {
        out<<m.mass;
        out<<" "<<m.volume;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, RigidMass<3, real>& m )
    {
        in>>m.mass;
        in>>m.volume;
        in>>m.inertiaMatrix;
        return in;
    }
    void operator *=(Real fact)
    {
        mass *= fact;
        inertiaMassMatrix *= fact;
        invInertiaMassMatrix /= fact;
    }
    void operator /=(Real fact)
    {
        mass /= fact;
        inertiaMassMatrix /= fact;
        invInertiaMassMatrix *= fact;
    }
};

template<typename real>
inline RigidDeriv<3,real> operator*(const RigidDeriv<3,real>& d, const RigidMass<3,real>& m)
{
    RigidDeriv<3,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
inline RigidDeriv<3,real> operator*(const RigidMass<3,real>& m, const RigidDeriv<3,real>& d)
{
    RigidDeriv<3,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
inline RigidDeriv<3,real> operator/(const RigidDeriv<3,real>& d, const RigidMass<3, real>& m)
{
    RigidDeriv<3,real> res;
    getVCenter(res) = getVCenter(d) / m.mass;
    getVOrientation(res) = m.invInertiaMassMatrix * getVOrientation(d);
    return res;
}


template<typename real>
class StdRigidTypes<3, real>
{
public:
    typedef real Real;
    typedef RigidCoord<3,real> Coord;
    typedef RigidDeriv<3,real> Deriv;
    typedef typename Coord::Vec3 Vec3;
    typedef typename Coord::Quat Quat;
    typedef Vec<3,Real> AngularVector;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef Vec<3,real> DPos;
    typedef Vec<3,real> DRot;
    static const DPos& getDPos(const Deriv& d) { return getVCenter(d); }
    static void setDPos(Deriv& d, const DPos& v) { getVCenter(d) = v; }
    static const DRot& getDRot(const Deriv& d) { return getVOrientation(d); }
    static void setDRot(Deriv& d, const DRot& v) { getVOrientation(d) = v; }

    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    typedef helper::vector<Coord> VecCoord;
    typedef helper::vector<Deriv> VecDeriv;
    typedef helper::vector<Real> VecReal;

    template<typename T>
    static void set(Coord& c, T x, T y, T z)
    {
        c.getCenter()[0] = (Real)x;
        c.getCenter()[1] = (Real)y;
        c.getCenter()[2] = (Real)z;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Coord& c)
    {
        x = (T)c.getCenter()[0];
        y = (T)c.getCenter()[1];
        z = (T)c.getCenter()[2];
    }

    // set linear and angular velocities
    template<typename T>
    static void set(Deriv& c, T x, T y, T z, T rx, T ry, T rz )
    {
        c.getLinear()[0] = (Real)x;
        c.getLinear()[1] = (Real)y;
        c.getLinear()[2] = (Real)z;
        c.getAngular()[0] = (Real)rx;
        c.getAngular()[1] = (Real)ry;
        c.getAngular()[2] = (Real)rz;
    }

    template<typename T>
    static void add(Coord& c, T x, T y, T z)
    {
        c.getCenter()[0] += (Real)x;
        c.getCenter()[1] += (Real)y;
        c.getCenter()[2] += (Real)z;
    }

    template<typename T>
    static void set(Deriv& c, T x, T y, T z)
    {
        getVCenter(c)[0] = (Real)x;
        getVCenter(c)[1] = (Real)y;
        getVCenter(c)[2] = (Real)z;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Deriv& c)
    {
        x = (T)getVCenter(c)[0];
        y = (T)getVCenter(c)[1];
        z = (T)getVCenter(c)[2];
    }

    template<typename T>
    static void add(Deriv& c, T x, T y, T z)
    {
        getVCenter(c)[0] += (Real)x;
        getVCenter(c)[1] += (Real)y;
        getVCenter(c)[2] += (Real)z;
    }

    static const char* Name();

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv( Real minMagnitude, Real maxMagnitude)
    {
        Deriv result;
        set( result, Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)),
                     Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)) );
        return result;
    }

    static Deriv coordDifference(const Coord& c1, const Coord& c2)
    {
        defaulttype::Vector3 vCenter = c1.getCenter() - c2.getCenter();
        defaulttype::Quat quat, quat1(c1.getOrientation()), quat2(c2.getOrientation());
        // Transformation between c2 and c1 frames
        quat = quat1*quat2.inverse();
        quat.normalize();
        defaulttype::Vector3 axis; defaulttype::Quat::value_type angle; quat.quatToAxis(axis, angle);
        axis*=angle;
        return Deriv(vCenter, axis);
    }

    static Coord interpolate(const helper::vector< Coord > & ancestors, const helper::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            // Position interpolation.
            c.getCenter() += ancestors[i].getCenter() * coefs[i];

            // Angle extraction from the orientation quaternion.
            helper::Quater<Real> q = ancestors[i].getOrientation();
            Real angle = acos(q[3]) * 2;

            // Axis extraction from the orientation quaternion.
            defaulttype::Vec<3,Real> v(q[0], q[1], q[2]);
            Real norm = v.norm();
            if (norm > 0.0005)
            {
                v.normalize();

                // The scale factor is applied to the angle
                angle *= coefs[i];

                // Corresponding quaternion is computed, then added to the interpolated point orientation.
                q.axisToQuat(v, angle);
                q.normalize();

                c.getOrientation() += q;
            }
        }

        c.getOrientation().normalize();

        return c;
    }

    static Deriv interpolate(const helper::vector< Deriv > & ancestors, const helper::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    /// inverse rigid transform
    static Coord inverse(const Coord& c)
    {
        CRot qinv = c.getOrientation().inverse();
        return Coord( -(qinv.rotate(c.getCenter())),qinv );
    }

    /// matrix product
    static Coord mult ( const Coord& a, const Coord& b )
    {
        return a.mult(b);
    }

    /// double cross product: a * ( b * c )
    static Vec3 crosscross ( const Vec3& a, const Vec3& b, const Vec3& c)
    {
        return cross( a, cross( b,c ));
    }

    /// create a rotation from Euler angles. For homogeneity with 2D.
    static Quat rotationEuler( Real x, Real y, Real z){ return Quat::fromEuler(x,y,z); }

};


#ifndef SOFA_FLOAT
typedef StdRigidTypes<3,double> Rigid3dTypes;
typedef RigidMass<3,double> Rigid3dMass;
#endif

typedef StdRigidTypes<3,float> Rigid3fTypes;
//#ifndef SOFA_DOUBLE
typedef RigidMass<3,float> Rigid3fMass;
//#endif

/// We now use template aliases so we do not break backward compatibility.
#ifndef SOFA_FLOAT
template<> inline const char* Rigid3dTypes::Name() { return "Rigid3d"; }
#endif
template<> inline const char* Rigid3fTypes::Name() { return "Rigid3f"; }


#ifdef SOFA_FLOAT
typedef Rigid3fTypes Rigid3Types;
typedef Rigid3fMass Rigid3Mass;
#else
typedef Rigid3dTypes Rigid3Types;
typedef Rigid3dMass Rigid3Mass;
#endif

typedef Rigid3Types RigidTypes;


//=============================================================================
// 2D Rigids
//=============================================================================

template<typename real>
class RigidDeriv<2,real>
{
public:
    typedef real value_type;
    typedef int size_type;
    typedef real Real;
    typedef Vec<2,Real> Pos;
    typedef Real Rot;
    typedef Vec<2,Real> Vec2;
    typedef Vec<3,Real> VecAll;

private:
    Vec2 vCenter;
    Real vOrientation;

public:
    friend class RigidCoord<2,real>;

    RigidDeriv()
    {
        clear();
    }

    template<typename real2>
    RigidDeriv(const Vec<2,real2> &velCenter, const real2 &velOrient)
        : vCenter(velCenter), vOrientation((Real)velOrient)
    {}

    template<typename real2>
    RigidDeriv(const Vec<3,real2> &v)
        : vCenter(Vec<2,real2>(v.data())), vOrientation((Real)v[2])
    {}

    void clear()
    {
        vCenter.clear();
        vOrientation=0;
    }

    template<typename real2>
    void operator=(const RigidDeriv<2,real2>& c)
    {
        vCenter = c.getVCenter();
        vOrientation = (Real)c.getVOrientation();
    }

    template<typename real2>
    void operator=(const Vec<2,real2>& v)
    {
        vCenter = v;
    }

    template<typename real2>
    void operator=(const Vec<3,real2>& v)
    {
        vCenter = v;
        vOrientation = (Real)v[2];
    }

    void operator+=(const RigidDeriv<2,real>& a)
    {
        vCenter += a.vCenter;
        vOrientation += a.vOrientation;
    }

    void operator-=(const RigidDeriv<2,real>& a)
    {
        vCenter -= a.vCenter;
        vOrientation -= a.vOrientation;
    }

    RigidDeriv<2,real> operator+(const RigidDeriv<2,real>& a) const
    {
        RigidDeriv<2,real> d;
        d.vCenter = vCenter + a.vCenter;
        d.vOrientation = vOrientation + a.vOrientation;
        return d;
    }

    RigidDeriv<2,real> operator-(const RigidDeriv<2,real>& a) const
    {
        RigidDeriv<2,real> d;
        d.vCenter = vCenter - a.vCenter;
        d.vOrientation = vOrientation - a.vOrientation;
        return d;
    }

    template<typename real2>
    void operator*=(real2 a)
    {
        vCenter *= a;
        vOrientation *= (Real)a;
    }

    template<typename real2>
    void operator/=(real2 a)
    {
        vCenter /= a;
        vOrientation /= (Real)a;
    }



    RigidDeriv<2,real> operator-() const
    {
        return RigidDeriv<2,real>(-vCenter, -vOrientation);
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const RigidDeriv<2,real>& a) const
    {
        return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]
                +vOrientation*a.vOrientation;
    }

    /// Euclidean norm
    Real norm() const
    {
        return helper::rsqrt( vCenter*vCenter + vOrientation*vOrientation);
    }

    Vec2& getVCenter() { return vCenter; }
    Real& getVOrientation() { return vOrientation; }
    const Vec2& getVCenter() const { return vCenter; }
    const Real& getVOrientation() const { return vOrientation; }

    Vec2& getLinear() { return vCenter; }
    Real& getAngular() { return vOrientation; }
    const Vec2& getLinear() const { return vCenter; }
    const Real& getAngular() const { return vOrientation; }

    VecAll getVAll() const
    {
        return VecAll(vCenter, vOrientation);
    }

    /// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
    Vec2 velocityAtRotatedPoint(const Vec2& p) const
    {
        return vCenter + Vec2(-p[1], p[0]) * vOrientation;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const RigidDeriv<2,real>& v )
    {
        out<<v.vCenter<<" "<<v.vOrientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, RigidDeriv<2,real>& v )
    {
        in>>v.vCenter>>v.vOrientation;
        return in;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 3 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 2 };

    real* ptr() { return vCenter.ptr(); }
    const real* ptr() const { return vCenter.ptr(); }

    static unsigned int size() {return 3;}

    /// Access to i-th element.
    real& operator[](int i)
    {
        if (i<2)
            return this->vCenter(i);
        else
            return this->vOrientation;
    }

    /// Const access to i-th element.
    const real& operator[](int i) const
    {
        if (i<2)
            return this->vCenter(i);
        else
            return this->vOrientation;
    }


    /// @name Tests operators
    /// @{

    bool operator==(const RigidDeriv<2,real>& b) const
    {
        return vCenter == b.vCenter && vOrientation == b.vOrientation;
    }

    bool operator!=(const RigidDeriv<2,real>& b) const
    {
        return vCenter != b.vCenter || vOrientation != b.vOrientation;
    }

    /// @}
};

template<typename real, typename real2>
inline RigidDeriv<2,real> operator*(RigidDeriv<2,real> r, real2 a)
{
    r *= a;
    return r;
}

template<typename real, typename real2>
inline RigidDeriv<2,real> operator/(RigidDeriv<2,real> r, real2 a)
{
    r /= a;
    return r;
}

/// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
template<typename R, typename T>
Vec<2,R> velocityAtRotatedPoint(const RigidDeriv<2,T>& v, const Vec<2,R>& p)
{
    return getVCenter(v) + Vec<2,R>(-p[1], p[0]) * getVOrientation(v);
}



template<typename real>
class RigidCoord<2,real>
{
public:
    typedef real value_type;
    typedef int size_type;
    typedef real Real;
    typedef Vec<2,Real> Pos;
    typedef Real Rot;
    typedef Vec<2,Real> Vec2;
    typedef Mat<3,3,Real> HomogeneousMat;
    typedef Vec<3,Real> HomogeneousVec;
private:
    Vec2 center;
    Real orientation;
public:
    RigidCoord (const Vec2 &posCenter, const Real &orient)
        : center(posCenter), orientation(orient) {}
    RigidCoord () { clear(); }

    void clear() { center.clear(); orientation = 0; }

    /**
     * @brief Random rigid transform composed of 2 random translations and a random angle
     * @param a Range of each random value: (-a,+a)
     * @return random rigid transform
     */
    static RigidCoord rand(SReal a)
    {
        RigidCoord t;
        t.center = Pos( SReal(helper::drand(a)), SReal(helper::drand(a)));
        t.orientation = SReal(helper::drand(a));
        return t;
    }

    void operator +=(const RigidDeriv<2,real>& a)
    {
        center += getVCenter(a);
        orientation += getVOrientation(a);
    }

    RigidCoord<2,real> operator + (const RigidDeriv<2,real>& a) const
    {
        RigidCoord<2,real> c = *this;
        c.center += getVCenter(a);
        c.orientation += getVOrientation(a);
        return c;
    }

    RigidCoord<2,real> operator -(const RigidCoord<2,real>& a) const
    {
        return RigidCoord<2,real>(this->center - a.getCenter(), this->orientation - a.orientation);
    }

    RigidCoord<2,real> operator +(const RigidCoord<2,real>& a) const
    {
        return RigidCoord<2,real>(this->center + a.getCenter(), this->orientation + a.orientation);
    }


    RigidCoord<2,real> operator-() const
    {
        return RigidCoord<2,real>( -this->center, this->orientation.inverse() );
    }


    void operator +=(const RigidCoord<2,real>& a)
    {
        center += a.getCenter();
        orientation += a.getOrientation();
    }

    template<typename real2>
    void operator*=(real2 a)
    {
        center *= a;
        orientation *= (Real)a;
    }

    template<typename real2>
    void operator/=(real2 a)
    {
        center /= a;
        orientation /= (Real)a;
    }

    template<typename real2>
    RigidCoord<2,real> operator*(real2 a) const
    {
        RigidCoord<2,real> r = *this;
        r *= a;
        return r;
    }

    template<typename real2>
    RigidCoord<2,real> operator/(real2 a) const
    {
        RigidCoord<2,real> r = *this;
        r /= a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const RigidCoord<2,real>& a) const
    {
        return center[0]*a.center[0]+center[1]*a.center[1]
                +orientation*a.orientation;
    }

    /// Squared norm
    Real norm2() const
    {
        Real angle = fmod( (Real) orientation, (Real) M_PI*2 );
        return center*center + angle*angle;
    }

    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt(norm2());
    }

    Vec2& getCenter () { return center; }
    Real& getOrientation () { return orientation; }
    const Vec2& getCenter () const { return center; }
    const Real& getOrientation () const { return orientation; }

    Vec2 rotate(const Vec2& v) const
    {
        Real s = sin(orientation);
        Real c = cos(orientation);
        return Vec2(c*v[0]-s*v[1],
                s*v[0]+c*v[1]);
    }
    Vec2 inverseRotate(const Vec2& v) const
    {
        Real s = sin(-orientation);
        Real c = cos(-orientation);
        return Vec2(c*v[0]-s*v[1],
                s*v[0]+c*v[1]);
    }

    Vec2 translate(const Vec2& v) const
    {
        return v + center;
    }

    static RigidCoord<2,real> identity()
    {
        RigidCoord<2,real> c;
        return c;
    }

    /// Apply a transformation with respect to itself
    void multRight( const RigidCoord<2,real>& c )
    {
        center += /*orientation.*/rotate(c.getCenter());
        orientation = orientation + c.getOrientation();
    }

    /// compute the product with another frame on the right
    RigidCoord<2,real> mult( const RigidCoord<2,real>& c ) const
    {
        RigidCoord<2,real> r;
        r.center = center + /*orientation.*/rotate( c.center );
        r.orientation = orientation + c.getOrientation();
        return r;
    }

    template<class Mat>
    void writeRotationMatrix( Mat& m) const
    {
        m[0][0] = (typename Mat::Real)cos(orientation); m[0][1] = (typename Mat::Real)-sin(orientation);
        m[1][0] = (typename Mat::Real)sin(orientation); m[1][1] = (typename Mat::Real) cos(orientation);
    }

    /// Set from the given matrix
    template<class Mat>
    void fromMatrix(const Mat& m)
    {
        center[0] = m[0][2];
        center[1] = m[1][2];
        orientation = atan2(m[1][0],m[0][0]);
    }

    /// Write to the given matrix
    template<class Mat>
    void toMatrix( Mat& m) const
    {
        m.identity();
        writeRotationMatrix( m );
        m[0][2] = center[0];
        m[1][2] = center[1];
    }


    /// create a homogeneous vector from a 2d vector
    template <typename V>
    static HomogeneousVec toHomogeneous( V v, real r=1. ){
        return HomogeneousVec( v[0], v[1], r );
    }
    /// create a 2d vector from a homogeneous vector
    template <typename V>
    static Vec2 fromHomogeneous( V v ){
        return Vec2( v[0], v[1] );
    }


    /// Write the OpenGL transformation matrix
    void writeOpenGlMatrix( float m[16] ) const
    {
        //orientation.writeOpenGlMatrix(m);
        m[0] = cos(orientation);
        m[1] = sin(orientation);
        m[2] = 0;
        m[3] = 0;
        m[4] = -sin(orientation);
        m[5] = cos(orientation);
        m[6] = 0;
        m[7] = 0;
        m[8] = 0;
        m[9] = 0;
        m[10] = 1;
        m[11] = 0;
        m[12] = (float)center[0];
        m[13] = (float)center[1];
        m[14] = (float)center[2];
        m[15] = 1;
    }

    /// compute the projection of a vector from the parent frame to the child
    Vec2 vectorToChild( const Vec2& v ) const
    {
        return /*orientation.*/inverseRotate(v);
    }

    /// Apply the transform to a point, i.e. project a point from the child frame to the parent frame (translation and rotation)
    Vec2 projectPoint( const Vec2& v ) const
    {
        return rotate(v)+center;
    }

    /// Apply the transform to a vector, i.e. project a vector from the child frame to the parent frame (rotation only, no translation added)
    Vec2 projectVector( const Vec2& v ) const
    {
        return rotate(v);
    }

    /// Apply the inverse transform to a point, i.e. project a point from the parent frame to the child frame (translation and rotation)
    Vec2 unprojectPoint( const Vec2& v ) const
    {
        return inverseRotate(v-center);
    }

    ///  Apply the inverse transform to a vector, i.e. project a vector from the parent frame to the child frame (rotation only, no translation)
    Vec2 unprojectVector( const Vec2& v ) const
    {
        return inverseRotate(v);
    }

    /// obsolete. Use projectPoint.
    Vec2 pointToParent( const Vec2& v ) const { return projectPoint(v); }
    /// obsolete. Use unprojectPoint.
    Vec2 pointToChild( const Vec2& v ) const { return unprojectPoint(v); }


    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const RigidCoord<2,real>& v )
    {
        out<<v.center<<" "<<v.orientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, RigidCoord<2,real>& v )
    {
        in>>v.center>>v.orientation;
        return in;
    }
    static int max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 3 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 2 };

    real* ptr() { return center.ptr(); }
    const real* ptr() const { return center.ptr(); }

    static unsigned int size() {return 3;}

    /// Access to i-th element.
    real& operator[](int i)
    {
        if (i<2)
            return this->center(i);
        else
            return this->orientation;
    }

    /// Const access to i-th element.
    const real& operator[](int i) const
    {
        if (i<2)
            return this->center(i);
        else
            return this->orientation;
    }

    /// @name Tests operators
    /// @{

    bool operator==(const RigidCoord<2,real>& b) const
    {
        return center == b.center && orientation == b.orientation;
    }

    bool operator!=(const RigidCoord<2,real>& b) const
    {
        return center != b.center || orientation != b.orientation;
    }

    /// @}
};

template<class real>
class RigidMass<2, real>
{
public:
    typedef real value_type;
    typedef real Real;
    Real mass,volume;
    Real inertiaMatrix;	      // Inertia matrix of the object
    Real inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Real invInertiaMatrix;	  // inverse of inertiaMatrix
    Real invInertiaMassMatrix; // inverse of inertiaMassMatrix
    RigidMass(Real m=1)
    {
        mass = m;
        volume = 1;
        inertiaMatrix = 1;
        recalc();
    }
    void operator=(Real m)
    {
        mass = m;
        recalc();
    }
    void operator+=(Real m)
    {
        mass += m;
        recalc();
    }
    void operator-=(Real m)
    {
        mass -= m;
        recalc();
    }
    // operator to cast to const Real
    operator const Real() const
    {
        return mass;
    }
    /// Mass for a circle
    RigidMass(Real m, Real radius)
    {
        mass = m;
        volume = radius*radius*R_PI;
        inertiaMatrix = (radius*radius)/2;
        recalc();
    }
    /// Mass for a rectangle
    RigidMass(Real m, Real xwidth, Real ywidth)
    {
        mass = m;
        volume = xwidth*xwidth + ywidth*ywidth;
        inertiaMatrix = volume/12;
        recalc();
    }

    void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        invInertiaMatrix = 1/(inertiaMatrix);
        invInertiaMassMatrix = 1/(inertiaMassMatrix);
    }
    inline friend std::ostream& operator << (std::ostream& out, const RigidMass<2,Real>& m )
    {
        out<<m.mass;
        out<<" "<<m.volume;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, RigidMass<2,Real>& m )
    {
        in>>m.mass;
        in>>m.volume;
        in>>m.inertiaMatrix;
        return in;
    }
    void operator *=(Real fact)
    {
        mass *= fact;
        inertiaMassMatrix *= fact;
        invInertiaMassMatrix /= fact;
    }
    void operator /=(Real fact)
    {
        mass /= fact;
        inertiaMassMatrix /= fact;
        invInertiaMassMatrix *= fact;
    }
};

template<typename real>
inline RigidDeriv<2,real> operator*(const RigidDeriv<2,real>& d, const RigidMass<2,real>& m)
{
    RigidDeriv<2,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
inline RigidDeriv<2,real> operator/(const RigidDeriv<2,real>& d, const RigidMass<2, real>& m)
{
    RigidDeriv<2,real> res;
    getVCenter(res) = getVCenter(d) / m.mass;
    getVOrientation(res) = m.invInertiaMassMatrix * getVOrientation(d);
    return res;
}


/** Degrees of freedom of 2D rigid bodies.
*/
template<typename real>
class StdRigidTypes<2, real>
{
public:
    typedef real Real;
    typedef Vec<2,real> Vec2;

    typedef RigidDeriv<2,Real> Deriv;
    typedef RigidCoord<2,Real> Coord;
    typedef Real AngularVector;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef Vec<2,real> DPos;
    typedef real DRot;
    static const DPos& getDPos(const Deriv& d) { return getVCenter(d); }
    static void setDPos(Deriv& d, const DPos& v) { getVCenter(d) = v; }
    static const DRot& getDRot(const Deriv& d) { return getVOrientation(d); }
    static void setDRot(Deriv& d, const DRot& v) { getVOrientation(d) = v; }

    static const char* Name();

    typedef helper::vector<Coord> VecCoord;
    typedef helper::vector<Deriv> VecDeriv;
    typedef helper::vector<Real> VecReal;

    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    template<typename T>
    static void set(Coord& c, T x, T y, T)
    {
        c.getCenter()[0] = (Real)x;
        c.getCenter()[1] = (Real)y;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Coord& c)
    {
        x = (T)c.getCenter()[0];
        y = (T)c.getCenter()[1];
        z = (T)0;
    }

    template<typename T>
    static void add(Coord& c, T x, T y, T)
    {
        c.getCenter()[0] += (Real)x;
        c.getCenter()[1] += (Real)y;
    }

    template<typename T>
    static void set(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] = (Real)x;
        c.getVCenter()[1] = (Real)y;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Deriv& c)
    {
        x = (T)c.getVCenter()[0];
        y = (T)c.getVCenter()[1];
        z = (T)0;
    }

    // Set linear and angular velocities, in 6D for uniformity with 3D
    template<typename T>
    static void set(Deriv& c, T x, T y, T, T vrot, T, T )
    {
        c.getVCenter()[0] = (Real)x;
        c.getVCenter()[1] = (Real)y;
        c.getVOrientation() = (Real) vrot;
    }

    template<typename T>
    static void add(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] += (Real)x;
        c.getVCenter()[1] += (Real)y;
    }

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv( Real minMagnitude, Real maxMagnitude )
    {
        Deriv result;
        set( result, Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)),
                     Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)));
        return result;
    }

    static Coord interpolate(const helper::vector< Coord > & ancestors, const helper::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            c += ancestors[i] * coefs[i];
        }

        return c;
    }

    static Deriv interpolate(const helper::vector< Deriv > & ancestors, const helper::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (unsigned int i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    /// specialized version of the double cross product: a * ( b * c ) for the variation of torque applied to the frame due to a small rotation with constant force.
    static Real crosscross ( const Vec2& f, const Real& dtheta, const Vec2& OP)
    {
        return dtheta * dot( f,OP );
    }

    /// specialized version of the double cross product: a * ( b * c ) for point acceleration
    static Vec2 crosscross ( const Real& omega, const Real& dtheta, const Vec2& OP)
    {
        return OP * omega * (-dtheta);
    }

    /// create a rotation from Euler angles (only the first is used). For homogeneity with 3D.
    static CRot rotationEuler( Real x, Real , Real ){ return CRot(x); }


};



#ifndef SOFA_FLOAT
typedef StdRigidTypes<2,double> Rigid2dTypes;
typedef RigidMass<2,double> Rigid2dMass;
template<> inline const char* Rigid2dTypes::Name() { return "Rigid2d"; }
#endif
#ifndef SOFA_DOUBLE
typedef StdRigidTypes<2,float> Rigid2fTypes;
typedef RigidMass<2,float> Rigid2fMass;
template<> inline const char* Rigid2fTypes::Name() { return "Rigid2f"; }
#endif


#ifdef SOFA_FLOAT
typedef Rigid2fTypes Rigid2Types;
typedef Rigid2fMass Rigid2Mass;
#else
typedef Rigid2dTypes Rigid2Types;
typedef Rigid2dMass Rigid2Mass;
#endif



// Specialization of the defaulttype::DataTypeInfo type traits template

template<int N, typename real>
struct DataTypeInfo< sofa::defaulttype::RigidDeriv<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::RigidDeriv<N,real>, sofa::defaulttype::RigidDeriv<N,real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "RigidDeriv<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

template<int N, typename real>
struct DataTypeInfo< sofa::defaulttype::RigidCoord<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::RigidCoord<N,real>, sofa::defaulttype::RigidCoord<N,real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "RigidCoord<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES


#ifndef SOFA_FLOAT
template<> struct DataTypeName< defaulttype::Rigid2dTypes::Coord > { static const char* name() { return "Rigid2dTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Rigid2dTypes::Deriv > { static const char* name() { return "Rigid2dTypes::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid3dTypes::Coord > { static const char* name() { return "Rigid3dTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Rigid3dTypes::Deriv > { static const char* name() { return "Rigid3dTypes::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid2dMass > { static const char* name() { return "Rigid2dMass"; } };
template<> struct DataTypeName< defaulttype::Rigid3dMass > { static const char* name() { return "Rigid3dMass"; } };
#endif
#ifndef SOFA_DOUBLE
template<> struct DataTypeName< defaulttype::Rigid2fTypes::Coord > { static const char* name() { return "Rigid2fTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Rigid2fTypes::Deriv > { static const char* name() { return "Rigid2fTypes::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid3fTypes::Coord > { static const char* name() { return "Rigid3fTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Rigid3fTypes::Deriv > { static const char* name() { return "Rigid3fTypes::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid2fMass > { static const char* name() { return "Rigid2fMass"; } };
template<> struct DataTypeName< defaulttype::Rigid3fMass > { static const char* name() { return "Rigid3fMass"; } };
#endif



/// \endcond



/** @name Helpers
 *  Helper Functions to more easily create tests and check the results.
 */
//@{

/** Velocity of a rigid body at a given point, based on its angular velocity and its linear velocity at another point.
  \param omega angular velocity
  \param v known linear velocity
  \param pv point where the linear velocity is known
  \param p point where we compute the velocity
  */
template <class Vec3>
static Vec3 rigidVelocity( const Vec3& omega, const Vec3& v, const Vec3& pv, const Vec3& p ) { return v + cross( omega, p-pv ); }

/// Apply the given translation and rotation to each entry of vector v
template<class V1, class Vec, class Rot>
static void displace( V1& v, Vec translation, Rot rotation )
{
    for(std::size_t i=0; i<v.size(); i++)
        v[i] = translation + rotation.rotate(v[i]);
}

/// Apply the given translation and rotation to each entry of vector v
template<class V1, class Rot>
static void rotate( V1& v, Rot rotation )
{
    for(std::size_t i=0; i<v.size(); i++)
        v[i] = rotation.rotate(v[i]);
}

/// Apply a rigid transformation (translation, Euler angles) to the given points and their associated velocities.
template<class V1, class V2>
static void rigidTransform ( V1& points, V2& velocities, SReal tx, SReal ty, SReal tz, SReal rx, SReal ry, SReal rz )
{
    typedef defaulttype::Vec<3,SReal> Vec3;
    typedef helper::Quater<SReal> Quat;
    Vec3 translation(tx,ty,tz);
    Quat rotation = Quat::createQuaterFromEuler(Vec3(rx,ry,rz));
    displace(points,translation,rotation);
    rotate(velocities,rotation);
}
//@}


}

} // namespace sofa


#endif
