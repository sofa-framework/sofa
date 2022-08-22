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

#include <sofa/defaulttype/fwd.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>
#include <sofa/type/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/random.h>
#include <istream>
#include <ostream>
#include <cstdlib>
#include <cmath>

#if !defined(NDEBUG)
#include <sofa/helper/logging/Messaging.h>
#endif

namespace sofa::defaulttype
{

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
    typedef sofa::Size Size;
    typedef real Real;
    typedef type::Vec<3,Real> Pos;
    typedef type::Vec<3,Real> Rot;
    typedef type::Vec<3,Real> Vec3;
    typedef type::Vec<6,Real> VecAll;
    typedef type::Quat<Real> Quat;

protected:
    Vec3 vCenter;
    Vec3 vOrientation;

public:
    friend class RigidCoord<3,real>;

    constexpr RigidDeriv()
    {
        clear();
    }

    explicit constexpr RigidDeriv(type::NoInit)
        : vCenter(type::NOINIT)
        , vOrientation(type::NOINIT)
    {

    }

    constexpr RigidDeriv(const Vec3 &velCenter, const Vec3 &velOrient)
        : vCenter(velCenter), vOrientation(velOrient)
    {}

    template<typename real2>
    constexpr RigidDeriv(const RigidDeriv<3,real2>& c)
        : vCenter(c.getVCenter()), vOrientation(c.getVOrientation())
    {}

    template<typename real2>
    constexpr RigidDeriv(const type::Vec<6,real2> &v)
        : vCenter(type::Vec<3,real2>(v.data())), vOrientation(type::Vec<3,real2>(v.data()+3))
    {}

    template<typename real2>
    constexpr RigidDeriv(const real2* ptr)
        :vCenter(ptr),vOrientation(ptr+3)
    {
    }

    constexpr void clear()
    {
        vCenter.clear();
        vOrientation.clear();
    }

    template<typename real2>
    constexpr void operator=(const RigidDeriv<3,real2>& c)
    {
        vCenter = c.getVCenter();
        vOrientation = c.getVOrientation();
    }

    template<typename real2>
    constexpr void operator=(const type::Vec<3,real2>& v)
    {
        vCenter = v;
    }

    template<typename real2>
    constexpr void operator=(const type::Vec<6,real2>& v)
    {
        vCenter = v;
        vOrientation = type::Vec<3,real2>(v.data()+3);
    }

    constexpr void operator+=(const RigidDeriv& a)
    {
        vCenter += a.vCenter;
        vOrientation += a.vOrientation;
    }

    constexpr void operator-=(const RigidDeriv& a)
    {
        vCenter -= a.vCenter;
        vOrientation -= a.vOrientation;
    }

    constexpr RigidDeriv<3,real> operator+(const RigidDeriv<3,real>& a) const
    {
        RigidDeriv d;
        d.vCenter = vCenter + a.vCenter;
        d.vOrientation = vOrientation + a.vOrientation;
        return d;
    }

    template<typename real2>
    constexpr void operator*=(real2 a)
    {
        vCenter *= a;
        vOrientation *= a;
    }

    template<typename real2>
    constexpr void operator/=(real2 a)
    {
        vCenter /= a;
        vOrientation /= a;
    }



    constexpr RigidDeriv<3,real> operator-() const
    {
        return RigidDeriv(-vCenter, -vOrientation);
    }

    constexpr RigidDeriv<3,real> operator-(const RigidDeriv<3,real>& a) const
    {
        return RigidDeriv<3,real>(this->vCenter - a.vCenter, this->vOrientation-a.vOrientation);
    }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    constexpr Real operator*(const RigidDeriv<3,real>& a) const
    {
        return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]+vCenter[2]*a.vCenter[2]
                +vOrientation[0]*a.vOrientation[0]+vOrientation[1]*a.vOrientation[1]
                +vOrientation[2]*a.vOrientation[2];
    }


    /// Euclidean norm
    constexpr real norm() const
    {
        return helper::rsqrt( vCenter*vCenter + vOrientation*vOrientation);
    }


    constexpr Vec3& getVCenter() { return vCenter; }
    constexpr Vec3& getVOrientation() { return vOrientation; }
    constexpr const Vec3& getVCenter() const { return vCenter; }
    constexpr const Vec3& getVOrientation() const { return vOrientation; }

    constexpr Vec3& getLinear() { return vCenter; }
    constexpr  const Vec3& getLinear() const { return vCenter; }
    constexpr Vec3& getAngular() { return vOrientation; }
    constexpr const Vec3& getAngular() const { return vOrientation; }

    constexpr VecAll getVAll() const
    {
        return VecAll(vCenter, vOrientation);
    }

    /// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
    constexpr Vec3 velocityAtRotatedPoint(const Vec3& p) const
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
    static constexpr sofa::Size total_size = 6;

    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    static constexpr sofa::Size spatial_dimensions = 3;

    constexpr real* ptr() { return vCenter.ptr(); }
    constexpr const real* ptr() const { return vCenter.ptr(); }

    static constexpr Size size() {return 6;}

    /// Access to i-th element.
    constexpr real& operator[](Size i)
    {
        if (i<3)
            return this->vCenter(i);
        else
            return this->vOrientation(i-3);
    }

    /// Const access to i-th element.
    constexpr const real& operator[](Size i) const
    {
        if (i<3)
            return this->vCenter(i);
        else
            return this->vOrientation(i-3);
    }

    /// @name Tests operators
    /// @{

    constexpr bool operator==(const RigidDeriv<3,real>& b) const
    {
        return vCenter == b.vCenter && vOrientation == b.vOrientation;
    }

    constexpr bool operator!=(const RigidDeriv<3,real>& b) const
    {
        return vCenter != b.vCenter || vOrientation != b.vOrientation;
    }

    /// @}

};

template<typename real, typename real2>
RigidDeriv<3,real> operator*(RigidDeriv<3, real> r, real2 a)
{
    r*=a;
    return r;
}

template<typename real, typename real2>
RigidDeriv<3,real> operator/(RigidDeriv<3, real> r,real2 a)
{
    r/=a;
    return r;
}

template<sofa::Size N,typename T>
typename RigidDeriv<N,T>::Pos& getLinear(RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<sofa::Size N, typename T>
const typename RigidDeriv<N,T>::Pos& getLinear(const RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<sofa::Size N, typename T>
typename RigidDeriv<N,T>::Rot& getAngular(RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<sofa::Size N, typename T>
const typename RigidDeriv<N,T>::Rot& getAngular(const RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<sofa::Size N,typename T>
typename RigidDeriv<N,T>::Pos& getVCenter(RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<sofa::Size N, typename T>
const typename RigidDeriv<N,T>::Pos& getVCenter(const RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<sofa::Size N, typename T>
typename RigidDeriv<N,T>::Rot& getVOrientation(RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<sofa::Size N, typename T>
const typename RigidDeriv<N,T>::Rot& getVOrientation(const RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

/// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
template<typename T, typename R>
constexpr type::Vec<3,T> velocityAtRotatedPoint(const RigidDeriv<3,R>& v, const type::Vec<3,T>& p)
{
    return getLinear(v) + cross( getAngular(v),p );
}

template<typename real>
class RigidCoord<3,real>
{
public:
    typedef real value_type;
    typedef sofa::Size Size;
    typedef real Real;
    typedef type::Vec<3,Real> Pos;
    typedef type::Quat<Real> Rot;
    typedef type::Vec<3,Real> Vec3;
    typedef type::Quat<Real> Quat;
    typedef RigidDeriv<3,Real> Deriv;
    typedef type::Mat<4,4,Real> HomogeneousMat;
    typedef type::Vec<4,Real> HomogeneousVec;

protected:
    Vec3 center;
    Quat orientation;
public:
    constexpr RigidCoord(const Vec3& posCenter, const Quat& orient)
        : center(posCenter), orientation(orient)
    {
    }

    constexpr RigidCoord () 
    { 
        clear(); 
    }

    template<typename real2>
    constexpr RigidCoord(const RigidCoord<3,real2>& c)
        : center(c.getCenter()), orientation(c.getOrientation())
    {
    }


    constexpr void clear()
    { 
        center.clear(); 
        orientation.clear(); 
    }

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
    constexpr void operator=(const RigidCoord<3,real2>& c)
    {
        center = c.getCenter();
        orientation = c.getOrientation();
    }

    constexpr void operator =(const Vec3& p)
    {
        center = p;
    }

    constexpr void operator +=(const Deriv& dg) {
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

    constexpr RigidCoord<3,real> operator+(const Deriv& dg) const {
        RigidCoord c = *this;
        c += dg;
        return c;
    }


    constexpr RigidCoord<3,real> operator-(const RigidCoord<3,real>& a) const
    {
        return RigidCoord<3,real>(this->center - a.getCenter(), a.orientation.inverse() * this->orientation);
    }

    constexpr RigidCoord<3,real> operator+(const RigidCoord<3,real>& a) const
    {
        return RigidCoord<3,real>(this->center + a.getCenter(), a.orientation * this->orientation);
    }

    constexpr RigidCoord<3,real> operator-() const
    {
        return RigidCoord<3,real>( -this->center, this->orientation.inverse() );
    }

    constexpr void operator +=(const RigidCoord<3,real>& a)
    {
        center += a.getCenter();
        orientation *= a.getOrientation();
    }

    template<typename real2>
    constexpr void operator*=(real2 a)
    {
        center *= a;
    }

    template<typename real2>
    constexpr void operator/=(real2 a)
    {
        center /= a;
    }

    template<typename real2>
    constexpr RigidCoord<3,real> operator*(real2 a) const
    {
        RigidCoord r = *this;
        r*=a;
        return r;
    }

    template<typename real2>
    constexpr RigidCoord<3,real> operator/(real2 a) const
    {
        RigidCoord r = *this;
        r/=a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    constexpr Real operator*(const RigidCoord<3,real>& a) const
    {
        return center[0]*a.center[0]+center[1]*a.center[1]+center[2]*a.center[2]
                +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
                +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
    }

    /** Squared norm. For the rotation we use the xyz components of the quaternion.
    Note that this is not equivalent to the angle, so a 2d rotation and the equivalent 3d rotation have different norms.
      */
    constexpr real norm2() const
    {
        return center*center
                + orientation[0]*orientation[0]
                + orientation[1]*orientation[1]
                + orientation[2]*orientation[2]; // xyzw quaternion has null x,y,z if rotation is null
    }

    /// Euclidean norm
    constexpr real norm() const
    {
        return helper::rsqrt(norm2());
    }


    constexpr Vec3& getCenter () { return center; }
    constexpr Quat& getOrientation () { return orientation; }
    constexpr const Vec3& getCenter () const { return center; }
    constexpr const Quat& getOrientation () const { return orientation; }

    static constexpr RigidCoord<3,real> identity()
    {
        RigidCoord c;
        return c;
    }

    constexpr Vec3 rotate(const Vec3& v) const
    {
        return orientation.rotate(v);
    }
    
    constexpr Vec3 inverseRotate(const Vec3& v) const
    {
        return orientation.inverseRotate(v);
    }

    constexpr Vec3 translate(const Vec3& v) const
    {
        return v + center;
    }

    /// Apply a transformation with respect to itself
    constexpr void multRight( const RigidCoord<3,real>& c )
    {
        center += orientation.rotate(c.getCenter());
        orientation = orientation * c.getOrientation();
    }

    /// compute the product with another frame on the right
    constexpr RigidCoord<3,real> mult( const RigidCoord<3,real>& c ) const
    {
        RigidCoord r;
        r.center = center + orientation.rotate( c.center );
        r.orientation = orientation * c.getOrientation();
        return r;
    }

    /// Set from the given matrix
    template<class Mat>
    constexpr void fromMatrix(const Mat& m)
    {
        center[0] = m[0][3];
        center[1] = m[1][3];
        center[2] = m[2][3];
        sofa::type::Mat<3,3,Real> rot; rot = m;
        orientation.fromMatrix(rot);
    }

    /// Write to the given matrix
    template<class Mat>
    constexpr void toMatrix( Mat& m) const
    {
        m.identity();
        orientation.toMatrix(m);
        m[0][3] = (typename Mat::Real)center[0];
        m[1][3] = (typename Mat::Real)center[1];
        m[2][3] = (typename Mat::Real)center[2];
    }

    constexpr void toHomogeneousMatrix( HomogeneousMat& m) const
    {
        m.identity();
        orientation.toHomogeneousMatrix(m);
        m[0][3] = center[0];
        m[1][3] = center[1];
        m[2][3] = center[2];
    }

    /// create a homogeneous vector from a 3d vector
    template <typename V>
    static constexpr HomogeneousVec toHomogeneous( V v, real r=1. ){
        return HomogeneousVec( v[0], v[1], v[2], r );
    }
    /// create a 3d vector from a homogeneous vector
    template <typename V>
    static constexpr Vec3 fromHomogeneous( V v ){
        return Vec3( v[0], v[1], v[2] );
    }


    template<class Mat>
    constexpr void writeRotationMatrix( Mat& m) const
    {
        orientation.toMatrix(m);
    }

    /// Write the OpenGL transformation matrix
    constexpr void writeOpenGlMatrix( float m[16] ) const
    {
        orientation.writeOpenGlMatrix(m);
        m[12] = (float)center[0];
        m[13] = (float)center[1];
        m[14] = (float)center[2];
    }

    /// Apply the transform to a point, i.e. project a point from the child frame to the parent frame (translation and rotation)
    constexpr Vec3 projectPoint( const Vec3& v ) const
    {
        return orientation.rotate(v)+center;
    }

    /// Apply the transform to a vector, i.e. project a vector from the child frame to the parent frame (rotation only, no translation added)
    constexpr Vec3 projectVector( const Vec3& v ) const
    {
        return orientation.rotate(v);
    }

    /// Apply the inverse transform to a point, i.e. project a point from the parent frame to the child frame (translation and rotation)
    constexpr Vec3 unprojectPoint( const Vec3& v ) const
    {
        return orientation.inverseRotate(v-center);
    }

    ///  Apply the inverse transform to a vector, i.e. project a vector from the parent frame to the child frame (rotation only, no translation)
    constexpr Vec3 unprojectVector( const Vec3& v ) const
    {
        return orientation.inverseRotate(v);
    }

    /// obsolete. Use projectPoint.
    constexpr Vec3 pointToParent( const Vec3& v ) const { return projectPoint(v); }
    /// obsolete. Use unprojectPoint.
    constexpr Vec3 pointToChild( const Vec3& v ) const { return unprojectPoint(v); }


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
#if !defined(NDEBUG)
        if (!v.orientation.isNormalized())
        {
            std::stringstream text;
            text << "Rigid Object with invalid quaternion (non-unitary norm)! Normalising quaternion value... " << msgendl;
            text << "Previous value was: " << v.orientation << msgendl ;
            v.orientation.normalize();
            text << "New value is: " << v.orientation;
            msg_warning("Rigid") << text.str();
        }
#endif // NDEBUG
        return in;
    }
    static constexpr Size max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr sofa::Size total_size = 7;
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    static constexpr sofa::Size spatial_dimensions = 3;

    constexpr real* ptr() { return center.ptr(); }
    constexpr const real* ptr() const { return center.ptr(); }

    static constexpr Size size() {return 7;}

    /// Access to i-th element.
    constexpr real& operator[](Size i)
    {
        if (i<3)
            return this->center(i);
        else
            return this->orientation[i-3];
    }

    /// Const access to i-th element.
    constexpr const real& operator[](Size i) const
    {
        if (i<3)
            return this->center(i);
        else
            return this->orientation[i-3];
    }

    /// @name Tests operators
    /// @{

    constexpr bool operator==(const RigidCoord<3,real>& b) const
    {
        return center == b.center && orientation == b.orientation;
    }

    constexpr bool operator!=(const RigidCoord<3,real>& b) const
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
    typedef type::Mat<3,3,Real> Mat3x3;
    Real mass,volume;
    Mat3x3 inertiaMatrix;	      // Inertia matrix of the object
    Mat3x3 inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat3x3 invInertiaMatrix;	  // inverse of inertiaMatrix
    Mat3x3 invInertiaMassMatrix; // inverse of inertiaMassMatrix
    
    constexpr RigidMass(Real m=1)
    {
        mass = m;
        volume = 1;
        inertiaMatrix.identity();
        recalc();
    }
    
    constexpr void operator=(Real m)
    {
        mass = m;
        recalc();
    }
    
    constexpr void operator+=(Real m)
    {
        mass += m;
        recalc();
    }
    
    constexpr void operator-=(Real m)
    {
        mass -= m;
        recalc();
    }
   
    // operator to cast to const Real
    constexpr operator const Real() const
    {
        return mass;
    }
    
    constexpr void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        const bool canInvert1 = invInertiaMatrix.invert(inertiaMatrix);
        const bool canInvert2 = invInertiaMassMatrix.invert(inertiaMassMatrix);
        assert(canInvert1);
        assert(canInvert2);
        SOFA_UNUSED(canInvert1);
        SOFA_UNUSED(canInvert2);
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
    
    constexpr void operator *=(Real fact)
    {
        mass *= fact;
        inertiaMassMatrix *= fact;
        invInertiaMassMatrix /= fact;
    }
    
    constexpr void operator /=(Real fact)
    {
        mass /= fact;
        inertiaMassMatrix /= fact;
        invInertiaMassMatrix *= fact;
    }
};

template<typename real>
RigidDeriv<3,real> operator*(const RigidDeriv<3,real>& d, const RigidMass<3,real>& m)
{
    RigidDeriv<3,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
RigidDeriv<3,real> operator*(const RigidMass<3,real>& m, const RigidDeriv<3,real>& d)
{
    RigidDeriv<3,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
RigidDeriv<3,real> operator/(const RigidDeriv<3,real>& d, const RigidMass<3, real>& m)
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
    typedef type::Vec<3,Real> AngularVector;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;
    static constexpr sofa::Size deriv_total_size = Deriv::total_size;

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static constexpr const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static constexpr void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static constexpr const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static constexpr void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef type::Vec<3,real> DPos;
    typedef type::Vec<3,real> DRot;
    static constexpr const DPos& getDPos(const Deriv& d) { return getVCenter(d); }
    static constexpr void setDPos(Deriv& d, const DPos& v) { getVCenter(d) = v; }
    static constexpr const DRot& getDRot(const Deriv& d) { return getVOrientation(d); }
    static constexpr void setDRot(Deriv& d, const DRot& v) { getVOrientation(d) = v; }

    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    typedef type::vector<Coord> VecCoord;
    typedef type::vector<Deriv> VecDeriv;
    typedef type::vector<Real> VecReal;

    template<typename T>
    static constexpr void set(Coord& c, T x, T y, T z)
    {
        c.getCenter()[0] = static_cast<Real>(x);
        c.getCenter()[1] = static_cast<Real>(y);
        c.getCenter()[2] = static_cast<Real>(z);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Coord& c)
    {
        x = static_cast<T>(c.getCenter()[0]);
        y = static_cast<T>(c.getCenter()[1]);
        z = static_cast<T>(c.getCenter()[2]);
    }

    // set linear and angular velocities
    template<typename T>
    static constexpr void set(Deriv& c, T x, T y, T z, T rx, T ry, T rz )
    {
        c.getLinear()[0] = static_cast<Real>(x);
        c.getLinear()[1] = static_cast<Real>(y);
        c.getLinear()[2] = static_cast<Real>(z);
        c.getAngular()[0] = static_cast<Real>(rx);
        c.getAngular()[1] = static_cast<Real>(xy);
        c.getAngular()[2] = static_cast<Real>(xz);
    }

    template<typename T>
    static constexpr void add(Coord& c, T x, T y, T z)
    {
        c.getCenter()[0] += static_cast<Real>(x);
        c.getCenter()[1] += static_cast<Real>(y);
        c.getCenter()[2] += static_cast<Real>(z);
    }

    template<typename T>
    static constexpr void set(Deriv& c, T x, T y, T z)
    {
        getVCenter(c)[0] = static_cast<Real>(x);
        getVCenter(c)[1] = static_cast<Real>(y);
        getVCenter(c)[2] = static_cast<Real>(z);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Deriv& c)
    {
        x = static_cast<T>(getVCenter(c)[0]);
        y = static_cast<T>(getVCenter(c)[1]);
        z = static_cast<T>(getVCenter(c)[2]);
    }

    template<typename T>
    static constexpr void add(Deriv& c, T x, T y, T z)
    {
        getVCenter(c)[0] += static_cast<Real>(x);
        getVCenter(c)[1] += static_cast<Real>(y);
        getVCenter(c)[2] += static_cast<Real>(z);
    }

    static constexpr const char* Name();

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv( Real minMagnitude, Real maxMagnitude)
    {
        Deriv result;
        set( result, Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)),
                     Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)) );
        return result;
    }

    static constexpr Deriv coordDifference(const Coord& c1, const Coord& c2)
    {
        type::Vector3 vCenter = c1.getCenter() - c2.getCenter();
        type::Quat<SReal> quat, quat1(c1.getOrientation()), quat2(c2.getOrientation());
        // Transformation between c2 and c1 frames
        quat = quat1*quat2.inverse();
        quat.normalize();
        type::Vec3 axis; 
        type::Quat<SReal>::value_type angle; 
        quat.quatToAxis(axis, angle);
        axis*=angle;
        return Deriv(vCenter, axis);
    }

    static Coord interpolate(const type::vector< Coord > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            // Position interpolation.
            c.getCenter() += ancestors[i].getCenter() * coefs[i];

            // Angle extraction from the orientation quaternion.
            type::Quat<Real> q = ancestors[i].getOrientation();
            Real angle = acos(q[3]) * 2;

            // Axis extraction from the orientation quaternion.
            type::Vec<3,Real> v(q[0], q[1], q[2]);
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

    static Deriv interpolate(const type::vector< Deriv > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    /// inverse rigid transform
    static constexpr Coord inverse(const Coord& c)
    {
        CRot qinv = c.getOrientation().inverse();
        return Coord( -(qinv.rotate(c.getCenter())),qinv );
    }

    /// matrix product
    static constexpr Coord mult ( const Coord& a, const Coord& b )
    {
        return a.mult(b);
    }

    /// double cross product: a * ( b * c )
    static constexpr Vec3 crosscross ( const Vec3& a, const Vec3& b, const Vec3& c)
    {
        return cross( a, cross( b,c ));
    }

    /// create a rotation from Euler angles. For homogeneity with 2D.
    static Quat rotationEuler( Real x, Real y, Real z){ return Quat::fromEuler(x,y,z); }

};

typedef StdRigidTypes<3,double> Rigid3dTypes;
typedef RigidMass<3,double> Rigid3dMass;

typedef StdRigidTypes<3,float> Rigid3fTypes;
typedef RigidMass<3,float> Rigid3fMass;

/// We now use template aliases so we do not break backward compatibility.
template<> constexpr const char* Rigid3dTypes::Name() { return "Rigid3d"; }
template<> constexpr const char* Rigid3fTypes::Name() { return "Rigid3f"; }

typedef StdRigidTypes<3,SReal> Rigid3Types;  ///< un-defined precision type
typedef StdRigidTypes<3,SReal> RigidTypes;   ///< alias (beurk)
typedef RigidMass<3,SReal>     Rigid3Mass;   ///< un-defined precision type

//=============================================================================
// 2D Rigids
//=============================================================================

template<typename real>
class RigidDeriv<2,real>
{
public:
    typedef real value_type;
    typedef sofa::Size Size;
    typedef real Real;
    typedef type::Vec<2,Real> Pos;
    typedef Real Rot;
    typedef type::Vec<2,Real> Vec2;
    typedef type::Vec<3,Real> VecAll;

private:
    Vec2 vCenter;
    Real vOrientation;

public:
    friend class RigidCoord<2,real>;

    explicit constexpr RigidDeriv(type::NoInit)
        : vCenter(type::NOINIT)
        , vOrientation(type::NOINIT)
    {

    }

    constexpr RigidDeriv()
    {
        clear();
    }

    template<typename real2>
    constexpr RigidDeriv(const type::Vec<2,real2> &velCenter, const real2 &velOrient)
        : vCenter(velCenter), vOrientation((Real)velOrient)
    {}

    template<typename real2>
    constexpr RigidDeriv(const type::Vec<3,real2> &v)
        : vCenter(type::Vec<2,real2>(v.data())), vOrientation((Real)v[2])
    {}

    constexpr void clear()
    {
        vCenter.clear();
        vOrientation=0;
    }

    template<typename real2>
    constexpr void operator=(const RigidDeriv<2,real2>& c)
    {
        vCenter = c.getVCenter();
        vOrientation = (Real)c.getVOrientation();
    }

    template<typename real2>
    constexpr void operator=(const type::Vec<2,real2>& v)
    {
        vCenter = v;
    }

    template<typename real2>
    constexpr void operator=(const type::Vec<3,real2>& v)
    {
        vCenter = v;
        vOrientation = (Real)v[2];
    }

    constexpr void operator+=(const RigidDeriv<2,real>& a)
    {
        vCenter += a.vCenter;
        vOrientation += a.vOrientation;
    }

    constexpr void operator-=(const RigidDeriv<2,real>& a)
    {
        vCenter -= a.vCenter;
        vOrientation -= a.vOrientation;
    }

    constexpr RigidDeriv<2,real> operator+(const RigidDeriv<2,real>& a) const
    {
        RigidDeriv<2,real> d;
        d.vCenter = vCenter + a.vCenter;
        d.vOrientation = vOrientation + a.vOrientation;
        return d;
    }

    constexpr RigidDeriv<2,real> operator-(const RigidDeriv<2,real>& a) const
    {
        RigidDeriv<2,real> d;
        d.vCenter = vCenter - a.vCenter;
        d.vOrientation = vOrientation - a.vOrientation;
        return d;
    }

    template<typename real2>
    constexpr void operator*=(real2 a)
    {
        vCenter *= a;
        vOrientation *= (Real)a;
    }

    template<typename real2>
    constexpr void operator/=(real2 a)
    {
        vCenter /= a;
        vOrientation /= (Real)a;
    }



    constexpr RigidDeriv<2,real> operator-() const
    {
        return RigidDeriv<2,real>(-vCenter, -vOrientation);
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    constexpr Real operator*(const RigidDeriv<2,real>& a) const
    {
        return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]
                +vOrientation*a.vOrientation;
    }

    /// Euclidean norm
    constexpr Real norm() const
    {
        return helper::rsqrt( vCenter*vCenter + vOrientation*vOrientation);
    }

    constexpr Vec2& getVCenter() { return vCenter; }
    constexpr Real& getVOrientation() { return vOrientation; }
    constexpr const Vec2& getVCenter() const { return vCenter; }
    constexpr const Real& getVOrientation() const { return vOrientation; }

    constexpr Vec2& getLinear() { return vCenter; }
    constexpr Real& getAngular() { return vOrientation; }
    constexpr const Vec2& getLinear() const { return vCenter; }
    constexpr const Real& getAngular() const { return vOrientation; }

    constexpr VecAll getVAll() const
    {
        return VecAll(vCenter, vOrientation);
    }

    /// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
    constexpr Vec2 velocityAtRotatedPoint(const Vec2& p) const
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
    static constexpr sofa::Size total_size = 3;

    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    static constexpr sofa::Size spatial_dimensions = 2;
    
    constexpr real* ptr() { return vCenter.ptr(); }
    constexpr const real* ptr() const { return vCenter.ptr(); }

    static constexpr Size size() {return 3;}

    /// Access to i-th element.
    constexpr real& operator[](Size i)
    {
        if (i<2)
            return this->vCenter(i);
        else
            return this->vOrientation;
    }

    /// Const access to i-th element.
    constexpr const real& operator[](Size i) const
    {
        if (i<2)
            return this->vCenter(i);
        else
            return this->vOrientation;
    }


    /// @name Tests operators
    /// @{

    constexpr bool operator==(const RigidDeriv<2,real>& b) const
    {
        return vCenter == b.vCenter && vOrientation == b.vOrientation;
    }

    constexpr bool operator!=(const RigidDeriv<2,real>& b) const
    {
        return vCenter != b.vCenter || vOrientation != b.vOrientation;
    }

    /// @}
};

template<typename real, typename real2>
RigidDeriv<2,real> operator*(RigidDeriv<2,real> r, real2 a)
{
    r *= a;
    return r;
}

template<typename real, typename real2>
RigidDeriv<2,real> operator/(RigidDeriv<2,real> r, real2 a)
{
    r /= a;
    return r;
}

/// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
template<typename R, typename T>
constexpr type::Vec<2,R> velocityAtRotatedPoint(const RigidDeriv<2,T>& v, const type::Vec<2,R>& p)
{
    return getVCenter(v) + type::Vec<2,R>(-p[1], p[0]) * getVOrientation(v);
}



template<typename real>
class RigidCoord<2,real>
{
public:
    typedef real value_type;
    typedef sofa::Size Size;
    typedef real Real;
    typedef type::Vec<2,Real> Pos;
    typedef Real Rot;
    typedef type::Vec<2,Real> Vec2;
    typedef type::Mat<3,3,Real> HomogeneousMat;
    typedef type::Vec<3,Real> HomogeneousVec;
private:
    Vec2 center;
    Real orientation;
public:
    constexpr RigidCoord (const Vec2 &posCenter, const Real &orient)
        : center(posCenter), orientation(orient) 
    {}

    constexpr RigidCoord () 
    { 
        clear(); 
    }

    void clear() 
    { 
        center.clear(); 
        orientation = 0; 
    }

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

    constexpr void operator +=(const RigidDeriv<2,real>& a)
    {
        center += getVCenter(a);
        orientation += getVOrientation(a);
    }

    constexpr RigidCoord<2,real> operator + (const RigidDeriv<2,real>& a) const
    {
        RigidCoord<2,real> c = *this;
        c.center += getVCenter(a);
        c.orientation += getVOrientation(a);
        return c;
    }

    constexpr RigidCoord<2,real> operator -(const RigidCoord<2,real>& a) const
    {
        return RigidCoord<2,real>(this->center - a.getCenter(), this->orientation - a.orientation);
    }

    constexpr RigidCoord<2,real> operator +(const RigidCoord<2,real>& a) const
    {
        return RigidCoord<2,real>(this->center + a.getCenter(), this->orientation + a.orientation);
    }


    constexpr RigidCoord<2,real> operator-() const
    {
        return RigidCoord<2,real>( -this->center, this->orientation.inverse() );
    }


    constexpr void operator +=(const RigidCoord<2,real>& a)
    {
        center += a.getCenter();
        orientation += a.getOrientation();
    }

    template<typename real2>
    constexpr void operator*=(real2 a)
    {
        center *= a;
        orientation *= (Real)a;
    }

    template<typename real2>
    constexpr void operator/=(real2 a)
    {
        center /= a;
        orientation /= (Real)a;
    }

    template<typename real2>
    constexpr RigidCoord<2,real> operator*(real2 a) const
    {
        RigidCoord<2,real> r = *this;
        r *= a;
        return r;
    }

    template<typename real2>
    constexpr RigidCoord<2,real> operator/(real2 a) const
    {
        RigidCoord<2,real> r = *this;
        r /= a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    constexpr Real operator*(const RigidCoord<2,real>& a) const
    {
        return center[0]*a.center[0]+center[1]*a.center[1]
                +orientation*a.orientation;
    }

    /// Squared norm
    constexpr Real norm2() const
    {
        Real angle = fmod( (Real) orientation, (Real) M_PI*2 );
        return center*center + angle*angle;
    }

    /// Euclidean norm
    constexpr real norm() const
    {
        return helper::rsqrt(norm2());
    }

    constexpr Vec2& getCenter () { return center; }
    constexpr  Real& getOrientation () { return orientation; }
    constexpr const Vec2& getCenter () const { return center; }
    constexpr const Real& getOrientation () const { return orientation; }

    constexpr Vec2 rotate(const Vec2& v) const
    {
        Real s = sin(orientation);
        Real c = cos(orientation);
        return Vec2(c*v[0]-s*v[1],
                s*v[0]+c*v[1]);
    }
    
    constexpr Vec2 inverseRotate(const Vec2& v) const
    {
        Real s = sin(-orientation);
        Real c = cos(-orientation);
        return Vec2(c*v[0]-s*v[1],
                s*v[0]+c*v[1]);
    }

    constexpr Vec2 translate(const Vec2& v) const
    {
        return v + center;
    }

    static constexpr RigidCoord<2,real> identity()
    {
        RigidCoord<2,real> c;
        return c;
    }

    /// Apply a transformation with respect to itself
    constexpr void multRight( const RigidCoord<2,real>& c )
    {
        center += /*orientation.*/rotate(c.getCenter());
        orientation = orientation + c.getOrientation();
    }

    /// compute the product with another frame on the right
    constexpr RigidCoord<2,real> mult( const RigidCoord<2,real>& c ) const
    {
        RigidCoord<2,real> r;
        r.center = center + /*orientation.*/rotate( c.center );
        r.orientation = orientation + c.getOrientation();
        return r;
    }

    template<class Mat>
    constexpr void writeRotationMatrix( Mat& m) const
    {
        m[0][0] = (typename Mat::Real)cos(orientation); m[0][1] = (typename Mat::Real)-sin(orientation);
        m[1][0] = (typename Mat::Real)sin(orientation); m[1][1] = (typename Mat::Real) cos(orientation);
    }

    /// Set from the given matrix
    template<class Mat>
    constexpr void fromMatrix(const Mat& m)
    {
        center[0] = m[0][2];
        center[1] = m[1][2];
        orientation = atan2(m[1][0],m[0][0]);
    }

    /// Write to the given matrix
    template<class Mat>
    constexpr void toMatrix( Mat& m) const
    {
        m.identity();
        writeRotationMatrix( m );
        m[0][2] = center[0];
        m[1][2] = center[1];
    }


    /// create a homogeneous vector from a 2d vector
    template <typename V>
    static constexpr HomogeneousVec toHomogeneous( V v, real r=1. ){
        return HomogeneousVec( v[0], v[1], r );
    }
    /// create a 2d vector from a homogeneous vector
    template <typename V>
    static constexpr Vec2 fromHomogeneous( V v ){
        return Vec2( v[0], v[1] );
    }


    /// Write the OpenGL transformation matrix
    constexpr void writeOpenGlMatrix( float m[16] ) const
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
    constexpr Vec2 vectorToChild( const Vec2& v ) const
    {
        return /*orientation.*/inverseRotate(v);
    }

    /// Apply the transform to a point, i.e. project a point from the child frame to the parent frame (translation and rotation)
    constexpr Vec2 projectPoint( const Vec2& v ) const
    {
        return rotate(v)+center;
    }

    /// Apply the transform to a vector, i.e. project a vector from the child frame to the parent frame (rotation only, no translation added)
    constexpr Vec2 projectVector( const Vec2& v ) const
    {
        return rotate(v);
    }

    /// Apply the inverse transform to a point, i.e. project a point from the parent frame to the child frame (translation and rotation)
    constexpr Vec2 unprojectPoint( const Vec2& v ) const
    {
        return inverseRotate(v-center);
    }

    ///  Apply the inverse transform to a vector, i.e. project a vector from the parent frame to the child frame (rotation only, no translation)
    constexpr Vec2 unprojectVector( const Vec2& v ) const
    {
        return inverseRotate(v);
    }

    /// obsolete. Use projectPoint.
    constexpr Vec2 pointToParent( const Vec2& v ) const { return projectPoint(v); }
    /// obsolete. Use unprojectPoint.
    constexpr Vec2 pointToChild( const Vec2& v ) const { return unprojectPoint(v); }


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

    static constexpr Size max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr sofa::Size total_size = 3;

    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    static constexpr sofa::Size spatial_dimensions = 2;

    constexpr real* ptr() { return center.ptr(); }
    constexpr const real* ptr() const { return center.ptr(); }

    static constexpr Size size() {return 3;}

    /// Access to i-th element.
    constexpr real& operator[](Size i)
    {
        if (i<2)
            return this->center(i);
        else
            return this->orientation;
    }

    /// Const access to i-th element.
    constexpr const real& operator[](Size i) const
    {
        if (i<2)
            return this->center(i);
        else
            return this->orientation;
    }

    /// @name Tests operators
    /// @{

    constexpr bool operator==(const RigidCoord<2,real>& b) const
    {
        return center == b.center && orientation == b.orientation;
    }

    constexpr bool operator!=(const RigidCoord<2,real>& b) const
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
    
    constexpr RigidMass(Real m=1)
    {
        mass = m;
        volume = 1;
        inertiaMatrix = 1;
        recalc();
    }
    
    constexpr void operator=(Real m)
    {
        mass = m;
        recalc();
    }
    
    constexpr void operator+=(Real m)
    {
        mass += m;
        recalc();
    }
    
    constexpr void operator-=(Real m)
    {
        mass -= m;
        recalc();
    }
    
    // operator to cast to const Real
    constexpr operator const Real() const
    {
        return mass;
    }
    
    /// Mass for a circle
    constexpr RigidMass(Real m, Real radius)
    {
        mass = m;
        volume = radius*radius*R_PI;
        inertiaMatrix = (radius*radius)/2;
        recalc();
    }

    /// Mass for a rectangle
    constexpr RigidMass(Real m, Real xwidth, Real ywidth)
    {
        mass = m;
        volume = xwidth*xwidth + ywidth*ywidth;
        inertiaMatrix = volume/12;
        recalc();
    }

    constexpr void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        if (inertiaMatrix == 0.)
        {
            throw std::runtime_error("Attempt to divide by zero");
        }
        invInertiaMatrix = 1. / inertiaMatrix;
        if (inertiaMassMatrix == 0.)
        {
            throw std::runtime_error("Attempt to divide by zero");
        }
        invInertiaMassMatrix = 1. / inertiaMassMatrix;
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

    constexpr void operator *=(Real fact)
    {
        mass *= fact;
        inertiaMassMatrix *= fact;
        invInertiaMassMatrix /= fact;
    }

    constexpr void operator /=(Real fact)
    {
        mass /= fact;
        inertiaMassMatrix /= fact;
        invInertiaMassMatrix *= fact;
    }
};

template<typename real>
constexpr RigidDeriv<2,real> operator*(const RigidDeriv<2,real>& d, const RigidMass<2,real>& m)
{
    RigidDeriv<2,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
constexpr RigidDeriv<2,real> operator/(const RigidDeriv<2,real>& d, const RigidMass<2, real>& m)
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
    typedef type::Vec<2,real> Vec2;

    typedef RigidDeriv<2,Real> Deriv;
    typedef RigidCoord<2,Real> Coord;
    typedef Real AngularVector;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;
    static constexpr sofa::Size deriv_total_size = Deriv::total_size;

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
    static constexpr const CPos& getCPos(const Coord& c) { return c.getCenter(); }
    static constexpr void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
    static constexpr const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
    static constexpr void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef type::Vec<2,real> DPos;
    typedef real DRot;
    static constexpr const DPos& getDPos(const Deriv& d) { return getVCenter(d); }
    static constexpr void setDPos(Deriv& d, const DPos& v) { getVCenter(d) = v; }
    static constexpr const DRot& getDRot(const Deriv& d) { return getVOrientation(d); }
    static constexpr void setDRot(Deriv& d, const DRot& v) { getVOrientation(d) = v; }

    static constexpr const char* Name();

    typedef type::vector<Coord> VecCoord;
    typedef type::vector<Deriv> VecDeriv;
    typedef type::vector<Real> VecReal;

    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    template<typename T>
    static constexpr void set(Coord& c, T x, T y, T)
    {
        c.getCenter()[0] = static_cast<Real>(x);
        c.getCenter()[1] = static_cast<Real>(y);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Coord& c)
    {
        x = static_cast<T>(c.getCenter()[0]);
        y = static_cast<T>(c.getCenter()[1]);
        z = static_cast<T>(0);
    }

    template<typename T>
    static constexpr void add(Coord& c, T x, T y, T)
    {
        c.getCenter()[0] += static_cast<Real>(x);
        c.getCenter()[1] += static_cast<Real>(y);
    }

    template<typename T>
    static constexpr void set(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] = static_cast<Real>(x);
        c.getVCenter()[1] = static_cast<Real>(y);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Deriv& c)
    {
        x = static_cast<T>(c.getVCenter()[0]);
        y = static_cast<T>(c.getVCenter()[1]);
        z = static_cast<T>(0);
    }

    // Set linear and angular velocities, in 6D for uniformity with 3D
    template<typename T>
    static constexpr void set(Deriv& c, T x, T y, T, T vrot, T, T )
    {
        c.getVCenter()[0] = static_cast<Real>(x);
        c.getVCenter()[1] = static_cast<Real>(y);
        c.getVOrientation() = static_cast<Real>(vrot);
    }

    template<typename T>
    static constexpr void add(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] += static_cast<Real>(x);
        c.getVCenter()[1] += static_cast<Real>(y);
    }

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv( Real minMagnitude, Real maxMagnitude )
    {
        Deriv result;
        set( result, Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)),
                     Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)));
        return result;
    }

    static Coord interpolate(const type::vector< Coord > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            c += ancestors[i] * coefs[i];
        }

        return c;
    }

    static Deriv interpolate(const type::vector< Deriv > & ancestors, const type::vector< Real > & coefs)
    {
        assert(ancestors.size() == coefs.size());

        Deriv d;

        for (sofa::Size i = 0; i < ancestors.size(); i++)
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }

    /// specialized version of the double cross product: a * ( b * c ) for the variation of torque applied to the frame due to a small rotation with constant force.
    static constexpr Real crosscross ( const Vec2& f, const Real& dtheta, const Vec2& OP)
    {
        return dtheta * dot( f,OP );
    }

    /// specialized version of the double cross product: a * ( b * c ) for point acceleration
    static constexpr Vec2 crosscross ( const Real& omega, const Real& dtheta, const Vec2& OP)
    {
        return OP * omega * (-dtheta);
    }

    /// create a rotation from Euler angles (only the first is used). For homogeneity with 3D.
    static constexpr CRot rotationEuler( Real x, Real , Real ){ return CRot(x); }


};



template<> constexpr const char* Rigid2dTypes::Name() { return "Rigid2d"; }
template<> constexpr const char* Rigid2fTypes::Name() { return "Rigid2f"; }

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
static constexpr Vec3 rigidVelocity( const Vec3& omega, const Vec3& v, const Vec3& pv, const Vec3& p ) { return v + cross( omega, p-pv ); }

/// Apply the given translation and rotation to each entry of vector v
template<class V1, class Vec, class Rot>
static constexpr void displace( V1& v, Vec translation, Rot rotation )
{
    for(sofa::Size i=0; i<v.size(); i++)
        v[i] = translation + rotation.rotate(v[i]);
}

/// Apply the given translation and rotation to each entry of vector v
template<class V1, class Rot>
static constexpr void rotate( V1& v, Rot rotation )
{
    for(sofa::Size i=0; i<v.size(); i++)
        v[i] = rotation.rotate(v[i]);
}

/// Apply a rigid transformation (translation, Euler angles) to the given points and their associated velocities.
template<class V1, class V2>
static constexpr void rigidTransform ( V1& points, V2& velocities, SReal tx, SReal ty, SReal tz, SReal rx, SReal ry, SReal rz )
{
    typedef type::Vec3 Vec3;
    typedef type::Quat<SReal> Quat;
    Vec3 translation(tx,ty,tz);
    Quat rotation = Quat::createQuaterFromEuler(Vec3(rx,ry,rz));
    displace(points,translation,rotation);
    rotate(velocities,rotation);
}
//@}

} // namespace sofa:: defaulttype
