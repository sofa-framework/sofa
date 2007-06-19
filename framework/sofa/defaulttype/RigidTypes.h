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
#ifndef SOFA_DEFAULTTYPE_RIGIDTYPES_H
#define SOFA_DEFAULTTYPE_RIGIDTYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/helper/vector.h>
#include <iostream>
using std::endl;

namespace sofa
{

namespace defaulttype
{

using sofa::helper::vector;

template<int N, typename real>
class StdRigidTypes;

template<int N, typename real>
class StdRigidMass;

//=============================================================================
// 3D Rigids
//=============================================================================

/** Degrees of freedom of 3D rigid bodies. Orientations are modeled using quaternions.
*/
template<typename real>
class StdRigidTypes<3, real>
{
public:
    typedef real Real;
    typedef Vec<3,Real> Vec3;
    typedef helper::Quater<Real> Quat;

    class Deriv
    {
    private:
        Vec3 vCenter;
        Vec3 vOrientation;
    public:
        friend class Coord;

        Deriv (const Vec3 &velCenter, const Vec3 &velOrient)
            : vCenter(velCenter), vOrientation(velOrient) {}
        Deriv () { clear(); }

        void clear() { vCenter.clear(); vOrientation.clear(); }

        void operator +=(const Deriv& a)
        {
            vCenter += a.vCenter;
            vOrientation += a.vOrientation;
        }

        Deriv operator + (const Deriv& a) const
        {
            Deriv d;
            d.vCenter = vCenter + a.vCenter;
            d.vOrientation = vOrientation + a.vOrientation;
            return d;
        }

        void operator*=(double a)
        {
            vCenter *= a;
            vOrientation *= a;
        }

        Deriv operator*(double a) const
        {
            Deriv r = *this;
            r*=a;
            return r;
        }

        Deriv operator - () const
        {
            return Deriv(-vCenter, -vOrientation);
        }

        /// dot product, mostly used to compute residuals as sqrt(x*x)
        double operator*(const Deriv& a) const
        {
            return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]+vCenter[2]*a.vCenter[2]
                    +vOrientation[0]*a.vOrientation[0]+vOrientation[1]*a.vOrientation[1]
                    +vOrientation[2]*a.vOrientation[2];
        }

        Vec3& getVCenter (void) { return vCenter; }
        Vec3& getVOrientation (void) { return vOrientation; }
        const Vec3& getVCenter (void) const { return vCenter; }
        const Vec3& getVOrientation (void) const { return vOrientation; }
        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Deriv& v )
        {
            out<<v.vCenter<<" "<<v.vOrientation;
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Deriv& v )
        {
            in>>v.vCenter>>v.vOrientation;
            return in;
        }
    };

    class Coord
    {
    private:
        Vec3 center;
        Quat orientation;
    public:
        Coord (const Vec3 &posCenter, const Quat &orient)
            : center(posCenter), orientation(orient) {}
        Coord () { clear(); }
        typedef real value_type;

        void clear() { center.clear(); orientation.clear(); }

        void operator +=(const Deriv& a)
        {
            center += a.getVCenter();
            orientation.normalize();
            Quat qDot = orientation.vectQuatMult(a.getVOrientation());
            for (int i = 0; i < 4; i++)
                orientation[i] += qDot[i] * 0.5f;
            orientation.normalize();
        }

        Coord operator + (const Deriv& a) const
        {
            Coord c = *this;
            c.center += a.getVCenter();
            c.orientation.normalize();
            Quat qDot = c.orientation.vectQuatMult(a.getVOrientation());
            for (int i = 0; i < 4; i++)
                c.orientation[i] += qDot[i] * 0.5f;
            c.orientation.normalize();
            return c;
        }

        void operator +=(const Coord& a)
        {
            //std::cout << "+="<<std::endl;
            center += a.getCenter();
            //orientation += a.getOrientation();
            //orientation.normalize();
        }

        void operator*=(double a)
        {
            //std::cout << "*="<<std::endl;
            center *= a;
            //orientation *= a;
        }

        Coord operator*(double a) const
        {
            Coord r = *this;
            r*=a;
            return r;
        }

        /// dot product, mostly used to compute residuals as sqrt(x*x)
        double operator*(const Coord& a) const
        {
            return center[0]*a.center[0]+center[1]*a.center[1]+center[2]*a.center[2]
                    +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
                    +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
        }

        Vec3& getCenter () { return center; }
        Quat& getOrientation () { return orientation; }
        const Vec3& getCenter () const { return center; }
        const Quat& getOrientation () const { return orientation; }

        static Coord identity()
        {
            Coord c;
            return c;
        }

        /// Apply a transformation with respect to itself
        void multRight( const Coord& c )
        {
            center += orientation.rotate(c.getCenter());
            orientation = orientation * c.getOrientation();
        }

        /// compute the product with another frame on the right
        Coord mult( const Coord& c ) const
        {
            Coord r;
            r.center = center + orientation.rotate( c.center );
            r.orientation = orientation * c.getOrientation();
            return r;
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

        /// compute the projection of a vector from the parent frame to the child
        Vec3 vectorToChild( const Vec3& v ) const
        {
            return orientation.inverseRotate(v);
        }

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Coord& v )
        {
            out<<v.center<<" "<<v.orientation;
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Coord& v )
        {
            in>>v.center>>v.orientation;
            return in;
        }
    };

    template <class T>
    class SparseData
    {
    public:
        SparseData(unsigned int _index, const T& _data): index(_index), data(_data) {};
        unsigned int index;
        T data;
    };

    typedef SparseData<Coord> SparseCoord;
    typedef SparseData<Deriv> SparseDeriv;

    typedef vector<SparseCoord> SparseVecCoord;
    typedef vector<SparseDeriv> SparseVecDeriv;

    //! All the Constraints applied to a state Vector
    typedef	vector<SparseVecDeriv> VecConst;

    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;

    static void set(Coord& c, double x, double y, double z)
    {
        c.getCenter()[0] = (Real)x;
        c.getCenter()[1] = (Real)y;
        c.getCenter()[2] = (Real)z;
    }

    static void get(double& x, double& y, double& z, const Coord& c)
    {
        x = c.getCenter()[0];
        y = c.getCenter()[1];
        z = c.getCenter()[2];
    }

    static void add(Coord& c, double x, double y, double z)
    {
        c.getCenter()[0] += (Real)x;
        c.getCenter()[1] += (Real)y;
        c.getCenter()[2] += (Real)z;
    }

    static void set(Deriv& c, double x, double y, double z)
    {
        c.getVCenter()[0] = (Real)x;
        c.getVCenter()[1] = (Real)y;
        c.getVCenter()[2] = (Real)z;
    }

    static void get(double& x, double& y, double& z, const Deriv& c)
    {
        x = c.getVCenter()[0];
        y = c.getVCenter()[1];
        z = c.getVCenter()[2];
    }

    static void add(Deriv& c, double x, double y, double z)
    {
        c.getVCenter()[0] += (Real)x;
        c.getVCenter()[1] += (Real)y;
        c.getVCenter()[2] += (Real)z;
    }

    static const char* Name();
};

template<typename real>
class StdRigidMass<3, real>
{
public:
    typedef real Real;
    typedef Mat<3,3,Real> Mat3x3;
    Real mass,volume;
    Mat3x3 inertiaMatrix;	      // Inertia matrix of the object
    Mat3x3 inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat3x3 invInertiaMatrix;	  // inverse of inertiaMatrix
    Mat3x3 invInertiaMassMatrix; // inverse of inertiaMassMatrix
    StdRigidMass(Real m=1)
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
    void recalc()
    {
        inertiaMassMatrix = inertiaMatrix * mass;
        invInertiaMatrix.invert(inertiaMatrix);
        invInertiaMassMatrix.invert(inertiaMassMatrix);
    }

    inline friend std::ostream& operator << (std::ostream& out, const StdRigidMass<3, real>& m )
    {
        out<<m.mass;
        out<<" "<<m.volume;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, StdRigidMass<3, real>& m )
    {
        in>>m.mass;
        in>>m.volume;
        in>>m.inertiaMatrix;
        return in;
    }
};

template<int N, typename real>
inline typename StdRigidTypes<N,real>::Deriv operator*(const typename StdRigidTypes<N,real>::Deriv& d, const StdRigidMass<N,real>& m)
{
    typename StdRigidTypes<N,real>::Deriv res;
    res.getVCenter() = d.getVCenter() * m.mass;
    res.getVOrientation() = m.inertiaMassMatrix * d.getVOrientation();
    return res;
}

template<int N, typename real>
inline typename StdRigidTypes<N, real>::Deriv operator/(const typename StdRigidTypes<N, real>::Deriv& d, const StdRigidMass<N, real>& m)
{
    typename StdRigidTypes<N, real>::Deriv res;
    res.getVCenter() = d.getVCenter() / m.mass;
    res.getVOrientation() = m.invInertiaMassMatrix * d.getVOrientation();
    return res;
}


typedef StdRigidTypes<3,double> Rigid3dTypes;
typedef StdRigidTypes<3,float> Rigid3fTypes;
typedef Rigid3dTypes Rigid3Types;
typedef Rigid3Types RigidTypes;

typedef StdRigidMass<3,double> Rigid3dMass;
typedef StdRigidMass<3,float> Rigid3fMass;
typedef Rigid3dMass Rigid3Mass;
typedef Rigid3Mass RigidMass;

/// Note: Many scenes use Rigid as template for 3D double-precision rigid type. Changing it to Rigid3d would break backward compatibility.
template<> inline const char* Rigid3dTypes::Name() { return "Rigid"; }
template<> inline const char* Rigid3fTypes::Name() { return "Rigid3f"; }


//=============================================================================
// 2D Rigids
//=============================================================================

/** Degrees of freedom of 2D rigid bodies.
*/
template<typename real>
class StdRigidTypes<2, real>
{
public:
    typedef real Real;
    typedef Vec<2,real> Vec2;


    static const char* Name();

    class Deriv
    {
    private:
        Vec2 vCenter;
        Real vOrientation;
    public:
        friend class Coord;

        Deriv (const Vec2 &velCenter, const Real &velOrient)
            : vCenter(velCenter), vOrientation(velOrient) {}
        Deriv () { clear(); }

        void clear() { vCenter.clear(); vOrientation=0; }

        void operator +=(const Deriv& a)
        {
            vCenter += a.vCenter;
            vOrientation += a.vOrientation;
        }

        Deriv operator + (const Deriv& a) const
        {
            Deriv d;
            d.vCenter = vCenter + a.vCenter;
            d.vOrientation = vOrientation + a.vOrientation;
            return d;
        }

        void operator*=(double a)
        {
            vCenter *= (Real)a;
            vOrientation *= (Real)a;
        }

        Deriv operator*(double a) const
        {
            Deriv r = *this;
            r *= (Real)a;
            return r;
        }

        Deriv operator - () const
        {
            return Deriv(-vCenter, -vOrientation);
        }

        /// dot product, mostly used to compute residuals as sqrt(x*x)
        double operator*(const Deriv& a) const
        {
            return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]
                    +vOrientation*a.vOrientation;
        }

        Vec2& getVCenter (void) { return vCenter; }
        Real& getVOrientation (void) { return vOrientation; }
        const Vec2& getVCenter (void) const { return vCenter; }
        const Real& getVOrientation (void) const { return vOrientation; }
        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Deriv& v )
        {
            out<<v.vCenter<<" "<<v.vOrientation;
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Deriv& v )
        {
            in>>v.vCenter>>v.vOrientation;
            return in;
        }
    };

    class Coord
    {
    private:
        Vec2 center;
        Real orientation;
    public:
        Coord (const Vec2 &posCenter, const Real &orient)
            : center(posCenter), orientation(orient) {}
        Coord () { clear(); }
        typedef real value_type;

        void clear() { center.clear(); orientation = 0; }

        void operator +=(const Deriv& a)
        {
            center += a.getVCenter();
            orientation += a.getVOrientation();
        }

        Coord operator + (const Deriv& a) const
        {
            Coord c = *this;
            c.center += a.getVCenter();
            c.orientation += a.getVOrientation();
            return c;
        }

        void operator +=(const Coord& a)
        {
            std::cout << "+="<<std::endl;
            center += a.getCenter();
            orientation += a.getOrientation();
        }

        void operator*=(double a)
        {
            std::cout << "*="<<std::endl;
            center *= (Real)a;
            orientation *= (Real)a;
        }

        Coord operator*(double a) const
        {
            Coord r = *this;
            r *= (Real)a;
            return r;
        }

        /// dot product, mostly used to compute residuals as sqrt(x*x)
        double operator*(const Coord& a) const
        {
            return center[0]*a.center[0]+center[1]*a.center[1]
                    +orientation*a.orientation;
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

        static Coord identity()
        {
            Coord c;
            return c;
        }

        /// Apply a transformation with respect to itself
        void multRight( const Coord& c )
        {
            center += /*orientation.*/rotate(c.getCenter());
            orientation = orientation * c.getOrientation();
        }

        /// compute the product with another frame on the right
        Coord mult( const Coord& c ) const
        {
            Coord r;
            r.center = center + /*orientation.*/rotate( c.center );
            r.orientation = orientation * c.getOrientation();
            return r;
        }

        template<class Mat>
        void writeRotationMatrix( Mat& m) const
        {
            m[0][0] = cos((typename Mat::Real)orientation); m[0][1] = -sin((typename Mat::Real)orientation);
            m[1][0] = sin((typename Mat::Real)orientation); m[1][1] =  cos((typename Mat::Real)orientation);
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

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Coord& v )
        {
            out<<v.center<<" "<<v.orientation;
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Coord& v )
        {
            in>>v.center>>v.orientation;
            return in;
        }
    };

    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;


    template <class T>
    class SparseData
    {
    public:
        SparseData(unsigned int _index, const T& _data): index(_index), data(_data) {};
        unsigned int index;
        T data;
    };

    typedef SparseData<Coord> SparseCoord;
    typedef SparseData<Deriv> SparseDeriv;

    typedef vector<SparseCoord> SparseVecCoord;
    typedef vector<SparseDeriv> SparseVecDeriv;

    typedef	vector<SparseVecDeriv> VecConst;

    static void set(Coord& c, double x, double y, double)
    {
        c.getCenter()[0] = (Real)x;
        c.getCenter()[1] = (Real)y;
    }

    static void get(double& x, double& y, double& z, const Coord& c)
    {
        x = c.getCenter()[0];
        y = c.getCenter()[1];
        z = 0;
    }

    static void add(Coord& c, double x, double y, double)
    {
        c.getCenter()[0] += (Real)x;
        c.getCenter()[1] += (Real)y;
    }

    static void set(Deriv& c, double x, double y, double)
    {
        c.getVCenter()[0] = (Real)x;
        c.getVCenter()[1] = (Real)y;
    }

    static void get(double& x, double& y, double& z, const Deriv& c)
    {
        x = c.getVCenter()[0];
        y = c.getVCenter()[1];
        z = 0;
    }

    static void add(Deriv& c, double x, double y, double)
    {
        c.getVCenter()[0] += (Real)x;
        c.getVCenter()[1] += (Real)y;
    }

};

template<class real>
class StdRigidMass<2, real>
{
public:
    typedef real Real;
    Real mass,volume;
    Real inertiaMatrix;	      // Inertia matrix of the object
    Real inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Real invInertiaMatrix;	  // inverse of inertiaMatrix
    Real invInertiaMassMatrix; // inverse of inertiaMassMatrix
    StdRigidMass(Real m=1)
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
    /// Mass for a circle
    StdRigidMass(Real m, Real radius)
    {
        mass = m;
        volume = radius*radius*M_PI;
        inertiaMatrix = (radius*radius)/2;
        recalc();
    }
    /// Mass for a rectangle
    StdRigidMass(Real m, Real xwidth, Real ywidth)
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
    inline friend std::ostream& operator << (std::ostream& out, const StdRigidMass<2,Real>& m )
    {
        out<<m.mass;
        out<<" "<<m.volume;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, StdRigidMass<2,Real>& m )
    {
        in>>m.mass;
        in>>m.volume;
        in>>m.inertiaMatrix;
        return in;
    }
};

typedef StdRigidTypes<2,double> Rigid2dTypes;
typedef StdRigidTypes<2,float> Rigid2fTypes;
typedef Rigid2dTypes Rigid2Types;

typedef StdRigidMass<2,double> Rigid2dMass;
typedef StdRigidMass<2,float> Rigid2fMass;
typedef Rigid2dMass Rigid2Mass;

template<> inline const char* Rigid2dTypes::Name() { return "Rigid2d"; }
template<> inline const char* Rigid2fTypes::Name() { return "Rigid2f"; }

} // namespace defaulttype

namespace core
{
namespace componentmodel
{
namespace behavior
{

/** Return the inertia force applied to a body referenced in a moving coordinate system.
\param sv spatial velocity (omega, vorigin) of the coordinate system
\param a acceleration of the origin of the coordinate system
\param m mass of the body
\param x position of the body in the moving coordinate system
\param v velocity of the body in the moving coordinate system
This default implementation returns no inertia.
*/
template<class Coord, class Deriv, class Vec, class M, class SV>
Deriv inertiaForce( const SV& /*sv*/, const Vec& /*a*/, const M& /*m*/, const Coord& /*x*/, const Deriv& /*v*/ );

/// Specialization of the inertia force for defaulttype::Rigid3dTypes
template <>
inline defaulttype::StdRigidTypes<3, double>::Deriv inertiaForce<
defaulttype::StdRigidTypes<3, double>::Coord,
            defaulttype::StdRigidTypes<3, double>::Deriv,
            objectmodel::BaseContext::Vec3,
            defaulttype::StdRigidMass<3, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::StdRigidMass<3, double>& mass,
                    const defaulttype::StdRigidTypes<3, double>::Coord& x,
                    const defaulttype::StdRigidTypes<3, double>::Deriv& v
            )
{
    defaulttype::StdRigidTypes<3, double>::Vec3 omega( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::StdRigidTypes<3, double>::Vec3 origin = x.getCenter(), finertia, zero(0,0,0);

    finertia = -( aframe + omega.cross( omega.cross(origin) + v.getVCenter()*2 ))*mass.mass;
    return defaulttype::StdRigidTypes<3, double>::Deriv( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

/// Specialization of the inertia force for defaulttype::Rigid3fTypes
template <>
inline defaulttype::StdRigidTypes<3, float>::Deriv inertiaForce<
defaulttype::StdRigidTypes<3, float>::Coord,
            defaulttype::StdRigidTypes<3, float>::Deriv,
            objectmodel::BaseContext::Vec3,
            defaulttype::StdRigidMass<3, float>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::StdRigidMass<3, float>& mass,
                    const defaulttype::StdRigidTypes<3, float>::Coord& x,
                    const defaulttype::StdRigidTypes<3, float>::Deriv& v
            )
{
    defaulttype::StdRigidTypes<3, float>::Vec3 omega( (float)vframe.lineVec[0], (float)vframe.lineVec[1], (float)vframe.lineVec[2] );
    defaulttype::StdRigidTypes<3, float>::Vec3 origin = x.getCenter(), finertia, zero(0,0,0);

    finertia = -( aframe + omega.cross( omega.cross(origin) + v.getVCenter()*2 ))*mass.mass;
    return defaulttype::StdRigidTypes<3, float>::Deriv( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

} // namespace behavoir

} // namespace componentmodel

namespace objectmodel
{

// Specialization of Field::getValueTypeString() method to display smaller
// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<>
inline std::string FieldBase::typeName< defaulttype::Quat >(const defaulttype::Quat *) { return "Quat"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Quat > >(const std::vector< defaulttype::Quat > *) { return "vector<Quat>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Quat > >(const helper::vector< defaulttype::Quat > *) { return "vector<Quat>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Quatf >(const defaulttype::Quatf *) { return "Quatf"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Quatf > >(const std::vector< defaulttype::Quatf > *) { return "vector<Quatf>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Quatf > >(const helper::vector< defaulttype::Quatf > *) { return "vector<Quatf>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Rigid2dTypes::Coord >(const defaulttype::Rigid2dTypes::Coord *) { return "Rigid2dTypes::Coord"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Rigid2dTypes::Coord > >(const std::vector< defaulttype::Rigid2dTypes::Coord > *) { return "vector<Rigid2dTypes::Coord>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Rigid2dTypes::Coord > >(const helper::vector< defaulttype::Rigid2dTypes::Coord > *) { return "vector<Rigid2dTypes::Coord>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Rigid2dTypes::Deriv >(const defaulttype::Rigid2dTypes::Deriv *) { return "Rigid2dTypes::Deriv"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Rigid2dTypes::Deriv > >(const std::vector< defaulttype::Rigid2dTypes::Deriv > *) { return "vector<Rigid2dTypes::Deriv>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Rigid2dTypes::Deriv > >(const helper::vector< defaulttype::Rigid2dTypes::Deriv > *) { return "vector<Rigid2dTypes::Deriv>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Rigid2fTypes::Coord >(const defaulttype::Rigid2fTypes::Coord *) { return "Rigid2fTypes::Coord"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Rigid2fTypes::Coord > >(const std::vector< defaulttype::Rigid2fTypes::Coord > *) { return "vector<Rigid2fTypes::Coord>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Rigid2fTypes::Coord > >(const helper::vector< defaulttype::Rigid2fTypes::Coord > *) { return "vector<Rigid2fTypes::Coord>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Rigid2fTypes::Deriv >(const defaulttype::Rigid2fTypes::Deriv *) { return "Rigid2fTypes::Deriv"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Rigid2fTypes::Deriv > >(const std::vector< defaulttype::Rigid2fTypes::Deriv > *) { return "vector<Rigid2fTypes::Deriv>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Rigid2fTypes::Deriv > >(const helper::vector< defaulttype::Rigid2fTypes::Deriv > *) { return "vector<Rigid2fTypes::Deriv>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Rigid3dTypes::Coord >(const defaulttype::Rigid3dTypes::Coord *) { return "Rigid3dTypes::Coord"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Rigid3dTypes::Coord > >(const std::vector< defaulttype::Rigid3dTypes::Coord > *) { return "vector<Rigid3dTypes::Coord>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Rigid3dTypes::Coord > >(const helper::vector< defaulttype::Rigid3dTypes::Coord > *) { return "vector<Rigid3dTypes::Coord>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Rigid3dTypes::Deriv >(const defaulttype::Rigid3dTypes::Deriv *) { return "Rigid3dTypes::Deriv"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Rigid3dTypes::Deriv > >(const std::vector< defaulttype::Rigid3dTypes::Deriv > *) { return "vector<Rigid3dTypes::Deriv>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Rigid3dTypes::Deriv > >(const helper::vector< defaulttype::Rigid3dTypes::Deriv > *) { return "vector<Rigid3dTypes::Deriv>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Rigid3fTypes::Coord >(const defaulttype::Rigid3fTypes::Coord *) { return "Rigid3fTypes::Coord"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Rigid3fTypes::Coord > >(const std::vector< defaulttype::Rigid3fTypes::Coord > *) { return "vector<Rigid3fTypes::Coord>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Rigid3fTypes::Coord > >(const helper::vector< defaulttype::Rigid3fTypes::Coord > *) { return "vector<Rigid3fTypes::Coord>"; }

template<>
inline std::string FieldBase::typeName< defaulttype::Rigid3fTypes::Deriv >(const defaulttype::Rigid3fTypes::Deriv *) { return "Rigid3fTypes::Deriv"; }

template<>
inline std::string FieldBase::typeName< std::vector< defaulttype::Rigid3fTypes::Deriv > >(const std::vector< defaulttype::Rigid3fTypes::Deriv > *) { return "vector<Rigid3fTypes::Deriv>"; }

template<>
inline std::string FieldBase::typeName< helper::vector< defaulttype::Rigid3fTypes::Deriv > >(const helper::vector< defaulttype::Rigid3fTypes::Deriv > *) { return "vector<Rigid3fTypes::Deriv>"; }

/// \endcond

} // namespace objectmodel

} // namespace core

} // namespace sofa


#endif
