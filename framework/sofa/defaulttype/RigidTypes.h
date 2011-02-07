/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_RIGIDTYPES_H
#define SOFA_DEFAULTTYPE_RIGIDTYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <iostream>

namespace sofa
{

namespace defaulttype
{

using std::endl;
using sofa::helper::vector;

//template<int N, typename real>
//class RigidDeriv;

template<int N, typename real>
class RigidCoord;

template<int N, typename real>
class RigidMass;

template<int N, typename real>
class StdRigidTypes;

//=============================================================================
// 3D Rigids
//=============================================================================

///** Degrees of freedom of 3D rigid bodies. Orientations are modeled using quaternions.
//*/
//template<typename real>
//class RigidDeriv<3, real>
//{
//public:
//	typedef real value_type;
//    typedef real Real;
//    typedef Vec<3,Real> Pos;
//    typedef Vec<3,Real> Rot;
//    typedef Vec<3,Real> Vec3;
//    typedef helper::Quater<Real> Quat;
//
//protected:
//    Vec3 vCenter;
//    Vec3 vOrientation;
//public:
//    friend class RigidCoord<3,real>;
//
//    RigidDeriv(const Vec3 &velCenter, const Vec3 &velOrient)
//    : vCenter(velCenter), vOrientation(velOrient) {}
//    RigidDeriv() { clear(); }
//
//    template<typename real2>
//    RigidDeriv(const RigidDeriv<3,real2>& c)
//    : vCenter(c.getVCenter()), vOrientation(c.getVOrientation())
//    {
//    }
//
//    void clear() { vCenter.clear(); vOrientation.clear(); }
//
//    template<typename real2>
//    void operator =(const RigidDeriv<3,real2>& c)
//    {
//        vCenter = c.getVCenter();
//        vOrientation = c.getVOrientation();
//    }
//
//    void operator =(const Vec3& v)
//    {
//        vCenter = v;
//    }
//
//    void operator +=(const RigidDeriv& a)
//    {
//        vCenter += a.vCenter;
//        vOrientation += a.vOrientation;
//    }
//
//	void operator -=(const RigidDeriv& a)
//    {
//        vCenter -= a.vCenter;
//        vOrientation -= a.vOrientation;
//    }
//
//    RigidDeriv<3,real> operator + (const RigidDeriv<3,real>& a) const
//    {
//        RigidDeriv d;
//        d.vCenter = vCenter + a.vCenter;
//        d.vOrientation = vOrientation + a.vOrientation;
//        return d;
//    }
//
//    template<typename real2>
//    void operator*=(real2 a)
//    {
//        vCenter *= a;
//        vOrientation *= a;
//    }
//
//    template<typename real2>
//    void operator/=(real2 a)
//    {
//        vCenter /= a;
//        vOrientation /= a;
//    }
//
//    RigidDeriv<3,real> operator*(float a) const
//    {
//        RigidDeriv r = *this;
//        r*=a;
//        return r;
//    }
//
//    RigidDeriv<3,real> operator*(double a) const
//    {
//        RigidDeriv r = *this;
//        r*=a;
//        return r;
//    }
//
//    RigidDeriv<3,real> operator - () const
//    {
//        return RigidDeriv(-vCenter, -vOrientation);
//    }
//
//	RigidDeriv<3,real> operator - (const RigidDeriv<3,real>& a) const
//	{
//		return RigidDeriv<3,real>(this->vCenter - a.vCenter, this->vOrientation-a.vOrientation);
//	}
//
//
//    /// dot product, mostly used to compute residuals as sqrt(x*x)
//    Real operator*(const RigidDeriv<3,real>& a) const
//    {
//        return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]+vCenter[2]*a.vCenter[2]
//            +vOrientation[0]*a.vOrientation[0]+vOrientation[1]*a.vOrientation[1]
//            +vOrientation[2]*a.vOrientation[2];
//    }
//
//    Vec3& getVCenter (void) { return vCenter; }
//    Vec3& getVOrientation (void) { return vOrientation; }
//    const Vec3& getVCenter (void) const { return vCenter; }
//    const Vec3& getVOrientation (void) const { return vOrientation; }
//
//	 Vec3& getLinear () { return vCenter; }
//	 const Vec3& getLinear () const { return vCenter; }
//	 Vec3& getAngular () { return vOrientation; }
//	 const Vec3& getAngular () const { return vOrientation; }
//
//
//	 Vec3 velocityAtRotatedPoint(const Vec3& p) const
//	 {
//	     return vCenter - cross(p, vOrientation);
//	 }
//
//	 /// write to an output stream
//    inline friend std::ostream& operator << ( std::ostream& out, const RigidDeriv<3,real>& v ){
//        out<<v.vCenter<<" "<<v.vOrientation;
//        return out;
//    }
//    /// read from an input stream
//    inline friend std::istream& operator >> ( std::istream& in, RigidDeriv<3,real>& v ){
//        in>>v.vCenter>>v.vOrientation;
//        return in;
//    }
//
//    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
//    enum { total_size = 6 };
//    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
//    enum { spatial_dimensions = 3 };
//
//    real* ptr() { return vCenter.ptr(); }
//    const real* ptr() const { return vCenter.ptr(); }
//
//	static unsigned int size(){return 6;}
//
//	/// Access to i-th element.
//    real& operator[](int i)
//    {
//		if (i<3)
//			return this->vCenter(i);
//		else
//			return this->vOrientation(i-3);
//    }
//
//    /// Const access to i-th element.
//    const real& operator[](int i) const
//    {
//		if (i<3)
//			return this->vCenter(i);
//		else
//			return this->vOrientation(i-3);
//    }
//};


template<typename T>
Vec<3,T>& getLinear(Vec<6,T>& v)
{
    // We cannot use casts if the compiler adds extra members in Vec
    BOOST_STATIC_ASSERT(sizeof(Vec<6,T>) == 6*sizeof(T));
    BOOST_STATIC_ASSERT(sizeof(Vec<3,T>) == 3*sizeof(T));
    return *reinterpret_cast<Vec<3,T>*>( &v[0]);
}

template<typename T>
const Vec<3,T>& getLinear(const Vec<6,T>& v)
{
    // We cannot use casts if the compiler adds extra members in Vec
    BOOST_STATIC_ASSERT(sizeof(Vec<6,T>) == 6*sizeof(T));
    BOOST_STATIC_ASSERT(sizeof(Vec<3,T>) == 3*sizeof(T));
    return *reinterpret_cast<const Vec<3,T>*>( &v[0]);
}

template<typename T>
Vec<3,T>& getAngular(Vec<6,T>& v)
{
    // We cannot use casts if the compiler adds extra members in Vec
    BOOST_STATIC_ASSERT(sizeof(Vec<6,T>) == 6*sizeof(T));
    BOOST_STATIC_ASSERT(sizeof(Vec<3,T>) == 3*sizeof(T));
    return *reinterpret_cast<Vec<3,T>*>( &v[3]);
}

template<typename T>
const Vec<3,T>& getAngular(const Vec<6,T>& v)
{
    // We cannot use casts if the compiler adds extra members in Vec
    BOOST_STATIC_ASSERT(sizeof(Vec<6,T>) == 6*sizeof(T));
    BOOST_STATIC_ASSERT(sizeof(Vec<3,T>) == 3*sizeof(T));
    return *reinterpret_cast<const Vec<3,T>*>( &v[3]);
}

template<typename T>
Vec<3,T>& getVCenter(Vec<6,T>& v) { return getLinear(v); }

template<typename T>
Vec<3,T>& getVOrientation(Vec<6,T>& v) { return getAngular(v); }

template<typename T>
const Vec<3,T>& getVCenter(const Vec<6,T>& v) { return getLinear(v); }

template<typename T>
const Vec<3,T>& getVOrientation(const Vec<6,T>& v) { return getAngular(v); }


template<typename T, typename R>
Vec<3,T> velocityAtRotatedPoint(const Vec<6,R>& v, const Vec<3,T>& p)
{
    return getLinear(v) + cross( getAngular(v),p );
}




template<typename real>
class RigidCoord<3,real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Vec<3,Real> Pos;
    typedef helper::Quater<Real> Rot;
    typedef Vec<3,Real> Vec3;
    typedef helper::Quater<Real> Quat;

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

    template<typename real2>
    void operator =(const RigidCoord<3,real2>& c)
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

    void operator +=(const Vec<6,real>& a)
    {
        center += getVCenter(a);
        orientation.normalize();
        Quat qDot = orientation.vectQuatMult(getVOrientation(a));
        for (int i = 0; i < 4; i++)
            orientation[i] += qDot[i] * 0.5f;
        orientation.normalize();
    }

    RigidCoord<3,real> operator + (const Vec<6,real>& a) const
    {
        RigidCoord c = *this;
        c.center += getVCenter(a);
        c.orientation.normalize();
        Quat qDot = c.orientation.vectQuatMult(getVOrientation(a));
        for (int i = 0; i < 4; i++)
            c.orientation[i] += qDot[i] * 0.5f;
        c.orientation.normalize();
        return c;
    }

    RigidCoord<3,real> operator -(const RigidCoord<3,real>& a) const
    {
        return RigidCoord<3,real>(this->center - a.getCenter(), a.orientation.inverse() * this->orientation);
    }

    RigidCoord<3,real> operator +(const RigidCoord<3,real>& a) const
    {
        return RigidCoord<3,real>(this->center + a.getCenter(), a.orientation * this->orientation);
    }

    void operator +=(const RigidCoord<3,real>& a)
    {
        center += a.getCenter();
        orientation *= a.getOrientation();
    }

    template<typename real2>
    void operator*=(real2 a)
    {
        //std::cout << "*="<<std::endl;
        center *= a;
        //orientation *= a;
    }

    template<typename real2>
    void operator/=(real2 a)
    {
        //std::cout << "/="<<std::endl;
        center /= a;
        //orientation /= a;
    }

    template<typename real2>
    RigidCoord<3,real> operator*(real2 a) const
    {
        RigidCoord r = *this;
        r*=a;
        return r;
    }





    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const RigidCoord<3,real>& a) const
    {
        return center[0]*a.center[0]+center[1]*a.center[1]+center[2]*a.center[2]
                +orientation[0]*a.orientation[0]+orientation[1]*a.orientation[1]
                +orientation[2]*a.orientation[2]+orientation[3]*a.orientation[3];
    }

    /// Squared norm
    real norm2() const
    {
        real r = (this->center).elems[0]*(this->center).elems[0];
        for (int i=1; i<3; i++)
            r += (this->center).elems[i]*(this->center).elems[i];
        return r;
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
        Mat3x3d rot; rot = m;
        orientation.fromMatrix(rot);
    }

    /// Write to the given matrix
    template<class Mat>
    void toMatrix( Mat& m) const
    {
        m.identity();
        orientation.toMatrix(m);
        m[0][3] = center[0];
        m[1][3] = center[1];
        m[2][3] = center[2];
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

    /// Project a point from the child frame to the parent frame
    Vec3 pointToParent( const Vec3& v ) const
    {
        return orientation.rotate(v)+center;
    }

    /// Project a point from the parent frame to the child frame
    Vec3 pointToChild( const Vec3& v ) const
    {
        return orientation.inverseRotate(v-center);
    }

    /// compute the projection of a vector from the parent frame to the child
    Vec3 vectorToChild( const Vec3& v ) const
    {
        return orientation.inverseRotate(v);
    }

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
inline Vec<6,real> operator*(const Vec<6,real>& d, const RigidMass<3,real>& m)
{
    Vec<6,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
inline Vec<6,real> operator/(const Vec<6,real>& d, const RigidMass<3, real>& m)
{
    Vec<6,real> res;
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
    typedef Vec<6,real> Deriv;
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

    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;
    typedef vector<Real> VecReal;

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

};

typedef StdRigidTypes<3,double> Rigid3dTypes;
typedef StdRigidTypes<3,float> Rigid3fTypes;

typedef RigidMass<3,double> Rigid3dMass;
typedef RigidMass<3,float> Rigid3fMass;
//typedef Rigid3Mass RigidMass;

/// Note: Many scenes use Rigid as template for 3D double-precision rigid type. Changing it to Rigid3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* Rigid3dTypes::Name() { return "Rigid3d"; }
template<> inline const char* Rigid3fTypes::Name() { return "Rigid"; }
#else
template<> inline const char* Rigid3dTypes::Name() { return "Rigid"; }
template<> inline const char* Rigid3fTypes::Name() { return "Rigid3f"; }
#endif

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

//template<typename real>
//class RigidDeriv<2,real>
//{
//public:
//	typedef real value_type;
//    typedef real Real;
//    typedef Vec<2,Real> Pos;
//    typedef Real Rot;
//    typedef Vec<2,Real> Vec2;
//private:
//    Vec2 vCenter;
//    Real vOrientation;
//public:
//    friend class RigidCoord<2,real>;
//
//    RigidDeriv (const Vec2 &velCenter, const Real &velOrient)
//    : vCenter(velCenter), vOrientation(velOrient) {}
//    RigidDeriv () { clear(); }
//
//    void clear() { vCenter.clear(); vOrientation=0; }
//
//    void operator +=(const RigidDeriv<2,real>& a)
//    {
//        vCenter += a.vCenter;
//        vOrientation += a.vOrientation;
//    }
//
//    RigidDeriv<2,real> operator + (const RigidDeriv<2,real>& a) const
//    {
//        RigidDeriv<2,real> d;
//        d.vCenter = vCenter + a.vCenter;
//        d.vOrientation = vOrientation + a.vOrientation;
//        return d;
//    }
//
//	RigidDeriv<2,real> operator - (const RigidDeriv<2,real>& a) const
//    {
//        RigidDeriv<2,real> d;
//        d.vCenter = vCenter - a.vCenter;
//        d.vOrientation = vOrientation - a.vOrientation;
//        return d;
//    }
//
//    template<typename real2>
//    void operator*=(real2 a)
//    {
//        vCenter *= a;
//        vOrientation *= (Real)a;
//    }
//
//    template<typename real2>
//    void operator/=(real2 a)
//    {
//        vCenter /= a;
//        vOrientation /= (Real)a;
//    }
//
//    RigidDeriv<2,real> operator*(float a) const
//    {
//        RigidDeriv<2,real> r = *this;
//        r *= a;
//        return r;
//    }
//
//    RigidDeriv<2,real> operator*(double a) const
//    {
//        RigidDeriv<2,real> r = *this;
//        r *= a;
//        return r;
//    }
//
//    RigidDeriv<2,real> operator - () const
//    {
//        return RigidDeriv<2,real>(-vCenter, -vOrientation);
//    }
//
//    /// dot product, mostly used to compute residuals as sqrt(x*x)
//    Real operator*(const RigidDeriv<2,real>& a) const
//    {
//        return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]
//            +vOrientation*a.vOrientation;
//    }
//
//    Vec2& getVCenter (void) { return vCenter; }
//    Real& getVOrientation (void) { return vOrientation; }
//    const Vec2& getVCenter (void) const { return vCenter; }
//    const Real& getVOrientation (void) const { return vOrientation; }
//
//    Vec2 velocityAtRotatedPoint(const Vec2& p) const
//    {
//        return vCenter + Vec2(-p[1], p[0]) * vOrientation;
//    }
//
//    /// write to an output stream
//    inline friend std::ostream& operator << ( std::ostream& out, const RigidDeriv<2,real>& v )
//    {
//        out<<v.vCenter<<" "<<v.vOrientation;
//        return out;
//    }
//    /// read from an input stream
//    inline friend std::istream& operator >> ( std::istream& in, RigidDeriv<2,real>& v )
//    {
//        in>>v.vCenter>>v.vOrientation;
//        return in;
//    }
//
//    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
//    enum { total_size = 3 };
//    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
//    enum { spatial_dimensions = 2 };
//
//    real* ptr() { return vCenter.ptr(); }
//    const real* ptr() const { return vCenter.ptr(); }
//
//	static unsigned int size(){return 3;}
//
//	/// Access to i-th element.
//    real& operator[](int i)
//    {
//		if (i<2)
//			return this->vCenter(i);
//		else
//			return this->vOrientation;
//    }
//
//    /// Const access to i-th element.
//    const real& operator[](int i) const
//    {
//		if (i<2)
//			return this->vCenter(i);
//		else
//			return this->vOrientation;
//    }
//};

template<typename R>
Vec<2,R>& getVCenter ( Vec<3,R>& v )
{
    // We cannot use casts if the compiler adds extra members in Vec
    BOOST_STATIC_ASSERT(sizeof(Vec<3,R>) == 3*sizeof(R));
    BOOST_STATIC_ASSERT(sizeof(Vec<2,R>) == 2*sizeof(R));
    return *reinterpret_cast<Vec<2,R>*>(&v[0]);
}

template<typename R>
R& getVOrientation ( Vec<3,R>& v )
{
    return v[2];
}

template<typename R>
const Vec<2,R>& getVCenter ( const Vec<3,R>& v )
{
    // We cannot use casts if the compiler adds extra members in Vec
    BOOST_STATIC_ASSERT(sizeof(Vec<3,R>) == 3*sizeof(R));
    BOOST_STATIC_ASSERT(sizeof(Vec<2,R>) == 2*sizeof(R));
    return *reinterpret_cast<const Vec<2,R>*>(&v[0]);
}

template<typename R>
const R& getVOrientation ( const Vec<3,R>& v )
{
    return v[2];
}

template<typename R, typename T>
Vec<2,R> velocityAtRotatedPoint(const Vec<3,T>& v, const Vec<2,R>& p)
{
    return getVCenter(v) + Vec<2,R>(-p[1], p[0]) * getVOrientation(v);
}



template<typename real>
class RigidCoord<2,real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Vec<2,Real> Pos;
    typedef Real Rot;
    typedef Vec<2,Real> Vec2;
private:
    Vec2 center;
    Real orientation;
public:
    RigidCoord (const Vec2 &posCenter, const Real &orient)
        : center(posCenter), orientation(orient) {}
    RigidCoord () { clear(); }

    void clear() { center.clear(); orientation = 0; }

    void operator +=(const Vec<3,real>& a)
    {
        center += getVCenter(a);
        orientation += getVOrientation(a);
    }

    RigidCoord<2,real> operator + (const Vec<3,real>& a) const
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

    void operator +=(const RigidCoord<2,real>& a)
    {
        //         std::cout << "+="<<std::endl;
        center += a.getCenter();
        orientation += a.getOrientation();
    }

    template<typename real2>
    void operator*=(real2 a)
    {
        //         std::cout << "*="<<std::endl;
        center *= a;
        orientation *= (Real)a;
    }

    template<typename real2>
    void operator/=(real2 a)
    {
        //         std::cout << "/="<<std::endl;
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

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const RigidCoord<2,real>& a) const
    {
        return center[0]*a.center[0]+center[1]*a.center[1]
                +orientation*a.orientation;
    }

    /// Squared norm
    real norm2() const
    {
        return center[0]*center[0]+center[1]*center[1];
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
inline Vec<3,real> operator*(const Vec<3,real>& d, const RigidMass<2,real>& m)
{
    Vec<3,real> res;
    getVCenter(res) = getVCenter(d) * m.mass;
    getVOrientation(res) = m.inertiaMassMatrix * getVOrientation(d);
    return res;
}

template<typename real>
inline Vec<3,real> operator/(const Vec<3,real>& d, const RigidMass<2, real>& m)
{
    Vec<3,real> res;
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

    typedef Vec<3,real> Deriv;
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

    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;
    typedef vector<Real> VecReal;

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
        getVCenter(c)[0] = (Real)x;
        getVCenter(c)[1] = (Real)y;
    }

    template<typename T>
    static void get(T& x, T& y, T& z, const Deriv& c)
    {
        x = (T)getVCenter(c)[0];
        y = (T)getVCenter(c)[1];
        z = (T)0;
    }

    template<typename T>
    static void add(Deriv& c, T x, T y, T)
    {
        getVCenter(c)[0] += (Real)x;
        getVCenter(c)[1] += (Real)y;
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

};

typedef StdRigidTypes<2,double> Rigid2dTypes;
typedef StdRigidTypes<2,float> Rigid2fTypes;

typedef RigidMass<2,double> Rigid2dMass;
typedef RigidMass<2,float> Rigid2fMass;

template<> inline const char* Rigid2dTypes::Name() { return "Rigid2d"; }
template<> inline const char* Rigid2fTypes::Name() { return "Rigid2f"; }

#ifdef SOFA_FLOAT
typedef Rigid2fTypes Rigid2Types;
typedef Rigid2fMass Rigid2Mass;
#else
typedef Rigid2dTypes Rigid2Types;
typedef Rigid2dMass Rigid2Mass;
#endif



// Specialization of the defaulttype::DataTypeInfo type traits template

//        template<typename real>
//        struct DataTypeInfo< sofa::defaulttype::Vec<6,real> > : public FixedArrayTypeInfo< sofa::defaulttype::Vec<6,real>, sofa::defaulttype::Vec<6,real>::total_size >
//        {
//            static std::string name() { std::ostringstream o; o << "RigidDeriv<" << 6 << "," << DataTypeName<real>::name() << ">"; return o.str(); }
//        };
//
//        template<typename real>
//        struct DataTypeInfo< sofa::defaulttype::Vec<3,real> > : public FixedArrayTypeInfo< sofa::defaulttype::Vec<3,real>, sofa::defaulttype::Vec<3,real>::total_size >
//        {
//            static std::string name() { std::ostringstream o; o << "RigidDeriv<" << 3 << "," << DataTypeName<real>::name() << ">"; return o.str(); }
//        };

template<int N, typename real>
struct DataTypeInfo< sofa::defaulttype::RigidCoord<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::RigidCoord<N,real>, sofa::defaulttype::RigidCoord<N,real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "RigidCoord<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Rigid2fTypes::Coord > { static const char* name() { return "Rigid2fTypes::Coord"; } };
//        template<> struct DataTypeName< defaulttype::Rigid2fTypes::Deriv > { static const char* name() { return "Rigid2fTypes::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid2dTypes::Coord > { static const char* name() { return "Rigid2dTypes::Coord"; } };
//        template<> struct DataTypeName< defaulttype::Rigid2dTypes::Deriv > { static const char* name() { return "Rigid2dTypes::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid3fTypes::Coord > { static const char* name() { return "Rigid3fTypes::Coord"; } };
//        template<> struct DataTypeName< defaulttype::Rigid3fTypes::Deriv > { static const char* name() { return "Rigid3fTypes::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid3dTypes::Coord > { static const char* name() { return "Rigid3dTypes::Coord"; } };
//        template<> struct DataTypeName< defaulttype::Rigid3dTypes::Deriv > { static const char* name() { return "Rigid3dTypes::Deriv"; } };
template<> struct DataTypeName< defaulttype::Rigid2fMass > { static const char* name() { return "Rigid2fMass"; } };
template<> struct DataTypeName< defaulttype::Rigid2dMass > { static const char* name() { return "Rigid2dMass"; } };
template<> struct DataTypeName< defaulttype::Rigid3fMass > { static const char* name() { return "Rigid3fMass"; } };
template<> struct DataTypeName< defaulttype::Rigid3dMass > { static const char* name() { return "Rigid3dMass"; } };

/// \endcond


} // namespace defaulttype

namespace core
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
inline defaulttype::Vec<6, double> inertiaForce<
defaulttype::RigidCoord<3, double>,
            defaulttype::Vec<6, double>,
            objectmodel::BaseContext::Vec3,
            defaulttype::RigidMass<3, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::RigidMass<3, double>& mass,
                    const defaulttype::RigidCoord<3, double>& x,
                    const defaulttype::Vec<6, double>& v
            )
{
    defaulttype::Vec<3, double> omega( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::Vec<3, double> origin = x.getCenter(), finertia;

    finertia = -( aframe + omega.cross( omega.cross(origin) + getVCenter(v)*2 ))*mass.mass;
    return defaulttype::Vec<6, double>( finertia[0], finertia[1], finertia[2], 0., 0., 0. );
    /// \todo replace zero by Jomega.cross(omega)
}

/// Specialization of the inertia force for defaulttype::Rigid3fTypes
template <>
inline defaulttype::Vec<6, float> inertiaForce<
defaulttype::RigidCoord<3, float>,
            defaulttype::Vec<6, float>,
            objectmodel::BaseContext::Vec3,
            defaulttype::RigidMass<3, float>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::RigidMass<3, float>& mass,
                    const defaulttype::RigidCoord<3, float>& x,
                    const defaulttype::Vec<6, float>& v
            )
{
    defaulttype::Vec<3, float> omega( (float)vframe.lineVec[0], (float)vframe.lineVec[1], (float)vframe.lineVec[2] );
    defaulttype::Vec<3, float> origin = x.getCenter(), finertia;

    finertia = -( aframe + omega.cross( omega.cross(origin) + getVCenter(v)*2 ))*mass.mass;
    return defaulttype::Vec<6, float>( finertia[0], finertia[1], finertia[2], 0., 0., 0. );
    /// \todo replace zero by Jomega.cross(omega)
}

} // namespace behavoir

} // namespace core

} // namespace sofa


#endif
