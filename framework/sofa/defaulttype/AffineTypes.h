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
#ifndef SOFA_DEFAULTTYPE_AFFINETYPES_H
#define SOFA_DEFAULTTYPE_AFFINETYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/core/objectmodel/BaseContext.h>
//#include <sofa/core/behavior/Mass.h>
#ifdef SOFA_SMP
#include <sofa/defaulttype/SharedTypes.h>
#endif /* SOFA_SMP */
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <iostream>

namespace sofa
{

namespace defaulttype
{

using std::endl;
using sofa::helper::vector;

template<int N, typename real>

class AffineDeriv;

template<int N, typename real>

class AffineCoord;

template<int N, typename real>

class AffineMass;

template<int N, typename real>

class StdAffineTypes;

//=============================================================================
// 3D Affines
//=============================================================================



/** Degrees of freedom of 3D rigid bodies. Orientations are modeled using quaternions.
*/
template<typename real>

class AffineDeriv<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Vec<3, Real> Vec3;
    typedef Vec3 Pos;
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef Mat33 Affine;

protected:
    Vec3 vCenter;
    Mat33 vAffine;
public:

    friend class AffineCoord<3, real>;

    AffineDeriv ( const Vec3 &vCenter, const Mat33 &vAffine )
        : vCenter ( vCenter ), vAffine ( vAffine ) {}

    AffineDeriv() { clear(); }

    template<typename real2>
    AffineDeriv ( const AffineDeriv<3, real2>& c )
        : vCenter ( c.getVCenter() ), vAffine ( c.getVAffine() )
    {
    }

    void clear() { vCenter.clear(); vAffine.clear(); }

    template<typename real2>
    void operator = ( const AffineDeriv<3, real2>& c )
    {
        vCenter = c.getVCenter();
        vAffine = c.getVAffine();
    }

    void operator += ( const AffineDeriv& a )
    {
        vCenter += a.vCenter;
        vAffine += a.vAffine;
    }

    void operator -= ( const AffineDeriv& a )
    {
        vCenter -= a.vCenter;
        vAffine -= a.vAffine;
    }

    AffineDeriv<3, real> operator + ( const AffineDeriv<3, real>& a ) const
    {
        AffineDeriv d;
        d.vCenter = vCenter + a.vCenter;
        d.vAffine = vAffine + a.vAffine;
        return d;
    }

    template<typename real2>
    void operator*= ( real2 a )
    {
        vCenter *= (Real)a;
        vAffine *= (Real)a;
    }

    template<typename real2>
    void operator/= ( real2 a )
    {
        vCenter /= (Real)a;
        vAffine /= (Real)a;
    }

    AffineDeriv<3, real> operator* ( float a ) const
    {
        AffineDeriv r = *this;
        r *= a;
        return r;
    }

    AffineDeriv<3, real> operator* ( double a ) const
    {
        AffineDeriv r = *this;
        r *= a;
        return r;
    }

    AffineDeriv<3, real> operator - () const
    {
        Affine tmp;
        tmp[0][0] = - vAffine[0][0];
        tmp[0][1] = - vAffine[0][1];
        tmp[0][2] = - vAffine[0][2];
        tmp[1][0] = - vAffine[1][0];
        tmp[1][1] = - vAffine[1][1];
        tmp[1][2] = - vAffine[1][2];
        tmp[2][0] = - vAffine[2][0];
        tmp[2][1] = - vAffine[2][1];
        tmp[2][2] = - vAffine[2][2];
        return AffineDeriv ( -vCenter, tmp );
    }

    AffineDeriv<3, real> operator - ( const AffineDeriv<3, real>& a ) const
    {
        return AffineDeriv<3, real> ( this->vCenter - a.vCenter, this->vAffine - a.vAffine );
    }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator* ( const AffineDeriv<3, real>& a ) const
    {
        return vCenter[0]*a.vCenter[0] + vCenter[1]*a.vCenter[1] + vCenter[2]*a.vCenter[2]
                + vAffine(0,0)*a.vAffine(0,0) + vAffine(0,1)*a.vAffine(0,1) + vAffine(0,2)*a.vAffine(0,2)
                + vAffine(1,0)*a.vAffine(1,0) + vAffine(1,1)*a.vAffine(1,1) + vAffine(1,2)*a.vAffine(1,2)
                + vAffine(2,0)*a.vAffine(2,0) + vAffine(2,1)*a.vAffine(2,1) + vAffine(2,2)*a.vAffine(2,2);
    }

    Vec3& getVCenter ( void ) { return vCenter; }

    Mat33& getVAffine ( void ) { return vAffine; }

    const Vec3& getVCenter ( void ) const { return vCenter; }

    const Mat33& getVAffine ( void ) const { return vAffine; }

    Vec3& getLinear () { return vCenter; }

    const Vec3& getLinear () const { return vCenter; }

    Vec3 velocityAtRotatedPoint ( const Vec3& p ) const
    {
        return vCenter - vAffine * p;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const AffineDeriv<3, real>& v )
    {
        out << v.vCenter << " " << v.vAffine;
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, AffineDeriv<3, real>& v )
    {
        in >> v.vCenter >> v.vAffine;
        return in;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 12 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 3 };

    real* ptr() { return vCenter.ptr(); }

    const real* ptr() const { return vCenter.ptr(); }

    static unsigned int size() {return 12;}

    /// Access to i-th element.
    real& operator[] ( int i )
    {
        if ( i<3 )
            return this->vCenter ( i );
        else
            return this->vAffine((i-3)/3, (i-3)%3);
    }

    /// Const access to i-th element.
    const real& operator[] ( int i ) const
    {
        if ( i<3 )
            return this->vCenter ( i );
        else
            return this->vAffine((i-3)/3, (i-3)%3);
    }
};










template<typename real>

class AffineCoord<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Vec<3, Real> Vec3;
    typedef Vec3 Pos;
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef Mat33 Affine;

protected:
    Vec3 center;
    Affine affine;
public:
    AffineCoord ( const Vec3 &center, const Affine &affine )
        : center ( center ), affine ( affine ) {}

    AffineCoord () { clear(); }

    template<typename real2>
    AffineCoord ( const AffineCoord<3, real2>& c )
        : center ( c.getCenter() ), affine ( c.getAffine() )
    {
    }

    void clear() { center.clear(); affine.identity(); }

    template<typename real2>
    void operator = ( const AffineCoord<3, real2>& c )
    {
        center = c.getCenter();
        affine = c.getAffine();
    }

    void operator += ( const AffineDeriv<3, real>& a )
    {
        center += a.getVCenter();
        affine += a.getVAffine();
    }

    AffineCoord<3, real> operator + ( const AffineDeriv<3, real>& a ) const
    {
        AffineCoord c = *this;
        c.center += a.getVCenter();
        c.affine += a.getVAffine();
        return c;
    }

    // Can be improved by using the polar decomposition (composition of rotations and substraction of shear and stretch components)
    AffineCoord<3, real> operator - ( const AffineCoord<3, real>& a ) const
    {
        return AffineCoord<3, real> ( this->center - a.getCenter(), a.affine.inverse() * this->affine );
    }

    // Can be improved by using the polar decomposition (composition of rotations and addition of shear and stretch components)
    AffineCoord<3, real> operator + ( const AffineCoord<3, real>& a ) const
    {
        return AffineCoord<3, real> ( this->center + a.getCenter(), a.affine * this->affine );
    }

    void operator += ( const AffineCoord<3, real>& a )
    {
        center += a.getCenter();
        //affine += a.affine;
    }

    template<typename real2>
    void operator*= ( real2 a )
    {
        center *= a;
        //affine *= a;
    }

    template<typename real2>
    void operator/= ( real2 a )
    {
        center /= a;
        //affine /= a;
    }

    template<typename real2>
    AffineCoord<3, real> operator* ( real2 a ) const
    {
        AffineCoord r = *this;
        r *= a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator* ( const AffineCoord<3, real>& a ) const
    {
        return center[0]*a.center[0] + center[1]*a.center[1] + center[2]*a.center[2]
                + affine(0,0)*a.affine(0,0) + affine(0,1)*a.affine(0,1) + affine(0,2)*a.affine(0,2)
                + affine(1,0)*a.affine(1,0) + affine(1,1)*a.affine(1,1) + affine(1,2)*a.affine(1,2)
                + affine(2,0)*a.affine(2,0) + affine(2,1)*a.affine(2,1) + affine(2,2)*a.affine(2,2);
    }

    /// Squared norm
    real norm2() const
    {
        real r = ( this->center ).elems[0] * ( this->center ).elems[0];

        for ( int i = 1; i < 3; i++ )
            r += ( this->center ).elems[i] * ( this->center ).elems[i];

        return r;
    }

    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt ( norm2() );
    }


    Vec3& getCenter () { return center; }

    Mat33& getAffine () { return affine; }

    const Vec3& getCenter () const { return center; }

    const Mat33& getAffine () const { return affine; }

    static AffineCoord<3, real> identity()
    {
        AffineCoord c;
        return c;
    }

    Vec3 rotateAndDeform ( const Vec3& v ) const
    {
        return affine * v;
    }

    Vec3 inverseRotateAndDeform ( const Vec3& v ) const
    {
        return affine.inverse() * v;
    }
    /*
              /// Apply a transformation with respect to itself
              void multRight ( const AffineCoord<3, real>& c )
              {
                center += affine * c.getCenter();
                affine = affine * c.getAffine();
                //TODO decomposition polaire. => mult rot, ajoute S terme a terme.
              }

              /// compute the product with another frame on the right
              AffineCoord<3, real> mult ( const AffineCoord<3, real>& c ) const
                {
                  AffineCoord r;
                  r.center = center + orientation.rotate ( c.center );
                  r.orientation = orientation * c.getOrientation();
                  return r;
                }
    */
    /// Set from the given matrix
    template<class Mat>
    void fromMatrix ( const Mat& m )
    {
        center[0] = m[0][3];
        center[1] = m[1][3];
        center[2] = m[2][3];
        affine(0,0) = m[0][0];
        affine(0,1) = m[0][1];
        affine(0,2) = m[0][2];
        affine(1,0) = m[1][0];
        affine(1,1) = m[1][1];
        affine(1,2) = m[1][2];
        affine(2,0) = m[2][0];
        affine(2,1) = m[2][1];
        affine(2,2) = m[2][2];
    }

    /// Write to the given matrix
    template<class Mat>
    void toMatrix ( Mat& m ) const
    {
        m.identity();
        m[0][0] = affine(0,0);
        m[0][1] = affine(0,1);
        m[0][2] = affine(0,2);
        m[1][0] = affine(1,0);
        m[1][1] = affine(1,1);
        m[1][2] = affine(1,2);
        m[2][0] = affine(2,0);
        m[2][1] = affine(2,1);
        m[2][2] = affine(2,2);
        m[0][3] = center[0];
        m[1][3] = center[1];
        m[2][3] = center[2];
    }
    /*
              template<class Mat>
              void writeRotationMatrix ( Mat& m ) const
                {
                  orientation.toMatrix ( m );
                }
    */
    /// Write the OpenGL transformation matrix
    void writeOpenGlMatrix ( float m[16] ) const
    {
        m[0] = affine(0,0);
        m[4] = affine(0,1);
        m[8] = affine(0,2);
        m[1] = affine(1,0);
        m[5] = affine(1,1);
        m[9] = affine(1,2);
        m[2] = affine(2,0);
        m[6] = affine(2,1);
        m[10] = affine(2,2);
        m[3] = 0;
        m[7] = 0;
        m[11] = 0;
        m[12] = ( float ) center[0];
        m[13] = ( float ) center[1];
        m[14] = ( float ) center[2];
        m[15] = 1;
    }
    /*
              /// compute the projection of a vector from the parent frame to the child
              Vec3 vectorToChild ( const Vec3& v ) const
                {
                  return orientation.inverseRotate ( v );
                }
    */
    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const AffineCoord<3, real>& v )
    {
        out << v.center << " " << v.affine;
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, AffineCoord<3, real>& v )
    {
        in >> v.center >> v.affine;
        return in;
    }

    static int max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 12 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 3 };

    real* ptr() { return center.ptr(); }

    const real* ptr() const { return center.ptr(); }

    static unsigned int size() {return 12;}

    /// Access to i-th element.
    real& operator[] ( int i )
    {
        if ( i<3 )
            return this->center ( i );
        else
            return this->affine((i-3)/3, (i-3)%3);
    }

    /// Const access to i-th element.
    const real& operator[] ( int i ) const
    {
        if ( i<3 )
            return this->center ( i );
        else
            return this->affine((i-3)/3, (i-3)%3);
    }
};






template<typename real>

class AffineMass<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Mat<3, 3, Real> Mat3x3;
    Real mass, volume;
    Mat3x3 inertiaMatrix;       // Inertia matrix of the object
    Mat3x3 inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat3x3 invInertiaMatrix;    // inverse of inertiaMatrix
    Mat3x3 invInertiaMassMatrix; // inverse of inertiaMassMatrix
    AffineMass ( Real m = 1 )
    {
        mass = m;
        volume = 1;
        inertiaMatrix.identity();
        recalc();
    }

    void operator= ( Real m )
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
        invInertiaMatrix.invert ( inertiaMatrix );
        invInertiaMassMatrix.invert ( inertiaMassMatrix );
    }

    inline friend std::ostream& operator << ( std::ostream& out, const AffineMass<3, real>& m )
    {
        out << m.mass;
        out << " " << m.volume;
        out << " " << m.inertiaMatrix;
        return out;
    }

    inline friend std::istream& operator >> ( std::istream& in, AffineMass<3, real>& m )
    {
        in >> m.mass;
        in >> m.volume;
        in >> m.inertiaMatrix;
        return in;
    }

    void operator *= ( Real fact )
    {
        mass *= fact;
        inertiaMassMatrix *= fact;
        invInertiaMassMatrix /= fact;
    }

    void operator /= ( Real fact )
    {
        mass /= fact;
        inertiaMassMatrix /= fact;
        invInertiaMassMatrix *= fact;
    }
};

template<int N, typename real>
inline AffineDeriv<N, real> operator* ( const AffineDeriv<N, real>& d, const AffineMass<N, real>& m )
{
    AffineDeriv<N, real> res;
    res.getVCenter() = d.getVCenter() * m.mass;
    res.getVOrientation() = m.inertiaMassMatrix * d.getVOrientation();
    return res;
}

template<int N, typename real>
inline AffineDeriv<N, real> operator/ ( const AffineDeriv<N, real>& d, const AffineMass<N, real>& m )
{
    AffineDeriv<N, real> res;
    res.getVCenter() = d.getVCenter() / m.mass;
    res.getVOrientation() = m.invInertiaMassMatrix * d.getVOrientation();
    return res;
}






template<typename real>

class StdAffineTypes<3, real>
{
public:
    typedef real Real;
    typedef AffineCoord<3, real> Coord;
    typedef AffineDeriv<3, real> Deriv;
    typedef typename Coord::Vec3 Vec3;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Affine CAffine;
    static const CPos& getCPos ( const Coord& c ) { return c.getCenter(); }

    static void setCPos ( Coord& c, const CPos& v ) { c.getCenter() = v; }

    static const CAffine& getCRot ( const Coord& c ) { return c.getAffine(); }

    static void setCRot ( Coord& c, const CAffine& v ) { c.getAffine() = v; }

    typedef typename Deriv::Pos DPos;
    typedef typename Deriv::Affine DAffine;
    static const DPos& getDPos ( const Deriv& d ) { return d.getVCenter(); }

    static void setDPos ( Deriv& d, const DPos& v ) { d.getVCenter() = v; }

    static const DAffine& getDAffine ( const Deriv& d ) { return d.getVAffine(); }

    static void setDAffine ( Deriv& d, const DAffine& v ) { d.getVAffine() = v; }

    //  typedef SparseConstraint<Coord> SparseVecCoord;
    //  typedef SparseConstraint<Deriv> SparseVecDeriv;

    //! All the Constraints applied to a state Vector
#ifndef SOFA_SMP
    //  typedef vector<SparseVecDeriv> VecConst;
#else /* SOFA_SMP */
    //  typedef SharedVector<SparseVecDeriv> VecConst;
#endif /* SOFA_SMP */

    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

#ifndef SOFA_SMP
    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;
    typedef vector<Real> VecReal;
#else /* SOFA_SMP */
    typedef SharedVector<Coord> VecCoord;
    typedef SharedVector<Deriv> VecDeriv;
    typedef SharedVector<Real> VecReal;
#endif /* SOFA_SMP */

    template<typename T>
    static void set ( Coord& c, T x, T y, T z )
    {
        c.getCenter() [0] = ( Real ) x;
        c.getCenter() [1] = ( Real ) y;
        c.getCenter() [2] = ( Real ) z;
    }

    template<typename T>
    static void get ( T& x, T& y, T& z, const Coord& c )
    {
        x = ( T ) c.getCenter() [0];
        y = ( T ) c.getCenter() [1];
        z = ( T ) c.getCenter() [2];
    }

    template<typename T>
    static void add ( Coord& c, T x, T y, T z )
    {
        c.getCenter() [0] += ( Real ) x;
        c.getCenter() [1] += ( Real ) y;
        c.getCenter() [2] += ( Real ) z;
    }

    template<typename T>
    static void set ( Deriv& c, T x, T y, T z )
    {
        c.getVCenter() [0] = ( Real ) x;
        c.getVCenter() [1] = ( Real ) y;
        c.getVCenter() [2] = ( Real ) z;
    }

    template<typename T>
    static void get ( T& x, T& y, T& z, const Deriv& c )
    {
        x = ( T ) c.getVCenter() [0];
        y = ( T ) c.getVCenter() [1];
        z = ( T ) c.getVCenter() [2];
    }

    template<typename T>
    static void add ( Deriv& c, T x, T y, T z )
    {
        c.getVCenter() [0] += ( Real ) x;
        c.getVCenter() [1] += ( Real ) y;
        c.getVCenter() [2] += ( Real ) z;
    }

    static const char* Name();

    static Coord interpolate ( const helper::vector< Coord > & ancestors, const helper::vector< Real > & coefs )
    {
        assert ( ancestors.size() == coefs.size() );

        Coord c;

        for ( unsigned int i = 0; i < ancestors.size(); i++ )
        {
            // Position interpolation.
            c.getCenter() += ancestors[i].getCenter() * coefs[i];
            c.getAffine() += ancestors[i].getAffine() * coefs[i]; // Linear blend skinning (peut etre amelioré avec la decomposition polaire (cf. interpolation des Rigides pour le code original)
        }

        return c;
    }

    static Deriv interpolate ( const helper::vector< Deriv > & ancestors, const helper::vector< Real > & coefs )
    {
        assert ( ancestors.size() == coefs.size() );

        Deriv d;

        for ( unsigned int i = 0; i < ancestors.size(); i++ )
        {
            d += ancestors[i] * coefs[i];
        }

        return d;
    }
};




typedef StdAffineTypes<3, double> Affine3dTypes;
typedef StdAffineTypes<3, float> Affine3fTypes;

typedef AffineMass<3, double> Affine3dMass;
typedef AffineMass<3, float> Affine3fMass;
//typedef Affine3Mass AffineMass;

/// Note: Many scenes use Affine as template for 3D double-precision rigid type. Changing it to Affine3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* Affine3dTypes::Name() { return "Affine3d"; }

template<> inline const char* Affine3fTypes::Name() { return "Affine"; }

#else
template<> inline const char* Affine3dTypes::Name() { return "Affine"; }

template<> inline const char* Affine3fTypes::Name() { return "Affine3f"; }

#endif

#ifdef SOFA_FLOAT
typedef Affine3fTypes Affine3Types;
typedef Affine3fMass Affine3Mass;
#else
typedef Affine3dTypes Affine3Types;
typedef Affine3dMass Affine3Mass;
#endif
typedef Affine3Types AffineTypes;







/*
//=============================================================================
// 2D Affines
//=============================================================================

template<typename real>
class AffineDeriv<2,real>
{
public:
  typedef real value_type;
    typedef real Real;
    typedef Vec<2,Real> Pos;
    typedef Real Rot;
    typedef Vec<2,Real> Vec2;
private:
    Vec2 vCenter;
    Real vOrientation;
public:
    friend class AffineCoord<2,real>;

    AffineDeriv (const Vec2 &velCenter, const Real &velOrient)
    : vCenter(velCenter), vOrientation(velOrient) {}
    AffineDeriv () { clear(); }

    void clear() { vCenter.clear(); vOrientation=0; }

    void operator +=(const AffineDeriv<2,real>& a)
    {
        vCenter += a.vCenter;
        vOrientation += a.vOrientation;
    }

    AffineDeriv<2,real> operator + (const AffineDeriv<2,real>& a) const
    {
        AffineDeriv<2,real> d;
        d.vCenter = vCenter + a.vCenter;
        d.vOrientation = vOrientation + a.vOrientation;
        return d;
    }

  AffineDeriv<2,real> operator - (const AffineDeriv<2,real>& a) const
    {
        AffineDeriv<2,real> d;
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

    AffineDeriv<2,real> operator*(float a) const
    {
        AffineDeriv<2,real> r = *this;
        r *= a;
        return r;
    }

    AffineDeriv<2,real> operator*(double a) const
    {
        AffineDeriv<2,real> r = *this;
        r *= a;
        return r;
    }

    AffineDeriv<2,real> operator - () const
    {
        return AffineDeriv<2,real>(-vCenter, -vOrientation);
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const AffineDeriv<2,real>& a) const
    {
        return vCenter[0]*a.vCenter[0]+vCenter[1]*a.vCenter[1]
            +vOrientation*a.vOrientation;
    }

    Vec2& getVCenter (void) { return vCenter; }
    Real& getVOrientation (void) { return vOrientation; }
    const Vec2& getVCenter (void) const { return vCenter; }
    const Real& getVOrientation (void) const { return vOrientation; }

    Vec2 velocityAtRotatedPoint(const Vec2& p) const
    {
        return vCenter + Vec2(-p[1], p[0]) * vOrientation;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const AffineDeriv<2,real>& v )
    {
        out<<v.vCenter<<" "<<v.vOrientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, AffineDeriv<2,real>& v )
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

  static unsigned int size(){return 3;}

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
};

template<typename real>
class AffineCoord<2,real>
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
    AffineCoord (const Vec2 &posCenter, const Real &orient)
    : center(posCenter), orientation(orient) {}
    AffineCoord () { clear(); }

    void clear() { center.clear(); orientation = 0; }

    void operator +=(const AffineDeriv<2,real>& a)
    {
        center += a.getVCenter();
        orientation += a.getVOrientation();
    }

    AffineCoord<2,real> operator + (const AffineDeriv<2,real>& a) const
    {
        AffineCoord<2,real> c = *this;
        c.center += a.getVCenter();
        c.orientation += a.getVOrientation();
        return c;
    }

    AffineCoord<2,real> operator -(const AffineCoord<2,real>& a) const
    {
        return AffineCoord<2,real>(this->center - a.getCenter(), this->orientation - a.orientation);
    }

    AffineCoord<2,real> operator +(const AffineCoord<2,real>& a) const
    {
        return AffineCoord<2,real>(this->center + a.getCenter(), this->orientation + a.orientation);
    }

    void operator +=(const AffineCoord<2,real>& a)
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
    AffineCoord<2,real> operator*(real2 a) const
    {
        AffineCoord<2,real> r = *this;
        r *= a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const AffineCoord<2,real>& a) const
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

    static AffineCoord<2,real> identity()
    {
        AffineCoord<2,real> c;
        return c;
    }

    /// Apply a transformation with respect to itself
    void multRight( const AffineCoord<2,real>& c )
    {
        //center += orientation.rotate(c.getCenter());
        center += rotate(c.getCenter());
        orientation = orientation + c.getOrientation();
    }

    /// compute the product with another frame on the right
    AffineCoord<2,real> mult( const AffineCoord<2,real>& c ) const
    {
        AffineCoord<2,real> r;
        //r.center = center + orientation.rotate( c.center );
        r.center = center + rotate( c.center );
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
        //return orientation.inverseRotate(v);
        return inverseRotate(v);
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const AffineCoord<2,real>& v )
    {
        out<<v.center<<" "<<v.orientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, AffineCoord<2,real>& v )
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

  static unsigned int size(){return 3;}

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
class AffineMass<2, real>
{
public:
  typedef real value_type;
    typedef real Real;
    Real mass,volume;
    Real inertiaMatrix;       // Inertia matrix of the object
    Real inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Real invInertiaMatrix;    // inverse of inertiaMatrix
    Real invInertiaMassMatrix; // inverse of inertiaMassMatrix
    AffineMass(Real m=1)
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
    AffineMass(Real m, Real radius)
    {
        mass = m;
        volume = radius*radius*R_PI;
        inertiaMatrix = (radius*radius)/2;
        recalc();
    }
    /// Mass for a rectangle
    AffineMass(Real m, Real xwidth, Real ywidth)
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
    inline friend std::ostream& operator << (std::ostream& out, const AffineMass<2,Real>& m )
    {
        out<<m.mass;
        out<<" "<<m.volume;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, AffineMass<2,Real>& m )
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

/// Degrees of freedom of 2D rigid bodies.
template<typename real>
class StdAffineTypes<2, real>
{
public:
    typedef real Real;
    typedef Vec<2,real> Vec2;

    typedef AffineDeriv<2,Real> Deriv;
    typedef AffineCoord<2,Real> Coord;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Rot CRot;
  static const CPos& getCPos(const Coord& c) { return c.getCenter(); }
  static void setCPos(Coord& c, const CPos& v) { c.getCenter() = v; }
  static const CRot& getCRot(const Coord& c) { return c.getOrientation(); }
  static void setCRot(Coord& c, const CRot& v) { c.getOrientation() = v; }

    typedef typename Deriv::Pos DPos;
    typedef typename Deriv::Rot DRot;
  static const DPos& getDPos(const Deriv& d) { return d.getVCenter(); }
  static void setDPos(Deriv& d, const DPos& v) { d.getVCenter() = v; }
  static const DRot& getDRot(const Deriv& d) { return d.getVOrientation(); }
  static void setDRot(Deriv& d, const DRot& v) { d.getVOrientation() = v; }

    static const char* Name();

#ifndef SOFA_SMP
    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;
#else
    typedef SharedVector<Coord> VecCoord;
    typedef SharedVector<Deriv> VecDeriv;
#endif

    typedef SparseConstraint<Coord> SparseVecCoord;
    typedef SparseConstraint<Deriv> SparseVecDeriv;
#ifndef SOFA_SMP
    typedef vector<Real> VecReal;
#else
    typedef SharedVector<Real> VecReal;
#endif

#ifndef SOFA_SMP
    typedef vector<SparseVecDeriv> VecConst;
#else
    typedef SharedVector<SparseVecDeriv> VecConst;
#endif

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

    template<typename T>
    static void add(Deriv& c, T x, T y, T)
    {
        c.getVCenter()[0] += (Real)x;
        c.getVCenter()[1] += (Real)y;
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

};

typedef StdAffineTypes<2,double> Affine2dTypes;
typedef StdAffineTypes<2,float> Affine2fTypes;

typedef AffineMass<2,double> Affine2dMass;
typedef AffineMass<2,float> Affine2fMass;

template<> inline const char* Affine2dTypes::Name() { return "Affine2d"; }
template<> inline const char* Affine2fTypes::Name() { return "Affine2f"; }

#ifdef SOFA_FLOAT
typedef Affine2fTypes Affine2Types;
typedef Affine2fMass Affine2Mass;
#else
typedef Affine2dTypes Affine2Types;
typedef Affine2dMass Affine2Mass;
#endif
*/


// Specialization of the defaulttype::DataTypeInfo type traits template

template<int N, typename real>

struct DataTypeInfo< sofa::defaulttype::AffineDeriv<N, real> > : public FixedArrayTypeInfo< sofa::defaulttype::AffineDeriv<N, real>, sofa::defaulttype::AffineDeriv<N, real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "AffineDeriv<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

template<int N, typename real>

struct DataTypeInfo< sofa::defaulttype::AffineCoord<N, real> > : public FixedArrayTypeInfo< sofa::defaulttype::AffineCoord<N, real>, sofa::defaulttype::AffineCoord<N, real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "AffineCoord<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

//template<> struct DataTypeName< defaulttype::Affine2fTypes::Coord > { static const char* name() { return "Affine2fTypes::Coord"; } };
//template<> struct DataTypeName< defaulttype::Affine2fTypes::Deriv > { static const char* name() { return "Affine2fTypes::Deriv"; } };
//template<> struct DataTypeName< defaulttype::Affine2dTypes::Coord > { static const char* name() { return "Affine2dTypes::Coord"; } };
//template<> struct DataTypeName< defaulttype::Affine2dTypes::Deriv > { static const char* name() { return "Affine2dTypes::Deriv"; } };

template<> struct DataTypeName< defaulttype::Affine3fTypes::Coord > { static const char* name() { return "Affine3fTypes::Coord"; } };

template<> struct DataTypeName< defaulttype::Affine3fTypes::Deriv > { static const char* name() { return "Affine3fTypes::Deriv"; } };

template<> struct DataTypeName< defaulttype::Affine3dTypes::Coord > { static const char* name() { return "Affine3dTypes::Coord"; } };

template<> struct DataTypeName< defaulttype::Affine3dTypes::Deriv > { static const char* name() { return "Affine3dTypes::Deriv"; } };

//template<> struct DataTypeName< defaulttype::Affine2fMass > { static const char* name() { return "Affine2fMass"; } };
//template<> struct DataTypeName< defaulttype::Affine2dMass > { static const char* name() { return "Affine2dMass"; } };

template<> struct DataTypeName< defaulttype::Affine3fMass > { static const char* name() { return "Affine3fMass"; } };

template<> struct DataTypeName< defaulttype::Affine3dMass > { static const char* name() { return "Affine3dMass"; } };

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
Deriv inertiaForce ( const SV& /*sv*/, const Vec& /*a*/, const M& /*m*/, const Coord& /*x*/, const Deriv& /*v*/ );

/// Specialization of the inertia force for defaulttype::Affine3dTypes
template <>
inline defaulttype::AffineDeriv<3, double> inertiaForce <
defaulttype::AffineCoord<3, double>,
            defaulttype::AffineDeriv<3, double>,
            objectmodel::BaseContext::Vec3,
            defaulttype::AffineMass<3, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::AffineMass<3, double>& mass,
                    const defaulttype::AffineCoord<3, double>& x,
                    const defaulttype::AffineDeriv<3, double>& v
            )
{
    defaulttype::AffineDeriv<3, double>::Vec3 omega ( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::AffineDeriv<3, double>::Vec3 origin = x.getCenter(), finertia;
    defaulttype::AffineDeriv<3, double>::Mat33 zero;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() * 2 ) ) * mass.mass;
    return defaulttype::AffineDeriv<3, double> ( finertia, zero);
    /// \todo replace zero by Jomega.cross(omega)
}

/// Specialization of the inertia force for defaulttype::Affine3fTypes
template <>
inline defaulttype::AffineDeriv<3, float> inertiaForce <
defaulttype::AffineCoord<3, float>,
            defaulttype::AffineDeriv<3, float>,
            objectmodel::BaseContext::Vec3,
            defaulttype::AffineMass<3, float>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::AffineMass<3, float>& mass,
                    const defaulttype::AffineCoord<3, float>& x,
                    const defaulttype::AffineDeriv<3, float>& v
            )
{
    defaulttype::AffineDeriv<3, float>::Vec3 omega ( ( float ) vframe.lineVec[0], ( float ) vframe.lineVec[1], ( float ) vframe.lineVec[2] );
    defaulttype::AffineDeriv<3, float>::Vec3 origin = x.getCenter(), finertia;
    defaulttype::AffineDeriv<3, double>::Mat33 zero;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() * 2 ) ) * mass.mass;
    return defaulttype::AffineDeriv<3, float> ( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

} // namespace behavoir

} // namespace core

} // namespace sofa


#endif
