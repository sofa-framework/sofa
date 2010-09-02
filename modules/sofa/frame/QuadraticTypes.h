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
#ifndef SOFA_DEFAULTTYPE_QUADRATICTYPES_H
#define SOFA_DEFAULTTYPE_QUADRATICTYPES_H

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

class QuadraticDeriv;

template<int N, typename real>

class QuadraticCoord;

template<int N, typename real>

class QuadraticMass;

template<int N, typename real>

class StdQuadraticTypes;

//=============================================================================
// 3D Quadratics
//=============================================================================



/** Degrees of freedom of 3D rigid bodies. Orientations are modeled using quaternions.
*/
template<typename real>

class QuadraticDeriv<3, real>
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
    Mat33 vQuadratic;
    Mat33 vMixed;
public:

    friend class QuadraticCoord<3, real>;

    QuadraticDeriv ( const Vec3 &vCenter, const Mat33 &vAffine, const Mat33 &vQuadratic, const Mat33 &vMixed )
        : vCenter ( vCenter ), vAffine ( vAffine ), vQuadratic ( vQuadratic ), vMixed ( vMixed ) {}

    QuadraticDeriv() { clear(); }

    template<typename real2>
    QuadraticDeriv ( const QuadraticDeriv<3, real2>& c )
        : vCenter ( c.getVCenter() ), vAffine ( c.getVAffine() ), vQuadratic ( c.getVQuadratic() ), vMixed ( c.getVMixed() )
    {
    }

    void clear() { vCenter.clear(); vAffine.clear(); vQuadratic.clear(); vMixed.clear(); }

    template<typename real2>
    void operator = ( const QuadraticDeriv<3, real2>& c )
    {
        vCenter = c.getVCenter();
        vAffine = c.getVAffine();
        vQuadratic = c.getVQuadratic();
        vMixed = c.getVMixed();
    }

    void operator += ( const QuadraticDeriv& a )
    {
        vCenter += a.vCenter;
        vAffine += a.getVAffine();
        vQuadratic += a.getVQuadratic();
        vMixed += a.getVMixed();
    }

    void operator -= ( const QuadraticDeriv& a )
    {
        vCenter -= a.vCenter;
        vAffine -= a.getVAffine();
        vQuadratic -= a.getVQuadratic();
        vMixed -= a.getVMixed();
    }

    QuadraticDeriv<3, real> operator + ( const QuadraticDeriv<3, real>& a ) const
    {
        QuadraticDeriv d;
        d.vCenter = vCenter + a.vCenter;
        d.vAffine = vAffine + a.getVAffine();
        d.vQuadratic = vQuadratic + a.getVQuadratic();
        d.vMixed = vMixed + a.getVMixed();
        return d;
    }

    template<typename real2>
    void operator*= ( real2 a )
    {
        vCenter *= (Real)a;
        vAffine *= (Real)a;
        vQuadratic *= (Real)a;
        vMixed *= (Real)a;
    }

    template<typename real2>
    void operator/= ( real2 a )
    {
        vCenter /= (Real)a;
        vAffine /= (Real)a;
        vQuadratic /= (Real)a;
        vMixed /= (Real)a;
    }

    QuadraticDeriv<3, real> operator* ( float a ) const
    {
        QuadraticDeriv r = *this;
        r *= a;
        return r;
    }

    QuadraticDeriv<3, real> operator* ( double a ) const
    {
        QuadraticDeriv r = *this;
        r *= a;
        return r;
    }

    QuadraticDeriv<3, real> operator - () const
    {
        Mat33 matAffine;
        matAffine[0][0] = - vAffine[0][0];
        matAffine[0][1] = - vAffine[0][1];
        matAffine[0][2] = - vAffine[0][2];
        matAffine[1][0] = - vAffine[1][0];
        matAffine[1][1] = - vAffine[1][1];
        matAffine[1][2] = - vAffine[1][2];
        matAffine[2][0] = - vAffine[2][0];
        matAffine[2][1] = - vAffine[2][1];
        matAffine[2][2] = - vAffine[2][2];
        Mat33 matQuad;
        matQuad[0][0] = - vQuadratic[0][0];
        matQuad[0][1] = - vQuadratic[0][1];
        matQuad[0][2] = - vQuadratic[0][2];
        matQuad[1][0] = - vQuadratic[1][0];
        matQuad[1][1] = - vQuadratic[1][1];
        matQuad[1][2] = - vQuadratic[1][2];
        matQuad[2][0] = - vQuadratic[2][0];
        matQuad[2][1] = - vQuadratic[2][1];
        matQuad[2][2] = - vQuadratic[2][2];
        Mat33 matMixed;
        matMixed[0][0] = - vMixed[0][0];
        matMixed[0][1] = - vMixed[0][1];
        matMixed[0][2] = - vMixed[0][2];
        matMixed[1][0] = - vMixed[1][0];
        matMixed[1][1] = - vMixed[1][1];
        matMixed[1][2] = - vMixed[1][2];
        matMixed[2][0] = - vMixed[2][0];
        matMixed[2][1] = - vMixed[2][1];
        matMixed[2][2] = - vMixed[2][2];
        return QuadraticDeriv ( -vCenter, matAffine, matQuad, matMixed );
    }

    QuadraticDeriv<3, real> operator - ( const QuadraticDeriv<3, real>& a ) const
    {
        return QuadraticDeriv<3, real> ( this->vCenter - a.vCenter, this->vAffine - a.vAffine, this->vQuadratic - a.vQuadratic, this->vMixed - a.vMixed );
    }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator* ( const QuadraticDeriv<3, real>& a ) const
    {
        return vCenter[0]*a.vCenter[0] + vCenter[1]*a.vCenter[1] + vCenter[2]*a.vCenter[2]
                + vQuadratic(0,0)*a.vQuadratic(0,0) + vQuadratic(0,1)*a.vQuadratic(0,1) + vQuadratic(0,2)*a.vQuadratic(0,2)
                + vQuadratic(1,0)*a.vQuadratic(1,0) + vQuadratic(1,1)*a.vQuadratic(1,1) + vQuadratic(1,2)*a.vQuadratic(1,2)
                + vQuadratic(2,0)*a.vQuadratic(2,0) + vQuadratic(2,1)*a.vQuadratic(2,1) + vQuadratic(2,2)*a.vQuadratic(2,2);
        //TODO !!
    }

    Vec3& getVCenter ( void ) { return vCenter; }

    Mat33& getVAffine ( void ) { return vAffine; }

    Mat33& getVQuadratic ( void ) { return vQuadratic; }

    Mat33& getVMixed ( void ) { return vMixed; }

    const Vec3& getVCenter ( void ) const { return vCenter; }

    const Mat33& getVAffine ( void ) const { return vAffine; }

    const Mat33& getVQuadratic ( void ) const { return vQuadratic; }

    const Mat33& getVMixed ( void ) const { return vMixed; }

    Vec3& getLinear () { return vCenter; }

    const Vec3& getLinear () const { return vCenter; }

    Vec3 velocityAtRotatedPoint ( const Vec3& p ) const
    {
        return vCenter - vQuadratic * p; //TODO !!
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const QuadraticDeriv<3, real>& v )
    {
        out << v.vCenter << " " << v.vAffine << " " << v.vQuadratic << " " << v.vMixed;
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, QuadraticDeriv<3, real>& v )
    {
        in >> v.vCenter >> v.vAffine >> v.vQuadratic >> v.vMixed;
        return in;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 30 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 3 };

    real* ptr() { return vCenter.ptr(); }

    const real* ptr() const { return vCenter.ptr(); }

    static unsigned int size() {return 30;}

    /// Access to i-th element.
    real& operator[] ( int i )
    {
        if ( i<3 )
            return this->vCenter ( i );
        else if ( i < 12)
            return this->vAffine((i-3)/3, (i-3)%3);
        else if ( i < 21)
            return this->vQuadratic((i-12)/3, (i-12)%3);
        else
            return this->vMixed((i-21)/3, (i-21)%3);
    }

    /// Const access to i-th element.
    const real& operator[] ( int i ) const
    {
        if ( i<3 )
            return this->vCenter ( i );
        else if ( i < 12)
            return this->vAffine((i-3)/3, (i-3)%3);
        else if ( i < 21)
            return this->vQuadratic((i-12)/3, (i-12)%3);
        else
            return this->vMixed((i-21)/3, (i-21)%3);
    }
};










template<typename real>

class QuadraticCoord<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Vec<3, Real> Vec3;
    typedef Vec3 Pos;
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef Mat33 Quadratic;

protected:
    Vec3 center;
    Quadratic affine;
public:
    QuadraticCoord ( const Vec3 &center, const Quadratic &affine )
        : center ( center ), affine ( affine ) {}

    QuadraticCoord () { clear(); }

    template<typename real2>
    QuadraticCoord ( const QuadraticCoord<3, real2>& c )
        : center ( c.getCenter() ), affine ( c.getQuadratic() )
    {
    }

    void clear() { center.clear(); affine.identity(); }

    template<typename real2>
    void operator = ( const QuadraticCoord<3, real2>& c )
    {
        center = c.getCenter();
        affine = c.getQuadratic();
    }

    void operator += ( const QuadraticDeriv<3, real>& a )
    {
        center += a.getVCenter();
        affine += a.getVQuadratic();
    }

    QuadraticCoord<3, real> operator + ( const QuadraticDeriv<3, real>& a ) const
    {
        QuadraticCoord c = *this;
        c.center += a.getVCenter();
        c.affine += a.getVQuadratic();
        return c;
    }

    // Can be improved by using the polar decomposition (composition of rotations and substraction of shear and stretch components)
    QuadraticCoord<3, real> operator - ( const QuadraticCoord<3, real>& a ) const
    {
        return QuadraticCoord<3, real> ( this->center - a.getCenter(), a.affine.inverse() * this->affine );
    }

    // Can be improved by using the polar decomposition (composition of rotations and addition of shear and stretch components)
    QuadraticCoord<3, real> operator + ( const QuadraticCoord<3, real>& a ) const
    {
        return QuadraticCoord<3, real> ( this->center + a.getCenter(), a.affine * this->affine );
    }

    void operator += ( const QuadraticCoord<3, real>& a )
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
    QuadraticCoord<3, real> operator* ( real2 a ) const
    {
        QuadraticCoord r = *this;
        r *= a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator* ( const QuadraticCoord<3, real>& a ) const
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

    Mat33& getQuadratic () { return affine; }

    const Vec3& getCenter () const { return center; }

    const Mat33& getQuadratic () const { return affine; }

    static QuadraticCoord<3, real> identity()
    {
        QuadraticCoord c;
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
              void multRight ( const QuadraticCoord<3, real>& c )
              {
                center += affine * c.getCenter();
                affine = affine * c.getQuadratic();
                //TODO decomposition polaire. => mult rot, ajoute S terme a terme.
              }

              /// compute the product with another frame on the right
              QuadraticCoord<3, real> mult ( const QuadraticCoord<3, real>& c ) const
                {
                  QuadraticCoord r;
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
    inline friend std::ostream& operator << ( std::ostream& out, const QuadraticCoord<3, real>& v )
    {
        out << v.center << " " << v.affine;
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, QuadraticCoord<3, real>& v )
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

class QuadraticMass<3, real>
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
    QuadraticMass ( Real m = 1 )
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

    inline friend std::ostream& operator << ( std::ostream& out, const QuadraticMass<3, real>& m )
    {
        out << m.mass;
        out << " " << m.volume;
        out << " " << m.inertiaMatrix;
        return out;
    }

    inline friend std::istream& operator >> ( std::istream& in, QuadraticMass<3, real>& m )
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
inline QuadraticDeriv<N, real> operator* ( const QuadraticDeriv<N, real>& d, const QuadraticMass<N, real>& m )
{
    QuadraticDeriv<N, real> res;
    res.getVCenter() = d.getVCenter() * m.mass;
    res.getVOrientation() = m.inertiaMassMatrix * d.getVOrientation();
    return res;
}

template<int N, typename real>
inline QuadraticDeriv<N, real> operator/ ( const QuadraticDeriv<N, real>& d, const QuadraticMass<N, real>& m )
{
    QuadraticDeriv<N, real> res;
    res.getVCenter() = d.getVCenter() / m.mass;
    res.getVOrientation() = m.invInertiaMassMatrix * d.getVOrientation();
    return res;
}






template<typename real>

class StdQuadraticTypes<3, real>
{
public:
    typedef real Real;
    typedef QuadraticCoord<3, real> Coord;
    typedef QuadraticDeriv<3, real> Deriv;
    typedef typename Coord::Vec3 Vec3;

    enum { spatial_dimensions = Coord::spatial_dimensions };
    enum { coord_total_size = Coord::total_size };
    enum { deriv_total_size = Deriv::total_size };

    typedef typename Coord::Pos CPos;
    typedef typename Coord::Quadratic CQuadratic;
    static const CPos& getCPos ( const Coord& c ) { return c.getCenter(); }

    static void setCPos ( Coord& c, const CPos& v ) { c.getCenter() = v; }

    static const CQuadratic& getCRot ( const Coord& c ) { return c.getQuadratic(); }

    static void setCRot ( Coord& c, const CQuadratic& v ) { c.getQuadratic() = v; }

    typedef typename Deriv::Pos DPos;
    typedef typename Deriv::Quadratic DQuadratic;
    static const DPos& getDPos ( const Deriv& d ) { return d.getVCenter(); }

    static void setDPos ( Deriv& d, const DPos& v ) { d.getVCenter() = v; }

    static const DQuadratic& getDQuadratic ( const Deriv& d ) { return d.getVQuadratic(); }

    static void setDQuadratic ( Deriv& d, const DQuadratic& v ) { d.getVQuadratic() = v; }

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
            c.getQuadratic() += ancestors[i].getQuadratic() * coefs[i]; // Linear blend skinning (peut etre amelioré avec la decomposition polaire (cf. interpolation des Rigides pour le code original)
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




typedef StdQuadraticTypes<3, double> Quadratic3dTypes;
typedef StdQuadraticTypes<3, float> Quadratic3fTypes;

typedef QuadraticMass<3, double> Quadratic3dMass;
typedef QuadraticMass<3, float> Quadratic3fMass;
//typedef Quadratic3Mass QuadraticMass;

/// Note: Many scenes use Quadratic as template for 3D double-precision rigid type. Changing it to Quadratic3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* Quadratic3dTypes::Name() { return "Quadratic3d"; }

template<> inline const char* Quadratic3fTypes::Name() { return "Quadratic"; }

#else
template<> inline const char* Quadratic3dTypes::Name() { return "Quadratic"; }

template<> inline const char* Quadratic3fTypes::Name() { return "Quadratic3f"; }

#endif

#ifdef SOFA_FLOAT
typedef Quadratic3fTypes Quadratic3Types;
typedef Quadratic3fMass Quadratic3Mass;
#else
typedef Quadratic3dTypes Quadratic3Types;
typedef Quadratic3dMass Quadratic3Mass;
#endif
typedef Quadratic3Types QuadraticTypes;



// Specialization of the defaulttype::DataTypeInfo type traits template

template<int N, typename real>

struct DataTypeInfo< sofa::defaulttype::QuadraticDeriv<N, real> > : public FixedArrayTypeInfo< sofa::defaulttype::QuadraticDeriv<N, real>, sofa::defaulttype::QuadraticDeriv<N, real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticDeriv<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

template<int N, typename real>

struct DataTypeInfo< sofa::defaulttype::QuadraticCoord<N, real> > : public FixedArrayTypeInfo< sofa::defaulttype::QuadraticCoord<N, real>, sofa::defaulttype::QuadraticCoord<N, real>::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticCoord<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

//template<> struct DataTypeName< defaulttype::Quadratic2fTypes::Coord > { static const char* name() { return "Quadratic2fTypes::Coord"; } };
//template<> struct DataTypeName< defaulttype::Quadratic2fTypes::Deriv > { static const char* name() { return "Quadratic2fTypes::Deriv"; } };
//template<> struct DataTypeName< defaulttype::Quadratic2dTypes::Coord > { static const char* name() { return "Quadratic2dTypes::Coord"; } };
//template<> struct DataTypeName< defaulttype::Quadratic2dTypes::Deriv > { static const char* name() { return "Quadratic2dTypes::Deriv"; } };

template<> struct DataTypeName< defaulttype::Quadratic3fTypes::Coord > { static const char* name() { return "Quadratic3fTypes::Coord"; } };

template<> struct DataTypeName< defaulttype::Quadratic3fTypes::Deriv > { static const char* name() { return "Quadratic3fTypes::Deriv"; } };

template<> struct DataTypeName< defaulttype::Quadratic3dTypes::Coord > { static const char* name() { return "Quadratic3dTypes::Coord"; } };

template<> struct DataTypeName< defaulttype::Quadratic3dTypes::Deriv > { static const char* name() { return "Quadratic3dTypes::Deriv"; } };

//template<> struct DataTypeName< defaulttype::Quadratic2fMass > { static const char* name() { return "Quadratic2fMass"; } };
//template<> struct DataTypeName< defaulttype::Quadratic2dMass > { static const char* name() { return "Quadratic2dMass"; } };

template<> struct DataTypeName< defaulttype::Quadratic3fMass > { static const char* name() { return "Quadratic3fMass"; } };

template<> struct DataTypeName< defaulttype::Quadratic3dMass > { static const char* name() { return "Quadratic3dMass"; } };

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

/// Specialization of the inertia force for defaulttype::Quadratic3dTypes
template <>
inline defaulttype::QuadraticDeriv<3, double> inertiaForce <
defaulttype::QuadraticCoord<3, double>,
            defaulttype::QuadraticDeriv<3, double>,
            objectmodel::BaseContext::Vec3,
            defaulttype::QuadraticMass<3, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::QuadraticMass<3, double>& mass,
                    const defaulttype::QuadraticCoord<3, double>& x,
                    const defaulttype::QuadraticDeriv<3, double>& v
            )
{
    defaulttype::QuadraticDeriv<3, double>::Vec3 omega ( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::QuadraticDeriv<3, double>::Vec3 origin = x.getCenter(), finertia;
    defaulttype::QuadraticDeriv<3, double>::Mat33 zero;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() * 2 ) ) * mass.mass;
    return defaulttype::QuadraticDeriv<3, double> ( finertia, zero);
    /// \todo replace zero by Jomega.cross(omega)
}

/// Specialization of the inertia force for defaulttype::Quadratic3fTypes
template <>
inline defaulttype::QuadraticDeriv<3, float> inertiaForce <
defaulttype::QuadraticCoord<3, float>,
            defaulttype::QuadraticDeriv<3, float>,
            objectmodel::BaseContext::Vec3,
            defaulttype::QuadraticMass<3, float>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::QuadraticMass<3, float>& mass,
                    const defaulttype::QuadraticCoord<3, float>& x,
                    const defaulttype::QuadraticDeriv<3, float>& v
            )
{
    defaulttype::QuadraticDeriv<3, float>::Vec3 omega ( ( float ) vframe.lineVec[0], ( float ) vframe.lineVec[1], ( float ) vframe.lineVec[2] );
    defaulttype::QuadraticDeriv<3, float>::Vec3 origin = x.getCenter(), finertia;
    defaulttype::QuadraticDeriv<3, double>::Mat33 zero;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() * 2 ) ) * mass.mass;
    return defaulttype::QuadraticDeriv<3, float> ( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

} // namespace behavoir

} // namespace core

} // namespace sofa


#endif
