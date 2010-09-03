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
#ifndef SOFA_FRAME_QUADRATICTYPES_H
#define SOFA_FRAME_QUADRATICTYPES_H

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
    typedef Vec<9, Real> Vec9;
    typedef Vec3 Pos;
    typedef defaulttype::Mat<3,9,Real> Mat39;
    typedef Mat39 Quadratic;

protected:
    Vec3 vCenter;
    Mat39 vQuadratic;
public:

    friend class QuadraticCoord<3, real>;

    QuadraticDeriv ( const Vec3 &vCenter, const Mat39 &vQuadratic )
        : vCenter ( vCenter ), vQuadratic ( vQuadratic ) {}

    QuadraticDeriv() { clear(); }

    template<typename real2>
    QuadraticDeriv ( const QuadraticDeriv<3, real2>& c )
        : vCenter ( c.getVCenter() ), vQuadratic ( c.getVQuadratic() )
    {
    }

    void clear() { vCenter.clear(); vQuadratic.clear(); }

    template<typename real2>
    void operator = ( const QuadraticDeriv<3, real2>& c )
    {
        vCenter = c.getVCenter();
        vQuadratic = c.getVQuadratic();
    }

    void operator += ( const QuadraticDeriv& a )
    {
        vCenter += a.vCenter;
        vQuadratic += a.getVQuadratic();
    }

    void operator -= ( const QuadraticDeriv& a )
    {
        vCenter -= a.vCenter;
        vQuadratic -= a.getVQuadratic();
    }

    QuadraticDeriv<3, real> operator + ( const QuadraticDeriv<3, real>& a ) const
    {
        QuadraticDeriv d;
        d.vCenter = vCenter + a.vCenter;
        d.vQuadratic = vQuadratic + a.getVQuadratic();
        return d;
    }

    template<typename real2>
    void operator*= ( real2 a )
    {
        vCenter *= (Real)a;
        vQuadratic *= (Real)a;
    }

    template<typename real2>
    void operator/= ( real2 a )
    {
        vCenter /= (Real)a;
        vQuadratic /= (Real)a;
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
        Mat39 matQuad;
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 9; ++j)
                matQuad[i][j] = - vQuadratic[i][j];
        return QuadraticDeriv ( -vCenter, matQuad);
    }

    QuadraticDeriv<3, real> operator - ( const QuadraticDeriv<3, real>& a ) const
    {
        return QuadraticDeriv<3, real> ( this->vCenter - a.vCenter, this->vQuadratic - a.vQuadratic);
    }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator* ( const QuadraticDeriv<3, real>& a ) const
    {
        Real result = vCenter[0]*a.vCenter[0] + vCenter[1]*a.vCenter[1] + vCenter[2]*a.vCenter[2];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 9; ++j)
                result += vQuadratic(i,j)*a.vQuadratic(i,j);
        return result;
    }

    Vec3& getVCenter ( void ) { return vCenter; }

    Mat39& getVQuadratic ( void ) { return vQuadratic; }

    const Vec3& getVCenter ( void ) const { return vCenter; }

    const Mat39& getVQuadratic ( void ) const { return vQuadratic; }

    Vec3& getLinear () { return vCenter; }

    const Vec3& getLinear () const { return vCenter; }

    Vec3 velocityAtRotatedPoint ( const Vec3& p ) const
    {
        Vec9 pq;
        for (unsigned int i = 0; i < 3; ++i)
            pq[i] = p[i];
        for (unsigned int i = 0; i < 3; ++i)
            pq[i+3] = p[i]*p[i];
        pq[6] = p[0]*p[1];
        pq[7] = p[1]*p[2];
        pq[8] = p[0]*p[2];
        return vCenter - vQuadratic * p;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const QuadraticDeriv<3, real>& v )
    {
        out << v.vCenter << " " << v.vQuadratic;
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, QuadraticDeriv<3, real>& v )
    {
        in >> v.vCenter >> v.vQuadratic;
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
        else
            return this->vQuadratic((i-3)/9, (i-3)%9);
    }

    /// Const access to i-th element.
    const real& operator[] ( int i ) const
    {
        if ( i<3 )
            return this->vCenter ( i );
        else
            return this->vQuadratic((i-3)/9, (i-3)%9);
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
    typedef defaulttype::Mat<3,9,Real> Mat39;
    typedef Mat39 Quadratic;

protected:
    Vec3 center;
    Mat39 quadratic;
public:
    QuadraticCoord ( const Vec3 &center, const Mat39 &quadratic )
        : center ( center ), quadratic ( quadratic ) {}

    QuadraticCoord () { clear(); }

    template<typename real2>
    QuadraticCoord ( const QuadraticCoord<3, real2>& c )
        : center ( c.getCenter() ), quadratic ( c.getQuadratic() )
    {
    }

    void clear() { center.clear(); quadratic.clear(); quadratic(0,0) = 1.0; quadratic(1,1) = 1.0; quadratic(2,2) = 1.0;} // Set the affine part to identity.

    template<typename real2>
    void operator = ( const QuadraticCoord<3, real2>& c )
    {
        center = c.getCenter();
        quadratic = c.getQuadratic();
    }

    void operator += ( const QuadraticDeriv<3, real>& a )
    {
        center += a.getVCenter();
        quadratic += a.getVQuadratic();
    }

    QuadraticCoord<3, real> operator + ( const QuadraticDeriv<3, real>& a ) const
    {
        QuadraticCoord c = *this;
        c.center += a.getVCenter();
        c.quadratic += a.getVQuadratic();
        return c;
    }
    /*
              // Can be improved by using the polar decomposition (composition of rotations and substraction of shear and stretch components)
              QuadraticCoord<3, real> operator - ( const QuadraticCoord<3, real>& a ) const
              {//TODO !!
                return QuadraticCoord<3, real> ( this->center - a.getCenter(), a.getQuadratic().inverse() * this->quadratic );
              }

              // Can be improved by using the polar decomposition (composition of rotations and addition of shear and stretch components)
              QuadraticCoord<3, real> operator + ( const QuadraticCoord<3, real>& a ) const
              {//TODO !!
                return QuadraticCoord<3, real> ( this->center + a.getCenter(), this->quadratic * a.getQuadratic());
              }
    */
    void operator += ( const QuadraticCoord<3, real>& a )
    {
        center += a.getCenter();
        //quadratic += a.quadratic;
    }

    template<typename real2>
    void operator*= ( real2 a )
    {
        center *= a;
        //quadratic *= a;
    }

    template<typename real2>
    void operator/= ( real2 a )
    {
        center /= a;
        //quadratic /= a;
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
        Real result = center[0]*a.center[0] + center[1]*a.center[1] + center[2]*a.center[2];
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 9; ++j)
                result += quadratic(i,j)*a.quadratic(i,j);
        return result;
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

    Mat39& getQuadratic () { return quadratic; }

    const Vec3& getCenter () const { return center; }

    const Mat39& getQuadratic () const { return quadratic; }

    static QuadraticCoord<3, real> identity()
    {
        QuadraticCoord c;
        return c;
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
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 9; ++j)
                quadratic(i,j) = m[i][j];
        center[0] = m[0][9];
        center[1] = m[1][9];
        center[2] = m[2][9];
    }

    /// Write to the given matrix
    template<class Mat>
    void toMatrix ( Mat& m ) const
    {
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 9; ++j)
                m[i][j] = quadratic(i,j);
        m[0][9] = center[0];
        m[1][9] = center[1];
        m[2][9] = center[2];
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
        // Use only Affine part to display frame
        m[0] = quadratic(0,0);
        m[4] = quadratic(0,1);
        m[8] = quadratic(0,2);
        m[1] = quadratic(1,0);
        m[5] = quadratic(1,1);
        m[9] = quadratic(1,2);
        m[2] = quadratic(2,0);
        m[6] = quadratic(2,1);
        m[10] = quadratic(2,2);
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
        out << v.center << " " << v.quadratic;
        return out;
    }

    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, QuadraticCoord<3, real>& v )
    {
        in >> v.center >> v.quadratic;
        return in;
    }

    static int max_size()
    {
        return 3;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 30 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 3 };

    real* ptr() { return center.ptr(); }

    const real* ptr() const { return center.ptr(); }

    static unsigned int size() {return 30;}

    /// Access to i-th element.
    real& operator[] ( int i )
    {
        if ( i<3 )
            return this->center ( i );
        else
            return this->quadratic((i-3)/9, (i-3)%9);
    }

    /// Const access to i-th element.
    const real& operator[] ( int i ) const
    {
        if ( i<3 )
            return this->center ( i );
        else
            return this->quadratic((i-3)/9, (i-3)%9);
    }
};






template<typename real>

class QuadraticMass<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Mat<30, 30, Real> Mat30x30;
    Real mass, volume;
    Mat30x30 inertiaMatrix;       // Inertia matrix of the object
    Mat30x30 inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat30x30 invInertiaMatrix;    // inverse of inertiaMatrix
    Mat30x30 invInertiaMassMatrix; // inverse of inertiaMassMatrix
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
/*
    template<int N, typename real>
    inline QuadraticDeriv<N, real> operator* ( const QuadraticDeriv<N, real>& d, const QuadraticMass<N, real>& m )
    {
      QuadraticDeriv<N, real> res;
      res.getVCenter() = d.getVCenter() * m.mass;
      res.getVQuadratic() = m.inertiaMassMatrix * d.getVQuadratic();
      return res;
    }

    template<int N, typename real>
    inline QuadraticDeriv<N, real> operator/ ( const QuadraticDeriv<N, real>& d, const QuadraticMass<N, real>& m )
    {
      QuadraticDeriv<N, real> res;
      res.getVCenter() = d.getVCenter() / m.mass;
      res.getVQuadratic() = m.invInertiaMassMatrix * d.getVQuadratic();
      return res;
    }
*/





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

    static const CQuadratic& getCQuad ( const Coord& c ) { return c.getQuadratic(); }

    static void setCQuad ( Coord& c, const CQuadratic& v ) { c.getQuadratic() = v; }

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
    defaulttype::QuadraticDeriv<3, double>::Mat39 zero;

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
    defaulttype::QuadraticDeriv<3, double>::Mat39 zero;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() * 2 ) ) * mass.mass;
    return defaulttype::QuadraticDeriv<3, float> ( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

} // namespace behavoir

} // namespace core

} // namespace sofa


#endif
