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
#ifndef FRAME_AFFINETYPES_H
#define FRAME_AFFINETYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/core/objectmodel/BaseContext.h>
//#include <sofa/core/behavior/Mass.h>
#ifdef SOFA_SMP
#include <sofa/defaulttype/SharedTypes.h>
#endif /* SOFA_SMP */
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <iostream>
#include "MappingTypes.h"

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


typedef Vec<3,double> V3d;
typedef Vec<3,float> V3f;
typedef Vec<12,double> VAffine3double;
typedef Vec<12,float>  VAffine3float;

template<typename T>
Vec<3,T>& getVCenter( Vec<12,T>& v ) { return *reinterpret_cast<Vec<3,T>*>(&v[0]); }

template<typename T>
const Vec<3,T>& getVCenter( const Vec<12,T>& v ) { return *reinterpret_cast<const Vec<3,T>*>(&v[0]); }

template<typename T>
Mat<3,3,T>& getVAffine( Vec<12,T>& v ) { return *reinterpret_cast<Mat<3,3,T>*>(&v[3]); }

template<typename T>
const Mat<3,3,T>& getVAffine( const Vec<12,T>& v ) { return *reinterpret_cast<const Mat<3,3,T>*>(&v[3]); }



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

    void operator += ( const Vec<12, real>& a )
    {
        center += getVCenter(a);
        affine += getVAffine(a);
    }

    AffineCoord<3, real> operator + ( const Vec<12, real>& a ) const
    {
        AffineCoord c = *this;
        c.center += getVCenter(a);
        c.affine += getVAffine(a);
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
        m[0] = (float)affine(0,0);
        m[4] = (float)affine(0,1);
        m[8] = (float)affine(0,2);
        m[1] = (float)affine(1,0);
        m[5] = (float)affine(1,1);
        m[9] = (float)affine(1,2);
        m[2] = (float)affine(2,0);
        m[6] = (float)affine(2,1);
        m[10] = (float)affine(2,2);
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
    /// Project a point from the child frame to the parent frame
    Vec3 pointToParent( const Vec3& v ) const
    {
        return getAffine()*v + getCenter();
    }

    /// Project a point from the parent frame to the child frame
    Vec3 pointToChild( const Vec3& v ) const
    {
        Mat33 affineInv;
        affineInv.invert( getAffine() );
        return affineInv * (v-center);
    }


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

//          static int max_size()
//          {
//            return 3;
//          }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 12 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    enum { spatial_dimensions = 3 };

    real* ptr() { return center.ptr(); }

    const real* ptr() const { return center.ptr(); }

    static unsigned int size() {return 12;}

    /// Access to i-th element.
    real& operator[](int i)
    {
        if (i<3)
            return this->center[i];
        else
            return this->affine((i-3)/3, (i-3)%3);
    }

    /// Const access to i-th element.
    const real& operator[](int i) const
    {
        if (i<3)
            return this->center[i];
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
    Real mass;
    Mat3x3 inertiaMatrix;       // Inertia matrix of the object
    Mat3x3 invInertiaMatrix;    // inverse of inertiaMatrix
    AffineMass ( Real m = 1 )
    {
        mass = m;
        inertiaMatrix.identity();
        invInertiaMatrix.identity();
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
        invInertiaMatrix.invert ( inertiaMatrix );
    }

    inline friend std::ostream& operator << ( std::ostream& out, const AffineMass<3, real>& m )
    {
        out << m.mass;
        out << " " << m.inertiaMatrix;
        return out;
    }

    inline friend std::istream& operator >> ( std::istream& in, AffineMass<3, real>& m )
    {
        in >> m.mass;
        in >> m.inertiaMatrix;
        return in;
    }

    void operator *= ( Real fact )
    {
        mass *= fact;
        inertiaMatrix *= fact;
        invInertiaMatrix /= fact;
    }

    void operator /= ( Real fact )
    {
        mass /= fact;
        inertiaMatrix /= fact;
        invInertiaMatrix *= fact;
    }
};
/*
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
*/





template<typename real>

class StdAffineTypes<3, real>
{
public:
    typedef real Real;
    typedef AffineCoord<3, real> Coord;
    typedef Vec<12, real> Deriv;
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
        getVCenter(c) [0] = ( Real ) x;
        getVCenter(c) [1] = ( Real ) y;
        getVCenter(c) [2] = ( Real ) z;
    }

    template<typename T>
    static void get ( T& x, T& y, T& z, const Deriv& c )
    {
        x = ( T ) getVCenter(c) [0];
        y = ( T ) getVCenter(c) [1];
        z = ( T ) getVCenter(c) [2];
    }

    template<typename T>
    static void add ( Deriv& c, T x, T y, T z )
    {
        getVCenter(c) [0] += ( Real ) x;
        getVCenter(c) [1] += ( Real ) y;
        getVCenter(c) [2] += ( Real ) z;
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

    /// matrix product
    static Coord mult ( const Coord& a, const Coord& b )
    {
        return Coord( a.getCenter() + a.getAffine()*b.getCenter(),  a.getAffine()*b.getAffine());
    }

    /// Compute an inverse transform
    static Coord inverse( const Coord& c )
    {
        CAffine m;
#ifdef DEBUG
        bool invertible = invertMatrix(m,c.getAffine());
        assert(invertible);
#else
        invertMatrix(m,c.getAffine());
#endif
        return Coord( -(m*c.getCenter()),m );
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
inline defaulttype::VAffine3double inertiaForce <
defaulttype::AffineCoord<3, double>,
            defaulttype::VAffine3double,
            objectmodel::BaseContext::Vec3,
            defaulttype::AffineMass<3, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::AffineMass<3, double>& mass,
                    const defaulttype::AffineCoord<3, double>& x,
                    const defaulttype::VAffine3double& v
            )
{
    defaulttype::Vec3d omega ( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::Vec3d origin = x.getCenter(), finertia;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + getVCenter(v) * 2 ) ) * mass.mass;
    defaulttype::VAffine3double result;
    result[0]=finertia[0]; result[1]=finertia[1]; result[2]=finertia[2];
    return result;
    /// \todo replace zero by Jomega.cross(omega)
}

/// Specialization of the inertia force for defaulttype::Affine3fTypes
template <>
inline defaulttype::VAffine3float inertiaForce <
defaulttype::AffineCoord<3, float>,
            defaulttype::VAffine3float,
            objectmodel::BaseContext::Vec3,
            defaulttype::AffineMass<3, float>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::AffineMass<3, float>& mass,
                    const defaulttype::AffineCoord<3, float>& x,
                    const defaulttype::VAffine3float& v
            )
{
    const defaulttype::Vec3f omega ( (float)vframe.lineVec[0], (float)vframe.lineVec[1], (float)vframe.lineVec[2] );
    defaulttype::Vec3f origin = x.getCenter(), finertia;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + getVCenter(v) * 2 ) ) * mass.mass;
    defaulttype::VAffine3float result;
    result[0]=finertia[0]; result[1]=finertia[1]; result[2]=finertia[2];
    return result;
    /// \todo replace zero by Jomega.cross(omega)
}

} // namespace behavoir

} // namespace core

} // namespace sofa


#endif
