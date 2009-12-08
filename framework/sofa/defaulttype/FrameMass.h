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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MASS_FRAMEMASS_H
#define SOFA_COMPONENT_MASS_FRAMEMASS_H

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace defaulttype
{

template<int N, typename real>
class FrameDeriv;

template<int N, typename real>
class FrameCoord;

template<int N, typename real>
class FrameMass;

template<int N, typename real>
class StdFrameTypes;

//=============================================================================
// 3D Frames
//=============================================================================

/** Degrees of freedom of 3D rigid bodies. Orientations are modeled using quaternions.
*/
template<typename real>
class FrameDeriv<3, real>: public RigidDeriv<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Vec<3,Real> Pos;
    typedef Vec<3,Real> Rot;
    typedef Vec<3,Real> Vec3;
    typedef helper::Quater<Real> Quat;

    friend class FrameCoord<3,real>;

    FrameDeriv ( const Vec3 &velCenter, const Vec3 &velOrient )
        : RigidDeriv<3, real> ( velCenter, velOrient ) {}
    FrameDeriv()
    {
        RigidDeriv<3, real>::clear();
    }

    template<typename real2>
    FrameDeriv ( const FrameDeriv<3,real2>& c )
        : RigidDeriv<3, real> ( c )
    {
    }


    template<typename real2>
    void operator = ( const FrameDeriv<3,real2>& c )
    {
        this->vCenter = c.getVCenter();
        this->vOrientation = c.getVOrientation();
    }

    void operator += ( const FrameDeriv& a )
    {
        this->vCenter += a.vCenter;
        this->vOrientation += a.vOrientation;
    }

    void operator -= ( const FrameDeriv& a )
    {
        this->vCenter -= a.vCenter;
        this->vOrientation -= a.vOrientation;
    }

    FrameDeriv<3,real> operator + ( const FrameDeriv<3,real>& a ) const
    {
        FrameDeriv d;
        d.vCenter = this->vCenter + a.vCenter;
        d.vOrientation = this->vOrientation + a.vOrientation;
        return d;
    }

    template<typename real2>
    void operator*= ( real2 a )
    {
        this->vCenter *= a;
        this->vOrientation *= a;
    }

    template<typename real2>
    void operator/= ( real2 a )
    {
        this->vCenter /= a;
        this->vOrientation /= a;
    }

    FrameDeriv<3,real> operator* ( float a ) const
    {
        FrameDeriv r = *this;
        r*=a;
        return r;
    }

    FrameDeriv<3,real> operator* ( double a ) const
    {
        FrameDeriv r = *this;
        r*=a;
        return r;
    }

    FrameDeriv<3,real> operator - () const
    {
        return FrameDeriv ( -this->vCenter, -this->vOrientation );
    }

    FrameDeriv<3,real> operator - ( const FrameDeriv<3,real>& a ) const
    {
        return FrameDeriv<3,real> ( this->vCenter - a.vCenter, this->vOrientation-a.vOrientation );
    }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator* ( const FrameDeriv<3,real>& a ) const
    {
        return this->vCenter[0]*a.vCenter[0]+this->vCenter[1]*a.vCenter[1]+this->vCenter[2]*a.vCenter[2]
                +this->vOrientation[0]*a.vOrientation[0]+this->vOrientation[1]*a.vOrientation[1]
                +this->vOrientation[2]*a.vOrientation[2];
    }

    Vec3& getVCenter ( void )
    {
        return this->vCenter;
    }
    Vec3& getVOrientation ( void )
    {
        return this->vOrientation;
    }
    const Vec3& getVCenter ( void ) const
    {
        return this->vCenter;
    }
    const Vec3& getVOrientation ( void ) const
    {
        return this->vOrientation;
    }

    Vec3& getLinear ()
    {
        return this->vCenter;
    }
    const Vec3& getLinear () const
    {
        return this->vCenter;
    }
    Vec3& getAngular ()
    {
        return this->vOrientation;
    }
    const Vec3& getAngular () const
    {
        return this->vOrientation;
    }



    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const FrameDeriv<3,real>& v )
    {
        out<<v.vCenter<<" "<<v.vOrientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, FrameDeriv<3,real>& v )
    {
        in>>v.vCenter>>v.vOrientation;
        return in;
    }

    enum { static_size = 3 };

    real* ptr()
    {
        return this->vCenter.ptr();
    }
    const real* ptr() const
    {
        return this->vCenter.ptr();
    }

    static unsigned int size()
    {
        return 6;
    }

    /// Access to i-th element.
    real& operator[] ( int i )
    {
        if ( i<3 )
            return this->vCenter ( i );
        else
            return this->vOrientation ( i-3 );
    }

    /// Const access to i-th element.
    const real& operator[] ( int i ) const
    {
        if ( i<3 )
            return this->vCenter ( i );
        else
            return this->vOrientation ( i-3 );
    }
};

template<typename real>
class FrameCoord<3,real>: public RigidCoord<3,real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Vec<3,Real> Pos;
    typedef helper::Quater<Real> Rot;
    typedef Vec<3,Real> Vec3;
    typedef helper::Quater<Real> Quat;

public:
    FrameCoord ( const Vec3 &posCenter, const Quat &orient )
        : RigidCoord<3,real> ( posCenter , orient ) {}
    FrameCoord ()
    {
        RigidCoord<3,real>::clear();
    }

    template<typename real2>
    FrameCoord ( const FrameCoord<3,real2>& c )
        : RigidCoord<3,real> ( c )
    {
    }


    template<typename real2>
    void operator = ( const FrameCoord<3,real2>& c )
    {
        this->center = c.getCenter();
        this->orientation = c.getOrientation();
    }

    //template<typename real2>
    //void operator =(const FrameCoord<3,real2>& c)
    //{
    //    this->center = c.getCenter();
    //    this->orientation = c.getOrientation();
    //}

    void operator += ( const FrameDeriv<3,real>& a )
    {
        this->center += a.getVCenter();
        this->orientation.normalize();
        Quat qDot = this->orientation.vectQuatMult ( a.getVOrientation() );
        for ( int i = 0; i < 4; i++ )
            this->orientation[i] += qDot[i] * 0.5f;
        this->orientation.normalize();
    }

    FrameCoord<3,real> operator + ( const FrameDeriv<3,real>& a ) const
    {
        FrameCoord c = *this;
        c.center += a.getVCenter();
        c.orientation.normalize();
        Quat qDot = c.orientation.vectQuatMult ( a.getVOrientation() );
        for ( int i = 0; i < 4; i++ )
            c.orientation[i] += qDot[i] * 0.5f;
        c.orientation.normalize();
        return c;
    }

    FrameCoord<3,real> operator - ( const FrameCoord<3,real>& a ) const
    {
        return FrameCoord<3,real> ( this->center - a.getCenter(), a.orientation.inverse() * this->orientation );
    }

    FrameCoord<3,real> operator + ( const FrameCoord<3,real>& a ) const
    {
        return FrameCoord<3,real> ( this->center + a.getCenter(), a.orientation * this->orientation );
    }

    void operator += ( const FrameCoord<3,real>& a )
    {
        this->center += a.getCenter();
        //	this->orientation += a.getOrientation();
        //	this->orientation.normalize();
    }

    template<typename real2>
    void operator*= ( real2 a )
    {
        //std::cout << "*="<<std::endl;
        this->center *= a;
        //this->orientation *= a;
    }

    template<typename real2>
    void operator/= ( real2 a )
    {
        //std::cout << "/="<<std::endl;
        this->center /= a;
        //this->orientation /= a;
    }

    template<typename real2>
    FrameCoord<3,real> operator* ( real2 a ) const
    {
        FrameCoord r = *this;
        r*=a;
        return r;
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator* ( const FrameCoord<3,real>& a ) const
    {
        return this->center[0]*a.center[0]+this->center[1]*a.center[1]+this->center[2]*a.center[2]
                +this->orientation[0]*a.orientation[0]+this->orientation[1]*a.orientation[1]
                +this->orientation[2]*a.orientation[2]+this->orientation[3]*a.orientation[3];
    }

    /// Squared norm
    real norm2() const
    {
        real r = ( this->center ).elems[0]* ( this->center ).elems[0];
        for ( int i=1; i<3; i++ )
            r += ( this->center ).elems[i]* ( this->center ).elems[i];
        return r;
    }

    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt ( norm2() );
    }


    Vec3& getCenter ()
    {
        return this->center;
    }
    Quat& getOrientation ()
    {
        return this->orientation;
    }
    const Vec3& getCenter () const
    {
        return this->center;
    }
    const Quat& getOrientation () const
    {
        return this->orientation;
    }

    static FrameCoord<3,real> identity()
    {
        FrameCoord c;
        return c;
    }

    Vec3 rotate ( const Vec3& v ) const
    {
        return this->orientation.rotate ( v );
    }
    Vec3 inverseRotate ( const Vec3& v ) const
    {
        return this->orientation.inverseRotate ( v );
    }

    /// Apply a transformation with respect to itself
    void multRight ( const FrameCoord<3,real>& c )
    {
        this->center += this->orientation.rotate ( c.getCenter() );
        this->orientation = this->orientation * c.getOrientation();
    }

    /// compute the product with another frame on the right
    FrameCoord<3,real> mult ( const FrameCoord<3,real>& c ) const
    {
        FrameCoord r;
        r.center = this->center + this->orientation.rotate ( c.center );
        r.orientation = this->orientation * c.getOrientation();
        return r;
    }

    /// Set from the given matrix
    template<class Mat>
    void fromMatrix ( const Mat& m )
    {
        this->center[0] = m[0][3];
        this->center[1] = m[1][3];
        this->center[2] = m[2][3];
        Mat3x3d rot;
        rot = m;
        this->orientation.fromMatrix ( rot );
    }

    /// Write to the given matrix
    template<class Mat>
    void toMatrix ( Mat& m ) const
    {
        m.identity();
        this->orientation.toMatrix ( m );
        m[0][3] = this->center[0];
        m[1][3] = this->center[1];
        m[2][3] = this->center[2];
    }

    template<class Mat>
    void writeRotationMatrix ( Mat& m ) const
    {
        this->orientation.toMatrix ( m );
    }

    /// Write the OpenGL transformation matrix
    void writeOpenGlMatrix ( float m[16] ) const
    {
        this->orientation.writeOpenGlMatrix ( m );
        m[12] = ( float ) this->center[0];
        m[13] = ( float ) this->center[1];
        m[14] = ( float ) this->center[2];
    }

    /// compute the projection of a vector from the parent frame to the child
    Vec3 vectorToChild ( const Vec3& v ) const
    {
        return this->orientation.inverseRotate ( v );
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const FrameCoord<3,real>& v )
    {
        out<<v.center<<" "<<v.orientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, FrameCoord<3,real>& v )
    {
        in>>v.center>>v.orientation;
        return in;
    }
    static int max_size()
    {
        return 3;
    }
    enum { static_size = 3 };

    real* ptr()
    {
        return this->center.ptr();
    }
    const real* ptr() const
    {
        return this->center.ptr();
    }

    static unsigned int size()
    {
        return 7;
    }

    /// Access to i-th element.
    real& operator[] ( int i )
    {
        if ( i<3 )
            return this->center ( i );
        else
            return this->orientation[i-3];
    }

    /// Const access to i-th element.
    const real& operator[] ( int i ) const
    {
        if ( i<3 )
            return this->center ( i );
        else
            return this->orientation[i-3];
    }

};

template<typename real>
class FrameMass<3, real>
{
public:
    typedef real value_type;
    typedef real Real;
    typedef typename StdFrameTypes<3,Real>::VecCoord VecCoord;
    typedef typename StdFrameTypes<3,Real>::VecDeriv VecDeriv;
    typedef typename StdFrameTypes<3,Real>::Coord Coord;
    typedef typename StdFrameTypes<3,Real>::Deriv Deriv;
    typedef Mat<36,6,Real> Mat36;
    typedef Mat<6,6,Real> Mat66;
    typedef vector<double> VD;
    Real mass,volume;
    Mat66 inertiaMatrix;	      // Inertia matrix of the object
    Mat66 inertiaMassMatrix;    // Inertia matrix of the object * mass of the object
    Mat66 invInertiaMatrix;	  // inverse of inertiaMatrix
    Mat66 invInertiaMassMatrix; // inverse of inertiaMassMatrix

    FrameMass ( Real m=1 )
    {
        mass = m;
        volume = 1;
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

    /// compute ma = M*a
    Deriv operator * ( const Deriv& a ) const
    {
        Vec6d va, vma;
        va[0] = a.getVCenter() [0];
        va[1] = a.getVCenter() [1];
        va[2] = a.getVCenter() [2];
        va[3] = a.getVOrientation() [0];
        va[4] = a.getVOrientation() [1];
        va[5] = a.getVOrientation() [2];

        vma = inertiaMassMatrix * va;

        Deriv ma;
        ma.getVCenter() [0] = vma[0];
        ma.getVCenter() [1] = vma[1];
        ma.getVCenter() [2] = vma[2];
        ma.getVOrientation() [0] = vma[3];
        ma.getVOrientation() [1] = vma[4];
        ma.getVOrientation() [2] = vma[5];

        return ma;
    }

    /// compute a = f/a
    Deriv operator / ( const Deriv& f ) const
    {
        Vec6d va, vma;
        vma[0] = f.getVCenter() [0];
        vma[1] = f.getVCenter() [1];
        vma[2] = f.getVCenter() [2];
        vma[3] = f.getVOrientation() [0];
        vma[4] = f.getVOrientation() [1];
        vma[5] = f.getVOrientation() [2];

        va = invInertiaMassMatrix * vma;

        Deriv a;
        a.getVCenter() [0] = va[0];
        a.getVCenter() [1] = va[1];
        a.getVCenter() [2] = va[2];
        a.getVOrientation() [0] = va[3];
        a.getVOrientation() [1] = va[4];
        a.getVOrientation() [2] = va[5];

        return a;
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

    inline friend std::ostream& operator << ( std::ostream& out, const FrameMass<3, real>& m )
    {
        out<<m.mass;
        out<<" "<<m.volume;
        out<<" "<<m.inertiaMatrix;
        return out;
    }
    inline friend std::istream& operator >> ( std::istream& in, FrameMass<3, real>& m )
    {
        in>>m.mass;
        in>>m.volume;
        in>>m.inertiaMatrix;
        return in;
    }

};


template<int N, typename real>
inline typename StdFrameTypes<N, real>::Deriv operator* ( const typename StdFrameTypes<N, real>::Deriv& d, const FrameMass<N,real>& m )
{
    return m * d;
}

template<int N, typename real>
inline typename StdFrameTypes<N, real>::Deriv operator/ ( const typename StdFrameTypes<N, real>::Deriv& d, const FrameMass<N, real>& m )
{
    return m / d;
}


template<typename real>
class StdFrameTypes<3, real>: public StdRigidTypes<3, real>
{
public:
    typedef FrameCoord<3,real> Coord;
    typedef FrameDeriv<3,real> Deriv;
    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;

    StdFrameTypes() : StdRigidTypes<3, real>()
    {};
    ~StdFrameTypes()
    {};




    static const char* Name();

};

typedef StdFrameTypes<3,double> Frame3dTypes;
typedef StdFrameTypes<3,float> Frame3fTypes;

typedef FrameMass<3,double> Frame3dMass;
typedef FrameMass<3,float> Frame3fMass;

/// Note: Many scenes use Rigid as template for 3D double-precision rigid type. Changing it to Rigid3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* Frame3dTypes::Name()
{
    return "Frame3d";
}
template<> inline const char* Frame3fTypes::Name()
{
    return "Frame";
}
#else
template<> inline const char* Frame3dTypes::Name()
{
    return "Frame";
}
template<> inline const char* Frame3fTypes::Name()
{
    return "Frame3f";
}
#endif

#ifdef SOFA_FLOAT
typedef Frame3fTypes Frame3Types;
typedef Frame3fMass Frame3Mass;
#else
typedef Frame3dTypes Frame3Types;
typedef Frame3dMass Frame3Mass;
#endif
typedef Frame3Types FrameTypes;



// Specialization of the defaulttype::DataTypeInfo type traits template

template<int N, typename real>
struct DataTypeInfo< sofa::defaulttype::FrameDeriv<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::FrameDeriv<N,real> >
{
    static std::string name()
    {
        std::ostringstream o;
        o << "FrameDeriv<" << N << "," << DataTypeName<real>::name() << ">";
        return o.str();
    }
};

template<int N, typename real>
struct DataTypeInfo< sofa::defaulttype::FrameCoord<N,real> > : public FixedArrayTypeInfo< sofa::defaulttype::FrameCoord<N,real> >
{
    static std::string name()
    {
        std::ostringstream o;
        o << "FrameCoord<" << N << "," << DataTypeName<real>::name() << ">";
        return o.str();
    }
};

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Frame3fTypes::Coord >
{
    static const char* name()
    {
        return "Frame3fTypes::Coord";
    }
};
template<> struct DataTypeName< defaulttype::Frame3fTypes::Deriv >
{
    static const char* name()
    {
        return "Frame3fTypes::Deriv";
    }
};
template<> struct DataTypeName< defaulttype::Frame3dTypes::Coord >
{
    static const char* name()
    {
        return "Frame3dTypes::Coord";
    }
};
template<> struct DataTypeName< defaulttype::Frame3dTypes::Deriv >
{
    static const char* name()
    {
        return "Frame3dTypes::Deriv";
    }
};
template<> struct DataTypeName< defaulttype::Frame3fMass >
{
    static const char* name()
    {
        return "Frame3fMass";
    }
};
template<> struct DataTypeName< defaulttype::Frame3dMass >
{
    static const char* name()
    {
        return "Frame3dMass";
    }
};

/// \endcond


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
Deriv inertiaForce ( const SV& /*sv*/, const Vec& /*a*/, const M& /*m*/, const Coord& /*x*/, const Deriv& /*v*/ );

/// Specialization of the inertia force for defaulttype::Frame3dTypes
template <>
inline defaulttype::FrameDeriv<3, double> inertiaForce<
defaulttype::FrameCoord<3, double>,
            defaulttype::FrameDeriv<3, double>,
            objectmodel::BaseContext::Vec3,
            defaulttype::FrameMass<3, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::FrameMass<3, double>& mass,
                    const defaulttype::FrameCoord<3, double>& x,
                    const defaulttype::FrameDeriv<3, double>& v
            )
{
    defaulttype::FrameDeriv<3, double>::Vec3 omega ( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::FrameDeriv<3, double>::Vec3 origin = x.getCenter(), finertia, zero ( 0,0,0 );

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() *2 ) ) *mass.mass;
    return defaulttype::FrameDeriv<3, double> ( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

/// Specialization of the inertia force for defaulttype::Frame3fTypes
template <>
inline defaulttype::FrameDeriv<3, float> inertiaForce<
defaulttype::FrameCoord<3, float>,
            defaulttype::FrameDeriv<3, float>,
            objectmodel::BaseContext::Vec3,
            defaulttype::FrameMass<3, float>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::FrameMass<3, float>& mass,
                    const defaulttype::FrameCoord<3, float>& x,
                    const defaulttype::FrameDeriv<3, float>& v
            )
{
    defaulttype::FrameDeriv<3, float>::Vec3 omega ( ( float ) vframe.lineVec[0], ( float ) vframe.lineVec[1], ( float ) vframe.lineVec[2] );
    defaulttype::FrameDeriv<3, float>::Vec3 origin = x.getCenter(), finertia, zero ( 0,0,0 );

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() *2 ) ) *mass.mass;
    return defaulttype::FrameDeriv<3, float> ( finertia, zero );
    /// \todo replace zero by Jomega.cross(omega)
}

} // namespace behavoir

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif

