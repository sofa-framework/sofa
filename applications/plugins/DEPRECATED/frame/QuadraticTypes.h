/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef FRAME_QUADRATICTYPES_H
#define FRAME_QUADRATICTYPES_H

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
#include <sofa/helper/decompose.h>

namespace sofa
{

namespace defaulttype
{
using std::endl;
using sofa::helper::vector;

/** DOF types for moving frames which generate a quadratic displacement field, with 30 scalar DOFs */
template<int N, typename real>
class StdQuadraticTypes
{
public:
    typedef real Real;
    typedef vector<Real> VecReal;
    static const int spatial_dimensions = N;
    typedef Mat<spatial_dimensions,spatial_dimensions*spatial_dimensions, Real> Quadratic;
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Vec<spatial_dimensions*spatial_dimensions, Real> QuadraticCoord;                   ///< Position or velocity of a point, and its second-degree polynomial values
    static const unsigned VSize = spatial_dimensions +  spatial_dimensions * spatial_dimensions*spatial_dimensions;  // number of entries
    typedef Mat<spatial_dimensions,spatial_dimensions,Real> Affine;


    class Deriv
    {
        Vec<VSize,Real> v;
    public:
        Deriv() { clear(); }
        Deriv( const Vec<VSize,Real>& d):v(d) {}
        Deriv( const SpatialCoord& c, const Quadratic& a) { getVCenter()=c; getVQuadratic()=a;}
        Deriv ( const SpatialCoord &center, const Affine &affine, const Affine &square=Affine(), const Affine &crossterms=Affine())
        {
            getVCenter() = center;
            for(unsigned int i=0; i<spatial_dimensions; ++i)
            {
                for(unsigned int j=0; j<spatial_dimensions; ++j)
                {
                    Quadratic& quadratic=getVQuadratic();
                    quadratic[i][j]=affine[i][j];
                    quadratic[i][j+3]=square[i][j];
                    quadratic[i][j+6]=crossterms[i][j];
                }
            }
        }
        void clear()
        {
            v.clear();
        }

        /// seen as a vector
        Vec<VSize,Real>& getVec() { return v; }
        const Vec<VSize,Real>& getVec() const { return v; }

        /// point
        SpatialCoord& getVCenter() { return *reinterpret_cast<SpatialCoord*>(&v[0]); }
        const SpatialCoord& getVCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

        /// local frame
        Quadratic& getVQuadratic() { return *reinterpret_cast<Quadratic*>(&v[spatial_dimensions]); }
        const Quadratic& getVQuadratic() const { return *reinterpret_cast<const Quadratic*>(&v[spatial_dimensions]); }

        Affine getAffine() const
        {
            Affine m;
            for (unsigned int i = 0; i < 3; ++i)
                for (unsigned int j = 0; j < 3; ++j)
                    m[i][j]=getVQuadratic()[i][j];
            return  m;
        }


        /// project to a rigid motion
        void setRigid()
        {
            Quadratic& q = getVQuadratic();
            // first matrix is skew-symmetric
            for(unsigned i=0; i<spatial_dimensions; i++)
            {
                q[i][i] = 0.0;
            }
            for(unsigned i=0; i<spatial_dimensions; i++)
            {
                for(unsigned j=i+1; j<spatial_dimensions; j++)
                {
                    q[i][j] = (q[i][j] - q[j][i]) *(Real)0.5;
                    q[j][i] = - q[i][j];
                }
            }
            // the rest is null (?)
            for(unsigned i=0; i<spatial_dimensions; i++)
            {
                for(unsigned j=spatial_dimensions; j<spatial_dimensions*spatial_dimensions; j++)
                {
                    q[i][j] = 0.;
                }
            }
        }


        static const unsigned spatial_dimensions = N;
        static const unsigned total_size = VSize;
        typedef Real value_type;



        Deriv operator +(const Deriv& a) const { return Deriv(v+a.v); }
        void operator +=(const Deriv& a) { v+=a.v; }

        Deriv operator -(const Deriv& a) const { return Deriv(v-a.v); }
        void operator -=(const Deriv& a) { v-=a.v; }


        template<typename real2>
        Deriv operator *(real2 a) const { return Deriv(v*a); }
        template<typename real2>
        void operator *=(real2 a) { v *= a; }

        template<typename real2>
        void operator /=(real2 a) { v /= a; }

        Deriv operator - () const { return Deriv(-v); }


        /// dot product, mostly used to compute residuals as sqrt(x*x)
        Real operator*(const Deriv& a) const
        {
            return v*a.v;
        }

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Deriv& c )
        {
            out<<c.getVCenter()<<" "<<c.getVQuadratic();
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Deriv& c )
        {
            in>>c.getVCenter()>>c.getVQuadratic();
            return in;
        }


        Real* ptr() { return v.ptr(); }
        const Real* ptr() const { return v.ptr(); }

        /// Vector size
        static unsigned size() { return VSize; }

        /// Access to i-th element.
        Real& operator[](int i)
        {
            return v[i];
        }

        /// Const access to i-th element.
        const Real& operator[](int i) const
        {
            return v[i];
        }
    };



    class Coord
    {
        Vec<VSize,Real> v;
    public:
        Coord() { clear(); }
        Coord( const Vec<VSize,Real>& d):v(d) {}
        Coord( const SpatialCoord& c, const Quadratic& a) { getCenter()=c; getQuadratic()=a;}
        Coord ( const SpatialCoord &center, const Affine &affine, const Affine &square=Affine(), const Affine &crossterms=Affine())
        {
            getCenter() = center;
            for(unsigned int i=0; i<spatial_dimensions; ++i)
            {
                for(unsigned int j=0; j<spatial_dimensions; ++j)
                {
                    Quadratic& quadratic=getQuadratic();
                    quadratic[i][j]=affine[i][j];
                    quadratic[i][j+3]=square[i][j];
                    quadratic[i][j+6]=crossterms[i][j];
                }
            }
        }
        void clear()
        {
            v.clear();
            for(unsigned i=0; i<spatial_dimensions; i++) getQuadratic()[i][i]=(Real)1.; // init affine part to identity
        }

        /// seen as a vector
        Vec<VSize,Real>& getVec() { return v; }
        const Vec<VSize,Real>& getVec() const { return v; }

        /// point
        SpatialCoord& getCenter() { return *reinterpret_cast<SpatialCoord*>(&v[0]); }
        const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

        /// local frame
        Quadratic& getQuadratic() { return *reinterpret_cast<Quadratic*>(&v[spatial_dimensions]); }
        const Quadratic& getQuadratic() const { return *reinterpret_cast<const Quadratic*>(&v[spatial_dimensions]); }

        Affine getAffine() const
        {
            Affine m;
            for (unsigned int i = 0; i < 3; ++i)
                for (unsigned int j = 0; j < 3; ++j)
                    m[i][j]=getQuadratic()[i][j];
            return  m;
        }

        /// project to a rigid motion
        void setRigid()
        {
            Quadratic& q = getQuadratic();
            // first matrix is pure rotation
            Affine a = getAffine(), rotation;
            helper::Decompose<Real>::polarDecomposition(a, rotation);
            for(unsigned i=0; i<spatial_dimensions; i++)
            {
                for(unsigned j=0; j<spatial_dimensions; j++)
                {
                    q[i][j] = rotation[i][j];
                }
            }
            // the rest is null (?)
            for(unsigned i=0; i<spatial_dimensions; i++)
            {
                for(unsigned j=spatial_dimensions; j<spatial_dimensions*spatial_dimensions; j++)
                {
                    q[i][j] = 0.;
                }
            }
        }


        static const unsigned spatial_dimensions = N;
        static const unsigned total_size = VSize;
        typedef Real value_type;



        Coord operator +(const Coord& a) const { return Coord(v+a.v); }
        void operator +=(const Coord& a) { v+=a.v; }

        Coord operator +(const Deriv& a) const { return Coord(v+a.getVec()); }
        void operator +=(const Deriv& a) { v+=a.getVec(); }

        Coord operator -(const Coord& a) const { return Coord(v-a.v); }
        void operator -=(const Coord& a) { v-=a.v; }


        template<typename real2>
        Coord operator *(real2 a) const { return Coord(v*a); }
        template<typename real2>
        void operator *=(real2 a) { v *= a; }

        template<typename real2>
        void operator /=(real2 a) { v /= a; }

        Coord operator - () const { return Coord(-v); }


        /// dot product, mostly used to compute residuals as sqrt(x*x)
        Real operator*(const Coord& a) const
        {
            return v*a.v;
        }

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Coord& c )
        {
            out<<c.getCenter()<<" "<<c.getQuadratic();
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Coord& c )
        {
            in>>c.getCenter()>>c.getQuadratic();
            return in;
        }


        Real* ptr() { return v.ptr(); }
        const Real* ptr() const { return v.ptr(); }

        /// Vector size
        static unsigned size() { return VSize; }

        /// Access to i-th element.
        Real& operator[](int i)
        {
            return v[i];
        }

        /// Const access to i-th element.
        const Real& operator[](int i) const
        {
            return v[i];
        }
        /// Write the OpenGL transformation matrix
        void writeOpenGlMatrix ( float m[16] ) const
        {
            BOOST_STATIC_ASSERT(spatial_dimensions == 3);
            m[0] = (float)getQuadratic()(0,0);
            m[4] = (float)getQuadratic()(0,1);
            m[8] = (float)getQuadratic()(0,2);
            m[1] = (float)getQuadratic()(1,0);
            m[5] = (float)getQuadratic()(1,1);
            m[9] = (float)getQuadratic()(1,2);
            m[2] = (float)getQuadratic()(2,0);
            m[6] = (float)getQuadratic()(2,1);
            m[10] = (float)getQuadratic()(2,2);
            m[3] = 0;
            m[7] = 0;
            m[11] = 0;
            m[12] = ( float ) getCenter()[0];
            m[13] = ( float ) getCenter()[1];
            m[14] = ( float ) getCenter()[2];
            m[15] = 1;
        }
        /// Project a point from the child frame to the parent frame
        SpatialCoord pointToParent( const SpatialCoord& v ) const
        {
            return getQuadratic()*v + getCenter();
        }

        /// Project a point from the parent frame to the child frame
        SpatialCoord pointToChild( const SpatialCoord& v ) const
        {
            Quadratic QuadraticInv;
            QuadraticInv.invert( getQuadratic() );
            return QuadraticInv * (v-getCenter());
        }
    };


    static QuadraticCoord convertToQuadraticCoord(const SpatialCoord& p)
    {
        return QuadraticCoord( p[0], p[1], p[2], p[0]*p[0], p[1]*p[1], p[2]*p[2], p[0]*p[1], p[1]*p[2], p[0]*p[2]);
    }


    typedef vector<Coord> VecCoord;
    typedef vector<Deriv> VecDeriv;
    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    static const char* Name();



    static const SpatialCoord& getCPos(const Coord& c) { return c.getCenter(); }
    static void setCPos(Coord& c, const SpatialCoord& v) { c.getCenter() = v; }

    static const SpatialCoord& getDPos(const Deriv& d) { return d.getVCenter(); }
    static void setDPos(Deriv& d, const SpatialCoord& v) { d.getVCenter() = v; }

    template<typename T>
    static void set ( Deriv& c, T x, T y, T z )
    {
        c.clear();
        c.getVCenter()[0] = ( Real ) x;
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
        c.clear();
        c.getVCenter() [0] += ( Real ) x;
        c.getVCenter() [1] += ( Real ) y;
        c.getVCenter() [2] += ( Real ) z;
    }

    template<typename T>
    static void set ( Coord& c, T x, T y, T z )
    {
        c.clear();
        c.getCenter()[0] = ( Real ) x;
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
        c.clear();
        c.getCenter() [0] += ( Real ) x;
        c.getCenter() [1] += ( Real ) y;
        c.getCenter() [2] += ( Real ) z;
    }



    static Deriv interpolate ( const helper::vector< Deriv > & ancestors, const helper::vector< Real > & coefs )
    {
        assert ( ancestors.size() == coefs.size() );

        Deriv c;

        for ( unsigned int i = 0; i < ancestors.size(); i++ )
        {
            c += ancestors[i] * coefs[i];  // Position and deformation gradient linear interpolation.
        }

        return c;
    }

    static Coord interpolate ( const helper::vector< Coord > & ancestors, const helper::vector< Real > & coefs )
    {
        assert ( ancestors.size() == coefs.size() );

        Coord c;

        for ( unsigned int i = 0; i < ancestors.size(); i++ )
        {
            c += ancestors[i] * coefs[i];  // Position and deformation gradient linear interpolation.
        }

        return c;
    }

    /// Compute an inverse transform. Only the linear (affine and translation) part is inverted !!
    static Coord inverse( const Coord& c )
    {
        Affine m;
#ifdef DEBUG
        bool invertible = invertMatrix(m,c.getAffine());
        assert(invertible);
#else
        invertMatrix(m,c.getAffine());
#endif
        return Coord( -(m*c.getCenter()),m );
    }
};



template<int N,typename real>
class QuadraticMass
{
public:
    typedef real value_type;
    typedef real Real;
    typedef Mat<N, N, Real> MatNN;
    Real mass;
    MatNN inertiaMatrix;       // Inertia matrix of the object
    MatNN invInertiaMatrix;    // inverse of inertiaMatrix
    QuadraticMass ( Real m = 1 )
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

    inline friend std::ostream& operator << ( std::ostream& out, const QuadraticMass& m )
    {
        out << m.mass;
        out << " " << m.inertiaMatrix;
        return out;
    }

    inline friend std::istream& operator >> ( std::istream& in, QuadraticMass& m )
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


typedef StdQuadraticTypes<3, double> Quadratic3dTypes;
typedef StdQuadraticTypes<3, float> Quadratic3fTypes;

typedef QuadraticMass<3, double> Quadratic3dMass;
typedef QuadraticMass<3, float> Quadratic3fMass;

/// Note: Many scenes use Quadratic as template for 3D double-precision rigid type. Changing it to Quadratic3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* Quadratic3dTypes::Name() { return "FrameQuadratic3d"; }
template<> inline const char* Quadratic3fTypes::Name() { return "FrameQuadratic"; }
#else
template<> inline const char* Quadratic3dTypes::Name() { return "FrameQuadratic"; }
template<> inline const char* Quadratic3fTypes::Name() { return "FrameQuadratic3f"; }
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
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3fTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3fTypes::Deriv, sofa::defaulttype::Quadratic3fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticCoordOrDeriv<" << sofa::defaulttype::Quadratic3fTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3fTypes::Real>::name() << ">"; return o.str(); }
};
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3dTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3dTypes::Deriv, sofa::defaulttype::Quadratic3dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticCoordOrDeriv<" << sofa::defaulttype::Quadratic3dTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3dTypes::Real>::name() << ">"; return o.str(); }
};
// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

//template<> struct DataTypeName< defaulttype::Quadratic2fTypes::Coord > { static const char* name() { return "FrameQuadratic2fTypes::Coord"; } };
//template<> struct DataTypeName< defaulttype::Quadratic2fTypes::Deriv > { static const char* name() { return "FrameQuadratic2fTypes::Deriv"; } };
//template<> struct DataTypeName< defaulttype::Quadratic2dTypes::Coord > { static const char* name() { return "FrameQuadratic2dTypes::Coord"; } };
//template<> struct DataTypeName< defaulttype::Quadratic2dTypes::Deriv > { static const char* name() { return "FrameQuadratic2dTypes::Deriv"; } };

template<> struct DataTypeName< defaulttype::Quadratic3fTypes::Coord > { static const char* name() { return "FrameQuadratic3fTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Quadratic3dTypes::Coord > { static const char* name() { return "FrameQuadratic3dTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Quadratic3fMass > { static const char* name() { return "FrameQuadratic3fMass"; } };
template<> struct DataTypeName< defaulttype::Quadratic3dMass > { static const char* name() { return "FrameQuadratic3dMass"; } };

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
inline defaulttype::Quadratic3dTypes::Deriv inertiaForce <
defaulttype::Quadratic3dTypes::Coord,
            defaulttype::Quadratic3dTypes::Deriv,
            objectmodel::BaseContext::Vec3,
            defaulttype::QuadraticMass<3, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::QuadraticMass<3, double>& mass,
                    const defaulttype::Quadratic3dTypes::Coord& x,
                    const defaulttype::Quadratic3dTypes::Deriv& v
            )
{
    defaulttype::Vec3d omega ( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::Vec3d origin = x.getCenter(), finertia;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() * 2 ) ) * mass.mass;
    defaulttype::Quadratic3dTypes::Deriv result;
    result[0]=finertia[0]; result[1]=finertia[1]; result[2]=finertia[2];
    return result;
    /// \todo replace zero by Jomega.cross(omega)
}

/// Specialization of the inertia force for defaulttype::Quadratic3fTypes
template <>
inline defaulttype::Quadratic3fTypes::Deriv inertiaForce <
defaulttype::Quadratic3fTypes::Coord,
            defaulttype::Quadratic3fTypes::Deriv,
            objectmodel::BaseContext::Vec3,
            defaulttype::QuadraticMass<3, float>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::QuadraticMass<3, float>& mass,
                    const defaulttype::Quadratic3fTypes::Coord& x,
                    const defaulttype::Quadratic3fTypes::Deriv& v
            )
{
    const defaulttype::Vec3f omega ( (float)vframe.lineVec[0], (float)vframe.lineVec[1], (float)vframe.lineVec[2] );
    defaulttype::Vec3f origin = x.getCenter(), finertia;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getVCenter() * 2 ) ) * mass.mass;
    defaulttype::Quadratic3fTypes::Deriv result;
    result[0]=finertia[0]; result[1]=finertia[1]; result[2]=finertia[2];
    return result;
    /// \todo replace zero by Jomega.cross(omega)
}

} // namespace behavoir

} // namespace core

} // namespace sofa



#endif
