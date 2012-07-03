/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FLEXIBLE_QUADRATICTYPES_H
#define FLEXIBLE_QUADRATICTYPES_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#ifdef SOFA_SMP
#include <sofa/defaulttype/SharedTypes.h>
#endif /* SOFA_SMP */

#include <sofa/defaulttype/Quat.h>

namespace sofa
{

namespace defaulttype
{

using std::endl;
using sofa::helper::vector;

/** DOF types for moving frames which generate a quadratic displacement field, with 30 scalar DOFs
*/
template<int _spatial_dimensions, typename _Real>
class StdQuadraticTypes
{
public:
    static const unsigned int spatial_dimensions = _spatial_dimensions;  ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int VSize = spatial_dimensions +  spatial_dimensions * spatial_dimensions*spatial_dimensions;  // number of entries
    enum { coord_total_size = VSize };
    enum { deriv_total_size = VSize };
    typedef _Real Real;
    typedef vector<Real> VecReal;

    // ------------    Types and methods defined for easier data access
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Vec<spatial_dimensions*spatial_dimensions, Real> QuadraticCoord;                   ///< Position or velocity of a point, and its second-degree polynomial values
    typedef Mat<spatial_dimensions,spatial_dimensions,Real> Affine;
    typedef Mat<spatial_dimensions,spatial_dimensions*spatial_dimensions, Real> Frame;

    class Deriv
    {
        Vec<VSize,Real> v;
    public:
        Deriv() { clear(); }
        Deriv( const Vec<VSize,Real>& d):v(d) {}
        Deriv( const SpatialCoord& c, const Frame& a) { getVCenter()=c; getVQuadratic()=a;}
        Deriv ( const SpatialCoord &center, const Affine &affine, const Affine &square=Affine(), const Affine &crossterms=Affine())
        {
            getVCenter() = center;
            for(unsigned int i=0; i<spatial_dimensions; ++i)
            {
                for(unsigned int j=0; j<spatial_dimensions; ++j)
                {
                    Frame& quadratic=getVQuadratic();
                    quadratic[i][j]=affine[i][j];
                    quadratic[i][j+3]=square[i][j];
                    quadratic[i][j+6]=crossterms[i][j];
                }
            }
        }

        void clear() { v.clear(); }

        static const unsigned int total_size = VSize;
        typedef Real value_type;

        static unsigned int size() { return VSize; }

        /// seen as a vector
        Vec<VSize,Real>& getVec() { return v; }
        const Vec<VSize,Real>& getVec() const { return v; }

        Real* ptr() { return v.ptr(); }
        const Real* ptr() const { return v.ptr(); }

        Real& operator[](int i) { return v[i]; }
        const Real& operator[](int i) const    { return v[i]; }

        /// point
        SpatialCoord& getVCenter() { return *reinterpret_cast<SpatialCoord*>(&v[0]); }
        const SpatialCoord& getVCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

        /// local frame
        Frame& getVQuadratic() { return *reinterpret_cast<Frame*>(&v[spatial_dimensions]); }
        const Frame& getVQuadratic() const { return *reinterpret_cast<const Frame*>(&v[spatial_dimensions]); }

        Affine getAffine() const
        {
            Affine m;
            for (unsigned int i = 0; i < 3; ++i)
                for (unsigned int j = 0; j < 3; ++j)
                    m[i][j]=getVQuadratic()[i][j];
            return  m;
        }


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
        Real operator*(const Deriv& a) const    { return v*a.v; }

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

        /// project to a rigid motion
        void setRigid()
        {
            Frame& q = getVQuadratic();
            // first matrix is skew-symmetric
            for(unsigned i=0; i<spatial_dimensions; i++) q[i][i] = 0.0;
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


    };

    typedef vector<Deriv> VecDeriv;
    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    static Deriv interpolate ( const helper::vector< Deriv > & ancestors, const helper::vector< Real > & coefs )
    {
        assert ( ancestors.size() == coefs.size() );
        Deriv c;
        for ( unsigned int i = 0; i < ancestors.size(); i++ )     c += ancestors[i] * coefs[i];
        return c;
    }


    class Coord
    {
        Vec<VSize,Real> v;
    public:
        Coord() { clear(); }
        Coord( const Vec<VSize,Real>& d):v(d) {}
        Coord( const SpatialCoord& c, const Frame& a) { getCenter()=c; getQuadratic()=a;}
        Coord ( const SpatialCoord &center, const Affine &affine, const Affine &square=Affine(), const Affine &crossterms=Affine())
        {
            getCenter() = center;
            for(unsigned int i=0; i<spatial_dimensions; ++i)
            {
                for(unsigned int j=0; j<spatial_dimensions; ++j)
                {
                    Frame& quadratic=getQuadratic();
                    quadratic[i][j]=affine[i][j];
                    quadratic[i][j+3]=square[i][j];
                    quadratic[i][j+6]=crossterms[i][j];
                }
            }
        }
        void clear() { v.clear(); for(unsigned i=0; i<spatial_dimensions; i++) getQuadratic()[i][i]=(Real)1.;  } // init affine part to identity

        static const unsigned int total_size = VSize;
        typedef Real value_type;

        static unsigned int size() { return VSize; }

        /// seen as a vector
        Vec<VSize,Real>& getVec() { return v; }
        const Vec<VSize,Real>& getVec() const { return v; }

        Real* ptr() { return v.ptr(); }
        const Real* ptr() const { return v.ptr(); }

        Real& operator[](int i) { return v[i]; }
        const Real& operator[](int i) const    { return v[i]; }


        /// point
        SpatialCoord& getCenter() { return *reinterpret_cast<SpatialCoord*>(&v[0]); }
        const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

        /// local frame
        Frame& getQuadratic() { return *reinterpret_cast<Frame*>(&v[spatial_dimensions]); }
        const Frame& getQuadratic() const { return *reinterpret_cast<const Frame*>(&v[spatial_dimensions]); }

        Affine getAffine() const
        {
            Affine m;
            for (unsigned int i = 0; i < 3; ++i)
                for (unsigned int j = 0; j < 3; ++j)
                    m[i][j]=getQuadratic()[i][j];
            return  m;
        }


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
        Real operator*(const Coord& a) const    { return v*a.v;    }

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
            Frame QuadraticInv;
            QuadraticInv.invert( getQuadratic() );
            return QuadraticInv * (v-getCenter());
        }


        /// project to a rigid motion
        void setRigid()
        {
            Frame& q = getQuadratic();
            // first matrix is pure rotation
            Affine a = getAffine(), rotation;
            polarDecomposition(a, rotation);
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
    };


    static QuadraticCoord convertToQuadraticCoord(const SpatialCoord& p)
    {
        return QuadraticCoord( p[0], p[1], p[2], p[0]*p[0], p[1]*p[1], p[2]*p[2], p[0]*p[1], p[1]*p[2], p[0]*p[2]);
    }

    typedef vector<Coord> VecCoord;

    static const char* Name();


    static Coord interpolate ( const helper::vector< Coord > & ancestors, const helper::vector< Real > & coefs )
    {
        assert ( ancestors.size() == coefs.size() );
        Coord c;
        for ( unsigned int i = 0; i < ancestors.size(); i++ ) c += ancestors[i] * coefs[i];  // Position and deformation gradient linear interpolation.
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


    /** @name Conversions
              * Convert to/from points in space
             */
    //@{
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
//@}


};

// ==========================================================================
// Mass

/** Mass associated with a quadratic deformable frame */
template<int _spatial_dimensions,typename _Real>
class QuadraticMass
{
public:
    typedef _Real Real;
    Real mass;
    // operator to cast to const Real
    operator const Real() const    {        return mass;    }

    typedef Real value_type;

    static const unsigned int spatial_dimensions = _spatial_dimensions;  ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int VSize = StdQuadraticTypes<spatial_dimensions,Real>::deriv_total_size;

    typedef Mat<VSize, VSize, Real> MatNN;

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
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3fTypes::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3fTypes::Coord, sofa::defaulttype::Quadratic3fTypes::Coord::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticCoord<" << sofa::defaulttype::Quadratic3fTypes::Coord::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3fTypes::Real>::name() << ">"; return o.str(); }
};
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3fTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3fTypes::Deriv, sofa::defaulttype::Quadratic3fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticDeriv<" << sofa::defaulttype::Quadratic3fTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3fTypes::Real>::name() << ">"; return o.str(); }
};
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3dTypes::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3dTypes::Coord, sofa::defaulttype::Quadratic3dTypes::Coord::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticCoord<" << sofa::defaulttype::Quadratic3dTypes::Coord::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3dTypes::Real>::name() << ">"; return o.str(); }
};
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3dTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3dTypes::Deriv, sofa::defaulttype::Quadratic3dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticDeriv<" << sofa::defaulttype::Quadratic3dTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3dTypes::Real>::name() << ">"; return o.str(); }
};
// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Quadratic3fTypes::Coord > { static const char* name() { return "Quadratic3fTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Quadratic3dTypes::Coord > { static const char* name() { return "Quadratic3dTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Quadratic3fMass > { static const char* name() { return "Quadratic3fMass"; } };
template<> struct DataTypeName< defaulttype::Quadratic3dMass > { static const char* name() { return "Quadratic3dMass"; } };

/// \endcond
} // namespace defaulttype


// ==========================================================================
// Mechanical Object

namespace component
{

namespace container
{

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_QuadraticTYPES_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Quadratic3dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Quadratic3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Quadratic3fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Quadratic3fTypes>;
#endif
#endif

} // namespace container

} // namespace component



} // namespace sofa



#endif
