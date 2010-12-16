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
#ifndef FRAME_DeformationGradientTYPES_H
#define FRAME_DeformationGradientTYPES_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/MapMapSparseMatrix.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/helper/PolarDecompose.h>
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
using helper::vector;


template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
class DeformationGradient;

template<int _spatial_dimensions, int _material_dimensions, typename _Real>
class DeformationGradient<_spatial_dimensions, _material_dimensions, 1, _Real>
{

public:
    static const unsigned spatial_dimensions = _spatial_dimensions;
    static const unsigned material_dimensions = _material_dimensions;
    enum {order = 1 };
    static const unsigned NumMatrices = order==0? 0 : (order==1? 1 : (order==2? 1 + material_dimensions : -1 ));
    static const unsigned VSize = spatial_dimensions +  NumMatrices * spatial_dimensions * spatial_dimensions;  // number of entries
    typedef _Real Real;
    typedef vector<Real> VecReal;
    typedef Vec<10,Real> SampleIntegVector;  ///< used to precompute the integration of deformation energy over a sample region

    // ------------    Types and methods defined for easier data access
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Mat<spatial_dimensions,spatial_dimensions, Real> MaterialFrame;      ///< Matrix representing a deformation gradient
    typedef Vec<spatial_dimensions, MaterialFrame> MaterialFrameGradient;                 ///< Gradient of a deformation gradient


    static const unsigned strain_size = spatial_dimensions * (1+spatial_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    typedef Vec<NumMatrices,StrainVec> Strain;  ///< Strain and its gradient, in vector form
    typedef Strain Stress;


protected:
    Vec<VSize,Real> v;

public:
    DeformationGradient() { v.clear(); }
    DeformationGradient( const Vec<VSize,Real>& d):v(d) {}
    void clear() { v.clear(); }

    /// seen as a vector
    Vec<VSize,Real>& getVec() { return v; }
    const Vec<VSize,Real>& getVec() const { return v; }

    /// point
    SpatialCoord& getCenter() { return *reinterpret_cast<SpatialCoord*>(&v[0]); }
    const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

    /// local frame (if order>=1)
    MaterialFrame& getMaterialFrame() { return *reinterpret_cast<MaterialFrame*>(&v[spatial_dimensions]); }
    const MaterialFrame& getMaterialFrame() const { return *reinterpret_cast<const MaterialFrame*>(&v[spatial_dimensions]); }

    static const unsigned total_size = VSize;
    typedef Real value_type;

    static void multStrain( Strain& s, Real r )
    {
        for(unsigned i=0; i<s.size(); i++)
            s[i] *= r;
    }

    static StrainVec getStrainVec(  const MaterialFrame& f ) // symmetric matrix
    {
        StrainVec s;
        unsigned ei=0;
        for(unsigned j=0; j<material_dimensions; j++)
        {
            for( unsigned k=j; k<material_dimensions; k++ )
            {
                s[ei] = f[j][k];
                ei++;
            }
        }
        return s;
    }

    static MaterialFrame getFrame( const StrainVec& s  )
    {
        MaterialFrame f;
        unsigned ei=0;
        for(unsigned j=0; j<material_dimensions; j++)
        {
            for( unsigned k=j; k<material_dimensions; k++ )
            {
                f[k][j] = f[j][k] = s[ei] ;
                ei++;
            }
        }
        return f;
    }

    void setStress( const Stress& stress  )
    {
        getMaterialFrame() = getFrame(stress[0]);
    }

    void getCorotationalStrain( MaterialFrame& rotation, Strain& strain ) const
    {
        MaterialFrame local_deformation_gradient;
        helper::polar_decomp(this->getMaterialFrame(), rotation, local_deformation_gradient); // decompose F=RD
        strain[0] = getStrainVec( local_deformation_gradient );
    }

    void getCorotationalStrainRate( Strain& strainRate, const MaterialFrame& rotation  ) const
    {
        // FF: assuming that the strain rate  can be decomposed using the same rotation as the strain
        strainRate[0] = getStrainVec( rotation.multTranspose(this->getMaterialFrame()) );
    }



    DeformationGradient operator +(const DeformationGradient& a) const { return DeformationGradient(v+a.v); }
    void operator +=(const DeformationGradient& a) { v+=a.v; }

    DeformationGradient operator -(const DeformationGradient& a) const { return DeformationGradient(v-a.v); }
    void operator -=(const DeformationGradient& a) { v-=a.v; }


    template<typename real2>
    DeformationGradient operator *(real2 a) const { return DeformationGradient(v*a); }
    template<typename real2>
    void operator *=(real2 a) { v *= a; }

    template<typename real2>
    void operator /=(real2 a) { v /= a; }

    DeformationGradient operator - () const { return DeformationGradient(-v); }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const DeformationGradient& a) const
    {
        return v*a.v;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const DeformationGradient& c )
    {
        out<<c.v;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, DeformationGradient& c )
    {
        in>>c.v;
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


template<int _spatial_dimensions, int _material_dimensions, typename _Real>
class DeformationGradient<_spatial_dimensions, _material_dimensions, 2, _Real>
{

public:
    static const unsigned spatial_dimensions = _spatial_dimensions;
    static const unsigned material_dimensions = _material_dimensions;
//            static const unsigned order = _order;  ///< 0: only a point, no gradient 1:deformation gradient, 2: deformation gradient and its gradient
    enum {order = 2 };
    static const unsigned NumMatrices = order==0? 0 : (order==1? 1 : (order==2? 1 + material_dimensions : -1 ));
    static const unsigned VSize = spatial_dimensions +  NumMatrices * spatial_dimensions * spatial_dimensions;  // number of entries
    typedef _Real Real;
    typedef vector<Real> VecReal;
    typedef Vec<35,Real> SampleIntegVector;  ///< used to precompute the integration of deformation energy over a sample region

    // ------------    Types and methods defined for easier data access
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Mat<spatial_dimensions,spatial_dimensions, Real> MaterialFrame;      ///< Matrix representing a deformation gradient
    typedef Vec<spatial_dimensions, MaterialFrame> MaterialFrameGradient;                 ///< Gradient of a deformation gradient


    static const unsigned strain_size = spatial_dimensions * (1+spatial_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;   ///< Strain in vector form
    typedef Vec<NumMatrices,StrainVec> Strain; ///< Strain and its gradient, in vector form
    typedef Strain Stress;


protected:
    Vec<VSize,Real> v;

public:
    DeformationGradient() { v.clear(); }
    DeformationGradient( const Vec<VSize,Real>& d):v(d) {}
    void clear() { v.clear(); }

    /// seen as a vector
    Vec<VSize,Real>& getVec() { return v; }
    const Vec<VSize,Real>& getVec() const { return v; }

    /// point
    SpatialCoord& getCenter() { return *reinterpret_cast<SpatialCoord*>(&v[0]); }
    const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

    /// local frame (if order>=1)
    MaterialFrame& getMaterialFrame() { return *reinterpret_cast<MaterialFrame*>(&v[spatial_dimensions]); }
    const MaterialFrame& getMaterialFrame() const { return *reinterpret_cast<const MaterialFrame*>(&v[spatial_dimensions]); }

    /// gradient of the local frame (if order>=2)
    MaterialFrameGradient& getMaterialFrameGradient() { return *reinterpret_cast<MaterialFrameGradient*>(&v[spatial_dimensions]); }
    const MaterialFrameGradient& getMaterialFrameGradient() const { return *reinterpret_cast<const MaterialFrameGradient*>(&v[spatial_dimensions]); }

    static const unsigned total_size = VSize;
    typedef Real value_type;

    static void multStrain( Strain& s, Real r )
    {
        for(unsigned i=0; i<s.size(); i++)
            s[i] *= r;
    }

    static StrainVec getStrainVec(  const MaterialFrame& f )
    {
        StrainVec s;
        unsigned ei=0;
        for(unsigned j=0; j<material_dimensions; j++)
        {
            for( unsigned k=j; k<material_dimensions; k++ )
            {
                s[ei] = f[j][k];
                ei++;
            }
        }
        return s;
    }

    static MaterialFrame getFrame( const StrainVec& s  )
    {
        MaterialFrame f;
        unsigned ei=0;
        for(unsigned j=0; j<material_dimensions; j++)
        {
            for( unsigned k=j; k<material_dimensions; k++ )
            {
                f[k][j] = f[j][k] = s[ei] ;
                ei++;
            }
        }
        return f;
    }

    void setStress( const Stress& stress  )
    {
        getMaterialFrame() = getFrame(stress[0]);
        MaterialFrameGradient& g= this->getMaterialFrameGradient();
        for(unsigned i=0; i<spatial_dimensions; i++ )
        {
            g[i] = getFrame( stress[1+i] ); // FF: assuming that the gradient of F can be decomposed using the same rotation as F
        }
    }

    void getCorotationalStrain( MaterialFrame& rotation, Strain& strain ) const
    {
        MaterialFrame local_deformation_gradient;
        helper::polar_decomp(this->getMaterialFrame(), rotation, local_deformation_gradient); // decompose F=RD
        strain[0] = getStrainVec( local_deformation_gradient );

        const MaterialFrameGradient& g= this->getMaterialFrameGradient();
        for(unsigned i=0; i<spatial_dimensions; i++ )
        {
            strain[1+i] = getStrainVec( rotation.multTranspose(g[i]) ); // FF: assuming that the gradient of F can be decomposed using the same rotation as F
        }
    }

    void getCorotationalStrainRate( Strain& strainRate, const MaterialFrame& rotation  ) const
    {
        // FF: assuming that the strain rate  can be decomposed using the same rotation as the strain
        strainRate[0] = getStrainVec( rotation.multTranspose(this->getMaterialFrame()) );
        const MaterialFrameGradient& g= this->getMaterialFrameGradient();
        for(unsigned i=0; i<spatial_dimensions; i++ )
        {
            strainRate[1+i] = getStrainVec( rotation.multTranspose(g[i]) );
        }
    }



    DeformationGradient operator +(const DeformationGradient& a) const { return DeformationGradient(v+a.v); }
    void operator +=(const DeformationGradient& a) { v+=a.v; }

    DeformationGradient operator -(const DeformationGradient& a) const { return DeformationGradient(v-a.v); }
    void operator -=(const DeformationGradient& a) { v-=a.v; }


    template<typename real2>
    DeformationGradient operator *(real2 a) const { return DeformationGradient(v*a); }
    template<typename real2>
    void operator *=(real2 a) { v *= a; }

    template<typename real2>
    void operator /=(real2 a) { v /= a; }

    DeformationGradient operator - () const { return DeformationGradient(-v); }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const DeformationGradient& a) const
    {
        return v*a.v;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const DeformationGradient& c )
    {
        out<<c.v;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, DeformationGradient& c )
    {
        in>>c.v;
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






/** Local deformation state of a material object.

  spatial_dimensions is the number of dimensions the object is moving in.
  material_dimensions is the number of internal dimensions of the object: 1 for a wire, 2 for a hull, 3 for a volumetric object
  order is the degree of the local displacement function: 0 for a simple displacement, 1 for a displacent and a nonrigid local frame, 2 for a displacent, a nonrigid local frame and the gradient of this frame.
  */
template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
struct DeformationGradientTypes
{
    static const unsigned spatial_dimensions = _spatial_dimensions;
    static const unsigned material_dimensions = _material_dimensions;
//            static const unsigned order = _order;  ///< 0: only a point, no gradient 1:deformation gradient, 2: deformation gradient and its gradient
    enum {order = _order };
    static const unsigned NumMatrices = order==0? 0 : (order==1? 1 : (order==2? 1 + material_dimensions : -1 ));
    static const unsigned VSize = spatial_dimensions +  NumMatrices * spatial_dimensions * spatial_dimensions;  // number of entries
    typedef _Real Real;
    typedef vector<Real> VecReal;

    // ------------    Types and methods defined for easier data access
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Mat<spatial_dimensions,spatial_dimensions, Real> MaterialFrame;      ///< Matrix representing a deformation gradient
    typedef Vec<spatial_dimensions, MaterialFrame> MaterialFrameGradient;                 ///< Gradient of a deformation gradient


    typedef DeformationGradient<spatial_dimensions,material_dimensions,order,Real> Coord;
    typedef vector<Coord> VecCoord;
    typedef Coord Deriv ;            ///< velocity and deformation gradient rate
    typedef vector<Deriv> VecDeriv;
    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    static const char* Name();




    template<typename T>
    static void set ( Coord& c, T x, T y, T z )
    {
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
        c.getCenter() [0] += ( Real ) x;
        c.getCenter() [1] += ( Real ) y;
        c.getCenter() [2] += ( Real ) z;
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

};






/** Mass associated with a sampling point
*/
template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
struct DeformationGradientMass
{
    typedef _Real Real;
    Real mass;  ///< Currently only a scalar mass, but a matrix should be used for more precision

    // operator to cast to const Real
    operator const Real() const
    {
        return mass;
    }

    template<int S, int M, int O, typename R>
    inline friend std::ostream& operator << ( std::ostream& out, const DeformationGradientMass<S,M,O,R>& m )
    {
        out << m.mass;
        return out;
    }

    template<int S, int M, int O, typename R>
    inline friend std::istream& operator >> ( std::istream& in, DeformationGradientMass<S,M,O,R>& m )
    {
        in >> m.mass;
        return in;
    }

    void operator *= ( Real fact )
    {
        mass *= fact;
    }

    void operator /= ( Real fact )
    {
        mass /= fact;
    }
};

template<int S, int M, int O, typename R>
inline typename DeformationGradientTypes<S,M,O,R>::Deriv operator* ( const typename DeformationGradientTypes<S,M,O,R>::Deriv& d, const DeformationGradientMass<S,M,O,R>& m )
{
    typename DeformationGradientTypes<S,M,O,R>::Deriv res;
    DeformationGradientTypes<S,M,O,R>::center(res) = DeformationGradientTypes<S,M,O,R>::center(d) * m.mass;
    return res;
}

template<int S, int M, int O, typename R>
inline typename DeformationGradientTypes<S,M,O,R>::Deriv operator/ ( const typename DeformationGradientTypes<S,M,O,R>::Deriv& d, const DeformationGradientMass<S,M,O,R>& m )
{
    typename DeformationGradientTypes<S,M,O,R>::Deriv res;
    DeformationGradientTypes<S,M,O,R>::center(res) = DeformationGradientTypes<S,M,O,R>::center(d) / m.mass;
    return res;
}





// ==========================================================================
// order 1

typedef DeformationGradientTypes<3, 3, 1, double> DeformationGradient331dTypes;
typedef DeformationGradientTypes<3, 3, 1, float>  DeformationGradient331fTypes;

typedef DeformationGradientMass<3, 3, 1, double> DeformationGradient331dMass;
typedef DeformationGradientMass<3, 3, 1, float>  DeformationGradient331fMass;

/// Note: Many scenes use DeformationGradient as template for 3D double-precision rigid type. Changing it to DeformationGradient3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* DeformationGradient331dTypes::Name() { return "DeformationGradient331d"; }

template<> inline const char* DeformationGradient331fTypes::Name() { return "DeformationGradient331"; }

#else
template<> inline const char* DeformationGradient331dTypes::Name() { return "DeformationGradient331"; }

template<> inline const char* DeformationGradient331fTypes::Name() { return "DeformationGradient331f"; }

#endif

#ifdef SOFA_FLOAT
typedef DeformationGradient331fTypes DeformationGradient331Types;
typedef DeformationGradient331fMass DeformationGradient331Mass;
#else
typedef DeformationGradient331dTypes DeformationGradient331Types;
typedef DeformationGradient331dMass DeformationGradient331Mass;
#endif

template<>
struct DataTypeInfo< DeformationGradient331fTypes::Deriv > : public FixedArrayTypeInfo< DeformationGradient331fTypes::Deriv, DeformationGradient331fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "DeformationGradient331<" << DataTypeName<float>::name() << ">"; return o.str(); }
};
template<>
struct DataTypeInfo< DeformationGradient331dTypes::Deriv > : public FixedArrayTypeInfo< DeformationGradient331dTypes::Deriv, DeformationGradient331dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "DeformationGradient331<" << DataTypeName<double>::name() << ">"; return o.str(); }
};




// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::DeformationGradient331fTypes::Coord > { static const char* name() { return "DeformationGradient331fTypes::CoordOrDeriv"; } };

//        template<> struct DataTypeName< defaulttype::DeformationGradient331fTypes::Deriv > { static const char* name() { return "DeformationGradient331fTypes::Deriv"; } };

template<> struct DataTypeName< defaulttype::DeformationGradient331dTypes::Coord > { static const char* name() { return "DeformationGradient331dTypes::CoordOrDeriv"; } };

//        template<> struct DataTypeName< defaulttype::DeformationGradient331dTypes::Deriv > { static const char* name() { return "DeformationGradient331dTypes::Deriv"; } };


template<> struct DataTypeName< defaulttype::DeformationGradient331fMass > { static const char* name() { return "DeformationGradient331fMass"; } };

template<> struct DataTypeName< defaulttype::DeformationGradient331dMass > { static const char* name() { return "DeformationGradient331dMass"; } };

/// \endcond






// ==========================================================================
// order 2


typedef DeformationGradientTypes<3, 3, 2, double> DeformationGradient332dTypes;
typedef DeformationGradientTypes<3, 3, 2, float>  DeformationGradient332fTypes;

typedef DeformationGradientMass<3, 3, 2, double> DeformationGradient332dMass;
typedef DeformationGradientMass<3, 3, 2, float>  DeformationGradient332fMass;

/// Note: Many scenes use DeformationGradient as template for 3D double-precision rigid type. Changing it to DeformationGradient3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* DeformationGradient332dTypes::Name() { return "DeformationGradient332d"; }

template<> inline const char* DeformationGradient332fTypes::Name() { return "DeformationGradient332"; }

#else
template<> inline const char* DeformationGradient332dTypes::Name() { return "DeformationGradient332"; }

template<> inline const char* DeformationGradient332fTypes::Name() { return "DeformationGradient332f"; }

#endif

#ifdef SOFA_FLOAT
typedef DeformationGradient332fTypes DeformationGradient332Types;
typedef DeformationGradient332fMass DeformationGradient332Mass;
#else
typedef DeformationGradient332dTypes DeformationGradient332Types;
typedef DeformationGradient332dMass DeformationGradient332Mass;
#endif

template<>
struct DataTypeInfo< DeformationGradient332fTypes::Deriv > : public FixedArrayTypeInfo< DeformationGradient332fTypes::Deriv, DeformationGradient332fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "DeformationGradient332<" << DataTypeName<float>::name() << ">"; return o.str(); }
};
template<>
struct DataTypeInfo< DeformationGradient332dTypes::Deriv > : public FixedArrayTypeInfo< DeformationGradient332dTypes::Deriv, DeformationGradient332dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "DeformationGradient332<" << DataTypeName<double>::name() << ">"; return o.str(); }
};




// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::DeformationGradient332fTypes::Coord > { static const char* name() { return "DeformationGradient332fTypes::CoordOrDeriv"; } };

//        template<> struct DataTypeName< defaulttype::DeformationGradient332fTypes::Deriv > { static const char* name() { return "DeformationGradient332fTypes::Deriv"; } };

template<> struct DataTypeName< defaulttype::DeformationGradient332dTypes::Coord > { static const char* name() { return "DeformationGradient332dTypes::CoordOrDeriv"; } };

//        template<> struct DataTypeName< defaulttype::DeformationGradient332dTypes::Deriv > { static const char* name() { return "DeformationGradient332dTypes::Deriv"; } };


template<> struct DataTypeName< defaulttype::DeformationGradient332fMass > { static const char* name() { return "DeformationGradient332fMass"; } };

template<> struct DataTypeName< defaulttype::DeformationGradient332dMass > { static const char* name() { return "DeformationGradient332dMass"; } };

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
template<class DeformationGradientT, class Vec, class M, class SV>
typename DeformationGradientT::Deriv inertiaForce ( const SV& /*sv*/, const Vec& /*a*/, const M& /*m*/, const typename DeformationGradientT::Coord& /*x*/, const  typename DeformationGradientT::Deriv& /*v*/ );

/// Specialization of the inertia force for defaulttype::DeformationGradient3dTypes
template <>
inline defaulttype::DeformationGradient332dTypes::Deriv inertiaForce <
defaulttype::DeformationGradient332dTypes,
            objectmodel::BaseContext::Vec3,
            defaulttype::DeformationGradientMass<3,3,2, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::DeformationGradientMass<3,3,2, double>& mass,
                    const defaulttype::DeformationGradient332dTypes::Coord& x,
                    const defaulttype::DeformationGradient332dTypes::Deriv& v
            )
{
    defaulttype::Vec3d omega ( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::Vec3d origin = x.getCenter(), finertia;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getCenter() * 2 ) ) * mass.mass;
    defaulttype::DeformationGradient332dTypes::Deriv result;
    result[0]=finertia[0]; result[1]=finertia[1]; result[2]=finertia[2];
    return result;
}

/// Specialization of the inertia force for defaulttype::DeformationGradient3dTypes
template <>
inline defaulttype::DeformationGradient332fTypes::Deriv inertiaForce <
defaulttype::DeformationGradient332fTypes,
            objectmodel::BaseContext::Vec3,
            defaulttype::DeformationGradientMass<3,3,2, double>,
            objectmodel::BaseContext::SpatialVector
            >
            (
                    const objectmodel::BaseContext::SpatialVector& vframe,
                    const objectmodel::BaseContext::Vec3& aframe,
                    const defaulttype::DeformationGradientMass<3,3,2, double>& mass,
                    const defaulttype::DeformationGradient332fTypes::Coord& x,
                    const defaulttype::DeformationGradient332fTypes::Deriv& v
            )
{
    const defaulttype::Vec3f omega ( (float)vframe.lineVec[0], (float)vframe.lineVec[1], (float)vframe.lineVec[2] );
    defaulttype::Vec3f origin = x.getCenter(), finertia;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + v.getCenter() * 2 ) ) * mass.mass;
    defaulttype::DeformationGradient332fTypes::Deriv result;
    result[0]=finertia[0]; result[1]=finertia[1]; result[2]=finertia[2];
    return result;
}


} // namespace behavoir

} // namespace core

} // namespace sofa


#endif
