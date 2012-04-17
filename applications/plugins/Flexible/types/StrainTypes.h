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
#ifndef FLEXIBLE_StrainTYPES_H
#define FLEXIBLE_StrainTYPES_H

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

namespace sofa
{

namespace defaulttype
{



using std::endl;
using helper::vector;

template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
class Strain;


/** Measure of strain, based on a deformation gradient. Corotational or Green-Lagrange. Can also represent the corresponding stress. Used by the materials to compute stress based on strain.
  */
template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
class StrainTypes
{
public:

    static const unsigned int spatial_dimensions = _spatial_dimensions;   ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int material_dimensions = _material_dimensions; ///< Number of dimensions of the material space (=number of axes of the deformable gradient): 3 for a volume object, 2 for a surface, 1 for a line.
    static const unsigned int order = _order;  ///< 0: point, 1:point+ strain, 2: point + strain + gradient, 3: point + strain + gradient + Hessian
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    static const unsigned int NumStrainVec = order==0? 0 : (order==1? 1 : ( order==2? 1 + spatial_dimensions : ( order==3? 1 + spatial_dimensions + spatial_dimensions*spatial_dimensions: 0 ))) ;
    static const unsigned int VSize = /*spatial_dimensions +*/ NumStrainVec * strain_size ;  // number of entries
    enum { coord_total_size = VSize };
    enum { deriv_total_size = VSize };
    typedef _Real Real;
    typedef vector<Real> VecReal;

    // ------------    Types and methods defined for easier data access
    typedef Vec<material_dimensions, Real> MaterialCoord;
    typedef vector<MaterialCoord> VecMaterialCoord;
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Mat<material_dimensions,material_dimensions,Real> StrainMat;    ///< Strain in matrix form
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    typedef Vec<spatial_dimensions, StrainVec> StrainGradient;          ///< Strain Gradient (for order > 1)
    typedef Mat<spatial_dimensions,spatial_dimensions, StrainVec> StrainHessian;     ///< Strain Hessian (for order > 2). @todo Could be optimized since hessian is symmetric

    /** Time derivative of a strain, or other vector-like associated quantities, such as stress.
    */
    class Deriv
    {
    protected:
        Vec<VSize,Real> v;

    public:
        Deriv() { v.clear(); }
        Deriv( const Vec<VSize,Real>& d):v(d) {}
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

        /// point, order 0
//        SpatialCoord& getCenter(){ return *reinterpret_cast<SpatialCoord*>(&v[0]); }
//        const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

        /// strain, order 1
        StrainVec& getStrain() { return *reinterpret_cast<StrainVec*>(&v[/*spatial_dimensions*/0]); }
        const StrainVec& getStrain() const { return *reinterpret_cast<const StrainVec*>(&v[/*spatial_dimensions*/0]); }

        /// gradient, order 2
        StrainGradient& getStrainGradient() { return *reinterpret_cast<StrainGradient*>(&v[/*spatial_dimensions+*/strain_size]); }
        const StrainGradient& getStrainGradient() const { return *reinterpret_cast<const StrainGradient*>(&v[/*spatial_dimensions+*/strain_size]); }

        /// Hessian, order 3
        StrainHessian& getStrainHessian() { return *reinterpret_cast<StrainHessian*>(&v[/*spatial_dimensions+*/strain_size+spatial_dimensions * strain_size]); }
        const StrainHessian& getStrainHessian() const { return *reinterpret_cast<const StrainHessian*>(&v[/*spatial_dimensions+*/strain_size+spatial_dimensions * strain_size]); }

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
        Real operator*(const Deriv& a) const    { return v*a.v;    }

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Deriv& c )
        {
            out<<c.v;
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Deriv& c )
        {
            in>>c.v;
            return in;
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

    /** strain, or other vector-like associated quantities, such as stress.
    */
    class Coord
    {
    protected:
        Vec<VSize,Real> v;

    public:
        Coord() { v.clear(); }
        Coord( const Vec<VSize,Real>& d):v(d) {}
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

        /// point, order 0
//        SpatialCoord& getCenter(){ return *reinterpret_cast<SpatialCoord*>(&v[0]); }
//        const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

        /// strain, order 1
        StrainVec& getStrain() { return *reinterpret_cast<StrainVec*>(&v[/*spatial_dimensions*/0]); }
        const StrainVec& getStrain() const { return *reinterpret_cast<const StrainVec*>(&v[/*spatial_dimensions*/0]); }

        /// gradient, order 2
        StrainGradient& getStrainGradient() { return *reinterpret_cast<StrainGradient*>(&v[/*spatial_dimensions+*/strain_size]); }
        const StrainGradient& getStrainGradient() const { return *reinterpret_cast<const StrainGradient*>(&v[/*spatial_dimensions+*/strain_size]); }

        /// Hessian, order 3
        StrainHessian& getStrainHessian() { return *reinterpret_cast<StrainHessian*>(&v[/*spatial_dimensions+*/strain_size+spatial_dimensions * strain_size]); }
        const StrainHessian& getStrainHessian() const { return *reinterpret_cast<const StrainHessian*>(&v[/*spatial_dimensions+*/strain_size+spatial_dimensions * strain_size]); }

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
            out<<c.v;
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Coord& c )
        {
            in>>c.v;
            return in;
        }


        /// Write the OpenGL transformation matrix
        void writeOpenGlMatrix ( float m[16] ) const
        {
            BOOST_STATIC_ASSERT(spatial_dimensions == 3);
            m[0] = (float)getStrain()(0);
            m[4] = (float)0;
            m[8] = (float)0;
            m[1] = (float)0;
            m[5] = (float)getStrain()(1);
            m[9] = (float)0;
            m[2] = (float)0;
            m[6] = (float)0;
            m[10] = (float)getStrain()(2);
            m[3] = 0;
            m[7] = 0;
            m[11] = 0;
            m[12] = /*( float ) getCenter()[0]*/0;
            m[13] =/* ( float ) getCenter()[1]*/0;
            m[14] = /*( float ) getCenter()[2]*/0;
            m[15] = 1;
        }
    };

    typedef vector<Coord> VecCoord;

    static const char* Name();

    static Coord interpolate ( const helper::vector< Coord > & ancestors, const helper::vector< Real > & coefs )
    {
        assert ( ancestors.size() == coefs.size() );
        Coord c;
        for ( unsigned int i = 0; i < ancestors.size(); i++ ) c += ancestors[i] * coefs[i];  // Position and deformation gradient linear interpolation.
        return c;
    }



    /** @name Conversions
              * Convert to/from points in space
             */
    //@{

    template<typename T>
    static void set ( Deriv& /*c*/, T /*x*/, T /*y*/, T /*z*/ )
    {
//        c.clear();
//        c.getCenter()[0] = ( Real ) x;
//        c.getCenter() [1] = ( Real ) y;
//        c.getCenter() [2] = ( Real ) z;
    }

    template<typename T>
    static void get ( T& /*x*/, T& /*y*/, T& /*z*/, const Deriv& /*c*/ )
    {
//        x = ( T ) c.getCenter() [0];
//        y = ( T ) c.getCenter() [1];
//        z = ( T ) c.getCenter() [2];
    }

    template<typename T>
    static void add ( Deriv& /*c*/, T /*x*/, T /*y*/, T /*z*/ )
    {
//        c.getCenter() [0] += ( Real ) x;
//        c.getCenter() [1] += ( Real ) y;
//        c.getCenter() [2] += ( Real ) z;
    }

    template<typename T>
    static void set ( Coord& /*c*/, T /*x*/, T /*y*/, T /*z*/ )
    {
//        c.clear();
//        c.getCenter()[0] = ( Real ) x;
//        c.getCenter() [1] = ( Real ) y;
//        c.getCenter() [2] = ( Real ) z;
    }

    template<typename T>
    static void get ( T& /*x*/, T& /*y*/, T& /*z*/, const Coord& /*c*/ )
    {
//        x = ( T ) c.getCenter() [0];
//        y = ( T ) c.getCenter() [1];
//        z = ( T ) c.getCenter() [2];
    }

    template<typename T>
    static void add ( Coord& /*c*/, T /*x*/, T /*y*/, T /*z*/ )
    {
//        c.getCenter() [0] += ( Real ) x;
//        c.getCenter() [1] += ( Real ) y;
//        c.getCenter() [2] += ( Real ) z;
    }
    //@}

};

// ==========================================================================
// Helpers


/// Convert a symetric matrix to voigt notation  exx=Fxx, eyy=Fyy, ezz=Fzz, exy=(Fxy+Fyx)/2 eyz=(Fyz+Fzy)/2, ezx=(Fxz+Fzx)/2,
template<int material_dimensions, typename Real>
static Vec<material_dimensions * (1+material_dimensions) / 2, Real> MatToVoigt(  const  Mat<material_dimensions,material_dimensions, Real>& f)
{
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    StrainVec s;
    unsigned int ei=0;
    for(unsigned int j=0; j<material_dimensions; j++)
        for( unsigned int k=0; k<material_dimensions-j; k++ )
            s[ei++] = (f[k][k+j]+f[k+j][k])*(Real)0.5;
    return s;
}

/// Voigt notation to symmetric matrix (F+F^T)/2  Fxx=exx, Fxy=Fyx=exy/2, etc.
template<typename Real>
static Mat<3,3, Real> VoigtToMat( const Vec<6, Real>& s  )
{
    static const unsigned int material_dimensions=3;
    Mat<material_dimensions,material_dimensions, Real> f;
    unsigned int ei=0;
    for(unsigned int j=0; j<material_dimensions; j++)
    {
        for( unsigned int k=0; k<material_dimensions-j; k++ )
        {
            f[k][k+j] = f[k+j][k] = s[ei] ;
            if(0!=j) {f[k][k+j] *= 0.5; f[k+j][k] *= 0.5;}
            ei++;
        }
    }
    return f;
}


// ==========================================================================
// Mass

/** Mass associated with a sampling point
      */
template<int _spatial_dimensions,int _material_dimensions, int _order, typename _Real>
struct StrainMass
{
    typedef _Real Real;
    Real mass;  ///< Currently only a scalar mass, but a matrix should be used for more precision
    // operator to cast to const Real
    operator const Real() const     {     return mass;    }
    template<int S, int M, int O, typename R>
    inline friend std::ostream& operator << ( std::ostream& out, const StrainMass<S,M,O,R>& m )    {        out << m.mass;        return out;    }
    template<int S, int M, int O, typename R>
    inline friend std::istream& operator >> ( std::istream& in, StrainMass<S,M,O,R>& m )    {       in >> m.mass;       return in;    }
    void operator *= ( Real fact )     {       mass *= fact;    }
    void operator /= ( Real fact )    {       mass /= fact;   }
};

//template<int S, int M, int O, typename R>
//inline StrainTypes<S,M,O,R>::Deriv operator* ( const StrainTypes<S,M,O,R>::Deriv& d, const StrainMass<S,M,O,R>& m )
//{
//    StrainTypes<S,M,O,R>::Deriv res;
//    StrainTypes<S,M,O,R>::center(res) = StrainTypes<S,M,O,R>::center(d) * m.mass;
//    return res;
//}

//template<int S, int M, int O, typename R>
//inline StrainTypes<S,M,O,R>::Deriv operator/ ( const StrainTypes<S,M,O,R>::Deriv& d, const StrainMass<S,M,O,R>& m )
//{
//    StrainTypes<S,M,O,R>::Deriv res;
//    StrainTypes<S,M,O,R>::center(res) = StrainTypes<S,M,O,R>::center(d) / m.mass;
//    return res;
//}




// ==========================================================================
// order 1

typedef StrainTypes<3, 3, 1, double> Strain331dTypes;
typedef StrainTypes<3, 3, 1, float>  Strain331fTypes;

typedef StrainMass<3, 3, 1, double> Strain331dMass;
typedef StrainMass<3, 3, 1, float>  Strain331fMass;

#ifdef SOFA_FLOAT
template<> inline const char* Strain331dTypes::Name() { return "Strain331d"; }
template<> inline const char* Strain331fTypes::Name() { return "Strain331"; }
#else
template<> inline const char* Strain331dTypes::Name() { return "Strain331"; }
template<> inline const char* Strain331fTypes::Name() { return "Strain331f"; }
#endif

#ifdef SOFA_FLOAT
typedef Strain331fTypes Strain331Types;
typedef Strain331fMass Strain331Mass;
#else
typedef Strain331dTypes Strain331Types;
typedef Strain331dMass Strain331Mass;
#endif

template<>
struct DataTypeInfo< Strain331fTypes::Deriv > : public FixedArrayTypeInfo< Strain331fTypes::Deriv, Strain331fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "Strain331<" << DataTypeName<float>::name() << ">"; return o.str(); }
};
template<>
struct DataTypeInfo< Strain331dTypes::Deriv > : public FixedArrayTypeInfo< Strain331dTypes::Deriv, Strain331dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "Strain331<" << DataTypeName<double>::name() << ">"; return o.str(); }
};


// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Strain331fTypes::Coord > { static const char* name() { return "Strain331fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::Strain331dTypes::Coord > { static const char* name() { return "Strain331dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::Strain331fMass > { static const char* name() { return "Strain331fMass"; } };
template<> struct DataTypeName< defaulttype::Strain331dMass > { static const char* name() { return "Strain331dMass"; } };

/// \endcond


// ==========================================================================
// order 2

typedef StrainTypes<3, 3, 2, double> Strain332dTypes;
typedef StrainTypes<3, 3, 2, float>  Strain332fTypes;

typedef StrainMass<3, 3, 2, double> Strain332dMass;
typedef StrainMass<3, 3, 2, float>  Strain332fMass;

#ifdef SOFA_FLOAT
template<> inline const char* Strain332dTypes::Name() { return "Strain332d"; }
template<> inline const char* Strain332fTypes::Name() { return "Strain332"; }
#else
template<> inline const char* Strain332dTypes::Name() { return "Strain332"; }
template<> inline const char* Strain332fTypes::Name() { return "Strain332f"; }
#endif

#ifdef SOFA_FLOAT
typedef Strain332fTypes Strain332Types;
typedef Strain332fMass Strain332Mass;
#else
typedef Strain332dTypes Strain332Types;
typedef Strain332dMass Strain332Mass;
#endif

template<>
struct DataTypeInfo< Strain332fTypes::Deriv > : public FixedArrayTypeInfo< Strain332fTypes::Deriv, Strain332fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "Strain332<" << DataTypeName<float>::name() << ">"; return o.str(); }
};
template<>
struct DataTypeInfo< Strain332dTypes::Deriv > : public FixedArrayTypeInfo< Strain332dTypes::Deriv, Strain332dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "Strain332<" << DataTypeName<double>::name() << ">"; return o.str(); }
};


// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Strain332fTypes::Coord > { static const char* name() { return "Strain332fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::Strain332dTypes::Coord > { static const char* name() { return "Strain332dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::Strain332fMass > { static const char* name() { return "Strain332fMass"; } };
template<> struct DataTypeName< defaulttype::Strain332dMass > { static const char* name() { return "Strain332dMass"; } };

/// \endcond


// ==========================================================================
// order 3

typedef StrainTypes<3, 3, 3, double> Strain333dTypes;
typedef StrainTypes<3, 3, 3, float>  Strain333fTypes;

typedef StrainMass<3, 3, 3, double> Strain333dMass;
typedef StrainMass<3, 3, 3, float>  Strain333fMass;

#ifdef SOFA_FLOAT
template<> inline const char* Strain333dTypes::Name() { return "Strain333d"; }
template<> inline const char* Strain333fTypes::Name() { return "Strain333"; }
#else
template<> inline const char* Strain333dTypes::Name() { return "Strain333"; }
template<> inline const char* Strain333fTypes::Name() { return "Strain333f"; }
#endif

#ifdef SOFA_FLOAT
typedef Strain333fTypes Strain333Types;
typedef Strain333fMass Strain333Mass;
#else
typedef Strain333dTypes Strain333Types;
typedef Strain333dMass Strain333Mass;
#endif

template<>
struct DataTypeInfo< Strain333fTypes::Deriv > : public FixedArrayTypeInfo< Strain333fTypes::Deriv, Strain333fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "Strain333<" << DataTypeName<float>::name() << ">"; return o.str(); }
};
template<>
struct DataTypeInfo< Strain333dTypes::Deriv > : public FixedArrayTypeInfo< Strain333dTypes::Deriv, Strain333dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "Strain333<" << DataTypeName<double>::name() << ">"; return o.str(); }
};


// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Strain333fTypes::Coord > { static const char* name() { return "Strain333fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::Strain333dTypes::Coord > { static const char* name() { return "Strain333dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::Strain333fMass > { static const char* name() { return "Strain333fMass"; } };
template<> struct DataTypeName< defaulttype::Strain333dMass > { static const char* name() { return "Strain333dMass"; } };

/// \endcond


} // namespace defaulttype




// ==========================================================================
// Mechanical Object

namespace component
{

namespace container
{

#if defined(WIN32) && !defined(FLEXIBLE_StrainTYPES_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Strain331dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Strain331dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Strain332dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Strain332dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Strain333dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Strain333dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Strain331fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Strain331fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Strain332fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Strain332fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Strain333fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Strain333fTypes>;
#endif
#endif

} // namespace container

} // namespace component



} // namespace sofa


#endif
