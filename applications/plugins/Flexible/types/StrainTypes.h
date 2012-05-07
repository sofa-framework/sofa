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

#include "../types/PolynomialBasis.h"
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

/**
    Generic class to implement a mechanical state representing strain
    strain is decomposed in a basis of an certain order.
    It is used here to implement different measures: Corotational, Green-Lagrange, Invariants of right Cauchy Green deformation tensor, etc.
    Can also represent the corresponding stress. Used by the materials to compute stress based on strain.
  */

template< int _spatial_dimensions, int _strain_size, int _order, typename _Real >
class BaseStrainTypes
{
public:
    typedef PolynomialBasis<_strain_size,_Real, _spatial_dimensions, _order> Basis;

    static const unsigned int spatial_dimensions = _spatial_dimensions;   ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int strain_size = _strain_size;
    typedef _Real Real;

    typedef typename Basis::T StrainVec;
    typedef typename Basis::Gradient StrainGradient;
    typedef typename Basis::Hessian StrainHessian;
    typedef typename Basis::TotalVec TotalVec;

    enum { VSize = Basis::total_size };  ///< number of entries
    enum { coord_total_size = VSize };
    enum { deriv_total_size = VSize };

    typedef vector<Real> VecReal;

    class Deriv
    {
    protected:
        Basis v;

    public:
        Deriv() { v.clear(); }
        Deriv( const Basis& d):v(d) {}
        Deriv( const TotalVec& d):v(d) {}
        void clear() { v.clear(); }

        static const unsigned int total_size = VSize;
        typedef Real value_type;

        static unsigned int size() { return VSize; }

        /// seen as a vector
        Real* ptr() { return v.ptr(); }
        const Real* ptr() const { return v.ptr(); }

        TotalVec& getVec() { return v.getVec(); }
        const TotalVec& getVec() const { return v.getVec(); }

        Real& operator[](int i) { return getVec()[i]; }
        const Real& operator[](int i) const    { return getVec()[i]; }

        /// basis
        Basis& getBasis() { return v; }
        const Basis& getBasis() const { return v; }

        StrainVec& getStrain() { return v.getVal(); }
        const StrainVec& getStrain() const { return v.getVal(); }

        StrainVec& getStrainGradient(int i) { return v.getGradient()[i]; }
        const StrainVec& getStrainGradient(int i) const { return v.getGradient()[i]; }

        StrainVec& getStrainHessian(int i,int j) { return v.getHessian()(i,j); }
        const StrainVec& getStrainHessian(int i,int j) const { return v.getHessian()(i,j); }


        Deriv operator +(const Deriv& a) const { return Deriv(getVec()+a.getVec()); }
        void operator +=(const Deriv& a) { getVec()+=a.getVec(); }

        Deriv operator -(const Deriv& a) const { return Deriv(getVec()-a.getVec()); }
        void operator -=(const Deriv& a) { getVec()-=a.getVec(); }

        template<typename real2>
        Deriv operator *(real2 a) const { return Deriv(getVec()*a); }
        template<typename real2>
        void operator *=(real2 a) { getVec() *= a; }

        template<typename real2>
        void operator /=(real2 a) { getVec() /= a; }

        Deriv operator - () const { return Deriv(-getVec()); }

        /// dot product, mostly used to compute residuals as sqrt(x*x)
        Real operator*(const Deriv& a) const    { return getVec()*a.getVec();    }

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Deriv& c )
        {
            out<<c.getVec();
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Deriv& c )
        {
            in>>c.getVec();
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

    class Coord
    {
    protected:
        Basis v;

    public:
        Coord() { v.clear(); }
        Coord( const Basis& d):v(d) {}
        Coord( const TotalVec& d):v(d) {}
        void clear() { v.clear(); }

        static const unsigned int total_size = VSize;
        typedef Real value_type;

        static unsigned int size() { return VSize; }

        /// seen as a vector
        Real* ptr() { return v.ptr(); }
        const Real* ptr() const { return v.ptr(); }

        TotalVec& getVec() { return v.getVec(); }
        const TotalVec& getVec() const { return v.getVec(); }

        Real& operator[](int i) { return getVec()[i]; }
        const Real& operator[](int i) const    { return getVec()[i]; }

        /// basis
        Basis& getBasis() { return v; }
        const Basis& getBasis() const { return v; }

        StrainVec& getStrain() { return v.getVal(); }
        const StrainVec& getStrain() const { return v.getVal(); }

        StrainVec& getStrainGradient(int i) { return v.getGradient()[i]; }
        const StrainVec& getStrainGradient(int i) const { return v.getGradient()[i]; }

        StrainVec& getStrainHessian(int i,int j) { return v.getHessian()(i,j); }
        const StrainVec& getStrainHessian(int i,int j) const { return v.getHessian()(i,j); }

        Coord operator +(const Coord& a) const { return Coord(getVec()+a.getVec()); }
        void operator +=(const Coord& a) { getVec()+=a.getVec(); }

        Coord operator +(const Deriv& a) const { return Coord(getVec()+a.getVec()); }
        void operator +=(const Deriv& a) { getVec()+=a.getVec(); }

        Coord operator -(const Coord& a) const { return Coord(getVec()-a.getVec()); }
        void operator -=(const Coord& a) { getVec()-=a.getVec(); }

        template<typename real2>
        Coord operator *(real2 a) const { return Coord(getVec()*a); }
        template<typename real2>
        void operator *=(real2 a) { getVec() *= a; }

        template<typename real2>
        void operator /=(real2 a) { getVec() /= a; }

        Coord operator - () const { return Coord(-getVec()); }

        /// dot product, mostly used to compute residuals as sqrt(x*x)
        Real operator*(const Coord& a) const    { return getVec()*a.getVec();    }

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Coord& c )
        {
            out<<c.getVec();
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Coord& c )
        {
            in>>c.getVec();
            return in;
        }

        /// Write the OpenGL transformation matrix
        void writeOpenGlMatrix ( float m[16] ) const
        {
            for(unsigned int i=0; i<15; i++) m[i]=0.; m[15]=1.;
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
    static void set ( Deriv& /*c*/, T /*x*/, T /*y*/, T /*z*/ )  {    }
    template<typename T>
    static void get ( T& /*x*/, T& /*y*/, T& /*z*/, const Deriv& /*c*/ ) {    }
    template<typename T>
    static void add ( Deriv& /*c*/, T /*x*/, T /*y*/, T /*z*/ )    {    }
    template<typename T>
    static void set ( Coord& /*c*/, T /*x*/, T /*y*/, T /*z*/ )    {    }
    template<typename T>
    static void get ( T& /*x*/, T& /*y*/, T& /*z*/, const Coord& /*c*/ )    {    }
    template<typename T>
    static void add ( Coord& /*c*/, T /*x*/, T /*y*/, T /*z*/ )    {    }
    //@}

};




// ==========================================================================
// Specialization for strain defined using Voigt notation


template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real >
class StrainTypes: public BaseStrainTypes<_spatial_dimensions,_material_dimensions * (1+_material_dimensions) / 2,_order,_Real>
{
public:
    typedef BaseStrainTypes<_spatial_dimensions,_material_dimensions * (1+_material_dimensions) / 2,_order,_Real> Inherit;

    typedef typename Inherit::Basis Basis;
    enum { spatial_dimensions = Inherit::spatial_dimensions };
    enum { strain_size = Inherit::strain_size };
    enum { VSize = Inherit::VSize } ;
    enum { coord_total_size = Inherit::coord_total_size };
    enum { deriv_total_size = Inherit::deriv_total_size };
    typedef typename Inherit::Real Real;
    typedef typename Inherit::VecReal VecReal;
    typedef typename Inherit::StrainVec StrainVec;
    typedef typename Inherit::StrainGradient StrainGradient;
    typedef typename Inherit::StrainHessian StrainHessian;
    typedef typename Inherit::VecDeriv VecDeriv;
    typedef typename Inherit::MatrixDeriv MatrixDeriv;
    typedef typename Inherit::VecCoord VecCoord;
    static const char* Name();

    static const unsigned int material_dimensions = _material_dimensions; ///< Number of dimensions of the material space (=number of axes of the deformable gradient): 3 for a volume object, 2 for a surface, 1 for a line.
    typedef Vec<material_dimensions, Real> MaterialCoord;
    typedef vector<MaterialCoord> VecMaterialCoord;
    typedef Mat<material_dimensions,material_dimensions,Real> StrainMat;    ///< Strain in matrix form
};

typedef StrainTypes<3, 3, 0, double> E331dTypes;
typedef StrainTypes<3, 3, 0, float>  E331fTypes;
typedef StrainTypes<3, 3, 1, double> E332dTypes;
typedef StrainTypes<3, 3, 1, float>  E332fTypes;
typedef StrainTypes<3, 3, 2, double> E333dTypes;
typedef StrainTypes<3, 3, 2, float>  E333fTypes;

#ifdef SOFA_FLOAT
template<> inline const char* E331dTypes::Name() { return "E331d"; }
template<> inline const char* E331fTypes::Name() { return "E331"; }
template<> inline const char* E332dTypes::Name() { return "E332d"; }
template<> inline const char* E332fTypes::Name() { return "E332"; }
template<> inline const char* E333dTypes::Name() { return "E333d"; }
template<> inline const char* E333fTypes::Name() { return "E333"; }
#else
template<> inline const char* E331dTypes::Name() { return "E331"; }
template<> inline const char* E331fTypes::Name() { return "E331f"; }
template<> inline const char* E332dTypes::Name() { return "E332"; }
template<> inline const char* E332fTypes::Name() { return "E332f"; }
template<> inline const char* E333dTypes::Name() { return "E333"; }
template<> inline const char* E333fTypes::Name() { return "E333f"; }
#endif

#ifdef SOFA_FLOAT
typedef E331fTypes E331Types;
typedef E332fTypes E332Types;
typedef E333fTypes E333Types;
#else
typedef E331dTypes E331Types;
typedef E332dTypes E332Types;
typedef E333dTypes E333Types;
#endif

template<> struct DataTypeInfo< E331fTypes::Deriv > : public FixedArrayTypeInfo< E331fTypes::Deriv, E331fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E331<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E331dTypes::Deriv > : public FixedArrayTypeInfo< E331dTypes::Deriv, E331dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E331<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E332fTypes::Deriv > : public FixedArrayTypeInfo< E332fTypes::Deriv, E332fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E332<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E332dTypes::Deriv > : public FixedArrayTypeInfo< E332dTypes::Deriv, E332dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E332<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E333fTypes::Deriv > : public FixedArrayTypeInfo< E333fTypes::Deriv, E333fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E333<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E333dTypes::Deriv > : public FixedArrayTypeInfo< E333dTypes::Deriv, E333dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E333<" << DataTypeName<double>::name() << ">"; return o.str(); } };

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::E331fTypes::Coord > { static const char* name() { return "E331fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E331dTypes::Coord > { static const char* name() { return "E331dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E332fTypes::Coord > { static const char* name() { return "E332fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E332dTypes::Coord > { static const char* name() { return "E332dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E333fTypes::Coord > { static const char* name() { return "E333fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E333dTypes::Coord > { static const char* name() { return "E333dTypes::CoordOrDeriv"; } };

/// \endcond


// ==========================================================================
// Specialization for Strain defined using Invariants of right Cauchy Green deformation tensor
/*
template<int _spatial_dimensions, int _order, typename _Real>
class InvariantsTypes: public BaseStrainTypes<_spatial_dimensions,3,_order,_Real>
{
public:
    typedef BaseStrainTypes<3,_order,_Real> Inherit;

    enum { spatial_dimensions = Inherit::spatial_dimensions };
    enum { order = Inherit::order };
    enum { strain_size = Inherit::strain_size };
    enum { NumStrainVec = Inherit::NumStrainVec } ;
    enum { VSize = Inherit::VSize } ;
    enum { coord_total_size = Inherit::coord_total_size };
    enum { deriv_total_size = Inherit::deriv_total_size };
    typedef typename Inherit::Real Real;
    typedef typename Inherit::VecReal VecReal;
    typedef typename Inherit::StrainVec StrainVec;
    typedef typename Inherit::StrainGradient StrainGradient;
    typedef typename Inherit::StrainHessian StrainHessian;
    typedef typename Inherit::VecDeriv VecDeriv;
    typedef typename Inherit::MatrixDeriv MatrixDeriv;
    typedef typename Inherit::VecCoord VecCoord;
    static const char* Name();

    static const unsigned int material_dimensions = _material_dimensions; ///< Number of dimensions of the material space (=number of axes of the deformable gradient): 3 for a volume object, 2 for a surface, 1 for a line.
    typedef Vec<material_dimensions, Real> MaterialCoord;
    typedef vector<MaterialCoord> VecMaterialCoord;
    typedef Mat<material_dimensions,material_dimensions,Real> StrainMat;    ///< Strain in matrix form
};
*/

typedef BaseStrainTypes<3, 3, 0, double> I331dTypes;
typedef BaseStrainTypes<3, 3, 0, float>  I331fTypes;
typedef BaseStrainTypes<3, 3, 1, double> I332dTypes;
typedef BaseStrainTypes<3, 3, 1, float>  I332fTypes;
typedef BaseStrainTypes<3, 3, 2, double> I333dTypes;
typedef BaseStrainTypes<3, 3, 2, float>  I333fTypes;

#ifdef SOFA_FLOAT
template<> inline const char* I331dTypes::Name() { return "I331d"; }
template<> inline const char* I331fTypes::Name() { return "I331"; }
template<> inline const char* I332dTypes::Name() { return "I332d"; }
template<> inline const char* I332fTypes::Name() { return "I332"; }
template<> inline const char* I333dTypes::Name() { return "I333d"; }
template<> inline const char* I333fTypes::Name() { return "I333"; }
#else
template<> inline const char* I331dTypes::Name() { return "I331"; }
template<> inline const char* I331fTypes::Name() { return "I331f"; }
template<> inline const char* I332dTypes::Name() { return "I332"; }
template<> inline const char* I332fTypes::Name() { return "I332f"; }
template<> inline const char* I333dTypes::Name() { return "I333"; }
template<> inline const char* I333fTypes::Name() { return "I333f"; }
#endif

#ifdef SOFA_FLOAT
typedef I331fTypes I331Types;
typedef I332fTypes I332Types;
typedef I333fTypes I333Types;
#else
typedef I331dTypes I331Types;
typedef I332dTypes I332Types;
typedef I333dTypes I333Types;
#endif

template<> struct DataTypeInfo< I331fTypes::Deriv > : public FixedArrayTypeInfo< I331fTypes::Deriv, I331fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "I331<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< I331dTypes::Deriv > : public FixedArrayTypeInfo< I331dTypes::Deriv, I331dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "I331<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< I332fTypes::Deriv > : public FixedArrayTypeInfo< I332fTypes::Deriv, I332fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "I332<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< I332dTypes::Deriv > : public FixedArrayTypeInfo< I332dTypes::Deriv, I332dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "I332<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< I333fTypes::Deriv > : public FixedArrayTypeInfo< I333fTypes::Deriv, I333fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "I333<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< I333dTypes::Deriv > : public FixedArrayTypeInfo< I333dTypes::Deriv, I333dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "I333<" << DataTypeName<double>::name() << ">"; return o.str(); } };

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::I331fTypes::Coord > { static const char* name() { return "I331fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::I331dTypes::Coord > { static const char* name() { return "I331dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::I332fTypes::Coord > { static const char* name() { return "I332fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::I332dTypes::Coord > { static const char* name() { return "I332dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::I333fTypes::Coord > { static const char* name() { return "I333fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::I333dTypes::Coord > { static const char* name() { return "I333dTypes::CoordOrDeriv"; } };

/// \endcond


// ==========================================================================
// Helpers

/// Convert a symetric matrix to voigt notation  exx=Fxx, eyy=Fyy, ezz=Fzz, exy=(Fxy+Fyx)/2 eyz=(Fyz+Fzy)/2, ezx=(Fxz+Fzx)/2,
template<int material_dimensions, typename Real>
static inline Vec<material_dimensions * (1+material_dimensions) / 2, Real> MatToVoigt(  const  Mat<material_dimensions,material_dimensions, Real>& f)
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
static inline Mat<3,3, Real> VoigtToMat( const Vec<6, Real>& s  )
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



} // namespace defaulttype




// ==========================================================================
// Mechanical Object

namespace component
{

namespace container
{

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_StrainTYPES_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::E331dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E331dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::E332dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E332dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::E333dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E333dTypes>;

extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::I331dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::I331dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::I332dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::I332dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::I333dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::I333dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::E331fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E331fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::E332fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E332fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::E333fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E333fTypes>;

extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::I331fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::I331fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::I332fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::I332fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::I333fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::I333fTypes>;
#endif
#endif

} // namespace container

} // namespace component



} // namespace sofa


#endif
