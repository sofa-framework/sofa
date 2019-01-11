/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef FLEXIBLE_StrainTYPES_H
#define FLEXIBLE_StrainTYPES_H

#include <Flexible/config.h>
#include "../types/PolynomialBasis.h"
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/random.h>
#ifdef SOFA_SMP
#include <sofa/defaulttype/SharedTypes.h>
#endif /* SOFA_SMP */

namespace sofa
{

namespace defaulttype
{

/**
    Generic class to implement a mechanical state representing strain
    strain is decomposed in a basis of an certain order.
    It is used here to implement different measures: Corotational, Green-Lagrange, Invariants of right Cauchy Green deformation tensor, etc.
    Can also represent the corresponding stress. Used by the materials to compute stress based on strain.
    _name template parameter is useful to be able to differenciate several BaseStrainTypes templated on the same _spatial_dimensions/_strain_size/_order/_Real
  */

template< int _spatial_dimensions, int _strain_size, int _order, typename _Real, char _name >
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
    enum { order = _order };

    typedef helper::vector<Real> VecReal;

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
        typedef int size_type;

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

        StrainVec& getStrainHessian(int i) { return v.getHessian().elems[i]; }
        const StrainVec& getStrainHessian(int i) const { return v.getHessian().elems[i]; }


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

        /// Euclidean norm
        Real norm() const
        {
            return getVec().norm();
        }

        /// @name Comparison operators
        /// @{

        bool operator==(const Deriv& o) const
        {
            return getVec() == o.getVec();
        }

        bool operator!=(const Deriv& o) const
        {
            return getVec() != o.getVec();
        }

        /// @}
    };

    typedef helper::vector<Deriv> VecDeriv;
    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    static Deriv interpolate ( const helper::vector< Deriv > & ancestors, const helper::vector< Real > & coefs )
    {
        assert ( ancestors.size() == coefs.size() );
        Deriv c;
        for ( unsigned int i = 0; i < ancestors.size(); i++ )     c += ancestors[i] * coefs[i];
        return c;
    }

    typedef Deriv Coord;

    typedef helper::vector<Coord> VecCoord;

    static const char* Name();

    /** @name Conversions
              * Convert to/from points in space
             */
    //@{
    template<typename T>
    static void set ( Deriv& /*c*/, T /*x*/, T /*y*/, T /*z*/ )  {    }
    template<typename T>
    static void get ( T& x, T& y, T& z, const Deriv& /*c*/ ) { x=y=z=0; /*std::cerr<<"WARNING: BaseStrainTypes::get(): a strain cannot be converted to spatial coordinates.\n"; */ }
    template<typename T>
    static void add ( Deriv& /*c*/, T /*x*/, T /*y*/, T /*z*/ )    {    }
    //@}

    static const TotalVec& getCPos(const Deriv& c) { return c.getVec(); }

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv( Real minMagnitude, Real maxMagnitude )
    {
        Deriv result;
        for( unsigned int i=0 ; i<VSize ; ++i )
            result[i] = Real(helper::drand(minMagnitude,maxMagnitude));
        return result;
    }

    /// for finite difference methods 
    static Deriv coordDifference(const Coord& c1, const Coord& c2)
    {
        return (Deriv)(c1-c2);
    }

};




// ==========================================================================
// Specialization for strain defined using Voigt notation


template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real, char _name='E' >
class StrainTypes: public BaseStrainTypes<_spatial_dimensions,_material_dimensions * (1+_material_dimensions) / 2,_order,_Real,_name>
{
public:
    typedef BaseStrainTypes<_spatial_dimensions,_material_dimensions * (1+_material_dimensions) / 2,_order,_Real,_name> Inherit;

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
    typedef helper::vector<MaterialCoord> VecMaterialCoord;
    typedef Mat<material_dimensions,material_dimensions,Real> StrainMat;    ///< Strain in matrix form
};

typedef StrainTypes<3, 3, 0, double> E331dTypes;
typedef StrainTypes<3, 3, 0, float>  E331fTypes;
typedef StrainTypes<3, 2, 0, double> E321dTypes;
typedef StrainTypes<3, 2, 0, float>  E321fTypes;
typedef StrainTypes<3, 1, 0, double> E311dTypes;
typedef StrainTypes<3, 1, 0, float>  E311fTypes;
typedef StrainTypes<3, 3, 1, double> E332dTypes;
typedef StrainTypes<3, 3, 1, float>  E332fTypes;
typedef StrainTypes<3, 3, 2, double> E333dTypes;
typedef StrainTypes<3, 3, 2, float>  E333fTypes;
typedef StrainTypes<2, 2, 0, double> E221dTypes;
typedef StrainTypes<2, 2, 0, float>  E221fTypes;
typedef StrainTypes<3, 3, 0, SReal> E331Types;
typedef StrainTypes<3, 2, 0, SReal> E321Types;
typedef StrainTypes<3, 1, 0, SReal> E311Types;
typedef StrainTypes<3, 3, 1, SReal> E332Types;
typedef StrainTypes<3, 3, 2, SReal> E333Types;
typedef StrainTypes<2, 2, 0, SReal> E221Types;

template<> inline const char* E331dTypes::Name() { return "E331d"; }
template<> inline const char* E331fTypes::Name() { return "E331f"; }
template<> inline const char* E321dTypes::Name() { return "E321d"; }
template<> inline const char* E321fTypes::Name() { return "E321f"; }
template<> inline const char* E311dTypes::Name() { return "E311d"; }
template<> inline const char* E311fTypes::Name() { return "E311f"; }
template<> inline const char* E332dTypes::Name() { return "E332d"; }
template<> inline const char* E332fTypes::Name() { return "E332f"; }
template<> inline const char* E333dTypes::Name() { return "E333d"; }
template<> inline const char* E333fTypes::Name() { return "E333f"; }
template<> inline const char* E221dTypes::Name() { return "E221d"; }
template<> inline const char* E221fTypes::Name() { return "E221f"; }


template<> struct DataTypeInfo< E331fTypes::Deriv > : public FixedArrayTypeInfo< E331fTypes::Deriv, E331fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E331<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E331dTypes::Deriv > : public FixedArrayTypeInfo< E331dTypes::Deriv, E331dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E331<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E321fTypes::Deriv > : public FixedArrayTypeInfo< E321fTypes::Deriv, E321fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E321<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E321dTypes::Deriv > : public FixedArrayTypeInfo< E321dTypes::Deriv, E321dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E321<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E311fTypes::Deriv > : public FixedArrayTypeInfo< E311fTypes::Deriv, E311fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E311<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E311dTypes::Deriv > : public FixedArrayTypeInfo< E311dTypes::Deriv, E311dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E311<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E332fTypes::Deriv > : public FixedArrayTypeInfo< E332fTypes::Deriv, E332fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E332<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E332dTypes::Deriv > : public FixedArrayTypeInfo< E332dTypes::Deriv, E332dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E332<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E333fTypes::Deriv > : public FixedArrayTypeInfo< E333fTypes::Deriv, E333fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E333<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E333dTypes::Deriv > : public FixedArrayTypeInfo< E333dTypes::Deriv, E333dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E333<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E221fTypes::Deriv > : public FixedArrayTypeInfo< E221fTypes::Deriv, E221fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E221<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< E221dTypes::Deriv > : public FixedArrayTypeInfo< E221dTypes::Deriv, E221dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "E221<" << DataTypeName<double>::name() << ">"; return o.str(); } };

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::E331fTypes::Coord > { static const char* name() { return "E331fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E331dTypes::Coord > { static const char* name() { return "E331dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E321fTypes::Coord > { static const char* name() { return "E321fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E321dTypes::Coord > { static const char* name() { return "E321dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E311fTypes::Coord > { static const char* name() { return "E311fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E311dTypes::Coord > { static const char* name() { return "E311dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E332fTypes::Coord > { static const char* name() { return "E332fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E332dTypes::Coord > { static const char* name() { return "E332dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E333fTypes::Coord > { static const char* name() { return "E333fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E333dTypes::Coord > { static const char* name() { return "E333dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E221fTypes::Coord > { static const char* name() { return "E221fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::E221dTypes::Coord > { static const char* name() { return "E221dTypes::CoordOrDeriv"; } };

/// \endcond


//// ==========================================================================
//// Specialization for diagonalized strain (3 principal stretches + additional terms to store anisotropy)
//// It is basically the same as the regular principal stretches 'U' but stored as diagional tensor (like 'E' with null non-diagonal terms).
//// The stress tensor is not diagonal (like 'E')
//// @todo store Coord as 3 principal stretches rather than diagional tensor


///// \endcond

// ==========================================================================
// Specialization for Strain defined using Invariants of right Cauchy Green deformation tensor


template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real, char _name='I' >
class InvariantStrainTypes: public BaseStrainTypes<_spatial_dimensions,/*_material_dimensions * (1+_material_dimensions) / 2*/ 3,_order,_Real,_name>
{
public:
    static const char* Name();
};

typedef InvariantStrainTypes<3, 3, 0, double> I331dTypes;
typedef InvariantStrainTypes<3, 3, 0, float>  I331fTypes;
typedef InvariantStrainTypes<3, 3, 0, SReal>  I331Types;

template<> inline const char* I331dTypes::Name() { return "I331d"; }
template<> inline const char* I331fTypes::Name() { return "I331f"; }

template<> struct DataTypeInfo< I331fTypes::Deriv > : public FixedArrayTypeInfo< I331fTypes::Deriv, I331fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "I331<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< I331dTypes::Deriv > : public FixedArrayTypeInfo< I331dTypes::Deriv, I331dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "I331<" << DataTypeName<double>::name() << ">"; return o.str(); } };


// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::I331fTypes::Coord > { static const char* name() { return "I331fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::I331dTypes::Coord > { static const char* name() { return "I331dTypes::CoordOrDeriv"; } };

/// \endcond




// ==========================================================================
// Specialization for Strain defined using the principal stretches


template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real, char _name='U' >
class PrincipalStretchesStrainTypes: public BaseStrainTypes<_spatial_dimensions,_material_dimensions,_order,_Real,_name>
{
public:
    static const char* Name();

    static const unsigned int material_dimensions = _material_dimensions; ///< Number of dimensions of the material space (=number of axes of the deformable gradient): 3 for a volume object, 2 for a surface, 1 for a line.
    typedef Vec<material_dimensions, _Real> MaterialCoord;
    typedef helper::vector<MaterialCoord> VecMaterialCoord;
    typedef Mat<material_dimensions,material_dimensions,_Real> StrainMat;    ///< Strain in matrix form
};

typedef PrincipalStretchesStrainTypes<3, 3, 0, double> U331dTypes;
typedef PrincipalStretchesStrainTypes<3, 3, 0, float>  U331fTypes;
typedef PrincipalStretchesStrainTypes<3, 2, 0, double> U321dTypes;
typedef PrincipalStretchesStrainTypes<3, 2, 0, float>  U321fTypes;
typedef PrincipalStretchesStrainTypes<3, 3, 0, SReal>  U331Types;
typedef PrincipalStretchesStrainTypes<3, 2, 0, SReal> U321Types;

template<> inline const char* U331dTypes::Name() { return "U331d"; }
template<> inline const char* U331fTypes::Name() { return "U331f"; }
template<> inline const char* U321dTypes::Name() { return "U321d"; }
template<> inline const char* U321fTypes::Name() { return "U321f"; }

template<> struct DataTypeInfo< U331fTypes::Deriv > : public FixedArrayTypeInfo< U331fTypes::Deriv, U331fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "U331<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< U331dTypes::Deriv > : public FixedArrayTypeInfo< U331dTypes::Deriv, U331dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "U331<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< U321fTypes::Deriv > : public FixedArrayTypeInfo< U321fTypes::Deriv, U321fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "U321<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< U321dTypes::Deriv > : public FixedArrayTypeInfo< U321dTypes::Deriv, U321dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "U321<" << DataTypeName<double>::name() << ">"; return o.str(); } };


// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::U331fTypes::Coord > { static const char* name() { return "U331fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::U331dTypes::Coord > { static const char* name() { return "U331dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::U321fTypes::Coord > { static const char* name() { return "U321fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::U321dTypes::Coord > { static const char* name() { return "U321dTypes::CoordOrDeriv"; } };

/// \endcond



// ==========================================================================
// Helpers

/// Convert a symmetric strain matrix to voigt notation  exx=Fxx, eyy=Fyy, ezz=Fzz, exy=(Fxy+Fyx) eyz=(Fyz+Fzy), ezx=(Fxz+Fzx)
template<int material_dimensions, int dim, typename Real>
static inline Vec<material_dimensions * (1+material_dimensions) / 2, Real> StrainMatToVoigt( const  Mat<dim,material_dimensions, Real>& f )
{
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    StrainVec s(NOINIT);
    unsigned int ei=0;
    for(unsigned int j=0; j<material_dimensions; j++)
        for( unsigned int k=0; k<material_dimensions-j; k++ )
            if(0!=j) s[ei++] = (f[k][k+j]+f[k+j][k]); // shear *2
            else s[ei++] = f[k][k]; // stretch
    return s;
}


/// Convert a diagonal matrix to principal stretches  exx=Fxx etc.
template<int material_dimensions, typename Real>
static inline Vec<material_dimensions, Real> MatToPrincipalStretches( const  Mat<material_dimensions,material_dimensions, Real>& f )
{
    assert( f.isDiagonal() );
    Vec<material_dimensions, Real> s(NOINIT);
    for(unsigned int i=0; i<material_dimensions; i++)
        s[i] = f[i][i];
    return s;
}


/// Voigt notation to symmetric matrix Fxx=sxx, Fxy=Fyx=sxy/2, etc. for strain tensors
template<typename Real>
static inline Mat<3,3, Real> StrainVoigtToMat( const Vec<6, Real>& s )
{
    static const unsigned int material_dimensions=3;
    Mat<material_dimensions,material_dimensions, Real> f(NOINIT);

    f[0][0] = s[0];
    f[1][1] = s[1];
    f[2][2] = s[2];
    f[0][1] = f[1][0] = s[3] * 0.5;
    f[1][2] = f[2][1] = s[4] * 0.5;
    f[0][2] = f[2][0] = s[5] * 0.5;

    return f;
}

/// Voigt notation to symmetric matrix Fxx=sxx, Fxy=Fyx=sxy/2, etc. for strain tensors
template<typename Real>
static inline Mat<2,2, Real> StrainVoigtToMat( const Vec<3, Real>& s )
{
    static const unsigned int material_dimensions=2;
    Mat<material_dimensions,material_dimensions, Real> f(NOINIT);

    f[0][0] = s[0];
    f[1][1] = s[1];
    f[0][1] = f[1][0] = s[2] * 0.5;

    return f;
}


/// Voigt notation to symmetric matrix Fxx=sxx, Fxy=Fyx=sxy/2, etc. for strain tensors
template<typename Real>
static inline Mat<1,1, Real> StrainVoigtToMat( const Vec<1, Real>& s )
{
    static const unsigned int material_dimensions=1;
    Mat<material_dimensions,material_dimensions, Real> f(NOINIT);
    f[0][0] = s[0];
    return f;
}


/// PrincipalStretches to diagonal matrix Fxx=sxx etc.
template<int material_dimensions,typename Real>
static inline Mat<material_dimensions,material_dimensions, Real> PrincipalStretchesToMat( const Vec<material_dimensions, Real>& s )
{
    Mat<material_dimensions,material_dimensions, Real> f(NOINIT);

    for( int i=0 ; i<material_dimensions ; ++i )
    {
        f[i][i] = s[i];
        for( int j=i+1 ; j<material_dimensions ; ++j )
        {
            f[i][j] = f[j][i] = 0;
        }
    }

    return f;
}



/// Convert a symmetric stress matrix to voigt notation  exx=Fxx, eyy=Fyy, ezz=Fzz, exy=(Fxy+Fyx)/2 eyz=(Fyz+Fzy)/2, ezx=(Fxz+Fzx)/2
template<int material_dimensions, typename Real>
static inline Vec<material_dimensions * (1+material_dimensions) / 2, Real> StressMatToVoigt( const  Mat<material_dimensions,material_dimensions, Real>& f )
{
    static const unsigned int strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    StrainVec s(NOINIT);
    unsigned int ei=0;
    for(unsigned int j=0; j<material_dimensions; j++)
        for( unsigned int k=0; k<material_dimensions-j; k++ )
            s[ei++] = (f[k][k+j]+f[k+j][k])*(Real)0.5;
    return s;
}


/// Voigt notation to symmetric matrix Fxx=sxx, Fxy=Fyx=sxy, etc. for stress tensors
template<typename Real>
static inline Mat<3,3, Real> StressVoigtToMat( const Vec<6, Real>& s )
{
    static const unsigned int material_dimensions=3;
    Mat<material_dimensions,material_dimensions, Real> f(NOINIT);

    f[0][0] = s[0];
    f[1][1] = s[1];
    f[2][2] = s[2];
    f[0][1] = f[1][0] = s[3];
    f[1][2] = f[2][1] = s[4];
    f[0][2] = f[2][0] = s[5];
    return f;
}

/// Voigt notation to symmetric matrix Fxx=sxx, Fxy=Fyx=sxy, etc. for stress tensors
template<typename Real>
static inline Mat<2,2, Real> StressVoigtToMat( const Vec<3, Real>& s )
{
    static const unsigned int material_dimensions=2;
    Mat<material_dimensions,material_dimensions, Real> f(NOINIT);
    /*unsigned int ei=0;
    for(unsigned int j=0; j<material_dimensions; j++){
        for( unsigned int k=0; k<material_dimensions-j; k++ )
            f[k][k+j] = f[k+j][k] = s[ei++] ;
    }*/
    f[0][0] = s[0];
    f[1][1] = s[1];
    f[0][1] = f[1][0] = s[2];
    return f;
}

/// Voigt notation to symmetric matrix Fxx=sxx, Fxy=Fyx=sxy, etc. for stress tensors
template<typename Real>
static inline Mat<1,1, Real> StressVoigtToMat( const Vec<1, Real>& s )
{
    static const unsigned int material_dimensions=1;
    Mat<material_dimensions,material_dimensions, Real> f(NOINIT);
    f[0][0] = s[0];
    return f;
}



// TODO: ADD  Mat*VoigtVec operators


/// \return 0.5 * ( A + At )
template<int N, class Real>
static defaulttype::Mat<N,N,Real> cauchyStrainTensor( const defaulttype::Mat<N,N,Real>& A )
{
    defaulttype::Mat<N,N,Real> B;
    for( int i=0 ; i<N ; i++ )
    {
        B[i][i] = A[i][i];
        for( int j=i+1 ; j<N ; j++ )
            B[i][j] = B[j][i] = (Real)0.5 * ( A[i][j] + A[j][i] );
    }
    return B;
}






} // namespace defaulttype




// ==========================================================================
// Mechanical Object

namespace component
{

namespace container
{

#if  !defined(FLEXIBLE_StrainTYPES_CPP)
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E331Types>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E332Types>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E333Types>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E321Types>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E311Types>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::E221Types>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::I331Types>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::U331Types>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::U321Types>;
#endif

} // namespace container

} // namespace component



} // namespace sofa


#endif
