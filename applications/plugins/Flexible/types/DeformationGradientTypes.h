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
#ifndef FLEXIBLE_DeformationGradientTYPES_H
#define FLEXIBLE_DeformationGradientTYPES_H

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

/** Local deformation state of a material object.
Template parameters are used to define the spatial dimensions, the material dimensions, and the order.
Order 1 corresponds to a traditional deformation gradient, while order 2 corresponds to an elaston.
In the names of the instanciated classes, the suffix corresponds to the parameters.
For instance, F332d moves in 3 spatial dimensions, is attached to a  volumetric object (3 dimensions), represents the deformation using an elaston (order=2), and encodes floating point numbers at double precision.
*/
template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
struct DefGradientTypes
{
    static const unsigned int spatial_dimensions = _spatial_dimensions;   ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int material_dimensions = _material_dimensions; ///< Number of dimensions of the material space (=number of axes of the deformable gradient): 3 for a volume object, 2 for a surface, 1 for a line.
    typedef _Real Real;

    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Mat<spatial_dimensions,material_dimensions, Real> Frame;      ///< Matrix representing a deformation gradient
    typedef PolynomialBasis<spatial_dimensions*material_dimensions, Real, spatial_dimensions, _order> Basis;    ///< deformation gradient, expressed on a certain basis
    typedef typename Basis::TotalVec BasisVec;                            ///< decomposed deformation gradient in a single vector

    enum { VSize = Basis::total_size };  ///< number of entries: point + decomposed deformation gradient
    enum { coord_total_size = VSize };
    enum { deriv_total_size = VSize };

    typedef helper::vector<Real> VecReal;

    /** Time derivative of a (generalized) deformation gradient, or other vector-like associated quantities, such as generalized forces.
    */
    class Deriv
    {
    protected:
        Basis b;

    public:
        Deriv() { b.clear(); }
        Deriv( const Basis& d):b(d) {}
        void clear() {  b.clear();}

        static const unsigned int total_size = VSize;
        typedef Real value_type;
        typedef int size_type;

        static unsigned int size() { return VSize; }

        /// seen as a vector
        Real* ptr() { return b.ptr(); }
        const Real* ptr() const { return b.ptr(); }

        BasisVec& getVec() { return b.getVec(); }
        const BasisVec& getVec() const { return b.getVec(); }

        Real& operator[](int i) { return getVec()[i]; }
        const Real& operator[](int i) const    { return getVec()[i]; }

        /// basis
        Basis& getBasis() { return b; }
        const Basis& getBasis() const { return b; }

        Frame& getF() { return *reinterpret_cast<Frame*>(&b.getVal()); }
        const Frame& getF() const { return *reinterpret_cast<const Frame*>(&b.getVal()); }

        Frame& getGradientF(int i) { return *reinterpret_cast<Frame*>(&b.getGradient()[i]); }
        const Frame& getGradientF(int i) const { return *reinterpret_cast<const Frame*>(&b.getGradient()[i]); }

        Frame& getHessianF(int i,int j) { return *reinterpret_cast<Frame*>(&b.getHessian()(i,j)); }
        const Frame& getHessianF(int i,int j) const { return *reinterpret_cast<const Frame*>(&b.getHessian()(i,j)); }

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
        Real operator*(const Deriv& a) const    { return getVec()*a.getVec(); }

        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Deriv& c )
        {
            out<<" "<<c.getVec();
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

//    void clear(){ v.clear(); b.clear(); for( unsigned int i = 0; i < material_dimensions; ++i) getF()[i][i] = (Real)1.0;}

    typedef helper::vector<Coord> VecCoord;

    static const char* Name();

    /** @name Conversions
              * Convert to/from points in space
             */
    //@{
    template<typename T>
    static void set ( Deriv& /*c*/, T /*x*/, T /*y*/, T /*z*/ )  {    }
    template<typename T>
    static void get ( T& x, T& y, T& z, const Deriv& /*c*/ ) { x=y=z=0; /*std::cerr<<"WARNING: DefGradientTypes::get(): a deformation gradient cannot be converted to spatial coordinates.\n"; */ }
    template<typename T>
    static void add ( Deriv& /*c*/, T /*x*/, T /*y*/, T /*z*/ )    {    }
    //@}

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



#ifndef SOFA_FLOAT
typedef DefGradientTypes<3, 3, 0, double> F331dTypes;
typedef DefGradientTypes<3, 3, 1, double> F332dTypes;
typedef DefGradientTypes<3, 2, 0, double> F321dTypes; // for planar deformations
typedef DefGradientTypes<3, 1, 0, double> F311dTypes; // for linear deformations
typedef DefGradientTypes<2, 2, 0, double> F221dTypes; // 2d planar deformations
template<> struct DataTypeInfo< F331dTypes::Deriv > : public FixedArrayTypeInfo< F331dTypes::Deriv, F331dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F331<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F332dTypes::Deriv > : public FixedArrayTypeInfo< F332dTypes::Deriv, F332dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F332<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F321dTypes::Deriv > : public FixedArrayTypeInfo< F321dTypes::Deriv, F321dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F321<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F311dTypes::Deriv > : public FixedArrayTypeInfo< F311dTypes::Deriv, F311dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F311<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F221dTypes::Deriv > : public FixedArrayTypeInfo< F221dTypes::Deriv, F221dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F221<" << DataTypeName<double>::name() << ">"; return o.str(); } };
#endif
#ifndef SOFA_DOUBLE
typedef DefGradientTypes<3, 3, 0, float>  F331fTypes;
typedef DefGradientTypes<3, 3, 1, float>  F332fTypes;
typedef DefGradientTypes<3, 2, 0, float>  F321fTypes; // for planar deformations
typedef DefGradientTypes<3, 1, 0, float>  F311fTypes; // for linear deformations
typedef DefGradientTypes<2, 2, 0, float>  F221fTypes; // 2d planar deformations
template<> struct DataTypeInfo< F331fTypes::Deriv > : public FixedArrayTypeInfo< F331fTypes::Deriv, F331fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F331<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F332fTypes::Deriv > : public FixedArrayTypeInfo< F332fTypes::Deriv, F332fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F332<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F321fTypes::Deriv > : public FixedArrayTypeInfo< F321fTypes::Deriv, F321fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F321<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F311fTypes::Deriv > : public FixedArrayTypeInfo< F311fTypes::Deriv, F311fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F311<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F221fTypes::Deriv > : public FixedArrayTypeInfo< F221fTypes::Deriv, F221fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F221<" << DataTypeName<float>::name() << ">"; return o.str(); } };
#endif


#ifdef SOFA_FLOAT
template<> inline const char* F331fTypes::Name() { return "F331"; }
template<> inline const char* F332fTypes::Name() { return "F332"; }
template<> inline const char* F321fTypes::Name() { return "F321"; }
template<> inline const char* F311fTypes::Name() { return "F311"; }
template<> inline const char* F221fTypes::Name() { return "F221"; }
#else
template<> inline const char* F331dTypes::Name() { return "F331"; }
template<> inline const char* F332dTypes::Name() { return "F332"; }
template<> inline const char* F321dTypes::Name() { return "F321"; }
template<> inline const char* F311dTypes::Name() { return "F311"; }
template<> inline const char* F221dTypes::Name() { return "F221"; }
#ifndef SOFA_DOUBLE
template<> inline const char* F331fTypes::Name() { return "F331f"; }
template<> inline const char* F332fTypes::Name() { return "F332f"; }
template<> inline const char* F321fTypes::Name() { return "F321f"; }
template<> inline const char* F311fTypes::Name() { return "F311f"; }
template<> inline const char* F221fTypes::Name() { return "F221f"; }
#endif
#endif

#ifdef SOFA_FLOAT
typedef F331fTypes F331Types;
typedef F332fTypes F332Types;
typedef F321fTypes F321Types;
typedef F311fTypes F311Types;
typedef F221fTypes F221Types;
#else
typedef F331dTypes F331Types;
typedef F332dTypes F332Types;
typedef F321dTypes F321Types;
typedef F311dTypes F311Types;
typedef F221dTypes F221Types;
#endif



// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES


#ifndef SOFA_FLOAT
template<> struct DataTypeName< defaulttype::F331dTypes::Coord > { static const char* name() { return "F331dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F332dTypes::Coord > { static const char* name() { return "F332dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F321dTypes::Coord > { static const char* name() { return "F321dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F311dTypes::Coord > { static const char* name() { return "F311dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F221dTypes::Coord > { static const char* name() { return "F221dTypes::CoordOrDeriv"; } };
#endif
#ifndef SOFA_DOUBLE
template<> struct DataTypeName< defaulttype::F331fTypes::Coord > { static const char* name() { return "F331fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F332fTypes::Coord > { static const char* name() { return "F332fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F321fTypes::Coord > { static const char* name() { return "F321fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F311fTypes::Coord > { static const char* name() { return "F311fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F221fTypes::Coord > { static const char* name() { return "F221fTypes::CoordOrDeriv"; } };
#endif




/// \endcond


} // namespace defaulttype




// ==========================================================================
// Mechanical Object

namespace component
{

namespace container
{

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_DeformationGradientTYPES_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F331dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F331dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F332dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F332dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F321dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F321dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F311dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F311dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F221dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F221dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F331fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F331fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F332fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F332fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F321fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F321fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F311fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F311fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F221fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F221fTypes>;
#endif
#endif

} // namespace container

} // namespace component




} // namespace sofa


#endif
