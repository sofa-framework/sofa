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
    static const unsigned order = _order;  ///< 0: only a point, no gradient 1:deformation gradient, 2: deformation gradient and its gradient
    static const unsigned NumMatrices = order==0? 0 : (order==1? 1 : (order==2? 1 + material_dimensions : -1 ));
    static const unsigned VSize = spatial_dimensions +  NumMatrices * spatial_dimensions * spatial_dimensions;  // number of entries
    typedef _Real Real;
    typedef vector<Real> VecReal;
    typedef Vec<VSize,Real> Coord;  ///< position and deformation gradient
    typedef vector<Coord> VecCoord;
    typedef Coord Deriv ;            ///< velocity and deformation gradient rate
    typedef vector<Deriv> VecDeriv;
    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    static const char* Name();


    // ------------    Types and methods defined for easier data access
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position on velocity of a point
    typedef Mat<spatial_dimensions,spatial_dimensions, Real> MaterialFrame;      ///< Matrix representing a deformation gradient
    typedef Vec<spatial_dimensions, MaterialFrame> MaterialFrameGradient;                 ///< Gradient of a deformation gradient

    /// position or velocity of the sampling point
    template<class CoordOrDeriv>
    static SpatialCoord& center( CoordOrDeriv& v ) { return *reinterpret_cast<SpatialCoord*>(&v[0]); } ///< position or velocity of the sampling point
    /// position or velocity of the sampling point
    template<class CoordOrDeriv>
    static const SpatialCoord& center( const CoordOrDeriv& v ) { return *reinterpret_cast<const SpatialCoord*>(&v[0]); } ///< position or velocity of the sampling point


    /// deformation gradient at the sampling point
    template<class CoordOrDeriv>
    static MaterialFrame& materialFrame( CoordOrDeriv& v ) { BOOST_STATIC_ASSERT(order>=1); return *reinterpret_cast<MaterialFrame*>(&v[spatial_dimensions]); }
    /// deformation gradient at the sampling point
    template<class CoordOrDeriv>
    static const MaterialFrame& materialFrame( const CoordOrDeriv& v ) {  BOOST_STATIC_ASSERT(order>=1); return *reinterpret_cast<const MaterialFrame*>(&v[spatial_dimensions]); }

    /// gradient of the deformation gradient at the sampling point
    template<class CoordOrDeriv>
    static MaterialFrameGradient& materialFrameGradient( CoordOrDeriv& v ) { BOOST_STATIC_ASSERT(order>=2); return *reinterpret_cast<MaterialFrame*>(&v[spatial_dimensions]); }
    /// gradient of the deformation gradient at the sampling point
    template<class CoordOrDeriv>
    static const MaterialFrameGradient& materialFrameGradient( const CoordOrDeriv& v ) { BOOST_STATIC_ASSERT(order>=2); return *reinterpret_cast<const MaterialFrame*>(&v[spatial_dimensions]); }
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

//        template<> struct DataTypeName< defaulttype::DeformationGradient331fTypes::Coord > { static const char* name() { return "DeformationGradient331fTypes::CoordOrDeriv"; } };

//        template<> struct DataTypeName< defaulttype::DeformationGradient331fTypes::Deriv > { static const char* name() { return "DeformationGradient331fTypes::Deriv"; } };

//        template<> struct DataTypeName< defaulttype::DeformationGradient331dTypes::Coord > { static const char* name() { return "DeformationGradient331dTypes::CoordOrDeriv"; } };

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

//                template<> struct DataTypeName< defaulttype::DeformationGradient332fTypes::Deriv > { static const char* name() { return "DeformationGradient332fTypes::Deriv"; } };

template<> struct DataTypeName< defaulttype::DeformationGradient332dTypes::Coord > { static const char* name() { return "DeformationGradient332dTypes::CoordOrDeriv"; } };

//                template<> struct DataTypeName< defaulttype::DeformationGradient332dTypes::Deriv > { static const char* name() { return "DeformationGradient332dTypes::Deriv"; } };


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
    defaulttype::Vec3d origin = defaulttype::DeformationGradient332dTypes::center(x), finertia;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + defaulttype::DeformationGradient332dTypes::center(v) * 2 ) ) * mass.mass;
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
    defaulttype::Vec3f omega ( vframe.lineVec[0], vframe.lineVec[1], vframe.lineVec[2] );
    defaulttype::Vec3f origin = defaulttype::DeformationGradient332fTypes::center(x), finertia;

    finertia = - ( aframe + omega.cross ( omega.cross ( origin ) + defaulttype::DeformationGradient332dTypes::center(v) * 2 ) ) * mass.mass;
    defaulttype::DeformationGradient332fTypes::Deriv result;
    result[0]=finertia[0]; result[1]=finertia[1]; result[2]=finertia[2];
    return result;
}


} // namespace behavoir

} // namespace core

} // namespace sofa


#endif
