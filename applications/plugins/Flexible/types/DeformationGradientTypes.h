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
#ifndef FLEXIBLE_DeformationGradientTYPES_H
#define FLEXIBLE_DeformationGradientTYPES_H

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
    static const unsigned int order = _order;  ///< 0: only a point, no gradient 1:deformation gradient, 2: deformation gradient and its gradient (=elaston)
    static const unsigned int NumMatrices = order==0? 0 : (order==1? 1 : (order==2? 1 + spatial_dimensions : -1 ));
    static const unsigned int VSize = spatial_dimensions +  NumMatrices * spatial_dimensions * material_dimensions;  // number of entries
    enum { coord_total_size = VSize };
    enum { deriv_total_size = VSize };
    typedef _Real Real;
    typedef vector<Real> VecReal;

    // ------------    Types and methods defined for easier data access
    typedef Vec<material_dimensions, Real> MaterialCoord;
    typedef vector<MaterialCoord> VecMaterialCoord;
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Mat<spatial_dimensions,material_dimensions, Real> MaterialFrame;      ///< Matrix representing a deformation gradient
    typedef Vec<spatial_dimensions, MaterialFrame> MaterialFrameGradient;                 ///< Gradient of a deformation gradient (for order 2)

    /** Time derivative of a (generalized) deformation gradient, or other vector-like associated quantities, such as generalized forces.
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

        /// point
        SpatialCoord& getCenter() { return *reinterpret_cast<SpatialCoord*>(&v[0]); }
        const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

        /// local frame (if order>=1)
        MaterialFrame& getMaterialFrame() { return *reinterpret_cast<MaterialFrame*>(&v[spatial_dimensions]); }
        const MaterialFrame& getMaterialFrame() const { return *reinterpret_cast<const MaterialFrame*>(&v[spatial_dimensions]); }

        /// gradient of the local frame (if order>=2)
        MaterialFrameGradient& getMaterialFrameGradient() { return *reinterpret_cast<MaterialFrameGradient*>(&v[spatial_dimensions+spatial_dimensions * material_dimensions]); }
        const MaterialFrameGradient& getMaterialFrameGradient() const { return *reinterpret_cast<const MaterialFrameGradient*>(&v[spatial_dimensions+spatial_dimensions * material_dimensions]); }

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

    /** Deformation gradient , or other vector-like associated quantities, such as generalized force
    */
    class Coord
    {
    protected:
        Vec<VSize,Real> v;

    public:
        Coord() { v.clear(); }
        Coord( const Vec<VSize,Real>& d):v(d) {}
        void clear() { v.clear(); for( unsigned int i = 0; i < spatial_dimensions; ++i) getMaterialFrame()[i][i] = (Real)1.0;}

        static const unsigned int total_size = VSize;
        typedef Real value_type;

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

        /// local frame (if order>=1)
        MaterialFrame& getMaterialFrame() { return *reinterpret_cast<MaterialFrame*>(&v[spatial_dimensions]); }
        const MaterialFrame& getMaterialFrame() const { return *reinterpret_cast<const MaterialFrame*>(&v[spatial_dimensions]); }

        /// gradient of the local frame (if order>=2)
        MaterialFrameGradient& getMaterialFrameGradient() { return *reinterpret_cast<MaterialFrameGradient*>(&v[spatial_dimensions+spatial_dimensions * material_dimensions]); }
        const MaterialFrameGradient& getMaterialFrameGradient() const { return *reinterpret_cast<const MaterialFrameGradient*>(&v[spatial_dimensions+spatial_dimensions * material_dimensions]); }

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
            m[0] = (float)getMaterialFrame()(0,0);
            m[4] = (float)getMaterialFrame()(0,1);
            m[8] = (float)getMaterialFrame()(0,2);
            m[1] = (float)getMaterialFrame()(1,0);
            m[5] = (float)getMaterialFrame()(1,1);
            m[9] = (float)getMaterialFrame()(1,2);
            m[2] = (float)getMaterialFrame()(2,0);
            m[6] = (float)getMaterialFrame()(2,1);
            m[10] = (float)getMaterialFrame()(2,2);
            m[3] = 0;
            m[7] = 0;
            m[11] = 0;
            m[12] = ( float ) getCenter()[0];
            m[13] = ( float ) getCenter()[1];
            m[14] = ( float ) getCenter()[2];
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
    static void set ( Deriv& c, T x, T y, T z )
    {
        c.clear();
        c.getCenter()[0] = ( Real ) x;
        c.getCenter() [1] = ( Real ) y;
        c.getCenter() [2] = ( Real ) z;
    }

    template<typename T>
    static void get ( T& x, T& y, T& z, const Deriv& c )
    {
        x = ( T ) c.getCenter() [0];
        y = ( T ) c.getCenter() [1];
        z = ( T ) c.getCenter() [2];
    }

    template<typename T>
    static void add ( Deriv& c, T x, T y, T z )
    {
        c.getCenter() [0] += ( Real ) x;
        c.getCenter() [1] += ( Real ) y;
        c.getCenter() [2] += ( Real ) z;
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
        c.getCenter() [0] += ( Real ) x;
        c.getCenter() [1] += ( Real ) y;
        c.getCenter() [2] += ( Real ) z;
    }
    //@}

};


// ==========================================================================


typedef DefGradientTypes<3, 3, 1, double> F331dTypes;
typedef DefGradientTypes<3, 3, 1, float>  F331fTypes;
typedef DefGradientTypes<3, 3, 2, double> F332dTypes;
typedef DefGradientTypes<3, 3, 2, float>  F332fTypes;


#ifdef SOFA_FLOAT
template<> inline const char* F331dTypes::Name() { return "F331d"; }
template<> inline const char* F331fTypes::Name() { return "F331"; }
template<> inline const char* F332dTypes::Name() { return "F332d"; }
template<> inline const char* F332fTypes::Name() { return "F332"; }
#else
template<> inline const char* F331dTypes::Name() { return "F331"; }
template<> inline const char* F331fTypes::Name() { return "F331f"; }
template<> inline const char* F332dTypes::Name() { return "F332"; }
template<> inline const char* F332fTypes::Name() { return "F332f"; }
#endif

#ifdef SOFA_FLOAT
typedef F331fTypes F331Types;
typedef F332fTypes F332Types;
#else
typedef F331dTypes F331Types;
typedef F332dTypes F332Types;
#endif

template<> struct DataTypeInfo< F331fTypes::Deriv > : public FixedArrayTypeInfo< F331fTypes::Deriv, F331fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F331<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F331dTypes::Deriv > : public FixedArrayTypeInfo< F331dTypes::Deriv, F331dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F331<" << DataTypeName<double>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F332fTypes::Deriv > : public FixedArrayTypeInfo< F332fTypes::Deriv, F332fTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F332<" << DataTypeName<float>::name() << ">"; return o.str(); } };
template<> struct DataTypeInfo< F332dTypes::Deriv > : public FixedArrayTypeInfo< F332dTypes::Deriv, F332dTypes::Deriv::total_size > {    static std::string name() { std::ostringstream o; o << "F332<" << DataTypeName<double>::name() << ">"; return o.str(); } };

// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::F331fTypes::Coord > { static const char* name() { return "F331fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F331dTypes::Coord > { static const char* name() { return "F331dTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F332fTypes::Coord > { static const char* name() { return "F332fTypes::CoordOrDeriv"; } };
template<> struct DataTypeName< defaulttype::F332dTypes::Coord > { static const char* name() { return "F332dTypes::CoordOrDeriv"; } };

/// \endcond


} // namespace defaulttype




// ==========================================================================
// Mechanical Object

namespace component
{

namespace container
{

#if defined(WIN32) && !defined(FLEXIBLE_DeformationGradientTYPES_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F331dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F331dTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F332dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F332dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F331fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F331fTypes>;
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::F332fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::F332fTypes>;
#endif
#endif

} // namespace container

} // namespace component




} // namespace sofa


#endif
