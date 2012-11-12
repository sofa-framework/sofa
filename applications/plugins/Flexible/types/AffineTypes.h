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
#ifndef FLEXIBLE_AffineTYPES_H
#define FLEXIBLE_AffineTYPES_H

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

#include <sofa/component/mass/AddMToMatrixFunctor.h>
#include <sofa/component/mass/UniformMass.h>

namespace sofa
{

namespace defaulttype
{

using std::endl;
using helper::vector;

/** DOF types associated with deformable frames. Each deformable frame generates an affine displacement field, with 12 independent degrees of freedom.
 */
template<int _spatial_dimensions, typename _Real>
class StdAffineTypes
{
public:
    static const unsigned int spatial_dimensions = _spatial_dimensions;  ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int VSize = spatial_dimensions +  spatial_dimensions * spatial_dimensions;  // number of entries
    enum { coord_total_size = VSize };
    enum { deriv_total_size = VSize };
    typedef _Real Real;
    typedef vector<Real> VecReal;

    // ------------    Types and methods defined for easier data access
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Mat<spatial_dimensions,spatial_dimensions, Real> Frame;       ///< Matrix representing a frame

    class Deriv : public Vec<VSize,Real>
    {
        typedef Vec<VSize,Real> MyVec;
    public:
        Deriv() { MyVec::clear(); }
        Deriv( const Vec<VSize,Real>& d):MyVec(d) {}
        Deriv( const SpatialCoord& c, const Frame& a) { getVCenter()=c; getVAffine()=a;}

        //static const unsigned int total_size = VSize;
        typedef Real value_type;

        /// point
        SpatialCoord& getVCenter() { return *reinterpret_cast<SpatialCoord*>(&this->elems[0]); }
        const SpatialCoord& getVCenter() const { return *reinterpret_cast<const SpatialCoord*>(&this->elems[0]); }

        /// local frame
        Frame& getVAffine() { return *reinterpret_cast<Frame*>(&this->elems[spatial_dimensions]); }
        const Frame& getVAffine() const { return *reinterpret_cast<const Frame*>(&this->elems[spatial_dimensions]); }

        /// project to a rigid motion
        void setRigid()
        {
            Frame& a = getVAffine();
            // make skew-symmetric
            for(unsigned i=0; i<spatial_dimensions; i++) a[i][i] = 0.0;
            for(unsigned i=0; i<spatial_dimensions; i++)
            {
                for(unsigned j=i+1; j<spatial_dimensions; j++)
                {
                    a[i][j] = (a[i][j] - a[j][i]) *(Real)0.5;
                    a[j][i] = - a[i][j];
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



    class Coord : public Vec<VSize,Real>
    {
        typedef Vec<VSize,Real> MyVec;

    public:
        Coord() { clear(); }
        Coord( const Vec<VSize,Real>& d):MyVec(d) {}
        Coord( const SpatialCoord& c, const Frame& a) { getCenter()=c; getAffine()=a;}
        void clear()  { MyVec::clear(); for(unsigned int i=0; i<spatial_dimensions; ++i) getAffine()[i][i]=(Real)1.0; } // init affine part to identity

        //static const unsigned int total_size = VSize;
        typedef Real value_type;

        /// point
        SpatialCoord& getCenter() { return *reinterpret_cast<SpatialCoord*>(&this->elems[0]); }
        const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&this->elems[0]); }

        /// local frame
        Frame& getAffine() { return *reinterpret_cast<Frame*>(&this->elems[spatial_dimensions]); }
        const Frame& getAffine() const { return *reinterpret_cast<const Frame*>(&this->elems[spatial_dimensions]); }


        /// write to an output stream
        inline friend std::ostream& operator << ( std::ostream& out, const Coord& c )
        {
            out<<c.getCenter()<<" "<<c.getAffine();
            return out;
        }
        /// read from an input stream
        inline friend std::istream& operator >> ( std::istream& in, Coord& c )
        {
            in>>c.getCenter()>>c.getAffine();
            return in;
        }

        /// Write the OpenGL transformation matrix
        void writeOpenGlMatrix ( float m[16] ) const
        {
            BOOST_STATIC_ASSERT(spatial_dimensions == 3);
            m[0] = (float)getAffine()(0,0);
            m[4] = (float)getAffine()(0,1);
            m[8] = (float)getAffine()(0,2);
            m[1] = (float)getAffine()(1,0);
            m[5] = (float)getAffine()(1,1);
            m[9] = (float)getAffine()(1,2);
            m[2] = (float)getAffine()(2,0);
            m[6] = (float)getAffine()(2,1);
            m[10] = (float)getAffine()(2,2);
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
            return getAffine()*v + getCenter();
        }

        /// Project a point from the parent frame to the child frame
        SpatialCoord pointToChild( const SpatialCoord& v ) const
        {
            Frame affineInv;
            affineInv.invert( getAffine() );
            return affineInv * (v-getCenter());
        }


        /// project to a rigid displacement
        void setRigid()
        {
            Frame rotation;
            polarDecomposition( getAffine(), rotation );
            getAffine() = rotation;
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

    static Coord inverse( const Coord& c )
    {
        Frame m;
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


typedef StdAffineTypes<3, double> Affine3dTypes;
typedef StdAffineTypes<3, float> Affine3fTypes;


/// Note: Many scenes use Affine as template for 3D double-precision rigid type. Changing it to Affine3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* Affine3dTypes::Name() { return "Affine3d"; }
template<> inline const char* Affine3fTypes::Name() { return "Affine"; }
#else
template<> inline const char* Affine3dTypes::Name() { return "Affine"; }
template<> inline const char* Affine3fTypes::Name() { return "Affine3f"; }
#endif

#ifdef SOFA_FLOAT
typedef Affine3fTypes Affine3Types;
#else
typedef Affine3dTypes Affine3Types;
#endif
typedef Affine3Types AffineTypes;


// Specialization of the defaulttype::DataTypeInfo type traits template
template<> struct DataTypeInfo< sofa::defaulttype::Affine3fTypes::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::Affine3fTypes::Coord, sofa::defaulttype::Affine3fTypes::Coord::total_size >
{
    static std::string name() { std::ostringstream o; o << "AffineCoord<" << sofa::defaulttype::Affine3fTypes::Coord::total_size << "," << DataTypeName<sofa::defaulttype::Affine3fTypes::Real>::name() << ">"; return o.str(); }
};
template<> struct DataTypeInfo< sofa::defaulttype::Affine3fTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Affine3fTypes::Deriv, sofa::defaulttype::Affine3fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "AffineDeriv<" << sofa::defaulttype::Affine3fTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Affine3fTypes::Real>::name() << ">"; return o.str(); }
};

template<> struct DataTypeInfo< sofa::defaulttype::Affine3dTypes::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::Affine3dTypes::Coord, sofa::defaulttype::Affine3dTypes::Coord::total_size >
{
    static std::string name() { std::ostringstream o; o << "AffineCoord<" << sofa::defaulttype::Affine3dTypes::Coord::total_size << "," << DataTypeName<sofa::defaulttype::Affine3dTypes::Real>::name() << ">"; return o.str(); }
};
template<> struct DataTypeInfo< sofa::defaulttype::Affine3dTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Affine3dTypes::Deriv, sofa::defaulttype::Affine3dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "AffineDeriv<" << sofa::defaulttype::Affine3dTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Affine3dTypes::Real>::name() << ">"; return o.str(); }
};


// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Affine3fTypes::Coord > { static const char* name() { return "Affine3fTypes::Coord"; } };
template<> struct DataTypeName< defaulttype::Affine3dTypes::Coord > { static const char* name() { return "Affine3dTypes::Coord"; } };

/// \endcond


} // namespace defaulttype



// ==========================================================================
// Mechanical Object

namespace component
{
namespace container
{


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_AffineTYPES_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Affine3dTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Affine3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_Flexible_API MechanicalObjectInternalData<defaulttype::Affine3fTypes>;
extern template class SOFA_Flexible_API MechanicalObject<defaulttype::Affine3fTypes>;
#endif
#endif

} // namespace container
} // namespace component


// ====================================================================
// AffineMass


namespace defaulttype
{

using std::endl;
using helper::vector;

/** Mass associated with an affine deformable frame */
template<int _spatial_dimensions,typename _Real>
class AffineMass : public Mat<StdAffineTypes<_spatial_dimensions,_Real>::deriv_total_size, StdAffineTypes<_spatial_dimensions,_Real>::deriv_total_size, _Real>
{
public:
    typedef _Real Real;

    static const unsigned int spatial_dimensions = _spatial_dimensions;  ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int VSize = StdAffineTypes<spatial_dimensions,Real>::deriv_total_size;

    typedef Mat<VSize, VSize, Real> MassMatrix;


    AffineMass() : MassMatrix(), m_invMassMatrix(NULL)
    {
        MassMatrix::identity();
    }

    /// build a uniform, diagonal matrix
    AffineMass( Real m ) : MassMatrix(), m_invMassMatrix(NULL)
    {
        setValue( m );
    }


    ~AffineMass()
    {
        if( m_invMassMatrix )
        {
            delete m_invMassMatrix;
            m_invMassMatrix = NULL;
        }
    }

    /// make a null inertia matrix
    void clear()
    {
        MassMatrix::clear();
        if( m_invMassMatrix ) m_invMassMatrix->clear();
    }


    static const char* Name();

    /// @returns the invert of the mass matrix
    const MassMatrix& getInverse() const
    {
        // optimization: compute the mass invert only once (needed by explicit solvers)
        if( !m_invMassMatrix )
        {
            m_invMassMatrix = new MassMatrix;
            m_invMassMatrix->invert( *this );
        }
        return *m_invMassMatrix;
    }

    /// set a uniform, diagonal matrix
    virtual void setValue( Real m )
    {
        for( unsigned i=0 ; i<VSize ; ++i ) (*this)(i,i) = m;
        updateInverse();
    }


    /// copy
    void operator= ( const MassMatrix& m )
    {
        *((MassMatrix*)this) = m;
        updateInverse();
    }

    /// this += m
    void operator+= ( const MassMatrix& m )
    {
        *((MassMatrix*)this) += m;
        updateInverse();
    }

    /// this -= m
    void operator-= ( const MassMatrix& m )
    {
        *((MassMatrix*)this) -= m;
        updateInverse();
    }

    /// this *= m
    void operator*= ( const MassMatrix& m )
    {
        *((MassMatrix*)this) *= m;
        updateInverse();
    }

    /// apply a factor to the mass matrix
    void operator*= ( Real m )
    {
        *((MassMatrix*)this) *= m;
        updateInverse();
    }

    /// apply a factor to the mass matrix
    void operator/= ( Real m )
    {
        *((MassMatrix*)this) /= m;
        updateInverse();
    }



    /// operator to cast to const Real, supposing the mass is uniform (and so diagonal)
    operator const Real() const
    {
        return (*this)(0,0);
    }


    /// @todo overload these functions so they can be able to update 'm_invMassMatrix' after modifying 'this'
    //void fill(real r)
    // transpose
    // transpose(m)
    // addtranspose
   //subtranspose



protected:

    mutable MassMatrix *m_invMassMatrix; ///< a pointer to the inverse of the mass matrix

    /// when the mass matrix is changed, if the inverse exists, it has to be updated
    /// @warning there are certainly cases (non overloaded functions or non virtual) that modify 'this' without updating 'm_invMassMatrix'. Another solution = keep a copy of 'this' each time the inversion is done, and when getInverse, if copy!=this -> update
    void updateInverse()
    {
        if( m_invMassMatrix ) m_invMassMatrix->invert( *this );
    }
};

template<int _spatial_dimensions,typename _Real>
inline typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv operator/(const typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv& d, const AffineMass<_spatial_dimensions, _Real>& m)
{
    return m.getInverse() * d;
}

template<int _spatial_dimensions,typename _Real>
inline typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv operator*(const AffineMass<_spatial_dimensions, _Real>& m,const typename StdAffineTypes<_spatial_dimensions,_Real>::Deriv& d)
{
    return d * m;
}

typedef AffineMass<3, double> Affine3dMass;
typedef AffineMass<3, float> Affine3fMass;


#ifdef SOFA_FLOAT
typedef Affine3fMass Affine3Mass;
#else
typedef Affine3dMass Affine3Mass;
#endif



// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

template<> struct DataTypeName< defaulttype::Affine3fMass > { static const char* name() { return "Affine3fMass"; } };
template<> struct DataTypeName< defaulttype::Affine3dMass > { static const char* name() { return "Affine3dMass"; } };

/// \endcond



} // namespace defaulttype

namespace component
{

namespace mass
{

template<int N, typename Real>
class AddMToMatrixFunctor< typename defaulttype::StdAffineTypes<N,Real>::Deriv, defaulttype::AffineMass<N,Real> >
{
public:
    void operator()(defaulttype::BaseMatrix * mat, const defaulttype::AffineMass<N,Real>& mass, int pos, double fact)
    {
//         cerr<<"WARNING: AddMToMatrixFunctor not implemented"<<endl;
        typedef defaulttype::AffineMass<N,Real> AffineMass;
        for( unsigned i=0; i<AffineMass::VSize; ++i )
            for( unsigned j=0; j<AffineMass::VSize; ++j )
            {
                mat->add(pos+i, pos+j, mass[i][j]*fact);
//            cerr<<"AddMToMatrixFunctor< defaulttype::Vec<N,Real>, defaulttype::Mat<N,N,Real> >::operator(), add "<< mass[i][j]*fact << " in " << pos+i <<","<< pos+j <<endl;
            }
    }
};

#ifndef SOFA_FLOAT
template <>
void UniformMass<defaulttype::Affine3dTypes, defaulttype::Affine3dMass>::draw( const core::visual::VisualParams* vparams );
template <>
double UniformMass<defaulttype::Affine3dTypes, defaulttype::Affine3dMass>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx ) const;
#endif
#ifndef SOFA_DOUBLE
template <>
void UniformMass<defaulttype::Affine3fTypes, defaulttype::Affine3fMass>::draw( const core::visual::VisualParams* vparams );
template <>
double UniformMass<defaulttype::Affine3fTypes, defaulttype::Affine3fMass>::getPotentialEnergy( const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& vx ) const;
#endif



} // namespace mass

} // namespace component






} // namespace sofa



#endif
