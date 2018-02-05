/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef FLEXIBLE_QUADRATICTYPES_H
#define FLEXIBLE_QUADRATICTYPES_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/decompose.h>
#ifdef SOFA_SMP
#include <sofa/defaulttype/SharedTypes.h>
#endif /* SOFA_SMP */

#include <sofa/defaulttype/Quat.h>

#include "DeformableFrameMass.h"

namespace sofa
{

namespace defaulttype
{


/** DOF types associated with 2nd order deformable frames. Each deformable frame generates an quadratic displacement field, with 30 independent degrees of freedom.
*/
template<int _spatial_dimensions, typename _Real>
class StdQuadraticTypes
{
public:
    static const unsigned int spatial_dimensions = _spatial_dimensions;  ///< Number of dimensions the frame is moving in, typically 3
    static const unsigned int num_cross_terms =  spatial_dimensions==1? 0 : ( spatial_dimensions==2? 1 : spatial_dimensions==3? 3 : 0);
    static const unsigned int num_quadratic_terms =  2*spatial_dimensions + num_cross_terms;
    static const unsigned int VSize = spatial_dimensions +  spatial_dimensions * num_quadratic_terms;  // number of entries
    enum { coord_total_size = VSize };
    enum { deriv_total_size = VSize };
    typedef _Real Real;
    typedef helper::vector<Real> VecReal;

    // ------------    Types and methods defined for easier data access
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Vec<num_quadratic_terms, Real> QuadraticCoord;                   ///< Position or velocity of a point, and its second-degree polynomial values
    typedef Mat<spatial_dimensions,num_cross_terms,Real> CrossM;
    typedef Mat<spatial_dimensions,spatial_dimensions,Real> Affine;
    typedef Mat<spatial_dimensions,num_quadratic_terms, Real> Frame;

    typedef SpatialCoord CPos;
    typedef SpatialCoord DPos;



    class Coord : public Vec<VSize,Real>
    {
        typedef Vec<VSize,Real> MyVec;

    public:

        enum { spatial_dimensions = _spatial_dimensions }; // different from Vec::spatial_dimensions == 30

        Coord() { clear(); }
        Coord( const Vec<VSize,Real>& d):MyVec(d) {}
        Coord( const SpatialCoord& c, const Frame& a) { getCenter()=c; getQuadratic()=a;}
        Coord ( const SpatialCoord &center, const Affine &affine, const Affine &square=Affine(), const CrossM &crossterms=CrossM())
        {
            getCenter() = center;
            for(unsigned int i=0; i<spatial_dimensions; ++i)
            {
                for(unsigned int j=0; j<spatial_dimensions; ++j)
                {
                    Frame& quadratic=getQuadratic();
                    quadratic[i][j]=affine[i][j];
                    quadratic[i][j+spatial_dimensions]=square[i][j];
                }
            }
            for(unsigned int i=0; i<spatial_dimensions; ++i)
            {
                for(unsigned int j=0; j<num_cross_terms; ++j)
                {
                    Frame& quadratic=getQuadratic();
                    quadratic[i][j+2*spatial_dimensions]=crossterms[i][j];
                }
            }
        }
        void clear() { MyVec::clear(); for(unsigned i=0; i<spatial_dimensions; i++) getQuadratic()[i][i]=(Real)1.0;  } // init affine part to identity

        typedef Real value_type;

        /// point
        SpatialCoord& getCenter() { return *reinterpret_cast<SpatialCoord*>(&this->elems[0]); }
        const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&this->elems[0]); }

        /// local frame
        Frame& getQuadratic() { return *reinterpret_cast<Frame*>(&this->elems[spatial_dimensions]); }
        const Frame& getQuadratic() const { return *reinterpret_cast<const Frame*>(&this->elems[spatial_dimensions]); }

        Affine getAffine() const
        {
            Affine m;
            for (unsigned int i = 0; i < spatial_dimensions; ++i)
                for (unsigned int j = 0; j < spatial_dimensions; ++j)
                    m[i][j]=getQuadratic()[i][j];
            return  m;
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

        /// Write the OpenGL transformation matrix
        void writeOpenGlMatrix ( float m[16] ) const
        {
            static_assert(spatial_dimensions == 3, "");
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
            return getQuadratic()*convertSpatialToQuadraticCoord(v) + getCenter();
        }

        /// Project a point from the parent frame to the child frame
        SpatialCoord pointToChild( const SpatialCoord& v ) const
        {
            Coord QuadraticInv = inverse(this);
            return QuadraticInv.pointToParent(v);

//            Frame QuadraticInv;
//            QuadraticInv.invert( getQuadratic() );
//            return QuadraticInv * (v-getCenter());
        }



        /// project to a rigid motion
        /// method: 0=polar (default), 1=SVD
        void setRigid( unsigned method=0 )
        {
            Frame& q = getQuadratic();
            // first matrix is pure rotation
            Affine rotation;

            if( method==1 ) // SVD
                helper::Decompose<Real>::polarDecomposition_stable(getAffine(), rotation);
            else // polar
                helper::Decompose<Real>::polarDecomposition(getAffine(), rotation);

            for(unsigned i=0; i<spatial_dimensions; i++)
                for(unsigned j=0; j<spatial_dimensions; j++)
                    q[i][j] = rotation[i][j];

            // the rest is null
            for(unsigned i=0; i<spatial_dimensions; i++)
                for(unsigned j=spatial_dimensions; j<num_quadratic_terms; j++)
                    q[i][j] = 0.;
        }

        template< int N, class Real2 > // N <= VSize
        void operator+=( const Vec<N,Real2>& p ) { for(int i=0;i<N;++i) this->elems[i] += (Real)p[i]; }
        template< int N, class Real2 > // N <= VSize
        void operator=( const Vec<N,Real2>& p ) { for(int i=0;i<N;++i) this->elems[i] = (Real)p[i]; }

    };


    typedef helper::vector<Coord> VecCoord;

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



    class Deriv : public Vec<VSize,Real>
    {
        typedef Vec<VSize,Real> MyVec;
    public:
        enum { spatial_dimensions = _spatial_dimensions }; // different from Vec::spatial_dimensions == 30

        Deriv() { MyVec::clear(); }
        Deriv( const Vec<VSize,Real>& d):MyVec(d) {}
        Deriv( const SpatialCoord& c, const Frame& a) { getVCenter()=c; getVQuadratic()=a;}
        Deriv ( const SpatialCoord &center, const Affine &affine, const Affine &square=Affine(), const CrossM &crossterms=CrossM())
        {
            getVCenter() = center;
            for(unsigned int i=0; i<spatial_dimensions; ++i)
            {
                for(unsigned int j=0; j<spatial_dimensions; ++j)
                {
                    Frame& quadratic=getVQuadratic();
                    quadratic[i][j]=affine[i][j];
                    quadratic[i][j+spatial_dimensions]=square[i][j];
                }
            }
            for(unsigned int i=0; i<spatial_dimensions; ++i)
            {
                for(unsigned int j=0; j<num_cross_terms; ++j)
                {
                    Frame& quadratic=getVQuadratic();
                    quadratic[i][j+2*spatial_dimensions]=crossterms[i][j];
                }
            }
        }

        typedef Real value_type;

        /// point
        SpatialCoord& getVCenter() { return *reinterpret_cast<SpatialCoord*>(&this->elems[0]); }
        const SpatialCoord& getVCenter() const { return *reinterpret_cast<const SpatialCoord*>(&this->elems[0]); }

        /// local frame
        Frame& getVQuadratic() { return *reinterpret_cast<Frame*>(&this->elems[spatial_dimensions]); }
        const Frame& getVQuadratic() const { return *reinterpret_cast<const Frame*>(&this->elems[spatial_dimensions]); }


        Affine getAffine() const
        {
            Affine m;
            for (unsigned int i = 0; i < spatial_dimensions; ++i)
                for (unsigned int j = 0; j < spatial_dimensions; ++j)
                    m[i][j]=getVQuadratic()[i][j];
            return  m;
        }

        /// get jacobian of the projection dQ/dM
        /// method: 0=polar (default), 1=SVD
        static void getJRigid(const Coord& c, Mat<VSize,VSize,Real>& J, unsigned method=0)
        {
            static const unsigned MSize = spatial_dimensions * spatial_dimensions;
            Mat<MSize,MSize,Real> dQOverdM;

            switch( method )
            {
                case 1: // SVD
                {
                    Affine U, V;
                    Vec<spatial_dimensions,Real> diag;
                    helper::Decompose<Real>::SVD_stable( c.getAffine(), U, diag, V ); // TODO this was already computed in setRigid...
                    helper::Decompose<Real>::polarDecomposition_stable_Gradient_dQOverdM( U, diag, V, dQOverdM );
                    break;
                }

                case 0: // polar
                default:
                {
                    Affine Q,S,invG;
                    helper::Decompose<Real>::polarDecomposition( c.getAffine(), Q, S ); // TODO this was already computed in setRigid...
                    helper::Decompose<Real>::polarDecompositionGradient_G(Q,S,invG); // TODO this was already computed in setRigid...
                    helper::Decompose<Real>::polarDecompositionGradient_dQOverdM(Q,invG,dQOverdM);
                }
            }

            // set all to 0 (quadratic terms are fully constrained)
            J.clear();

            // translation -> identity
            for(unsigned int i=0; i<spatial_dimensions; ++i)
                for(unsigned int j=0; j<spatial_dimensions; ++j)
                    J(i,j)=(i==j)?1.:0;

            // affine part
            for(unsigned int i=0; i<MSize; ++i)
                for(unsigned int j=0; j<MSize; ++j)
                    J(i+spatial_dimensions,j+spatial_dimensions)=dQOverdM(i,j);
        }

        /// project to a rigid motion
        /// method: 0=polar (default), 1=SVD, 2=approximation
        void setRigid(const Coord& c, unsigned method = 0 )
        {
            switch( method )
            {
                case 1: // SVD
                {
                    Affine U, V, dQ;
                    Vec<spatial_dimensions,Real> diag;
                    helper::Decompose<Real>::SVD_stable( c.getAffine(), U, diag, V );
                    helper::Decompose<Real>::polarDecomposition_stable_Gradient_dQ( U, diag, V, this->getAffine(), dQ );

                    Frame& q = getVQuadratic();
                    for(unsigned i=0; i<spatial_dimensions; i++)
                        for(unsigned j=0; j<spatial_dimensions; j++)
                            q[i][j] = dQ[i][j];

                    // the rest is null
                    for(unsigned i=0; i<spatial_dimensions; i++)
                        for(unsigned j=spatial_dimensions; j<num_quadratic_terms; j++)
                            q[i][j] = 0.;

                    break;
                }
                case 2: // approximation
                {
                    // good approximation of the solution with no inversion of a 6x6 matrix
                    // based on : dR ~ 0.5*(dA.A^-1 - A^-T dA^T) R
                    // the projection matrix is however non symmetric..

                    // Compute velocity tensor W = Adot.Ainv
                    Affine Ainv;  invertMatrix(Ainv,c.getAffine());
                    Affine W = getAffine() * Ainv;

                    // make it skew-symmetric
                    for(unsigned i=0; i<spatial_dimensions; i++) W[i][i] = 0.0;
                    for(unsigned i=0; i<spatial_dimensions; i++)
                    {
                        for(unsigned j=i+1; j<spatial_dimensions; j++)
                        {
                            W[i][j] = (W[i][j] - W[j][i]) *(Real)0.5;
                            W[j][i] = - W[i][j];
                        }
                    }

                    // retrieve global velocity : Rdot = W.R
                    Affine R;
                    helper::Decompose<Real>::polarDecomposition( c.getAffine() , R );
                    Affine Rdot = W*R;

                    // assign rigid velocity
                    Frame& q = getVQuadratic();
                    for(unsigned i=0; i<spatial_dimensions; i++)
                        for(unsigned j=0; j<spatial_dimensions; j++)
                            q[i][j] = Rdot[i][j];

                    // the rest is null
                    for(unsigned i=0; i<spatial_dimensions; i++)
                        for(unsigned j=spatial_dimensions; j<num_quadratic_terms; j++)
                            q[i][j] = 0.;
                }


                case 0: // polar
                default:
                {
                    Affine Q,S,invG,dQ;
                    helper::Decompose<Real>::polarDecomposition( c.getAffine(), Q, S );
                    helper::Decompose<Real>::polarDecompositionGradient_G(Q,S,invG);
                    helper::Decompose<Real>::polarDecompositionGradient_dQ(invG,Q,this->getAffine(),dQ);

                    Frame& q = getVQuadratic();
                    for(unsigned i=0; i<spatial_dimensions; i++)
                        for(unsigned j=0; j<spatial_dimensions; j++)
                            q[i][j] = dQ[i][j];

                    // the rest is null
                    for(unsigned i=0; i<spatial_dimensions; i++)
                        for(unsigned j=spatial_dimensions; j<num_quadratic_terms; j++)
                            q[i][j] = 0.;
                }
            }
        }


        template< int N, class Real2 > // N <= VSize
        void operator+=( const Vec<N,Real2>& p ) { for(int i=0;i<N;++i) this->elems[i] += (Real)p[i]; }
        template< int N, class Real2 > // N <= VSize
        void operator=( const Vec<N,Real2>& p ) { for(int i=0;i<N;++i) this->elems[i] = (Real)p[i]; }
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

    /// for finite difference methods 
    static Deriv coordDifference(const Coord& c1, const Coord& c2)
    {
        return (Deriv)(c1-c2);
    }


};


template<typename Real>
static Vec<2,Real> convertSpatialToQuadraticCoord(const Vec<1,Real>& p)
{
    return Vec<5,Real>( p[0], p[0]*p[0]);
}

template<typename Real>
static Vec<5,Real> convertSpatialToQuadraticCoord(const Vec<2,Real>& p)
{
    return Vec<5,Real>( p[0], p[1], p[0]*p[0], p[1]*p[1], p[0]*p[1]);
}

template<typename Real>
static Vec<9,Real> convertSpatialToQuadraticCoord(const Vec<3,Real>& p)
{
    return Vec<9,Real>( p[0], p[1], p[2], p[0]*p[0], p[1]*p[1], p[2]*p[2], p[0]*p[1], p[1]*p[2], p[0]*p[2]);
}


// returns dp^* / dp

template<typename Real>
static Mat<2,1,Real> SpatialToQuadraticCoordGradient(const Vec<1,Real>& p)
{
    Mat<2,1,Real> M;
    M(0,0)=1;     M(1,0)=2*p[0];
    return M;
}

template<typename Real>
static Mat<5,2,Real> SpatialToQuadraticCoordGradient(const Vec<2,Real>& p)
{
    Mat<5,2,Real> M;
    for(unsigned int i=0;i<2;i++) { M(i,i)=1;  M(i+2,i)=2*p[i];}
    M(4,0)=p[1];     M(4,1)=p[0];
    return M;
}

template<typename Real>
static Mat<9,3,Real> SpatialToQuadraticCoordGradient(const Vec<3,Real>& p)
{
    Mat<9,3,Real> M;
    for(unsigned int i=0;i<3;i++) { M(i,i)=1;  M(i+3,i)=2*p[i];}
    M(6,0)=p[1]; M(6,1)=p[0];
    M(7,1)=p[2]; M(7,2)=p[1];
    M(8,0)=p[2]; M(8,2)=p[0];
    return M;
}


#ifndef SOFA_FLOAT
typedef StdQuadraticTypes<3, double> Quadratic3dTypes;
#endif
#ifndef SOFA_DOUBLE
typedef StdQuadraticTypes<3, float> Quadratic3fTypes;
#endif

/// Note: Many scenes use Quadratic as template for 3D double-precision rigid type. Changing it to Quadratic3d would break backward compatibility.
#ifdef SOFA_FLOAT
template<> inline const char* Quadratic3fTypes::Name() { return "Quadratic"; }
#else
template<> inline const char* Quadratic3dTypes::Name() { return "Quadratic"; }
#ifndef SOFA_DOUBLE
template<> inline const char* Quadratic3fTypes::Name() { return "Quadratic3f"; }
#endif
#endif

#ifdef SOFA_FLOAT
typedef Quadratic3fTypes Quadratic3Types;
#else
typedef Quadratic3dTypes Quadratic3Types;
#endif
//typedef Quadratic3Types QuadraticTypes;


// Specialization of the defaulttype::DataTypeInfo type traits template
#ifndef SOFA_DOUBLE
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3fTypes::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3fTypes::Coord, sofa::defaulttype::Quadratic3fTypes::Coord::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticCoord<" << sofa::defaulttype::Quadratic3fTypes::Coord::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3fTypes::Real>::name() << ">"; return o.str(); }
};
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3fTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3fTypes::Deriv, sofa::defaulttype::Quadratic3fTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticDeriv<" << sofa::defaulttype::Quadratic3fTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3fTypes::Real>::name() << ">"; return o.str(); }
};
#endif

#ifndef SOFA_FLOAT
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3dTypes::Coord > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3dTypes::Coord, sofa::defaulttype::Quadratic3dTypes::Coord::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticCoord<" << sofa::defaulttype::Quadratic3dTypes::Coord::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3dTypes::Real>::name() << ">"; return o.str(); }
};
template<> struct DataTypeInfo< sofa::defaulttype::Quadratic3dTypes::Deriv > : public FixedArrayTypeInfo< sofa::defaulttype::Quadratic3dTypes::Deriv, sofa::defaulttype::Quadratic3dTypes::Deriv::total_size >
{
    static std::string name() { std::ostringstream o; o << "QuadraticDeriv<" << sofa::defaulttype::Quadratic3dTypes::Deriv::total_size << "," << DataTypeName<sofa::defaulttype::Quadratic3dTypes::Real>::name() << ">"; return o.str(); }
};
#endif
// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES


#ifndef SOFA_FLOAT
template<> struct DataTypeName< defaulttype::Quadratic3dTypes::Coord > { static const char* name() { return "Quadratic3dTypes::Coord"; } };
#endif
#ifndef SOFA_DOUBLE
template<> struct DataTypeName< defaulttype::Quadratic3fTypes::Coord > { static const char* name() { return "Quadratic3fTypes::Coord"; } };
#endif


/// \endcond



// ====================================================================
// QuadraticMass

#ifndef SOFA_FLOAT
typedef DeformableFrameMass<3, StdQuadraticTypes<3,double>::deriv_total_size, double> Quadratic3dMass;
#endif
#ifndef SOFA_DOUBLE
typedef DeformableFrameMass<3, StdQuadraticTypes<3,float>::deriv_total_size, float> Quadratic3fMass;
#endif


#ifdef SOFA_FLOAT
typedef Quadratic3fMass Quadratic3Mass;
#else
typedef Quadratic3dMass Quadratic3Mass;
#endif



// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES


#ifndef SOFA_FLOAT
template<> struct DataTypeName< defaulttype::Quadratic3dMass > { static const char* name() { return "Quadratic3dMass"; } };
#endif
#ifndef SOFA_DOUBLE
template<> struct DataTypeName< defaulttype::Quadratic3fMass > { static const char* name() { return "Quadratic3fMass"; } };
#endif

/// \endcond



} // namespace defaulttype



} // namespace sofa



#endif
