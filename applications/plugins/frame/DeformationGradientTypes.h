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
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace defaulttype
{

using std::endl;
using helper::vector;

template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
class DeformationGradient;

template<class DeformationGradientType, bool _iscorotational>
class CStrain;



template<class DeformationGradientType, bool _iscorotational>
class CStrain //< DeformationGradientType, _iscorotational>
{
public:
    CStrain() { }

    static const bool iscorotational = _iscorotational;

    typedef typename DeformationGradientType::Coord DeformationGradient;
    typedef typename DeformationGradientType::Real Real;

    static const unsigned material_dimensions = DeformationGradientType::material_dimensions;
    typedef Mat<material_dimensions,material_dimensions, Real> MaterialFrame;

    enum {order = DeformationGradientType::order }; // order = 1 -> deformationgradient // order = 2 -> deformationgradient + spatial derivatives

    static const unsigned strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    typedef Mat<strain_size,strain_size,Real> StrStr;

    static const unsigned strain_order = order==1? 0 : ( (order==2 && iscorotational)? 1 : ( (order==2 && !iscorotational)? 2 : 0 )) ;
    static const unsigned NumStrainVec = strain_order==0? 1 : ( strain_order==1? 1 + material_dimensions : ( strain_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : 0 )) ;
    typedef Vec<NumStrainVec,StrainVec> Strain;  ///< Strain and its gradient, in vector form
    typedef Strain Stress;

    static const unsigned strainenergy_order = 2*strain_order; 	///< twice the order of strain
    static const unsigned strainenergy_size = strainenergy_order==0? 1 : (
            strainenergy_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : (
                    (strainenergy_order==4 && material_dimensions==1)? 5 : (
                            (strainenergy_order==4 && material_dimensions==2)? 15 : (
                                    (strainenergy_order==4 && material_dimensions==3)? 35 : 1 ))));
    typedef Vec<strainenergy_size,Real> StrainEnergyVec;



    static StrainVec getStrainVec(  const MaterialFrame& f ) // symmetric matrix to voigt notation
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

    static StrainVec getStrainVec_assymetric(  const MaterialFrame& f ) // symmetric matrix (f+f^T)/2 to voigt notation
    {
        StrainVec s;
        unsigned ei=0;
        for(unsigned j=0; j<material_dimensions; j++)
        {
            for( unsigned k=j; k<material_dimensions; k++ )
            {
                s[ei] = (f[j][k]+f[k][j])*0.5;   //  TO DO: check in material if eij must be eij or 2eij
                ei++;
            }
        }
        return s;
    }

    static MaterialFrame getFrame( const StrainVec& s  ) // voigt notation to symmetric matrix
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




    //  void setStress( const Stress& stress  ) // replace deformation gradient by stress
    //{
    //    getMaterialFrame() = getFrame(stress[0]);
    //}

    //void getCorotationalStrain( MaterialFrame& rotation, Strain& strain ) const
    //{
    //        MaterialFrame local_deformation_gradient;
    //        helper::polar_decomp(this->getMaterialFrame(), rotation, local_deformation_gradient); // decompose F=RD
    //        strain[0] = getStrainVec( local_deformation_gradient );
    // }

    //void getCorotationalStrainRate( Strain& strainRate, const MaterialFrame& rotation  ) const {
    //    // FF: assuming that the strain rate  can be decomposed using the same rotation as the strain
    //        strainRate[0] = getStrainVec( rotation.multTranspose(this->getMaterialFrame()) );
    // }

    static void getStrain( const DeformationGradient& F, Strain& strain , MaterialFrame& rotation)
    {
        if(iscorotational) // cauchy strain (order 0 or 1) : e= [grad(R^T u)+grad(R^T u)^T ]/2 = [R^T F + F^T R ]/2 - I
        {
            MaterialFrame strainmat;
            helper::polar_decomp(F.getMaterialFrame(), rotation, strainmat); // decompose F=RD

            // order 0: e = [R^T F + F^T R ]/2 - I = D - I
            for(unsigned j=0; j<material_dimensions; j++) strainmat[j][j]-=1.;
            strain[0] = getStrainVec( strainmat );

            if(strain_order==0) return;
            // order 1 : de =  [R^T dF + dF^T R ]/2
            for(unsigned i=1; i<NumStrainVec; i++)
                strain[i] = getStrainVec_assymetric( rotation.multTranspose( F.getMaterialFrameGradient()[i-1] ) );
        }
        else // green-lagrange strain (order 0 or 2) : E= [F^T.F - I ]/2
        {
            unsigned ei=0;
            // order 0: E = [F^T.F - I ]/2
            MaterialFrame strainmat=F.getMaterialFrame().multTranspose( F.getMaterialFrame() );
            for(unsigned j=0; j<material_dimensions; j++) strainmat[j][j]-=1.;
            strainmat*=0.5;
            strain[ei] = getStrainVec( strainmat ); ei++;

            if(strain_order==0) return;
            // order 1: dE/dpi = [dFi^T.F +  F^T.dFi ]/2
            for(unsigned i=1; i<=material_dimensions; i++)
            {
                strainmat = F.getMaterialFrame().multTranspose( F.getMaterialFrameGradient()[i-1] );
                strain[ei] = getStrainVec_assymetric( strainmat ); ei++;
            }

            // order 2: dE/dpidpj = [dFi^T.dFj +  dFj^T.dFi ]/2
            for(unsigned i=0; i<material_dimensions; i++)
                for(unsigned j=i; j<material_dimensions; j++)
                {
                    strainmat = F.getMaterialFrameGradient()[i].multTranspose( F.getMaterialFrameGradient()[j] );
                    strain[ei] = getStrainVec_assymetric( strainmat ); ei++;
                }
        }
    }

    static void getStrainRate( const DeformationGradient& F, Strain& strain, const MaterialFrame& rotation)
    {
        if(iscorotational)
        {
            // order 0: e = [R^T Fr + Fr^T R ]/2
            strain[0] = getStrainVec_assymetric( rotation.multTranspose(F.getMaterialFrame()) );

            if(strain_order==0) return;
            // order 1 : de =  [R^T dFr + dFr^T R ]/2
            for(unsigned i=1; i<NumStrainVec; i++)  strain[i] = getStrainVec_assymetric( rotation.multTranspose( F.getMaterialFrameGradient()[i-1] ) );
        }
        else // green-lagrange strain (order 0 or 2) : E= [F^T.F - I ]/2
        {
            // to do
        }
    }



    static void mult( Strain& s, Real r )
    {
        for(unsigned i=0; i<s.size(); i++)
            s[i] *= r;
    }

    static Strain mult( const Strain& s, const Mat<strain_size,strain_size,Real>& H )
    // compute H.s -> returns a vector of the order of the input strain
    {
        Strain ret;
        for(unsigned i=0; i<s.size(); i++)
            ret[i] = H*s[i];
        return ret;
    }


    static StrainEnergyVec multTranspose(const Strain& s1 , const Strain& s2 )
    // compute s1^T.s2 -> returns a vector of twice the order of the strain
    // can be used to compute energy U=strain^T.stress/2
    //        or force wrt. dof i : Fi = -dU/di= dstrain/di ^T.stress
    //        or stiffness wrt. dof i and j : Kij = -dU/didj= dstrain/di ^T.dstress/dj
    {

    }

};



// specialization for vec3 -> should be zero (no strain) but compilation error with Vec<0,Real>..
template< bool _iscorotational>
class CStrain < defaulttype::ExtVec3dTypes, _iscorotational>
{
public:
    static const bool iscorotational = _iscorotational;
    typedef double Real;

    static const unsigned material_dimensions = 3;
    typedef Mat<material_dimensions,material_dimensions, Real> MaterialFrame;

    enum {order = 0 }; // order = 1 -> deformationgradient // order = 2 -> deformationgradient + spatial derivatives

    static const unsigned strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    typedef Mat<strain_size,strain_size,Real> StrStr;

    static const unsigned strain_order = order==1? 0 : ( (order==2 && iscorotational)? 1 : ( (order==2 && !iscorotational)? 2 : 0 )) ;
    static const unsigned NumStrainVec = strain_order==0? 1 : ( strain_order==1? 1 + material_dimensions : ( strain_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : 0 )) ;
    typedef Vec<NumStrainVec,StrainVec> Strain;  ///< Strain and its gradient, in vector form
    typedef Strain Stress;

    static const unsigned strainenergy_order = 2*strain_order; 	///< twice the order of strain
    static const unsigned strainenergy_size = strainenergy_order==0? 1 : (
            strainenergy_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : (
                    (strainenergy_order==4 && material_dimensions==1)? 5 : (
                            (strainenergy_order==4 && material_dimensions==2)? 15 : (
                                    (strainenergy_order==4 && material_dimensions==3)? 35 : 1 ))));
    typedef Vec<strainenergy_size,Real> StrainEnergyVec;
};
template< bool _iscorotational>
class CStrain < defaulttype::ExtVec3fTypes, _iscorotational>
{
public:
    static const bool iscorotational = _iscorotational;
    typedef float Real;

    static const unsigned material_dimensions = 3;
    typedef Mat<material_dimensions,material_dimensions, Real> MaterialFrame;

    enum {order = 0 }; // order = 1 -> deformationgradient // order = 2 -> deformationgradient + spatial derivatives

    static const unsigned strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    typedef Mat<strain_size,strain_size,Real> StrStr;

    static const unsigned strain_order = order==1? 0 : ( (order==2 && iscorotational)? 1 : ( (order==2 && !iscorotational)? 2 : 0 )) ;
    static const unsigned NumStrainVec = strain_order==0? 1 : ( strain_order==1? 1 + material_dimensions : ( strain_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : 0 )) ;
    typedef Vec<NumStrainVec,StrainVec> Strain;  ///< Strain and its gradient, in vector form
    typedef Strain Stress;

    static const unsigned strainenergy_order = 2*strain_order; 	///< twice the order of strain
    static const unsigned strainenergy_size = strainenergy_order==0? 1 : (
            strainenergy_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : (
                    (strainenergy_order==4 && material_dimensions==1)? 5 : (
                            (strainenergy_order==4 && material_dimensions==2)? 15 : (
                                    (strainenergy_order==4 && material_dimensions==3)? 35 : 1 ))));
    typedef Vec<strainenergy_size,Real> StrainEnergyVec;
};
template<  bool _iscorotational>
class CStrain < defaulttype::Vec3dTypes, _iscorotational>
{
public:
    static const bool iscorotational = _iscorotational;
    typedef double Real;

    static const unsigned material_dimensions = 3;
    typedef Mat<material_dimensions,material_dimensions, Real> MaterialFrame;

    enum {order = 0 }; // order = 1 -> deformationgradient // order = 2 -> deformationgradient + spatial derivatives

    static const unsigned strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    typedef Mat<strain_size,strain_size,Real> StrStr;

    static const unsigned strain_order = order==1? 0 : ( (order==2 && iscorotational)? 1 : ( (order==2 && !iscorotational)? 2 : 0 )) ;
    static const unsigned NumStrainVec = strain_order==0? 1 : ( strain_order==1? 1 + material_dimensions : ( strain_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : 0 )) ;
    typedef Vec<NumStrainVec,StrainVec> Strain;  ///< Strain and its gradient, in vector form
    typedef Strain Stress;

    static const unsigned strainenergy_order = 2*strain_order; 	///< twice the order of strain
    static const unsigned strainenergy_size = strainenergy_order==0? 1 : (
            strainenergy_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : (
                    (strainenergy_order==4 && material_dimensions==1)? 5 : (
                            (strainenergy_order==4 && material_dimensions==2)? 15 : (
                                    (strainenergy_order==4 && material_dimensions==3)? 35 : 1 ))));
    typedef Vec<strainenergy_size,Real> StrainEnergyVec;
};
template<  bool _iscorotational>
class CStrain < defaulttype::Vec3fTypes, _iscorotational>
{
public:
    static const bool iscorotational = _iscorotational;
    typedef float Real;

    static const unsigned material_dimensions = 3;
    typedef Mat<material_dimensions,material_dimensions, Real> MaterialFrame;

    enum {order = 0 }; // order = 1 -> deformationgradient // order = 2 -> deformationgradient + spatial derivatives

    static const unsigned strain_size = material_dimensions * (1+material_dimensions) / 2; ///< independent entries in the strain tensor
    typedef Vec<strain_size,Real> StrainVec;    ///< Strain in vector form
    typedef Mat<strain_size,strain_size,Real> StrStr;

    static const unsigned strain_order = order==1? 0 : ( (order==2 && iscorotational)? 1 : ( (order==2 && !iscorotational)? 2 : 0 )) ;
    static const unsigned NumStrainVec = strain_order==0? 1 : ( strain_order==1? 1 + material_dimensions : ( strain_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : 0 )) ;
    typedef Vec<NumStrainVec,StrainVec> Strain;  ///< Strain and its gradient, in vector form
    typedef Strain Stress;

    static const unsigned strainenergy_order = 2*strain_order; 	///< twice the order of strain
    static const unsigned strainenergy_size = strainenergy_order==0? 1 : (
            strainenergy_order==2? 1 + material_dimensions*(material_dimensions+3)/2 : (
                    (strainenergy_order==4 && material_dimensions==1)? 5 : (
                            (strainenergy_order==4 && material_dimensions==2)? 15 : (
                                    (strainenergy_order==4 && material_dimensions==3)? 35 : 1 ))));
    typedef Vec<strainenergy_size,Real> StrainEnergyVec;
};


/*
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
    DeformationGradient(){ v.clear(); }
    DeformationGradient( const Vec<VSize,Real>& d):v(d){}
    void clear(){ v.clear(); }

    /// seen as a vector
    Vec<VSize,Real>& getVec(){ return v; }
    const Vec<VSize,Real>& getVec() const { return v; }

    /// point
    SpatialCoord& getCenter(){ return *reinterpret_cast<SpatialCoord*>(&v[0]); }
    const SpatialCoord& getCenter() const { return *reinterpret_cast<const SpatialCoord*>(&v[0]); }

    /// local frame (if order>=1)
    MaterialFrame& getMaterialFrame(){ return *reinterpret_cast<MaterialFrame*>(&v[spatial_dimensions]); }
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
        for(unsigned j=0; j<material_dimensions; j++){
            for( unsigned k=j; k<material_dimensions; k++ ){
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
        for(unsigned j=0; j<material_dimensions; j++){
            for( unsigned k=j; k<material_dimensions; k++ ){
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

    void getCorotationalStrainRate( Strain& strainRate, const MaterialFrame& rotation  ) const {
        // FF: assuming that the strain rate  can be decomposed using the same rotation as the strain
            strainRate[0] = getStrainVec( rotation.multTranspose(this->getMaterialFrame()) );
     }



    DeformationGradient operator +(const DeformationGradient& a) const { return DeformationGradient(v+a.v); }
    void operator +=(const DeformationGradient& a){ v+=a.v; }

    DeformationGradient operator -(const DeformationGradient& a) const { return DeformationGradient(v-a.v); }
    void operator -=(const DeformationGradient& a){ v-=a.v; }


    template<typename real2>
    DeformationGradient operator *(real2 a) const { return DeformationGradient(v*a); }
    template<typename real2>
    void operator *=(real2 a){ v *= a; }

    template<typename real2>
    void operator /=(real2 a){ v /= a; }

    DeformationGradient operator - () const { return DeformationGradient(-v); }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    Real operator*(const DeformationGradient& a) const
    {
        return v*a.v;
    }

    /// write to an output stream
    inline friend std::ostream& operator << ( std::ostream& out, const DeformationGradient& c ){
        out<<c.v;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> ( std::istream& in, DeformationGradient& c ){
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

*/
template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
class DeformationGradient // <_spatial_dimensions, _material_dimensions, _order, _Real>
{

public:
    static const unsigned spatial_dimensions = _spatial_dimensions;
    static const unsigned material_dimensions = _material_dimensions;
    static const unsigned order = _order;  ///< 0: only a point, no gradient 1:deformation gradient, 2: deformation gradient and its gradient
    //       enum {order = 2 };
    static const unsigned NumMatrices = order==0? 0 : (order==1? 1 : (order==2? 1 + material_dimensions : -1 ));
    static const unsigned VSize = spatial_dimensions +  NumMatrices * material_dimensions * material_dimensions;  // number of entries
    typedef _Real Real;
    typedef vector<Real> VecReal;
    //        typedef Vec<35,Real> SampleIntegVector;  ///< used to precompute the integration of deformation energy over a sample region

    // ------------    Types and methods defined for easier data access
    typedef Vec<material_dimensions, Real> MaterialCoord;
    typedef vector<MaterialCoord> VecMaterialCoord;
    typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
    typedef Mat<material_dimensions,material_dimensions, Real> MaterialFrame;      ///< Matrix representing a deformation gradient
    typedef Vec<material_dimensions, MaterialFrame> MaterialFrameGradient;                 ///< Gradient of a deformation gradient

    //static const unsigned strain_size = spatial_dimensions * (1+spatial_dimensions) / 2; ///< independent entries in the strain tensor
    //typedef Vec<strain_size,Real> StrainVec;   ///< Strain in vector form
    //typedef Vec<NumMatrices,StrainVec> Strain; ///< Strain and its gradient, in vector form
    //typedef Strain Stress;


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

    //static void multStrain( Strain& s, Real r )
    //{
    //    for(unsigned i=0; i<s.size(); i++)
    //        s[i] *= r;
    //}

    //static StrainVec getStrainVec(  const MaterialFrame& f )
    //{
    //    StrainVec s;
    //    unsigned ei=0;
    //    for(unsigned j=0; j<material_dimensions; j++){
    //        for( unsigned k=j; k<material_dimensions; k++ ){
    //            s[ei] = f[j][k];
    //            ei++;
    //        }
    //    }
    //    return s;
    //}

    //static MaterialFrame getFrame( const StrainVec& s  )
    //{
    //    MaterialFrame f;
    //    unsigned ei=0;
    //    for(unsigned j=0; j<material_dimensions; j++){
    //        for( unsigned k=j; k<material_dimensions; k++ ){
    //            f[k][j] = f[j][k] = s[ei] ;
    //            ei++;
    //        }
    //    }
    //    return f;
    //}

    //void setStress( const Stress& stress  )
    //{
    //    getMaterialFrame() = getFrame(stress[0]);
    //    MaterialFrameGradient& g= this->getMaterialFrameGradient();
    //    for(unsigned i=0; i<spatial_dimensions; i++ ){
    //        g[i] = getFrame( stress[1+i] ); // FF: assuming that the gradient of F can be decomposed using the same rotation as F
    //    }
    //}

    // void getCorotationalStrain( MaterialFrame& rotation, Strain& strain ) const {
    //        MaterialFrame local_deformation_gradient;
    //        helper::polar_decomp(this->getMaterialFrame(), rotation, local_deformation_gradient); // decompose F=RD
    //        strain[0] = getStrainVec( local_deformation_gradient );

    //        const MaterialFrameGradient& g= this->getMaterialFrameGradient();
    //        for(unsigned i=0; i<spatial_dimensions; i++ ){
    //            strain[1+i] = getStrainVec( rotation.multTranspose(g[i]) ); // FF: assuming that the gradient of F can be decomposed using the same rotation as F
    //        }
    //}

    //void getCorotationalStrainRate( Strain& strainRate, const MaterialFrame& rotation  ) const {
    //    // FF: assuming that the strain rate  can be decomposed using the same rotation as the strain
    //        strainRate[0] = getStrainVec( rotation.multTranspose(this->getMaterialFrame()) );
    //        const MaterialFrameGradient& g= this->getMaterialFrameGradient();
    //        for(unsigned i=0; i<spatial_dimensions; i++ ){
    //            strainRate[1+i] = getStrainVec( rotation.multTranspose(g[i]) );
    //        }
    //}



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
    static const unsigned order = _order;  ///< 0: only a point, no gradient 1:deformation gradient, 2: deformation gradient and its gradient
    typedef _Real Real;
//            static const unsigned NumMatrices = order==0? 0 : (order==1? 1 : (order==2? 1 + material_dimensions : -1 ));
//            static const unsigned VSize = spatial_dimensions +  NumMatrices * spatial_dimensions * spatial_dimensions;  // number of entries
    typedef vector<Real> VecReal;
//
    // ------------    Types and methods defined for easier data access
//            typedef Vec<material_dimensions, Real> MaterialCoord;
//            typedef Vec<spatial_dimensions, Real> SpatialCoord;                   ///< Position or velocity of a point
//            typedef Mat<material_dimensions,spatial_dimensions, Real> MaterialFrame;      ///< Matrix representing a deformation gradient
//            typedef Vec<spatial_dimensions, MaterialFrame> MaterialFrameGradient;                 ///< Gradient of a deformation gradient


    typedef DeformationGradient<spatial_dimensions,material_dimensions,order,Real> Coord;
    typedef vector<Coord> VecCoord;
    typedef Coord Deriv ;            ///< velocity and deformation gradient rate
    typedef vector<Deriv> VecDeriv;
    typedef MapMapSparseMatrix<Deriv> MatrixDeriv;

    typedef typename Coord::MaterialCoord MaterialCoord; ///< Position or velocity of a point in the object
    typedef typename Coord::SpatialCoord SpatialCoord;   ///< Position or velocity of a point in space
    typedef typename Coord::MaterialFrame MaterialFrame; ///< Matrix representing a deformation gradient
    typedef typename Coord::MaterialFrameGradient MaterialFrameGradient;                 ///< Gradient of a deformation gradient

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
