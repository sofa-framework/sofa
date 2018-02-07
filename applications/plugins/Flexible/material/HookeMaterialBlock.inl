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
#ifndef FLEXIBLE_HookeMaterialBlock_INL
#define FLEXIBLE_HookeMaterialBlock_INL

#include "../material/HookeMaterialBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include "../types/StrainTypes.h"
#include "../types/PolynomialBasis.h"

namespace sofa
{

namespace defaulttype
{


//////////////////////////////////////////////////////////////////////////////////
////  material laws
//////////////////////////////////////////////////////////////////////////////////

template<typename _Real,unsigned int dim,unsigned int size>
class HookeLaw
{
public:
    typedef _Real Real;

    static const unsigned int material_dimensions = dim;
    static const unsigned int strain_size = size;

    std::vector<Real> Kparams;  /** Constants for the stiffness matrix (e.g. Lamé coeffs) */
    std::vector<Real> Cparams;  /** Constants for the compliance matrix (e.g. Young modulus, poisson, shear modulus)*/

    virtual void set(const std::vector<Real> &cparams)=0;
    virtual Eigen::Matrix<Real,size,size,Eigen::RowMajor> assembleK(const Real &vol) const =0;
    virtual void applyK(Vec<size,Real> &out, const Vec<size,Real> &in, const Real &vol) const =0;
    virtual Eigen::Matrix<Real,size,size,Eigen::RowMajor> assembleC(const Real &vol) const =0;
};


/// isotropic 3D/2D
template<typename Real,int dim,unsigned int size>
class IsotropicHookeLaw: public HookeLaw<Real,dim,size>
{
public:
    virtual void set(const std::vector<Real> &cparams)
    {
        Real youngM=cparams[0] , poissonR=cparams[1];

        Real lamd = youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        Real mu = 0.5*youngM/(1+poissonR);

        this->Cparams.assign(cparams.begin(),cparams.end());
        this->Kparams.resize(2); this->Kparams[0]=lamd; this->Kparams[1]=mu;
    }

    virtual Eigen::Matrix<Real,size,size,Eigen::RowMajor> assembleK(const Real &vol) const
    {
        typedef Eigen::Matrix<Real,size,size,Eigen::RowMajor> block;
        block K=block::Zero();
        if(!vol) return K;
        Real muVol = this->Kparams[1]*vol;
        for(unsigned int i=0; i<dim; i++)  K(i,i)-=muVol*2.0;
        for(unsigned int i=dim; i<size; i++) K(i,i)-=muVol;
        for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<dim; j++) K(i,j)-=this->Kparams[0]*vol;
        return K;
    }

    virtual void applyK(Vec<size,Real> &out, const Vec<size,Real> &in, const Real &vol) const
    {
        if(!vol) return;
        Real muVol = this->Kparams[1]*vol;
        for(unsigned int i=0; i<dim; i++)             out[i]-=in[i]*muVol*2.0;
        for(unsigned int i=dim; i<size; i++)          out[i]-=in[i]*muVol;
        if(this->Kparams[0])
        {
            Real tce = in[0]; for(unsigned int i=1; i<dim; i++) tce += in[i];  tce *= this->Kparams[0]*vol;
            for(unsigned int i=0; i<dim; i++) out[i]-=tce;
        }
    }

    virtual Eigen::Matrix<Real,size,size,Eigen::RowMajor> assembleC(const Real &vol) const
    {
        typedef Eigen::Matrix<Real,size,size,Eigen::RowMajor> block;
        block C=block::Zero();
        if(!vol) return C;
        Real invvolE = 1./(vol*this->Cparams[0]);
        for(unsigned int i=0; i<dim; i++)  C(i,i)-=invvolE;
        for(unsigned int i=dim; i<size; i++) C(i,i)-= invvolE * (1+this->Cparams[1]) * 2.0;;
        for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<i; j++) C(i,j) = C(j,i) = invvolE *this-> Cparams[1];
        return C;
    }
};



/// isotropic viscosity 3D/2D (= isotropic law with zero poisson ratio)
template<class Real,unsigned int dim,unsigned int size>
class ViscosityHookeLaw: public HookeLaw<Real,dim,size>
{
public:
    virtual void set(const std::vector<Real> &cparams)
    {
        this->Cparams.clear(); this->Cparams.push_back(cparams[0]);
        this->Kparams.clear(); this->Kparams.push_back(0.5*cparams[0]);
    }

    virtual Eigen::Matrix<Real,size,size,Eigen::RowMajor> assembleK(const Real &vol) const
    {
        static_assert( dim<=size, "" );
        typedef Eigen::Matrix<Real,size,size,Eigen::RowMajor> block;
        block K=block::Zero();
        if(!vol) return K;
        Real muVol = this->Kparams[0]*vol;
        for(unsigned int i=0; i<dim; i++)  K(i,i)-=muVol*2.0;
        for(unsigned int i=dim; i<size; i++) K(i,i)-=muVol;
        return K;
    }

    virtual void applyK(Vec<size,Real> &out, const Vec<size,Real> &in, const Real &vol) const
    {
        if(!vol) return;
        Real muVol = this->Kparams[0]*vol;
        for(unsigned int i=0; i<dim; i++)             out[i]-=in[i]*muVol*2.0;
        for(unsigned int i=dim; i<size; i++)          out[i]-=in[i]*muVol;
    }

    virtual Eigen::Matrix<Real,size,size,Eigen::RowMajor> assembleC(const Real &vol) const
    {
        typedef Eigen::Matrix<Real,size,size,Eigen::RowMajor> block;
        block C=block::Zero();
        if(!vol) return C;
        Real invvolE = 1./(vol*this->Cparams[0]);
        for(unsigned int i=0; i<dim; i++)  C(i,i)-=invvolE;
        for(unsigned int i=dim; i<size; i++) C(i,i)-=invvolE*2.0;
        return C;
    }
};




/// Orthotropic 3D
template<class Real,unsigned int dim,unsigned int size>
class OrthotropicHookeLaw: public HookeLaw<Real,dim,size> {};

template<class Real>
class OrthotropicHookeLaw<Real,3,6>: public HookeLaw<Real,3,6>
{
public:
    virtual void set(const std::vector<Real> &cparams)
    {
        Real youngMx=cparams[0]     ,youngMy=cparams[1]     ,youngMz=cparams[2] ,
             poissonRxy=cparams[3]  ,poissonRyz=cparams[4]  ,poissonRzx=cparams[5] ,
             shearMxy=cparams[6]    ,shearMyz=cparams[7]    ,shearMzx=cparams[8];

        Real coeff=1./( -youngMx*youngMy*youngMz + youngMz*youngMy*youngMy*poissonRxy*poissonRxy + youngMx*youngMz*youngMz*poissonRyz*poissonRyz + youngMy*youngMx*youngMx*poissonRzx*poissonRzx + 2*youngMx*youngMy*youngMz*poissonRxy*poissonRyz*poissonRzx);
        Real C11=youngMz*youngMx*youngMx*(youngMz*poissonRyz*poissonRyz-youngMy)*coeff;
        Real C22=youngMx*youngMy*youngMy*(youngMx*poissonRzx*poissonRzx-youngMz)*coeff;
        Real C33=youngMy*youngMz*youngMz*(youngMy*poissonRxy*poissonRxy-youngMx)*coeff;
        Real C12=-youngMx*youngMy*youngMz*(youngMx*poissonRyz*poissonRzx + youngMy*poissonRxy)*coeff;
        Real C23=-youngMx*youngMy*youngMz*(youngMy*poissonRxy*poissonRzx + youngMz*poissonRyz)*coeff;
        Real C13=-youngMx*youngMy*youngMz*(youngMz*poissonRxy*poissonRyz + youngMx*poissonRzx)*coeff;;
        Real C44=shearMyz;
        Real C55=shearMzx;
        Real C66=shearMxy;

        this->Cparams.assign(cparams.begin(),cparams.end());
        this->Kparams.resize(9);
        this->Kparams[0]=C11; this->Kparams[1]=C22; this->Kparams[2]=C33;
        this->Kparams[3]=C12; this->Kparams[4]=C23; this->Kparams[5]=C13;
        this->Kparams[6]=C44; this->Kparams[7]=C55; this->Kparams[8]=C66;
    }

    virtual Eigen::Matrix<Real,6,6,Eigen::RowMajor> assembleK(const Real &vol) const
    {
        Eigen::Matrix<Real,6,6,Eigen::RowMajor> K;
        K<<    -vol*this->Kparams[0] ,-vol*this->Kparams[3] ,-vol*this->Kparams[5] , 0                      , 0                     , 0,
               -vol*this->Kparams[3] ,-vol*this->Kparams[1] ,-vol*this->Kparams[4] , 0                      , 0                     , 0,
               -vol*this->Kparams[5] ,-vol*this->Kparams[4] ,-vol*this->Kparams[2] , 0                      , 0                     , 0,
                0                    , 0                    , 0                    ,-vol*this->Kparams[6]   , 0                     , 0,
                0                    , 0                    , 0                    , 0                      ,-vol*this->Kparams[7]  , 0,
                0                    , 0                    , 0                    , 0                      , 0                     ,-vol*this->Kparams[8];
        return K;
    }

    virtual void applyK(Vec<6,Real> &out, const Vec<6,Real> &in, const Real &vol) const
    {
        if(!vol) return;
        out[0]-=vol*(this->Kparams[0]*in[0]+this->Kparams[3]*in[1]+this->Kparams[5]*in[2]);
        out[1]-=vol*(this->Kparams[3]*in[0]+this->Kparams[1]*in[1]+this->Kparams[4]*in[2]);
        out[2]-=vol*(this->Kparams[5]*in[0]+this->Kparams[4]*in[1]+this->Kparams[2]*in[2]);
        out[3]-=vol*this->Kparams[6]*in[3];
        out[4]-=vol*this->Kparams[7]*in[4];
        out[5]-=vol*this->Kparams[8]*in[5];
    }

    virtual Eigen::Matrix<Real,6,6,Eigen::RowMajor> assembleC(const Real &vol) const
    {
        Eigen::Matrix<Real,6,6,Eigen::RowMajor> C;
        C<<    -1./(vol*this->Cparams[0])                   ,  this->Cparams[3]/(vol*this->Cparams[0])  ,  this->Cparams[5]/(vol*this->Cparams[2])  , 0                    , 0                 , 0,
                this->Cparams[3]/(vol*this->Cparams[0])     , -1./(vol*this->Cparams[1])                ,  this->Cparams[4]/(vol*this->Cparams[1])  , 0                    , 0                 , 0,
                this->Cparams[5]/(vol*this->Cparams[2])     ,  this->Cparams[4]/(vol*this->Cparams[1])  , -1./(vol*this->Cparams[2])                , 0                    , 0                 , 0,
                0                        , 0                           , 0                           ,-1./(vol*vol*this->Cparams[7])    , 0                             , 0,
                0                        , 0                           , 0                           , 0                                ,-1./(vol*vol*this->Cparams[8]) , 0,
                0                        , 0                           , 0                           , 0                                , 0                             ,-1./(vol*this->Cparams[6]);
        return C;
    }
};



/// transverse isotropic 3D (supposing e1 is the axis of symmetry)
template<class Real,unsigned int dim,unsigned int size>
class TransverseHookeLaw: public HookeLaw<Real,dim,size> {};

template<typename Real>
class TransverseHookeLaw<Real,3,6>: public HookeLaw<Real,3,6>
{
public:
    virtual void set(const std::vector<Real> &cparams)
    {
        Real youngMx=cparams[0]     ,youngMy=cparams[1]     ,
             poissonRxy=cparams[2]  ,poissonRyz=cparams[3]  ,
             shearMxy=cparams[4];

        Real coeff1=1./( -youngMx + youngMx*poissonRyz + 2*youngMy*poissonRxy*poissonRxy);
        Real C11=youngMx*youngMx*(poissonRyz-1.)*coeff1;
        Real C12=-youngMx*youngMy*poissonRxy*coeff1;
        Real coeff2=1./( -youngMx + youngMx*poissonRyz*poissonRyz + 2*youngMy*poissonRxy*poissonRxy*(1+poissonRyz) );
        Real C22=youngMy*(youngMy*poissonRxy*poissonRxy-youngMx)*coeff2;
        Real C23=-youngMy*(youngMy*poissonRxy*poissonRxy+youngMx*poissonRyz)*coeff2;
        Real C55=shearMxy;

        this->Cparams.assign(cparams.begin(),cparams.end());
        this->Kparams.resize(5);
        this->Kparams[0]=C11; this->Kparams[1]=C22;
        this->Kparams[2]=C12; this->Kparams[3]=C23;
        this->Kparams[4]=C55;
    }

    virtual Eigen::Matrix<Real,6,6,Eigen::RowMajor> assembleK(const Real &vol) const
    {
        Eigen::Matrix<Real,6,6,Eigen::RowMajor> K;
        K<<    -vol*this->Kparams[0] ,-vol*this->Kparams[2] ,-vol*this->Kparams[2]  , 0                                               , 0                        , 0,
               -vol*this->Kparams[2] ,-vol*this->Kparams[1] ,-vol*this->Kparams[3]  , 0                                               , 0                        , 0,
               -vol*this->Kparams[2] ,-vol*this->Kparams[3] ,-vol*this->Kparams[1]  , 0                                               , 0                        , 0,
                0                    , 0                    , 0                     , -vol*(this->Kparams[1] - this->Kparams[3])*0.5  , 0                        , 0,
                0                    , 0                    , 0                     , 0                                               ,-vol*this->Kparams[4]     , 0,
                0                    , 0                    , 0                     , 0                                               , 0                        ,-vol*this->Kparams[4];
        return K;
    }

    virtual void applyK(Vec<6,Real> &out, const Vec<6,Real> &in, const Real &vol) const
    {
        if(!vol) return;
        out[0]-=vol*(this->Kparams[0]*in[0]+this->Kparams[2]*(in[1]+in[2]) ) ;
        out[1]-=vol*(this->Kparams[2]*in[0]+this->Kparams[1]*in[1]+this->Kparams[3]*in[2]);
        out[2]-=vol*(this->Kparams[2]*in[0]+this->Kparams[3]*in[1]+this->Kparams[1]*in[2]);
        out[3]-=vol*(this->Kparams[1] - this->Kparams[3])*in[3]*0.5;
        out[4]-=vol*this->Kparams[4]*in[4];
        out[5]-=vol*this->Kparams[4]*in[5];
    }

    virtual Eigen::Matrix<Real,6,6,Eigen::RowMajor> assembleC(const Real &vol) const
    {
        Eigen::Matrix<Real,6,6,Eigen::RowMajor> C;
        C<<    -1./(vol*this->Cparams[0])              ,  this->Cparams[2]/(vol*this->Cparams[0])  ,  this->Cparams[2]/(vol*this->Cparams[0])  , 0             , 0           , 0,
                this->Cparams[2]/(vol*this->Cparams[0]), -1./(vol*this->Cparams[1])                ,  this->Cparams[3]/(vol*this->Cparams[1])  , 0             , 0           , 0,
                this->Cparams[2]/(vol*this->Cparams[0]),  this->Cparams[3]/(vol*this->Cparams[1])  , -1./(vol*this->Cparams[1])                , 0             , 0           , 0,
                0                                       , 0                                          , 0                                       ,-2*(1+this->Cparams[3])/(vol*this->Cparams[1])   , 0                           , 0,
                0                                       , 0                                          , 0                                       , 0                                               ,-1./(vol*this->Cparams[4])   , 0,
                0                                       , 0                                          , 0                                       , 0                                               , 0                           ,-1./(vol*this->Cparams[4]);
        return C;
    }
};


/// Orthotropic 2D = transverse isotropic 2D

template<typename Real>
class OrthotropicHookeLaw<Real,2,3>: public HookeLaw<Real,2,3>
{
public:
    virtual void set(const std::vector<Real> &cparams)
    {
        Real youngMx=cparams[0]     ,youngMy=cparams[1],
             poissonRxy=cparams[2]  ,shearMxy=cparams[3];

        Real coeff=1./(youngMx-youngMy*poissonRxy*poissonRxy);
        Real C11=youngMx*youngMx*coeff;
        Real C22=youngMx*youngMy*coeff;
        Real C12=youngMx*youngMy*poissonRxy*coeff;
        Real C33=shearMxy;

        this->Cparams.assign(cparams.begin(),cparams.end());
        this->Kparams.resize(4);
        this->Kparams[0]=C11; this->Kparams[1]=C22;
        this->Kparams[2]=C12; this->Kparams[3]=C33;
    }

    virtual Eigen::Matrix<Real,3,3,Eigen::RowMajor> assembleK(const Real &vol) const
    {
        Eigen::Matrix<Real,3,3,Eigen::RowMajor> K;
        K<<    -vol*this->Kparams[0] ,-vol*this->Kparams[2] , 0,
               -vol*this->Kparams[2] ,-vol*this->Kparams[1] , 0,
                0                    , 0                    ,-vol*this->Kparams[3];
        return K;
    }

    virtual void applyK(Vec<3,Real> &out, const Vec<3,Real> &in, const Real &vol) const
    {
        if(!vol) return;
        out[0]-=vol*(this->Kparams[0]*in[0]+this->Kparams[2]*in[1]);
        out[1]-=vol*(this->Kparams[2]*in[0]+this->Kparams[1]*in[1]);
        out[2]-=vol*this->Kparams[3]*in[2];
    }

    virtual Eigen::Matrix<Real,3,3,Eigen::RowMajor> assembleC(const Real &vol) const
    {
        Eigen::Matrix<Real,3,3,Eigen::RowMajor> C;
        C<<    -1./(vol*this->Cparams[0])               ,  this->Cparams[2]/(vol*this->Cparams[0]) , 0,
                this->Cparams[2]/(vol*this->Cparams[0]) ,-1./(vol*this->Cparams[1])                , 0,
                0                                       , 0                                        ,-1./(vol*this->Cparams[3]);
        return C;
    }
};





//////////////////////////////////////////////////////////////////////////////////
////  Class HookeMaterialFactors
////  To store all factors needed by HookeMaterialBlock
////  These factors depends on the strain order
//////////////////////////////////////////////////////////////////////////////////

// order 0
template<int spatial_dimensions, int strain_size, class Real, int order> class HookeMaterialFactors
{
protected:

    Real _vol;  ///< volume

public:

    template<class VType> inline void set( const VType volume )
    {
        _vol = (*volume)[0];
    }

    inline const Real& vol() const { return _vol; }
    inline Real& vol() { return _vol; }

    inline Vec<spatial_dimensions,Real> order1() const { return Vec<spatial_dimensions,Real>(); } // need to exist to compile but should never be called
    inline Vec<strain_size,Real> order2() const { return Vec<strain_size,Real>(); } // need to exist to compile but should never be called
    inline Mat<spatial_dimensions,strain_size,Real> order3() const { return Mat<spatial_dimensions,strain_size,Real>(); } // need to exist to compile but should never be called
    inline MatSym<strain_size,Real> order4() const { return MatSym<strain_size,Real>(); } // need to exist to compile but should never be called
};

// order 1
template<int spatial_dimensions, int strain_size, class Real> class HookeMaterialFactors<spatial_dimensions,strain_size,Real,1> : public HookeMaterialFactors<spatial_dimensions,strain_size,Real,0>
{
protected:

    typedef HookeMaterialFactors<spatial_dimensions,strain_size,Real,0> Inherit;

    Vec<spatial_dimensions,Real> _order1;              ///< (i) corresponds to the volume factor val*gradient(i)
    Vec<strain_size,Real> _order2;                     ///< (i) corresponds to the volume factor val*hessian(i)

public:

    template<class VType> inline void set( const VType volume )
    {
        Inherit::set( volume );
        _order1 = getOrder1Factors(*volume);
        _order2 = getOrder2Factors(*volume);
    }

    inline const Vec<spatial_dimensions,Real>& order1() const { return _order1; }
    inline const Vec<strain_size,Real>& order2() const { return _order2; }
};

// order 2
template<int spatial_dimensions, int strain_size, class Real> class HookeMaterialFactors<spatial_dimensions,strain_size,Real,2> : public HookeMaterialFactors<spatial_dimensions,strain_size,Real,1>
{
protected:

    typedef HookeMaterialFactors<spatial_dimensions,strain_size,Real,1> Inherit;

    Mat<spatial_dimensions,strain_size,Real> _order3;  ///< (i,j) corresponds to the volume factor gradient(i)*hessian(j)
    MatSym<strain_size,Real> _order4;                  ///< (i,j) corresponds to the volume factor hessian(i)*hessian(j)

public:

    template<class VType> inline void set( const VType volume )
    {
        Inherit::set( volume );
        _order3 = getOrder3Factors(*volume);
        _order4 = getOrder4Factors(*volume);
    }

    inline const Mat<spatial_dimensions,strain_size,Real>& order3() const { return _order3; }
    inline const MatSym<strain_size,Real>& order4() const { return _order4; }
};


//////////////////////////////////////////////////////////////////////////////////
////  template implementation
//////////////////////////////////////////////////////////////////////////////////

/**
  Template class used to implement one material block for Hooke Material.

In matrix form :
           stress = lambda.trace(strain) I  + 2.mu.strain
           W = vol*(lambda/2 tr(strain)^2 + mu tr(strain^2))

       In Voigt notation: (e1=exx, e2=eyy, e3=ezz, e4=2exy, e5=2eyz, e6=2ezx, s1=sxx, s2=syy, s3=szz, s4=sxy, s5=syz, s6=szx)
            stress = H.strain
            W = vol*stress.strain/2
            f = - H.vol.strain - viscosity.vol.strainRate
            df = (- H.vol.kfactor - viscosity.vol.bfactor) dstrain

See http://en.wikipedia.org/wiki/Hooke%27s_law
  **/

template<class _StrainType, class LawType>
class HookeMaterialBlock :
    public  BaseMaterialBlock< _StrainType >
{
public:
    typedef _StrainType T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    enum { material_dimensions = T::material_dimensions };
    enum { strain_size = T::strain_size };
    enum { spatial_dimensions = T::spatial_dimensions };
    enum { order = T::order };
    typedef typename T::StrainVec StrainVec;

    typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;

    HookeMaterialFactors<spatial_dimensions,strain_size,Real,order> factors;

    static const bool constantK=true;

    /** store constants and basic operators */
    LawType hooke;
    ViscosityHookeLaw<Real,material_dimensions,strain_size> viscosity;


    /// Initialize based on the material parameters: Young modulus, Poisson Ratio, Lamé coefficients (which are redundant with Young modulus and Poisson ratio) and viscosity (stress/strain rate).
    void init( const std::vector<Real> &params, const Real &visc )
    {
        factors.vol()=1.;
        if(this->volume) factors.set( this->volume );
        hooke.set(params);
        std::vector<Real> v; v.push_back(visc); viscosity.set(v);
    }


    Real getPotentialEnergy(const Coord& x) const
    {
        Deriv f; const Deriv v;
        addForce( f , x , v );
        Real W=0;
        W-=dot(f.getStrain(),x.getStrain())*0.5;

        if( order > 0 )
        {
            for(unsigned int i=0; i<spatial_dimensions; i++) W-=dot(f.getStrainGradient(i),x.getStrainGradient(i))*0.5;

            if( order > 1 )
            {
                for(unsigned int i=0; i<spatial_dimensions; i++) for(unsigned int j=i; j<spatial_dimensions; j++) W-=dot(f.getStrainHessian(i,j),x.getStrainHessian(i,j))*0.5;
            }
        }
        return W;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& v) const
    {
        // order 0
        hooke.applyK(f.getStrain(),x.getStrain(),factors.vol());

        if( order > 0 )
        {
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)  hooke.applyK(f.getStrain(),x.getStrainGradient(i),factors.order1()[i]);
            for(unsigned int i=0; i<spatial_dimensions; i++)  hooke.applyK(f.getStrainGradient(i),x.getStrain(),factors.order1()[i]);
            // order 2
            unsigned int count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    hooke.applyK(f.getStrainGradient(i),x.getStrainGradient(j),factors.order2()[count]);
                    if(i!=j) hooke.applyK(f.getStrainGradient(j),x.getStrainGradient(i),factors.order2()[count]);
                    count++;
                }

            if( order > 1 )
            {
                count = 0;
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=i; j<spatial_dimensions; j++)
                    {
                        hooke.applyK(f.getStrain(),x.getStrainHessian(i,j),factors.order2()[count]);
                        hooke.applyK(f.getStrainHessian(i,j),x.getStrain(),factors.order2()[count]);
                        count++;
                    }
                // order 3
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        hooke.applyK(f.getStrainGradient(i),x.getStrainHessian(j),factors.order3()(i,j));
                        hooke.applyK(f.getStrainHessian(j),x.getStrainGradient(i),factors.order3()(i,j));
                    }
                // order 4
                for(unsigned int i=0; i<strain_size; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        hooke.applyK(f.getStrainHessian(i),x.getStrainHessian(j),factors.order4()(i,j));
                    }
            }
        }


        if(viscosity.Cparams[0])
        {
            // order 0
            viscosity.applyK(f.getStrain(),v.getStrain(),factors.vol());

            if( order > 0 )
            {
                // order 1
                for(unsigned int i=0; i<spatial_dimensions; i++)  viscosity.applyK(f.getStrain(),v.getStrainGradient(i),factors.order1()[i]);
                for(unsigned int i=0; i<spatial_dimensions; i++)  viscosity.applyK(f.getStrainGradient(i),v.getStrain(),factors.order1()[i]);
                // order 2
                unsigned int count =0;
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=i; j<spatial_dimensions; j++)
                    {
                        viscosity.applyK(f.getStrainGradient(i),v.getStrainGradient(j),factors.order2()[count]);
                        if(i!=j) viscosity.applyK(f.getStrainGradient(j),v.getStrainGradient(i),factors.order2()[count]);
                        count++;
                    }

                if( order > 1 )
                {
                    count =0;
                    for(unsigned int i=0; i<spatial_dimensions; i++)
                        for(unsigned int j=i; j<spatial_dimensions; j++)
                        {
                            viscosity.applyK(f.getStrain(),v.getStrainHessian(i,j),factors.order2()[count]);
                            viscosity.applyK(f.getStrainHessian(i,j),v.getStrain(),factors.order2()[count]);
                            count++;
                        }

                    // order 3
                    for(unsigned int i=0; i<spatial_dimensions; i++)
                        for(unsigned int j=0; j<strain_size; j++)
                        {
                            viscosity.applyK(f.getStrainGradient(i),v.getStrainHessian(j),factors.order3()(i,j));
                            viscosity.applyK(f.getStrainHessian(j),v.getStrainGradient(i),factors.order3()(i,j));
                        }
                    // order 4
                    for(unsigned int i=0; i<strain_size; i++)
                        for(unsigned int j=0; j<strain_size; j++)
                        {
                            viscosity.applyK(f.getStrainHessian(i),v.getStrainHessian(j),factors.order4()(i,j));
                        }
                }
            }
        }
    }

    void addDForce( Deriv&   df , const Deriv&   dx, const SReal& kfactor, const SReal& bfactor ) const
    {
        // order 0
        hooke.applyK(df.getStrain(),dx.getStrain(),factors.vol()*kfactor);

        if( order > 0 )
        {
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)  hooke.applyK(df.getStrain(),dx.getStrainGradient(i),factors.order1()[i]*kfactor);
            for(unsigned int i=0; i<spatial_dimensions; i++)  hooke.applyK(df.getStrainGradient(i),dx.getStrain(),factors.order1()[i]*kfactor);
            // order 2
            unsigned int count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    hooke.applyK(df.getStrainGradient(i),dx.getStrainGradient(j),factors.order2()[count]*kfactor);
                    if(i!=j) hooke.applyK(df.getStrainGradient(j),dx.getStrainGradient(i),factors.order2()[count]*kfactor);
                    count++;
                }
            if( order > 1 )
            {
                count = 0;
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=i; j<spatial_dimensions; j++)
                    {
                        hooke.applyK(df.getStrain(),dx.getStrainHessian(i,j),factors.order2()[count]*kfactor);
                        hooke.applyK(df.getStrainHessian(i,j),dx.getStrain(),factors.order2()[count]*kfactor);
                        count++;
                    }
                // order 3
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        hooke.applyK(df.getStrainGradient(i),dx.getStrainHessian(j),factors.order3()(i,j)*kfactor);
                        hooke.applyK(df.getStrainHessian(j),dx.getStrainGradient(i),factors.order3()(i,j)*kfactor);
                    }
                // order 4
                for(unsigned int i=0; i<strain_size; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        hooke.applyK(df.getStrainHessian(i),dx.getStrainHessian(j),factors.order4()(i,j)*kfactor);
                    }
            }
        }


        if(viscosity.Cparams[0])
        {
            // order 0
            viscosity.applyK(df.getStrain(),dx.getStrain(),factors.vol()*bfactor);

            if( order > 0 )
            {
                // order 1
                for(unsigned int i=0; i<spatial_dimensions; i++)  viscosity.applyK(df.getStrain(),dx.getStrainGradient(i),factors.order1()[i]*bfactor);
                for(unsigned int i=0; i<spatial_dimensions; i++)  viscosity.applyK(df.getStrainGradient(i),dx.getStrain(),factors.order1()[i]*bfactor);
                // order 2
                unsigned int count = 0;
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=i; j<spatial_dimensions; j++)
                    {
                        viscosity.applyK(df.getStrainGradient(i),dx.getStrainGradient(j),factors.order2()[count]*bfactor);
                        if(i!=j) viscosity.applyK(df.getStrainGradient(j),dx.getStrainGradient(i),factors.order2()[count]*bfactor);
                        count++;
                    }

                if( order > 1 )
                {
                    count = 0;
                    for(unsigned int i=0; i<spatial_dimensions; i++)
                        for(unsigned int j=i; j<spatial_dimensions; j++)
                        {
                            viscosity.applyK(df.getStrain(),dx.getStrainHessian(i,j),factors.order2()[count]*bfactor);
                            viscosity.applyK(df.getStrainHessian(i,j),dx.getStrain(),factors.order2()[count]*bfactor);
                            count++;
                        }
                    // order 3
                    for(unsigned int i=0; i<spatial_dimensions; i++)
                        for(unsigned int j=0; j<strain_size; j++)
                        {
                            viscosity.applyK(df.getStrainGradient(i),dx.getStrainHessian(j),factors.order3()(i,j)*bfactor);
                            viscosity.applyK(df.getStrainHessian(j),dx.getStrainGradient(i),factors.order3()(i,j)*bfactor);
                        }
                    // order 4
                    for(unsigned int i=0; i<strain_size; i++)
                        for(unsigned int j=0; j<strain_size; j++)
                        {
                            viscosity.applyK(df.getStrainHessian(i),dx.getStrainHessian(j),factors.order4()(i,j)*bfactor);
                        }
                }
            }
        }
    }



    MatBlock getK() const
    {
        MatBlock K = MatBlock();
        EigenMap eK(&K[0][0],MatBlock::nbLines,MatBlock::nbCols);

        // order 0
        eK.block(0,0,strain_size,strain_size) = hooke.assembleK(factors.vol());

        if( order > 0 )
        {
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)   eK.block(strain_size*(i+1),0,strain_size,strain_size) = hooke.assembleK(factors.order1()[i]);
            for(unsigned int i=0; i<spatial_dimensions; i++)   eK.block(0,strain_size*(i+1),strain_size,strain_size) = hooke.assembleK(factors.order1()[i]);
            // order 2
            unsigned int count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    eK.block(strain_size*(i+1),strain_size*(j+1),strain_size,strain_size) = hooke.assembleK(factors.order2()[count]);
                    if(i!=j) eK.block(strain_size*(j+1),strain_size*(i+1),strain_size,strain_size) = hooke.assembleK(factors.order2()[count]);
                    count++;
                }

            if( order > 1 )
            {
                unsigned int offset = (spatial_dimensions+1)*strain_size;
                for(unsigned int j=0; j<strain_size; j++)
                {
                    eK.block(0,offset+strain_size*j,strain_size,strain_size) = hooke.assembleK(factors.order2()[j]);
                    eK.block(offset+strain_size*j,0,strain_size,strain_size) = hooke.assembleK(factors.order2()[j]);
                }

                // order 3
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        eK.block(strain_size*(i+1),offset+strain_size*j,strain_size,strain_size) = hooke.assembleK(factors.order3()(i,j));
                        eK.block(offset+strain_size*j,strain_size*(i+1),strain_size,strain_size) = hooke.assembleK(factors.order3()(i,j));
                    }
                // order 4
                for(unsigned int i=0; i<strain_size; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        eK.block(offset+strain_size*i,offset+strain_size*j,strain_size,strain_size) = hooke.assembleK(factors.order4()(i,j));
                    }
            }
        }


        return K;
    }

    MatBlock getC() const
    {
        MatBlock C ;
        if( order > 0 ) C.invert(-getK());
        else
        {
            EigenMap eC(&C[0][0]);
            eC.block(0,0,strain_size,strain_size) = -hooke.assembleC(factors.vol());
        }
        return C;
    }

    MatBlock getB() const
    {
        MatBlock B = MatBlock();
        EigenMap eB(&B[0][0]);
        // order 0
        eB.block(0,0,strain_size,strain_size) = viscosity.assembleK(factors.vol());

        if( order > 0 )
        {
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)   eB.block(strain_size*(i+1),0,strain_size,strain_size) = viscosity.assembleK(factors.order1()[i]);
            for(unsigned int i=0; i<spatial_dimensions; i++)   eB.block(0,strain_size*(i+1),strain_size,strain_size) = viscosity.assembleK(factors.order1()[i]);
            // order 2
            unsigned int count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    eB.block(strain_size*(i+1),strain_size*(j+1),strain_size,strain_size) = viscosity.assembleK(factors.order2()[count]);
                    if(i!=j) eB.block(strain_size*(j+1),strain_size*(i+1),strain_size,strain_size) = viscosity.assembleK(factors.order2()[count]);
                    count++;
                }

            if( order > 1 )
            {
                unsigned int offset = (spatial_dimensions+1)*strain_size;
                for(unsigned int j=0; j<strain_size; j++)
                {
                    eB.block(0,offset+strain_size*j,strain_size,strain_size) = viscosity.assembleK(factors.order2()[j]);
                    eB.block(offset+strain_size*j,0,strain_size,strain_size) = viscosity.assembleK(factors.order2()[j]);
                }

                // order 3
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        eB.block(strain_size*(i+1),offset+strain_size*j,strain_size,strain_size) = viscosity.assembleK(factors.order3()(i,j));
                        eB.block(offset+strain_size*j,strain_size*(i+1),strain_size,strain_size) = viscosity.assembleK(factors.order3()(i,j));
                    }
                // order 4
                for(unsigned int i=0; i<strain_size; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        eB.block(offset+strain_size*i,offset+strain_size*j,strain_size,strain_size) = viscosity.assembleK(factors.order4()(i,j));
                    }
            }
        }
        return B;
    }


};





} // namespace defaulttype
} // namespace sofa



#endif
