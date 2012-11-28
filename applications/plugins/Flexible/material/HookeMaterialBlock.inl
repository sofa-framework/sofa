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
////  helpers
//////////////////////////////////////////////////////////////////////////////////

template<class Real,int dim,int size>
static Eigen::Matrix<Real,size,size,Eigen::RowMajor> assembleK_Isotropic(const Real &mu,const Real &lambda,const Real &vol)
{
    typedef Eigen::Matrix<Real,size,size,Eigen::RowMajor> block;
    block K=block::Zero();
    if(!vol) return K;
    Real muVol = mu*vol;
    for(unsigned int i=0; i<dim; i++)  K(i,i)-=muVol*2.0;
    for(unsigned int i=dim; i<size; i++) K(i,i)-=muVol;
    for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<dim; j++) K(i,j)-=lambda*vol;
    return K;
}

template<class Real,int dim,int size>
void applyK_Isotropic(Vec<size,Real> &out, const Vec<size,Real> &in, const Real &mu,const Real &lambda,const Real &vol)
{
    if(!vol) return;
    Real muVol = mu*vol;
    for(unsigned int i=0; i<dim; i++)             out[i]-=in[i]*muVol*2.0;
    for(unsigned int i=dim; i<size; i++)          out[i]-=in[i]*muVol;
    if(lambda)
    {
        Real tce = in[0]; for(unsigned int i=1; i<dim; i++) tce += in[i];  tce *= lambda*vol;
        for(unsigned int i=0; i<dim; i++) out[i]-=tce;
    }
}


template<class Real,int dim,int size>
static Eigen::Matrix<Real,size,size,Eigen::RowMajor> assembleC_Isotropic(const Real &youngM,const Real &poissonR,const Real &vol)
{
    typedef Eigen::Matrix<Real,size,size,Eigen::RowMajor> block;
    block C=block::Zero();
    if(!vol) return C;
    Real invvolE = 1./(vol*youngM);
    for(unsigned int i=0; i<dim; i++)  C(i,i)-=invvolE;
    for(unsigned int i=dim; i<size; i++) C(i,i)-= 2 * invvolE * (1+poissonR);
    for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<i; j++) C(i,j) = C(j,i) = invvolE * poissonR;
    return C;
}


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

template<class _StrainType>
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

    //@{
    /** Constants for the stiffness matrix */
    Real lamd;  ///< Lamé first coef
    Real mu;  ///< Lamé second coef
    Real viscosity;  ///< stress/strain rate
    //@}

    //@{
    /** Constants for the compliance matrix */
    Real youngModulus;
    Real poissonRatio;
    //@}

    /// Initialize based on the material parameters: Young modulus, Poisson Ratio, Lamé coefficients (which are redundant with Young modulus and Poisson ratio) and viscosity (stress/strain rate).
    void init( const Real &youngM, const Real& poissonR, const Real &visc )
    {
        factors.vol()=1.;
        if(this->volume) factors.set( this->volume );

        // used in compliance:
        youngModulus = youngM;
        poissonRatio = poissonR;

        // lame coef. Used in stiffness:
        lamd = youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        mu = 0.5*youngM/(1+poissonR);
        viscosity = visc;
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
        applyK_Isotropic<Real,material_dimensions>(f.getStrain(),x.getStrain(),mu,lamd,factors.vol());

        if( order > 0 )
        {
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrain(),x.getStrainGradient(i),mu,lamd,factors.order1()[i]);
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainGradient(i),x.getStrain(),mu,lamd,factors.order1()[i]);
            // order 2
            unsigned int count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainGradient(i),x.getStrainGradient(j),mu,lamd,factors.order2()[count]);
                    if(i!=j) applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainGradient(j),x.getStrainGradient(i),mu,lamd,factors.order2()[count]);
                    count++;
                }

            if( order > 1 )
            {
                count = 0;
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=i; j<spatial_dimensions; j++)
                    {
                        applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrain(),x.getStrainHessian(i,j),mu,lamd,factors.order2()[count]);
                        applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainHessian(i,j),x.getStrain(),mu,lamd,factors.order2()[count]);
                        count++;
                    }
                // order 3
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainGradient(i),x.getStrainHessian(j),mu,lamd,factors.order3()(i,j));
                        applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainHessian(j),x.getStrainGradient(i),mu,lamd,factors.order3()(i,j));
                    }
                // order 4
                for(unsigned int i=0; i<strain_size; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainHessian(i),x.getStrainHessian(j),mu,lamd,factors.order4()(i,j));
                        applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainHessian(j),x.getStrainHessian(i),mu,lamd,factors.order4()(i,j));
                    }
            }
        }


        if(viscosity)
        {
            // order 0
            applyK_Isotropic<Real,material_dimensions>(f.getStrain(),v.getStrain(),viscosity,0,factors.vol());

            if( order > 0 )
            {
                // order 1
                for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrain(),v.getStrainGradient(i),viscosity,0,factors.order1()[i]);
                for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainGradient(i),v.getStrain(),viscosity,0,factors.order1()[i]);
                // order 2
                unsigned int count =0;
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=i; j<spatial_dimensions; j++)
                    {
                        applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainGradient(i),v.getStrainGradient(j),viscosity,0,factors.order2()[count]);
                        if(i!=j) applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainGradient(j),v.getStrainGradient(i),viscosity,0,factors.order2()[count]);
                        count++;
                    }

                if( order > 1 )
                {
                    count =0;
                    for(unsigned int i=0; i<spatial_dimensions; i++)
                        for(unsigned int j=i; j<spatial_dimensions; j++)
                        {
                            applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrain(),v.getStrainHessian(i,j),viscosity,0,factors.order2()[count]);
                            applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainHessian(i,j),v.getStrain(),viscosity,0,factors.order2()[count]);
                            count++;
                        }

                    // order 3
                    for(unsigned int i=0; i<spatial_dimensions; i++)
                        for(unsigned int j=0; j<strain_size; j++)
                        {
                            applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainGradient(i),v.getStrainHessian(j),viscosity,0,factors.order3()(i,j));
                            applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainHessian(j),v.getStrainGradient(i),viscosity,0,factors.order3()(i,j));
                        }
                    // order 4
                    for(unsigned int i=0; i<strain_size; i++)
                        for(unsigned int j=0; j<strain_size; j++)
                        {
                            applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainHessian(i),v.getStrainHessian(j),viscosity,0,factors.order4()(i,j));
                            applyK_Isotropic<Real,material_dimensions,strain_size>(f.getStrainHessian(j),v.getStrainHessian(i),viscosity,0,factors.order4()(i,j));
                        }
                }
            }
        }
    }

    void addDForce( Deriv&   df , const Deriv&   dx, const double& kfactor, const double& bfactor ) const
    {
        // order 0
        applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrain(),dx.getStrain(),mu,lamd,factors.vol()*kfactor);

        if( order > 0 )
        {
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrain(),dx.getStrainGradient(i),mu,lamd,factors.order1()[i]*kfactor);
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainGradient(i),dx.getStrain(),mu,lamd,factors.order1()[i]*kfactor);
            // order 2
            unsigned int count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainGradient(i),dx.getStrainGradient(j),mu,lamd,factors.order2()[count]*kfactor);
                    if(i!=j) applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainGradient(j),dx.getStrainGradient(i),mu,lamd,factors.order2()[count]*kfactor);
                    count++;
                }
            if( order > 1 )
            {
                count = 0;
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=i; j<spatial_dimensions; j++)
                    {
                        applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrain(),dx.getStrainHessian(i,j),mu,lamd,factors.order2()[count]*kfactor);
                        applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainHessian(i,j),dx.getStrain(),mu,lamd,factors.order2()[count]*kfactor);
                        count++;
                    }
                // order 3
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainGradient(i),dx.getStrainHessian(j),mu,lamd,factors.order3()(i,j)*kfactor);
                        applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainHessian(j),dx.getStrainGradient(i),mu,lamd,factors.order3()(i,j)*kfactor);
                    }
                // order 4
                for(unsigned int i=0; i<strain_size; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainHessian(i),dx.getStrainHessian(j),mu,lamd,factors.order4()(i,j)*kfactor);
                        applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainHessian(j),dx.getStrainHessian(i),mu,lamd,factors.order4()(i,j)*kfactor);
                    }
            }
        }


        if(viscosity)
        {
            // order 0
            applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),viscosity,0,factors.vol()*bfactor);

            if( order > 0 )
            {
                // order 1
                for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrain(),dx.getStrainGradient(i),viscosity,0,factors.order1()[i]*bfactor);
                for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainGradient(i),dx.getStrain(),viscosity,0,factors.order1()[i]*bfactor);
                // order 2
                unsigned int count = 0;
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=i; j<spatial_dimensions; j++)
                    {
                        applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainGradient(i),dx.getStrainGradient(j),viscosity,0,factors.order2()[count]*bfactor);
                        if(i!=j) applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainGradient(j),dx.getStrainGradient(i),viscosity,0,factors.order2()[count]*bfactor);
                        count++;
                    }

                if( order > 1 )
                {
                    count = 0;
                    for(unsigned int i=0; i<spatial_dimensions; i++)
                        for(unsigned int j=i; j<spatial_dimensions; j++)
                        {
                            applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrain(),dx.getStrainHessian(i,j),viscosity,0,factors.order2()[count]*bfactor);
                            applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainHessian(i,j),dx.getStrain(),viscosity,0,factors.order2()[count]*bfactor);
                            count++;
                        }
                    // order 3
                    for(unsigned int i=0; i<spatial_dimensions; i++)
                        for(unsigned int j=0; j<strain_size; j++)
                        {
                            applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainGradient(i),dx.getStrainHessian(j),viscosity,0,factors.order3()(i,j)*bfactor);
                            applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainHessian(j),dx.getStrainGradient(i),viscosity,0,factors.order3()(i,j)*bfactor);
                        }
                    // order 4
                    for(unsigned int i=0; i<strain_size; i++)
                        for(unsigned int j=0; j<strain_size; j++)
                        {
                            applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainHessian(i),dx.getStrainHessian(j),viscosity,0,factors.order4()(i,j)*bfactor);
                            applyK_Isotropic<Real,material_dimensions,strain_size>(df.getStrainHessian(j),dx.getStrainHessian(i),viscosity,0,factors.order4()(i,j)*bfactor);
                        }
                }
            }
        }
    }



    MatBlock getK() const
    {
        MatBlock K = MatBlock();
        EigenMap eK(&K[0][0]);
        // order 0
        eK.block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.vol());

        if( order > 0 )
        {
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)   eK.block(strain_size*(i+1),0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order1()[i]);
            for(unsigned int i=0; i<spatial_dimensions; i++)   eK.block(0,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order1()[i]);
            // order 2
            unsigned int count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    eK.block(strain_size*(i+1),strain_size*(j+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order2()[count]);
                    if(i!=j) eK.block(strain_size*(j+1),strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order2()[count]);
                    count++;
                }

            if( order > 1 )
            {
                unsigned int offset = (spatial_dimensions+1)*strain_size;
                for(unsigned int j=0; j<strain_size; j++)
                {
                    eK.block(0,offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order2()[j]);
                    eK.block(offset+strain_size*j,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order2()[j]);
                }

                // order 3
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        eK.block(strain_size*(i+1),offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order3()(i,j));
                        eK.block(offset+strain_size*j,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order3()(i,j));
                    }
                // order 4
                for(unsigned int i=0; i<strain_size; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        eK.block(offset+strain_size*i,offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order4()(i,j));
                        eK.block(offset+strain_size*j,offset+strain_size*i,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(mu,lamd,factors.order4()(i,j));
                    }
            }
        }
        return K;
    }

    MatBlock getC() const
    {
        // TO DO: check why C need to be multiplied by -1
        MatBlock C ;
        if( order > 0 ) C.invert(-getK());
        else
        {
            EigenMap eC(&C[0][0]);
            eC.block(0,0,strain_size,strain_size) = -assembleC_Isotropic<Real,material_dimensions,strain_size>(youngModulus,poissonRatio,factors.vol());
        }
        return C;
    }

    MatBlock getB() const
    {
        MatBlock B = MatBlock();
        EigenMap eB(&B[0][0]);
        // order 0
        eB.block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.vol());

        if( order > 0 )
        {
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)   eB.block(strain_size*(i+1),0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order1()[i]);
            for(unsigned int i=0; i<spatial_dimensions; i++)   eB.block(0,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order1()[i]);
            // order 2
            unsigned int count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    eB.block(strain_size*(i+1),strain_size*(j+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order2()[count]);
                    if(i!=j) eB.block(strain_size*(j+1),strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order2()[count]);
                    count++;
                }

            if( order > 1 )
            {
                unsigned int offset = (spatial_dimensions+1)*strain_size;
                for(unsigned int j=0; j<strain_size; j++)
                {
                    eB.block(0,offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order2()[j]);
                    eB.block(offset+strain_size*j,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order2()[j]);
                }

                // order 3
                for(unsigned int i=0; i<spatial_dimensions; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        eB.block(strain_size*(i+1),offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order3()(i,j));
                        eB.block(offset+strain_size*j,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order3()(i,j));
                    }
                // order 4
                for(unsigned int i=0; i<strain_size; i++)
                    for(unsigned int j=0; j<strain_size; j++)
                    {
                        eB.block(offset+strain_size*i,offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order4()(i,j));
                        eB.block(offset+strain_size*j,offset+strain_size*i,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions,strain_size>(viscosity,0,factors.order4()(i,j));
                    }
            }
        }
        return B;
    }


};






} // namespace defaulttype
} // namespace sofa



#endif
