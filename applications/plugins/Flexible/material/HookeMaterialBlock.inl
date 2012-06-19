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
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define E221(type)  StrainTypes<2,2,0,type>
#define E331(type)  StrainTypes<3,3,0,type>
#define E332(type)  StrainTypes<3,3,1,type>
#define E333(type)  StrainTypes<3,3,2,type>

#define D221(type)  DiagonalizedStrainTypes<2,2,0,type>
#define D331(type)  DiagonalizedStrainTypes<3,3,0,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////
template<class Real,int dim>
static Eigen::Matrix<Real,dim*(dim+1)/2,dim*(dim+1)/2,Eigen::RowMajor> assembleK_Isotropic(const Real &mu,const Real &lambda,const Real &vol)
{
    static const unsigned int size=dim*(dim+1)/2;
    typedef Eigen::Matrix<Real,size,size,Eigen::RowMajor> block;
    block K=block::Zero();
    if(!vol) return K;
    Real muVol = mu*vol;
    for(unsigned int i=0; i<dim; i++)  K(i,i)-=muVol*2.0;
    for(unsigned int i=dim; i<size; i++) K(i,i)-=muVol;
    for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<dim; j++) K(i,j)-=lambda*vol;
    return K;
}

template<class Real,int dim>
void applyK_Isotropic(Vec<dim*(dim+1)/2,Real> &out, const Vec<dim*(dim+1)/2,Real> &in, const Real &mu,const Real &lambda,const Real &vol)
{
    if(!vol) return;
    static const unsigned int size=dim*(dim+1)/2;
    Real muVol = mu*vol;
    for(unsigned int i=0; i<dim; i++)             out[i]-=in[i]*muVol*2.0;
    for(unsigned int i=dim; i<size; i++)          out[i]-=in[i]*muVol;
    if(lambda)
    {
        Real tce = in[0]; for(unsigned int i=1; i<dim; i++) tce += in[i];  tce *= lambda*vol;
        for(unsigned int i=0; i<dim; i++) out[i]-=tce;
    }
}


template<class Real,int dim>
static Eigen::Matrix<Real,dim*(dim+1)/2,dim*(dim+1)/2,Eigen::RowMajor> assembleC_Isotropic(const Real &youngM,const Real &poissonR,const Real &vol)
{
    static const unsigned int size=dim*(dim+1)/2;
    typedef Eigen::Matrix<Real,size,size,Eigen::RowMajor> block;
    block C=block::Zero();
    if(!vol) return C;
    Real volOverE = vol/youngM;
    for(unsigned int i=0; i<dim; i++)  C(i,i)-=volOverE;
    for(unsigned int i=dim; i<size; i++) C(i,i)-= 2 * volOverE * (1+poissonR);
    for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<i; j++) C(i,j) = C(j,i) = volOverE * poissonR;
    return C;
}




//////////////////////////////////////////////////////////////////////////////////
////  E331
//////////////////////////////////////////////////////////////////////////////////

/**
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

template<class _Real>
class HookeMaterialBlock< E331(_Real) > :
    public  BaseMaterialBlock< E331(_Real) >
{
public:
    typedef E331(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    enum { material_dimensions = T::material_dimensions };
    enum { strain_size = T::strain_size };
    enum { spatial_dimensions = T::spatial_dimensions };
    typedef typename T::StrainVec StrainVec;

    static const bool constantK=true;

    Real vol;  ///< volume

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
        vol=1.;
        if(this->volume) vol=(*this->volume)[0];

        // used in compliance:
        youngModulus = youngM;
        poissonRatio = poissonR;

        // lame coef. Used in stiffness:
        lamd = youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        mu = 0.5*youngM/(1+poissonR);
        viscosity = visc;
    }

    Real getPotentialEnergy(const Coord& x)
    {
        Deriv f,v;
        addForce( f , x , v);
        Real W=0;
        W-=dot(f.getStrain(),x.getStrain())*0.5;
        return W;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& v)
    {
        applyK_Isotropic<Real,material_dimensions>(f.getStrain(),x.getStrain(),mu,lamd,vol);
        applyK_Isotropic<Real,material_dimensions>(f.getStrain(),v.getStrain(),viscosity,0.0,vol);
    }

    void addDForce( Deriv&   df , const Deriv&   dx, const double& kfactor, const double& bfactor )
    {
        applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),mu,lamd,vol*kfactor);
        applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),viscosity,0.0,vol*bfactor);
    }


    MatBlock getK()
    {
        MatBlock K = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eK(&K[0][0]);
        eK.template block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,vol);
        return K;
    }

    MatBlock getC()
    {
        MatBlock C = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eC(&C[0][0]);
        eC.template block(0,0,strain_size,strain_size) = assembleC_Isotropic<Real,material_dimensions>(youngModulus,poissonRatio,vol);
        return C;
    }

    MatBlock getB()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        eB.template block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,vol);
        return B;
    }
};




//////////////////////////////////////////////////////////////////////////////////
////  E221
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class HookeMaterialBlock< E221(_Real) > :
    public  BaseMaterialBlock< E221(_Real) >
{
public:
    typedef E221(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    enum { material_dimensions = T::material_dimensions };
    enum { strain_size = T::strain_size };
    enum { spatial_dimensions = T::spatial_dimensions };
    typedef typename T::StrainVec StrainVec;

    static const bool constantK=true;

    Real vol;  ///< volume

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
        vol=1.;
        if(this->volume) vol=(*this->volume)[0];

        // used in compliance:
        youngModulus = youngM;
        poissonRatio = poissonR;

        // lame coef. Used in stiffness:
        lamd = youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        mu = 0.5*youngM/(1+poissonR);
        viscosity = visc;
    }

    Real getPotentialEnergy(const Coord& x)
    {
        Deriv f,v;
        addForce( f , x , v);
        Real W=0;
        W-=dot(f.getStrain(),x.getStrain())*0.5;
        return W;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& v)
    {
        applyK_Isotropic<Real,material_dimensions>(f.getStrain(),x.getStrain(),mu,lamd,vol);
        applyK_Isotropic<Real,material_dimensions>(f.getStrain(),v.getStrain(),viscosity,0.0,vol);
    }

    void addDForce( Deriv&   df , const Deriv&   dx, const double& kfactor, const double& bfactor )
    {
        applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),mu,lamd,vol*kfactor);
        applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),viscosity,0.0,vol*bfactor);
    }


    MatBlock getK()
    {
        MatBlock K = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eK(&K[0][0]);
        eK.template block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,vol);
        return K;
    }

    MatBlock getC()
    {
        MatBlock C = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eC(&C[0][0]);
        eC.template block(0,0,strain_size,strain_size) = assembleC_Isotropic<Real,material_dimensions>(youngModulus,poissonRatio,vol);
        return C;
    }

    MatBlock getB()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        eB.template block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,vol);
        return B;
    }
};


////////////////////////////////////////////////////////////////////////////////////
//////  specialization for DiagonalizedStrainTypes with a bit less computations...
////////////////////////////////////////////////////////////////////////////////////

//template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real >
//class HookeMaterialBlock< DiagonalizedStrainTypes<_spatial_dimensions,_material_dimensions,_order,_Real> > : public HookeMaterialBlock< StrainTypes<_spatial_dimensions,_material_dimensions,_order,_Real> >
//{
//public:

//    typedef DiagonalizedStrainTypes<_spatial_dimensions,_material_dimensions,_order,_Real> T;

//    typedef BaseMaterialBlock<T> Inherit;
//    typedef typename Inherit::Coord Coord;
//    typedef typename Inherit::Deriv Deriv;
//    typedef typename Inherit::MatBlock MatBlock;
//    typedef typename Inherit::Real Real;

//    enum { material_dimensions = T::material_dimensions };
//    enum { strain_size = T::strain_size };
//    enum { spatial_dimensions = T::spatial_dimensions };
//    typedef typename T::StrainVec StrainVec;

//    Real getPotentialEnergy(const Coord& x)
//    {
//        StrainVec stress;
//        for(unsigned int i=0;i<material_dimensions;i++)             stress[i]-=x.getStrain()[i]*this->mu2Vol;
//        for(unsigned int i=material_dimensions;i<strain_size;i++)   stress[i]-=(x.getStrain()[i]*this->mu2Vol)*0.5;
//        Real tce = x.getStrain()[0];
//        for(unsigned int i=1;i<material_dimensions;i++) tce += x.getStrain()[i];
//        tce *= this->lambdaVol;
//        for(unsigned int i=0;i<material_dimensions;i++) stress[i]+=tce;
//        Real W=dot(stress,x.getStrain())*0.5;
//        return W;
//    }

//    void addForce( Deriv& f , const Coord& x , const Deriv& v)
//    {
//        for(unsigned int i=0;i<material_dimensions;i++)             f.getStrain()[i]-=x.getStrain()[i]*this->mu2Vol + v.getStrain()[i]*this->viscosityVol;
//        for(unsigned int i=material_dimensions;i<strain_size;i++)   f.getStrain()[i]-=( /*x.getStrain()[i]*this->mu2Vol +*/ v.getStrain()[i]*this->viscosityVol)*0.5; // these strain entries are null
//        Real tce = x.getStrain()[0];
//        for(unsigned int i=1;i<material_dimensions;i++) tce += x.getStrain()[i];
//        tce *= this->lambdaVol;
//        for(unsigned int i=0;i<material_dimensions;i++) f.getStrain()[i]-=tce;
//    }

//    void addDForce( Deriv& df, const Deriv& dx, const double& kfactor, const double& bfactor )
//    {
//        for(unsigned int i=0;i<material_dimensions;i++)             df.getStrain()[i]-=dx.getStrain()[i]*this->mu2Vol*kfactor + dx.getStrain()[i]*this->viscosityVol*bfactor;
//        for(unsigned int i=material_dimensions;i<strain_size;i++)   df.getStrain()[i]-=(dx.getStrain()[i]*this->mu2Vol*kfactor + dx.getStrain()[i]*this->viscosityVol*bfactor)*0.5;
//        Real tce = dx.getStrain()[0];
//        for(unsigned int i=1;i<material_dimensions;i++) tce += dx.getStrain()[i];
//        tce *= this->lambdaVol * kfactor;
//        for(unsigned int i=0;i<material_dimensions;i++) df.getStrain()[i]-=tce;
//    }

//};



//////////////////////////////////////////////////////////////////////////////////
////  E332
//////////////////////////////////////////////////////////////////////////////////

template<class _Real>
class HookeMaterialBlock< E332(_Real) > :
    public  BaseMaterialBlock< E332(_Real) >
{
public:
    typedef E332(_Real) T;


    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    enum { material_dimensions = T::material_dimensions };
    enum { strain_size = T::strain_size };
    enum { spatial_dimensions = T::spatial_dimensions };
    typedef typename T::StrainVec StrainVec;

    static const bool constantK=true;

    Real vol;  ///< volume
    Vec<spatial_dimensions,Real> order1factors;              ///< (i) corresponds to the volume factor val*gradient(i)
    Vec<strain_size,Real> order2factors;                     ///< (i) corresponds to the volume factor val*hessian(i)

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
        vol=1.;
        if(this->volume)
        {
            vol=(*this->volume)[0];
            order1factors=getOrder1Factors(*this->volume);
            order2factors=getOrder2Factors(*this->volume);
        }

        // used in compliance:
        youngModulus = youngM;
        poissonRatio = poissonR;

        // lame coef. Used in stiffness:
        lamd = youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        mu = 0.5*youngM/(1+poissonR);
        viscosity = visc;
    }


    Real getPotentialEnergy(const Coord& x)
    {
        Deriv f,v;
        addForce( f , x , v);
        Real W=0;
        W-=dot(f.getStrain(),x.getStrain())*0.5;
        for(unsigned int i=0; i<spatial_dimensions; i++) W-=dot(f.getStrainGradient(i),x.getStrainGradient(i))*0.5;
        return W;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& v)
    {
        // order 0
        applyK_Isotropic<Real,material_dimensions>(f.getStrain(),x.getStrain(),mu,lamd,vol);
        // order 1
        for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(f.getStrain(),x.getStrainGradient(i),mu,lamd,order1factors[i]);
        for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),x.getStrain(),mu,lamd,order1factors[i]);
        // order 2
        unsigned int count = 0;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=i; j<spatial_dimensions; j++)
            {
                applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),x.getStrainGradient(j),mu,lamd,order2factors[count]);
                if(i!=j) applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(j),x.getStrainGradient(i),mu,lamd,order2factors[count]);
                count++;
            }

        if(viscosity)
        {
            // order 0
            applyK_Isotropic<Real,material_dimensions>(f.getStrain(),v.getStrain(),viscosity,0,vol);
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(f.getStrain(),v.getStrainGradient(i),viscosity,0,order1factors[i]);
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),v.getStrain(),viscosity,0,order1factors[i]);
            // order 2
            count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),v.getStrainGradient(j),viscosity,0,order2factors[count]);
                    if(i!=j) applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(j),v.getStrainGradient(i),viscosity,0,order2factors[count]);
                    count++;
                }
        }
    }

    void addDForce( Deriv&   df , const Deriv&   dx, const double& kfactor, const double& bfactor )
    {
        // order 0
        applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),mu,lamd,vol*kfactor);
        // order 1
        for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrainGradient(i),mu,lamd,order1factors[i]*kfactor);
        for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrain(),mu,lamd,order1factors[i]*kfactor);
        // order 2
        unsigned int count = 0;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=i; j<spatial_dimensions; j++)
            {
                applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrainGradient(j),mu,lamd,order2factors[count]*kfactor);
                if(i!=j) applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(j),dx.getStrainGradient(i),mu,lamd,order2factors[count]*kfactor);
                count++;
            }

        if(viscosity)
        {
            // order 0
            applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),viscosity,0,vol*bfactor);
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrainGradient(i),viscosity,0,order1factors[i]*bfactor);
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrain(),viscosity,0,order1factors[i]*bfactor);
            // order 2
            count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrainGradient(j),viscosity,0,order2factors[count]*bfactor);
                    if(i!=j) applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(j),dx.getStrainGradient(i),viscosity,0,order2factors[count]*bfactor);
                    count++;
                }
        }
    }


    MatBlock getK()
    {
        MatBlock K = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eK(&K[0][0]);
        // order 0
        eK.template block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,vol);
        // order 1
        for(unsigned int i=0; i<spatial_dimensions; i++)   eK.template block(strain_size*(i+1),0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order1factors[i]);
        for(unsigned int i=0; i<spatial_dimensions; i++)   eK.template block(0,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order1factors[i]);
        // order 2
        unsigned int count = 0;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=i; j<spatial_dimensions; j++)
            {
                eK.template block(strain_size*(i+1),strain_size*(j+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order2factors[count]);
                if(i!=j) eK.template block(strain_size*(j+1),strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order2factors[count]);
                count++;
            }
        return K;
    }

    MatBlock getC()
    {
        MatBlock C ;
        C.invert(getK());
        return C;
    }

    MatBlock getB()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        eB.template block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,vol);
        // order 1
        for(unsigned int i=0; i<spatial_dimensions; i++)   eB.template block(strain_size*(i+1),0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order1factors[i]);
        for(unsigned int i=0; i<spatial_dimensions; i++)   eB.template block(0,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order1factors[i]);
        // order 2
        unsigned int count = 0;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=i; j<spatial_dimensions; j++)
            {
                eB.template block(strain_size*(i+1),strain_size*(j+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order2factors[count]);
                if(i!=j) eB.template block(strain_size*(j+1),strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order2factors[count]);
                count++;
            }
        return B;
    }


};




//////////////////////////////////////////////////////////////////////////////////
////  E333
//////////////////////////////////////////////////////////////////////////////////


template<class _Real>
class HookeMaterialBlock< E333(_Real) > :
    public  BaseMaterialBlock< E333(_Real) >
{
public:
    typedef E333(_Real) T;

    typedef BaseMaterialBlock<T> Inherit;
    typedef typename Inherit::Coord Coord;
    typedef typename Inherit::Deriv Deriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::Real Real;

    enum { material_dimensions = T::material_dimensions };
    enum { strain_size = T::strain_size };
    enum { spatial_dimensions = T::spatial_dimensions };
    typedef typename T::StrainVec StrainVec;

    static const bool constantK=true;

    Real vol;  ///< volume
    Vec<spatial_dimensions,Real> order1factors;              ///< (i) corresponds to the volume factor val*gradient(i)
    Vec<strain_size,Real> order2factors;                     ///< (i) corresponds to the volume factor val*hessian(i)
    Mat<spatial_dimensions,strain_size,Real> order3factors;  ///< (i,j) corresponds to the volume factor gradient(i)*hessian(j)
    MatSym<strain_size,Real> order4factors;                  ///< (i,j) corresponds to the volume factor hessian(i)*hessian(j)

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
        vol=1.;
        if(this->volume)
        {
            vol=(*this->volume)[0];
            order1factors=getOrder1Factors(*this->volume);
            order2factors=getOrder2Factors(*this->volume);
            order3factors=getOrder3Factors(*this->volume);
            order4factors=getOrder4Factors(*this->volume);
        }

        // used in compliance:
        youngModulus = youngM;
        poissonRatio = poissonR;

        // lame coef. Used in stiffness:
        lamd = youngM*poissonR/((1-2*poissonR)*(1+poissonR)) ;
        mu = 0.5*youngM/(1+poissonR);
        viscosity = visc;
    }


    Real getPotentialEnergy(const Coord& x)
    {
        Deriv f; const Deriv v;
        addForce( f , x , v);
        Real W=0;
        W-=dot(f.getStrain(),x.getStrain())*0.5;
        for(unsigned int i=0; i<spatial_dimensions; i++) W-=dot(f.getStrainGradient(i),x.getStrainGradient(i))*0.5;
        for(unsigned int i=0; i<spatial_dimensions; i++) for(unsigned int j=i; j<spatial_dimensions; j++) W-=dot(f.getStrainHessian(i,j),x.getStrainHessian(i,j))*0.5;
        return W;
    }

    void addForce( Deriv& f , const Coord& x , const Deriv& v)
    {
        // order 0
        applyK_Isotropic<Real,material_dimensions>(f.getStrain(),x.getStrain(),mu,lamd,vol);
        // order 1
        for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(f.getStrain(),x.getStrainGradient(i),mu,lamd,order1factors[i]);
        for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),x.getStrain(),mu,lamd,order1factors[i]);
        // order 2
        unsigned int count = 0;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=i; j<spatial_dimensions; j++)
            {
                applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),x.getStrainGradient(j),mu,lamd,order2factors[count]);
                if(i!=j) applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(j),x.getStrainGradient(i),mu,lamd,order2factors[count]);
                applyK_Isotropic<Real,material_dimensions>(f.getStrain(),x.getStrainHessian(i,j),mu,lamd,order2factors[count]);
                applyK_Isotropic<Real,material_dimensions>(f.getStrainHessian(i,j),x.getStrain(),mu,lamd,order2factors[count]);
                count++;
            }
        // order 3
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=0; j<strain_size; j++)
            {
                applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),x.getStrainHessian(j),mu,lamd,order3factors(i,j));
                applyK_Isotropic<Real,material_dimensions>(f.getStrainHessian(j),x.getStrainGradient(i),mu,lamd,order3factors(i,j));
            }
        // order 4
        for(unsigned int i=0; i<strain_size; i++)
            for(unsigned int j=0; j<strain_size; j++)
            {
                applyK_Isotropic<Real,material_dimensions>(f.getStrainHessian(i),x.getStrainHessian(j),mu,lamd,order4factors(i,j));
                applyK_Isotropic<Real,material_dimensions>(f.getStrainHessian(j),x.getStrainHessian(i),mu,lamd,order4factors(i,j));
            }


        if(viscosity)
        {
            // order 0
            applyK_Isotropic<Real,material_dimensions>(f.getStrain(),v.getStrain(),viscosity,0,vol);
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(f.getStrain(),v.getStrainGradient(i),viscosity,0,order1factors[i]);
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),v.getStrain(),viscosity,0,order1factors[i]);
            // order 2
            count =0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),v.getStrainGradient(j),viscosity,0,order2factors[count]);
                    if(i!=j) applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(j),v.getStrainGradient(i),viscosity,0,order2factors[count]);
                    applyK_Isotropic<Real,material_dimensions>(f.getStrain(),v.getStrainHessian(i,j),viscosity,0,order2factors[count]);
                    applyK_Isotropic<Real,material_dimensions>(f.getStrainHessian(i,j),v.getStrain(),viscosity,0,order2factors[count]);
                    count++;
                }
            // order 3
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=0; j<strain_size; j++)
                {
                    applyK_Isotropic<Real,material_dimensions>(f.getStrainGradient(i),v.getStrainHessian(j),viscosity,0,order3factors(i,j));
                    applyK_Isotropic<Real,material_dimensions>(f.getStrainHessian(j),v.getStrainGradient(i),viscosity,0,order3factors(i,j));
                }
            // order 4
            for(unsigned int i=0; i<strain_size; i++)
                for(unsigned int j=0; j<strain_size; j++)
                {
                    applyK_Isotropic<Real,material_dimensions>(f.getStrainHessian(i),v.getStrainHessian(j),viscosity,0,order4factors(i,j));
                    applyK_Isotropic<Real,material_dimensions>(f.getStrainHessian(j),v.getStrainHessian(i),viscosity,0,order4factors(i,j));
                }
        }
    }

    void addDForce( Deriv&   df , const Deriv&   dx, const double& kfactor, const double& bfactor )
    {
        // order 0
        applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),mu,lamd,vol*kfactor);
        // order 1
        for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrainGradient(i),mu,lamd,order1factors[i]*kfactor);
        for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrain(),mu,lamd,order1factors[i]*kfactor);
        // order 2
        unsigned int count = 0;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=i; j<spatial_dimensions; j++)
            {
                applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrainGradient(j),mu,lamd,order2factors[count]*kfactor);
                if(i!=j) applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(j),dx.getStrainGradient(i),mu,lamd,order2factors[count]*kfactor);
                applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrainHessian(i,j),mu,lamd,order2factors[count]*kfactor);
                applyK_Isotropic<Real,material_dimensions>(df.getStrainHessian(i,j),dx.getStrain(),mu,lamd,order2factors[count]*kfactor);
                count++;
            }
        // order 3
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=0; j<strain_size; j++)
            {
                applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrainHessian(j),mu,lamd,order3factors(i,j)*kfactor);
                applyK_Isotropic<Real,material_dimensions>(df.getStrainHessian(j),dx.getStrainGradient(i),mu,lamd,order3factors(i,j)*kfactor);
            }
        // order 4
        for(unsigned int i=0; i<strain_size; i++)
            for(unsigned int j=0; j<strain_size; j++)
            {
                applyK_Isotropic<Real,material_dimensions>(df.getStrainHessian(i),dx.getStrainHessian(j),mu,lamd,order4factors(i,j)*kfactor);
                applyK_Isotropic<Real,material_dimensions>(df.getStrainHessian(j),dx.getStrainHessian(i),mu,lamd,order4factors(i,j)*kfactor);
            }


        if(viscosity)
        {
            // order 0
            applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrain(),viscosity,0,vol*bfactor);
            // order 1
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrainGradient(i),viscosity,0,order1factors[i]*bfactor);
            for(unsigned int i=0; i<spatial_dimensions; i++)  applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrain(),viscosity,0,order1factors[i]*bfactor);
            // order 2
            count = 0;
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=i; j<spatial_dimensions; j++)
                {
                    applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrainGradient(j),viscosity,0,order2factors[count]*bfactor);
                    if(i!=j) applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(j),dx.getStrainGradient(i),viscosity,0,order2factors[count]*bfactor);
                    applyK_Isotropic<Real,material_dimensions>(df.getStrain(),dx.getStrainHessian(i,j),viscosity,0,order2factors[count]*bfactor);
                    applyK_Isotropic<Real,material_dimensions>(df.getStrainHessian(i,j),dx.getStrain(),viscosity,0,order2factors[count]*bfactor);
                    count++;
                }
            // order 3
            for(unsigned int i=0; i<spatial_dimensions; i++)
                for(unsigned int j=0; j<strain_size; j++)
                {
                    applyK_Isotropic<Real,material_dimensions>(df.getStrainGradient(i),dx.getStrainHessian(j),viscosity,0,order3factors(i,j)*bfactor);
                    applyK_Isotropic<Real,material_dimensions>(df.getStrainHessian(j),dx.getStrainGradient(i),viscosity,0,order3factors(i,j)*bfactor);
                }
            // order 4
            for(unsigned int i=0; i<strain_size; i++)
                for(unsigned int j=0; j<strain_size; j++)
                {
                    applyK_Isotropic<Real,material_dimensions>(df.getStrainHessian(i),dx.getStrainHessian(j),viscosity,0,order4factors(i,j)*bfactor);
                    applyK_Isotropic<Real,material_dimensions>(df.getStrainHessian(j),dx.getStrainHessian(i),viscosity,0,order4factors(i,j)*bfactor);
                }
        }
    }



    MatBlock getK()
    {
        MatBlock K = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eK(&K[0][0]);
        // order 0
        eK.template block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,vol);
        // order 1
        for(unsigned int i=0; i<spatial_dimensions; i++)   eK.template block(strain_size*(i+1),0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order1factors[i]);
        for(unsigned int i=0; i<spatial_dimensions; i++)   eK.template block(0,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order1factors[i]);
        // order 2
        unsigned int count = 0;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=i; j<spatial_dimensions; j++)
            {
                eK.template block(strain_size*(i+1),strain_size*(j+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order2factors[count]);
                if(i!=j) eK.template block(strain_size*(j+1),strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order2factors[count]);
                count++;
            }
        // order 3
        unsigned int offset = (spatial_dimensions+1)*strain_size;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=0; j<strain_size; j++)
            {
                eK.template block(strain_size*(i+1),offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order3factors(i,j));
                eK.template block(offset+strain_size*j,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order3factors(i,j));
            }
        // order 4
        for(unsigned int i=0; i<strain_size; i++)
            for(unsigned int j=0; j<strain_size; j++)
            {
                eK.template block(offset+strain_size*i,offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order4factors(i,j));
                eK.template block(offset+strain_size*j,offset+strain_size*i,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(mu,lamd,order4factors(i,j));
            }
        return K;
    }

    MatBlock getC()
    {
        MatBlock C ;
        C.invert(getK());
        return C;
    }

    MatBlock getB()
    {
        MatBlock B = MatBlock();
        typedef Eigen::Map<Eigen::Matrix<Real,T::deriv_total_size,T::deriv_total_size,Eigen::RowMajor> > EigenMap;
        EigenMap eB(&B[0][0]);
        // order 0
        eB.template block(0,0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,vol);
        // order 1
        for(unsigned int i=0; i<spatial_dimensions; i++)   eB.template block(strain_size*(i+1),0,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order1factors[i]);
        for(unsigned int i=0; i<spatial_dimensions; i++)   eB.template block(0,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order1factors[i]);
        // order 2
        unsigned int count = 0;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=i; j<spatial_dimensions; j++)
            {
                eB.template block(strain_size*(i+1),strain_size*(j+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order2factors[count]);
                if(i!=j) eB.template block(strain_size*(j+1),strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order2factors[count]);
                count++;
            }
        // order 3
        unsigned int offset = (spatial_dimensions+1)*strain_size;
        for(unsigned int i=0; i<spatial_dimensions; i++)
            for(unsigned int j=0; j<strain_size; j++)
            {
                eB.template block(strain_size*(i+1),offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order3factors(i,j));
                eB.template block(offset+strain_size*j,strain_size*(i+1),strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order3factors(i,j));
            }
        // order 4
        for(unsigned int i=0; i<strain_size; i++)
            for(unsigned int j=0; j<strain_size; j++)
            {
                eB.template block(offset+strain_size*i,offset+strain_size*j,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order4factors(i,j));
                eB.template block(offset+strain_size*j,offset+strain_size*i,strain_size,strain_size) = assembleK_Isotropic<Real,material_dimensions>(viscosity,0,order4factors(i,j));
            }
        return B;
    }


};





} // namespace defaulttype
} // namespace sofa



#endif
