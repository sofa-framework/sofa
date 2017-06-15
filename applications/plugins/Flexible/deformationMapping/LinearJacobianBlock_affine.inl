/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef FLEXIBLE_LinearJacobianBlock_affine_INL
#define FLEXIBLE_LinearJacobianBlock_affine_INL

#include "LinearJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "../types/AffineTypes.h"
#include "../types/QuadraticTypes.h"
#include "../types/DeformationGradientTypes.h"

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  Affine3 -> Vec3
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Affine3(InReal) , V3(OutReal) > :
    public  BaseJacobianBlock< Affine3(InReal) , V3(OutReal) >
{
public:
    typedef Affine3(InReal) In;
    typedef V3(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    /**
    Mapping:   \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0) = w.t + w.A.q0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.

    Jacobian:    \f$ dp = w.dt + w.dA.q0\f$
      */

    static const bool constant=true;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa;   ///< =  w.q0      =  dp/dA

    void init( const LinearJacobianBlock<In,Out>& block) // copy
    {
        Pt=block.Pt;
        Pa=block.Pa;
    }

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa=In::inverse(InPos).pointToParent(SPos)*Pt;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result +=  data.getCenter() * Pt + data.getAffine() * Pa;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += data.getVCenter() * Pt + data.getVAffine() * Pa;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data * Pt ;
        for (unsigned int j = 0; j < dim; ++j) result.getVAffine()[j] += Pa * data[j];
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) J(i,i)=Pt;
        for(unsigned int i=0; i<dim; ++i) for (unsigned int j=0; j<dim; ++j) J(j,i+(j+1)*dim)=Pa[i];
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};



//////////////////////////////////////////////////////////////////////////////////
////  Affine3 -> ExtVec3   same as Vec3 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Affine3(InReal) , EV3(OutReal) > :
    public  BaseJacobianBlock< Affine3(InReal) , EV3(OutReal) >
{
public:
    typedef Affine3(InReal) In;
    typedef EV3(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    /**
    Mapping:   \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0) = w.t + w.A.q0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.

    Jacobian:    \f$ dp = w.dt + w.dA.q0\f$
    */

    static const bool constant=true;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa;   ///< =  w.q0      =  dp/dA

    void init( const LinearJacobianBlock<In,Out>& block) // copy
    {
        Pt=block.Pt;
        Pa=block.Pa;
    }

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa=In::inverse(InPos).pointToParent(SPos)*Pt;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result +=  data.getCenter() * Pt + data.getAffine() * Pa;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += data.getVCenter() * Pt + data.getVAffine() * Pa;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data * Pt ;
        for (unsigned int j = 0; j < dim; ++j) result.getVAffine()[j] += Pa * data[j];
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) J(i,i)=Pt;
        for(unsigned int i=0; i<dim; ++i) for (unsigned int j=0; j<dim; ++j) J(j,i+(j+1)*dim)=Pa[i];
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Affine3 -> F331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Affine3(InReal) , F331(OutReal) > :
    public  BaseJacobianBlock< Affine3(InReal) , F331(OutReal) >
{
public:
    typedef Affine3(InReal) In;
    typedef F331(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0).grad w.M + w.A.A0^{-1}.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.( q0.grad w + w.A0^{-1} ).M\f$
    */

    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa;      ///< =   q0.grad w.M + w.A0^{-1}.M   =  dF/dA

    void init( const LinearJacobianBlock<In,Out>& block) // copy
    {
        Ft=block.Ft;
        PFa=block.PFa;
    }

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;
        InCoord inverseInitialTransform = In::inverse(InPos);   // A0^{-1}
        SpatialCoord vectorInLocalCoordinates = inverseInitialTransform.pointToParent(SPos);  // q0
        PFa.getF()=covMN(vectorInLocalCoordinates,Ft) + inverseInitialTransform.getAffine() * F0 * w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getAffine()*PFa.getF();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data.getVCenter(),Ft) + data.getVAffine()*PFa.getF();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getF() * Ft ;

        for (unsigned int j = 0; j < dim; ++j)
        {
            result.getVAffine()[j] += PFa.getF() * (data.getF()[j]);
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dim; ++l)    J(j+l*mdim,i+dim+l*dim)=PFa.getF()[i][j];
        return J;
    }


    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Affine3 -> F321  same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Affine3(InReal) , F321(OutReal) > :
    public  BaseJacobianBlock< Affine3(InReal) , F321(OutReal) >
{
public:
    typedef Affine3(InReal) In;
    typedef F321(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0).grad w.M + w.A.A0^{-1}.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.( q0.grad w + w.A0^{-1} ).M\f$
    */

    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa;      ///< =   q0.grad w.M + w.A0^{-1}.M   =  dF/dA

    void init( const LinearJacobianBlock<In,Out>& block) // copy
    {
        Ft=block.Ft;
        PFa=block.PFa;
    }

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;
        InCoord inverseInitialTransform = In::inverse(InPos);   // A0^{-1}
        SpatialCoord vectorInLocalCoordinates = inverseInitialTransform.pointToParent(SPos);  // q0
        PFa.getF()=covMN(vectorInLocalCoordinates,Ft) + inverseInitialTransform.getAffine() * F0 * w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getAffine()*PFa.getF();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data.getVCenter(),Ft) + data.getVAffine()*PFa.getF();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getF() * Ft ;

        for (unsigned int j = 0; j < dim; ++j)
        {
            result.getVAffine()[j] += PFa.getF() * (data.getF()[j]);
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dim; ++l)    J(j+l*mdim,i+dim+l*dim)=PFa.getF()[i][j];
        return J;
    }


    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};



//////////////////////////////////////////////////////////////////////////////////
////  Affine3 -> F311  same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Affine3(InReal) , F311(OutReal) > :
    public  BaseJacobianBlock< Affine3(InReal) , F311(OutReal) >
{
public:
    typedef Affine3(InReal) In;
    typedef F311(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0).grad w.M + w.A.A0^{-1}.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.( q0.grad w + w.A0^{-1} ).M\f$
    */

    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa;      ///< =   q0.grad w.M + w.A0^{-1}.M   =  dF/dA

    void init( const LinearJacobianBlock<In,Out>& block) // copy
    {
        Ft=block.Ft;
        PFa=block.PFa;
    }

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;
        InCoord inverseInitialTransform = In::inverse(InPos);   // A0^{-1}
        SpatialCoord vectorInLocalCoordinates = inverseInitialTransform.pointToParent(SPos);  // q0
        PFa.getF()=covMN(vectorInLocalCoordinates,Ft) + inverseInitialTransform.getAffine() * F0 * w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getAffine()*PFa.getF();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data.getVCenter(),Ft) + data.getVAffine()*PFa.getF();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getF() * Ft ;

        for (unsigned int j = 0; j < dim; ++j)
        {
            result.getVAffine()[j] += PFa.getF() * (data.getF()[j]);
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dim; ++l)    J(j+l*mdim,i+dim+l*dim)=PFa.getF()[i][j];
        return J;
    }


    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};

//////////////////////////////////////////////////////////////////////////////////
////  Affine3 -> F332
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Affine3(InReal) , F332(OutReal) > :
    public  BaseJacobianBlock< Affine3(InReal) , F332(OutReal) >
{
public:
    typedef Affine3(InReal) In;
    typedef F332(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;
    typedef Mat<dim,mdim,Real> mHessian;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0).grad w.M + w.A.A0^{-1}.M  \f$
        - \f$ (grad F)_k = [ (t+A.q0).(grad2 w)_k^T + A.[(grad w)_k.A0^{-1} +  A0^{-1}_k.grad w] ].M \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
        - _k denotes component/column k
    Jacobian:
        - \f$ d F = [dt.grad w + dA.( q0.grad w + w.A0^{-1} )].M \f$
        - \f$ d (grad F)_k = [dt.(grad2 w)_k^T + dA.[q0.(grad2 w)_k^T + (grad w)_k.A0^{-1} +  A0^{-1}_k.grad w] ].M \f$
    */

    static const bool constant=true;

    mGradient Ft;       ///< =   grad w     =  d F/dt
    mHessian dFt;      ///< =   (grad2 w)_k^T   =  d (grad F)_k/dt
    OutCoord PFdFa;      ///< =   q0.grad w + w.A0^{-1}, [q0.(grad2 w)_k^T + (grad w)_k.A0^{-1} +  A0^{-1}_k.grad w]   =  dF/dA , d (grad F)_k/dA

    void init( const LinearJacobianBlock<In,Out>& block) // copy
    {
        Ft=block.Ft;
        dFt=block.dFt;
        PFdFa=block.PFdFa;
    }

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& ddw)
    {
        Ft=F0.transposed()*dw;
        dFt=ddw.transposed()*F0;

        InCoord inverseInitialTransform = In::inverse(InPos);   // A0^{-1}
        SpatialCoord vectorInLocalCoordinates = inverseInitialTransform.pointToParent(SPos);  // q0
        PFdFa.getF()=covMN(vectorInLocalCoordinates,Ft) + inverseInitialTransform.getAffine() * F0 * w;

        Mat<dim,dim> AOinv = inverseInitialTransform.getAffine();
        Mat<dim,dim> AOinvT = AOinv.transposed();
        Mat<dim,mdim> AOinvM; for (unsigned int k = 0; k < dim; ++k) AOinvM[k]=F0.transposed()*AOinv[k];
        for (unsigned int k = 0; k < dim; ++k) PFdFa.getGradientF(k) = covMN( vectorInLocalCoordinates, dFt[k]) + AOinvM * dw[k] + covMN(AOinvT[k],Ft);
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getAffine()*PFdFa.getF();
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN( data.getCenter(), dFt[k]) + data.getAffine() * PFdFa.getGradientF(k);
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data.getVCenter(),Ft) + data.getVAffine()*PFdFa.getF();
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN(data.getVCenter(),dFt[k]) + data.getVAffine() * PFdFa.getGradientF(k);
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getF() * Ft ;
        for (unsigned int k = 0; k < dim; ++k) result.getVCenter() += data.getGradientF(k) * dFt[k] ;

        for (unsigned int j = 0; j < dim; ++j)
        {
            result.getVAffine()[j] += PFdFa.getF() * (data.getF()[j]);
            for (unsigned int k = 0; k < dim; ++k) result.getVAffine()[j] += PFdFa.getGradientF(k) * (data.getGradientF(k)[j]);
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dim; ++l)    J(j+l*mdim,i+dim+l*dim)=PFdFa.getF()[i][j];
        unsigned int offset=dim*mdim;
        for(unsigned int k=0; k<dim; ++k)
        {
            for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+offset+i*mdim,i)=dFt[k][j];
            for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dim; ++l)    J(j+offset+l*mdim,i+dim+l*dim)=PFdFa.getGradientF(k)[i][j];
            offset+=dim*mdim;
        }
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Affine3 -> Affine3 = F331 with dw=0
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Affine3(InReal) , Affine3(OutReal) > :
    public  BaseJacobianBlock< Affine3(InReal) , Affine3(OutReal) >
{
public:
    typedef Affine3(InReal) In;
    typedef Affine3(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    /**
    Mapping:
        - \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0) = w.t + w.A.q0  \f$
        - \f$ F = w.A.A0^{-1}.F0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0,F0 is the position of p,F in the reference configuration.
        - q0 is the local position of p0.
    Jacobian:
        - \f$ dp = w.dt + w.dA.q0\f$
        - \f$ d F = w.dA.A0^{-1}.F0\f$
    */

    static const bool constant=true;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa;      ///< =   w.q0      =  dp/dA  , w.A0^{-1}.F0   =  dF/dA

    void init( const LinearJacobianBlock<In,Out>& block) // copy
    {
        Pt=block.Pt;
        Pa=block.Pa;
    }

    void init( const InCoord& InPos, const OutCoord& OutPos, const SpatialCoord& /*SPos*/, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        InCoord inverseInitialTransform = In::inverse(InPos);   // A0^{-1}
        Pa.getCenter()=inverseInitialTransform.pointToParent(OutPos.getCenter())*w;
        Pa.getAffine()=inverseInitialTransform.getAffine()*OutPos.getAffine()*w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getCenter() +=  data.getCenter() * Pt + data.getAffine() * Pa.getCenter();
        result.getAffine() +=  data.getAffine() * Pa.getAffine() ;
        for (unsigned int j = 0; j < dim; ++j) result.getAffine()[j][j] -= Pt; // this term cancels the initial identity affine matrix
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getVCenter() += data.getVCenter() * Pt + data.getVAffine() * Pa.getCenter();
        result.getVAffine() += data.getVAffine() * Pa.getAffine();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getVCenter() * Pt ;
        for (unsigned int j = 0; j < dim; ++j) result.getVAffine()[j] += Pa.getCenter() * (data.getVCenter())[j] + Pa.getAffine() * (data.getVAffine()[j]);
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) J(i,i)=Pt;
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<dim; ++j)
            {
                J(j,i+(j+1)*dim)=Pa.getCenter()[i];
                for(unsigned int l=0; l<dim; ++l)   J(j+(l+1)*dim,i+dim+l*dim)=Pa.getAffine()[i][j];
            }
        return J;
    }


    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};

} // namespace defaulttype
} // namespace sofa



#endif
