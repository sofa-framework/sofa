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
#ifndef FLEXIBLE_MLSJacobianBlock_affine_INL
#define FLEXIBLE_MLSJacobianBlock_affine_INL

#include "MLSJacobianBlock.h"
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
class MLSJacobianBlock< Affine3(InReal) , V3(OutReal) > :
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

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    /**
    Mapping:   \f$ p = w.t + A.A0^{-1}.(p*-w.t0) + w.p0- p*   = w.t + A.q0 + C \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)

    Jacobian:    \f$ dp = w.dt + dA.q0\f$
      */

    static const bool constant=true;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa;   ///< =  q0      =  dp/dA
    OutCoord C;   ///< =  w.p0- p*      =  constant term



    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Basis& p, const Gradient& /*dp*/, const Hessian& /*ddp*/)
    {
        Pt=p[0];
        Pa=In::inverse(InPos).getAffine()*(BasisToCoord(p)-InPos.getCenter()*Pt);
        C=SPos*Pt-BasisToCoord(p);
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result +=  data.getCenter() * Pt + data.getAffine() * Pa + C;
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
class MLSJacobianBlock< Affine3(InReal) , EV3(OutReal) > :
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

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    /**
    Mapping:   \f$ p = w.t + A.A0^{-1}.(p*-w.t0) + w.p0- p*   = w.t + A.q0 + C \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)

    Jacobian:    \f$ dp = w.dt + dA.q0\f$
      */

    static const bool constant=true;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa;   ///< =  q0      =  dp/dA
    OutCoord C;   ///< =  w.p0- p*      =  constant term



    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Basis& p, const Gradient& /*dp*/, const Hessian& /*ddp*/)
    {
        Pt=p[0];
        Pa=In::inverse(InPos).getAffine()*(BasisToCoord(p)-InPos.getCenter()*Pt);
        C=SPos*Pt-BasisToCoord(p);
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result +=  data.getCenter() * Pt + data.getAffine() * Pa + C;
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
class MLSJacobianBlock< Affine3(InReal) , F331(OutReal) > :
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

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;



    /**
    Mapping:   \f$ F = grad p.M = t.grad w.M + A.A0^{-1}.grad(p*-w.t0).M + (p0.grad w + w.I - grad p*).M \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.A0^{-1}.grad(p*-w.t0).M\f$
      */


    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa;      ///< =   A0^{-1}.(grad p*- t0.grad w).M   =  dF/dA
    OutCoord C;   ///< =  (p0.grad w + w.I - grad p*).M      =  constant term

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dp, const Hessian& /*ddp*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dp[i][0];
        Ft=F0.transposed()*dw;
        Mat<dim,dim,Real> gradps; for (unsigned int j = 0; j < dim; ++j) for (unsigned int k = 0; k < dim; ++k) gradps(j,k)=dp[k][j+1];
        PFa.getF()=In::inverse(InPos).getAffine()*(gradps* F0 - covMN(InPos.getCenter(),Ft) );
        Mat<dim,dim,Real> wI; for (unsigned int j = 0; j < dim; ++j) wI(j,j)=p[0];
        C.getF()=covMN(SPos,Ft) + (wI-gradps)*F0 ;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getAffine()*PFa.getF() + C.getF();
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
class MLSJacobianBlock< Affine3(InReal) , F321(OutReal) > :
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

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;



    /**
    Mapping:   \f$ F = grad p.M = t.grad w.M + A.A0^{-1}.grad(p*-w.t0).M + (p0.grad w + w.I - grad p*).M \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.A0^{-1}.grad(p*-w.t0).M\f$
      */


    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa;      ///< =   A0^{-1}.(grad p*- t0.grad w).M   =  dF/dA
    OutCoord C;   ///< =  (p0.grad w + w.I - grad p*).M      =  constant term

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dp, const Hessian& /*ddp*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dp[i][0];
        Ft=F0.transposed()*dw;
        Mat<dim,dim,Real> gradps; for (unsigned int j = 0; j < dim; ++j) for (unsigned int k = 0; k < dim; ++k) gradps(j,k)=dp[k][j+1];
        PFa.getF()=In::inverse(InPos).getAffine()*(gradps* F0 - covMN(InPos.getCenter(),Ft) );
        Mat<dim,dim,Real> wI; for (unsigned int j = 0; j < dim; ++j) wI(j,j)=p[0];
        C.getF()=covMN(SPos,Ft) + (wI-gradps)*F0 ;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getAffine()*PFa.getF() + C.getF();
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
class MLSJacobianBlock< Affine3(InReal) , F311(OutReal) > :
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

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;



    /**
    Mapping:   \f$ F = grad p.M = t.grad w.M + A.A0^{-1}.grad(p*-w.t0).M + (p0.grad w + w.I - grad p*).M \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.A0^{-1}.grad(p*-w.t0).M\f$
      */


    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa;      ///< =   A0^{-1}.(grad p*- t0.grad w).M   =  dF/dA
    OutCoord C;   ///< =  (p0.grad w + w.I - grad p*).M      =  constant term

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dp, const Hessian& /*ddp*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dp[i][0];
        Ft=F0.transposed()*dw;
        Mat<dim,dim,Real> gradps; for (unsigned int j = 0; j < dim; ++j) for (unsigned int k = 0; k < dim; ++k) gradps(j,k)=dp[k][j+1];
        PFa.getF()=In::inverse(InPos).getAffine()*(gradps* F0 - covMN(InPos.getCenter(),Ft) );
        Mat<dim,dim,Real> wI; for (unsigned int j = 0; j < dim; ++j) wI(j,j)=p[0];
        C.getF()=covMN(SPos,Ft) + (wI-gradps)*F0 ;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getAffine()*PFa.getF() + C.getF();
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
class MLSJacobianBlock< Affine3(InReal) , F332(OutReal) > :
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

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;
    typedef Mat<dim,mdim,Real> mHessian;


    /**
    Mapping:
        - \f$ F = grad p.M = t.grad w.M + A.A0^{-1}.grad(p*-w.t0).M + (p0.grad w + w.I - grad p*).M \f$
        - \f$ (grad F)_k = t.(grad2 w)_k^T.M + A.A0^{-1}.[ grad2(p*)_k - t0.grad2(w)_k].M   +  ( I_k.grad w + p0.(grad2 w)_k + (grad w)_k.I - (grad2 p*)_k).M
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)
        - grad denotes spatial derivatives
        - _k denotes component/column k
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.A0^{-1}.grad(p*-w.t0).M\f$
        - \f$ d(grad F)_k = dt.(grad2 w)_k^T.M + dA.A0^{-1}.[ grad2(p*)_k - t0.grad2(w)_k].M
      */



    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    mHessian dFt;      ///< =   (grad2 w)_k^T.M   =  d (grad F)_k/dt
    OutCoord PFdFa;      ///< =   A0^{-1}.(grad p*- t0.grad w).M, [A0^{-1}.[ grad2(p*)_k - t0.grad2(w)_k].M]   =  dF/dA , d (grad F)_k/dA
    OutCoord C;   ///< =  (p0.grad w + w.I - grad p*).M , [( I_k.grad w + p0.(grad2 w)_k + (grad w)_k.I - (grad2 p*)_k).M ]       =  constant term

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dp, const Hessian& ddp)
    {
        Mat<dim,dim,Real> A0inv=In::inverse(InPos).getAffine();

        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dp[i][0];
        Ft=F0.transposed()*dw;
        Mat<dim,dim,Real> gradps; for (unsigned int i = 0; i < dim; ++i) for (unsigned int j = 0; j < dim; ++j) gradps(i,j)=dp[j][i+1];
        PFdFa.getF()=A0inv*(gradps* F0 - covMN(InPos.getCenter(),Ft) );
        Mat<dim,dim,Real> wI; for (unsigned int j = 0; j < dim; ++j) wI(j,j)=p[0];
        C.getF()=covMN(SPos,Ft) + (wI-gradps)*F0 ;

        Mat<dim,dim,Real> ddw; for (unsigned int i = 0; i < dim; ++i) for (unsigned int j = 0; j < dim; ++j) ddw(i,j)=ddp(i,j)[0];
        dFt=ddw.transposed()*F0;

        for (unsigned int k = 0; k < dim; ++k)
        {
            for (unsigned int i = 0; i < dim; ++i) for (unsigned int j = 0; j < dim; ++j) gradps(i,j)=ddp(j,k)[i+1];
            PFdFa.getGradientF(k) = A0inv*(gradps* F0 - covMN(InPos.getCenter(),dFt[k]) );
            for (unsigned int j = 0; j < dim; ++j) wI(j,j)=dw[k];
            SpatialCoord one; one[k]=1;
            C.getGradientF(k)=covMN(SPos,dFt[k]) + covMN(one,Ft) + (wI-gradps)*F0 ;
        }
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getAffine()*PFdFa.getF() + C.getF();
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN( data.getCenter(), dFt[k]) + data.getAffine() * PFdFa.getGradientF(k) + C.getGradientF(k) ;
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


} // namespace defaulttype
} // namespace sofa



#endif
