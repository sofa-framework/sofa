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
#ifndef FLEXIBLE_MLSJacobianBlock_point_INL
#define FLEXIBLE_MLSJacobianBlock_point_INL

#include "MLSJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>
#include "../types/DeformationGradientTypes.h"

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> Vec3
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< V3(InReal) , V3(OutReal) > :
    public  BaseJacobianBlock< V3(InReal) , V3(OutReal) >
{
public:
    typedef V3(InReal) In;
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
    Mapping:   \f$ p = w.t + w.(p0-t0)  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.

    Jacobian:    \f$ dp = w.dt \f$
      */

    static const bool constant=true;

    OutCoord C;   ///< =  w.(p0-t0)  =  constant term
    Real Pt;      ///< =   w         =  dp/dt


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Basis& p, const Gradient& /*dp*/, const Hessian& /*ddp*/)
    {
        Pt=p[0];
        C=(SPos-InPos)*Pt;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result +=  data * Pt + C;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += data * Pt ;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data * Pt ;
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; i++) J(i,i)=Pt;
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};

//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> ExtVec3   same as Vec3 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< V3(InReal) , EV3(OutReal) > :
    public  BaseJacobianBlock< V3(InReal) , EV3(OutReal) >
{
public:
    typedef V3(InReal) In;
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
    Mapping:   \f$ p = w.t + w.(p0-t0)  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.

    Jacobian:    \f$ dp = w.dt \f$
      */

    static const bool constant=true;

    OutCoord C;   ///< =  w.(p0-t0)  =  constant term
    Real Pt;      ///< =   w         =  dp/dt


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Basis& p, const Gradient& /*dp*/, const Hessian& /*ddp*/)
    {
        Pt=p[0];
        C=(SPos-InPos)*Pt;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result +=  data * Pt + C;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += data * Pt ;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data * Pt ;
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; i++) J(i,i)=Pt;
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> F331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< V3(InReal) , F331(OutReal) > :
    public  BaseJacobianBlock< V3(InReal) , F331(OutReal) >
{
public:
    typedef V3(InReal) In;
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
    Mapping:
        - \f$ F = grad p . M = (t+p0-t0).grad w.M + w.M  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M \f$
      */

    static const bool constant=true;

    OutCoord C;       ///< =  (p0-t0).grad w.M + w.M   =  constant term
    mGradient Ft;  ///< =   grad w.M     =  d F/dt

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dP, const Hessian& /*ddP*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dP[i][0];
        Ft=F0.transposed()*dw;
        C.getF()=covMN(SPos-InPos,Ft);
        C.getF()+=F0*p[0];
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data,Ft) + C.getF();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data,Ft) ;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data.getF() * Ft ;
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<mdim; j++) J(j+i*mdim,i)=Ft[j];
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> F321   same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< V3(InReal) , F321(OutReal) > :
    public  BaseJacobianBlock< V3(InReal) , F321(OutReal) >
{
public:
    typedef V3(InReal) In;
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
    Mapping:
        - \f$ F = grad p . M = (t+p0-t0).grad w.M + w.M  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M \f$
      */

    static const bool constant=true;

    OutCoord C;       ///< =  (p0-t0).grad w.M + w.M   =  constant term
    mGradient Ft;  ///< =   grad w.M     =  d F/dt

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dP, const Hessian& /*ddP*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dP[i][0];
        Ft=F0.transposed()*dw;
        C.getF()=covMN(SPos-InPos,Ft);
        C.getF()+=F0*p[0];
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data,Ft) + C.getF();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data,Ft) ;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data.getF() * Ft ;
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<mdim; j++) J(j+i*mdim,i)=Ft[j];
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> F311   same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< V3(InReal) , F311(OutReal) > :
    public  BaseJacobianBlock< V3(InReal) , F311(OutReal) >
{
public:
    typedef V3(InReal) In;
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
    Mapping:
        - \f$ F = grad p . M = (t+p0-t0).grad w.M + w.M  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M \f$
      */

    static const bool constant=true;

    OutCoord C;       ///< =  (p0-t0).grad w.M + w.M   =  constant term
    mGradient Ft;  ///< =   grad w.M     =  d F/dt

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dP, const Hessian& /*ddP*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dP[i][0];
        Ft=F0.transposed()*dw;
        C.getF()=covMN(SPos-InPos,Ft);
        C.getF()+=F0*p[0];
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data,Ft) + C.getF();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data,Ft) ;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data.getF() * Ft ;
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<mdim; j++) J(j+i*mdim,i)=Ft[j];
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> F332
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< V3(InReal) , F332(OutReal) > :
    public  BaseJacobianBlock< V3(InReal) , F332(OutReal) >
{
public:
    typedef V3(InReal) In;
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
        - \f$ F = grad p.M = (t+p0-t0).grad w.M + w.M  \f$
        - \f$ (grad F)_k = ( (t+p0-t0).(grad2 w)_k^T + [(grad w)_k.I +  I_k.grad w] ).M \f$ the second term can be removed because \f$ \sum_i grad w_i =0\f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - grad denotes spatial derivatives
        - _k denotes component/column k
    Jacobian:
        - \f$ d F = dt.grad w.M \f$
        - \f$ d (grad F)_k = dt.(grad2 w)_k^T.M \f$
      */

    static const bool constant=true;

    OutCoord C;       ///< =  w.(p0-t0)   ,  (p0-t0).grad w + w.I,  (p0-t0).(grad2 w)_k^T + [(grad w)_k.I +  I_k.grad w]   =  constant term
    Real Pt;           ///< =   w     =  dp/dt
    mGradient Ft;  ///< =   grad w.M     =  d F/dt
    mHessian dFt;  ///< =   (grad2 w)_k^T.M   =  d (grad F)_k/dt

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dP, const Hessian& ddP)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dP[i][0];
        Ft=F0.transposed()*dw;
        C.getF()=covMN(SPos-InPos,Ft);
        C.getF()+=F0*p[0];

        Mat<dim,dim,Real> ddw;
        for (unsigned int i = 0; i < dim; ++i) for (unsigned int j = 0; j < dim; ++j) ddw(i,j)=ddP(i,j)[0];
        dFt=ddw.transposed()*F0;
        for (unsigned int k = 0; k < dim; ++k)
        {
            C.getGradientF(k)=covMN(SPos-InPos,dFt[k]);
        }
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data,Ft) + C.getF();
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN( data, dFt[k]) + C.getGradientF(k);
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data,Ft) ;
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN(data,dFt[k]) ;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data.getF() * Ft ;
        for (unsigned int k = 0; k < dim; ++k) result += data.getGradientF(k) * dFt[k] ;
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<mdim; j++) J(j+i*mdim,i)=Ft[j];
        unsigned int offset=mdim*dim;
        for (unsigned int k = 0; k < dim; ++k)
        {
            for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<mdim; j++) J(j+i*mdim,i)=Ft[j];
            for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<mdim; j++) J(j+offset+i*mdim,i)=dFt[k][j];
            offset+=mdim*dim;
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
