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
#ifndef FLEXIBLE_MLSJacobianBlock_rigid_INL
#define FLEXIBLE_MLSJacobianBlock_rigid_INL

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
////  Rigid3 -> Vec3
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< Rigid3(InReal) , V3(OutReal) > :
    public  BaseJacobianBlock< Rigid3(InReal) , V3(OutReal) >
{
public:
    typedef Rigid3(InReal) In;
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
    enum { adim = InDeriv::total_size - dim };  // size of angular velocity vector

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:   \f$ p = w.t + A.A0^{-1}.(p*-w.t0) + w.p0- p*   = w.t + A.q0 + C \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)

    Jacobian:    \f$ dp = w.dt + Omega x q0 \f$
      */

    static const bool constant=false;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa0;   ///< =  q0    : weighted point in local frame
    OutCoord Pa;   ///< =  Omega x q0        : rotated point
    OutCoord C;   ///< =  w.p0- p*      =  constant term


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Basis& p, const Gradient& /*dp*/, const Hessian& /*ddp*/)
    {
        Pt=p[0];
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        Pa0=AOinv*(BasisToCoord(p)-InPos.getCenter()*Pt);
        C=SPos*Pt-BasisToCoord(p);

        Pa= InPos.rotate(Pa0);
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        Pa= data.rotate(Pa0);
        result +=  data.getCenter() * Pt + Pa + C;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += getLinear(data) * Pt + cross(getAngular(data), Pa);
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        getLinear(result) += data * Pt ;
        getAngular(result) += cross(Pa, data);
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) J(i,i)=Pt;

        cpMatrix W=-crossProductMatrix(Pa);
        for(unsigned int l=0; l<adim; ++l) for (unsigned int i=0; i<dim; ++i) J(i,l+dim)=W(i,l);
        return J;
    }

    // TO DO : implement this !!
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> ExtVec3   same as Vec3 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< Rigid3(InReal) , EV3(OutReal) > :
    public  BaseJacobianBlock< Rigid3(InReal) , EV3(OutReal) >
{
public:
    typedef Rigid3(InReal) In;
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
    enum { adim = InDeriv::total_size - dim };  // size of angular velocity vector

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:   \f$ p = w.t + A.A0^{-1}.(p*-w.t0) + w.p0- p*   = w.t + A.q0 + C \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)

    Jacobian:    \f$ dp = w.dt + Omega x q0 \f$
      */

    static const bool constant=false;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa0;   ///< =  q0    : weighted point in local frame
    OutCoord Pa;   ///< =  Omega x q0        : rotated point
    OutCoord C;   ///< =  w.p0- p*      =  constant term


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Basis& p, const Gradient& /*dp*/, const Hessian& /*ddp*/)
    {
        Pt=p[0];
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        Pa0=AOinv*(BasisToCoord(p)-InPos.getCenter()*Pt);
        C=SPos*Pt-BasisToCoord(p);

        Pa= InPos.rotate(Pa0);
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        Pa= data.rotate(Pa0);
        result +=  data.getCenter() * Pt + Pa + C;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += getLinear(data) * Pt + cross(getAngular(data), Pa);
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        getLinear(result) += data * Pt ;
        getAngular(result) += cross(Pa, data);
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) J(i,i)=Pt;

        cpMatrix W=-crossProductMatrix(Pa);
        for(unsigned int l=0; l<adim; ++l) for (unsigned int i=0; i<dim; ++i) J(i,l+dim)=W(i,l);
        return J;
    }

    // TO DO : implement this !!
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> F331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< Rigid3(InReal) , F331(OutReal) > :
    public  BaseJacobianBlock< Rigid3(InReal) , F331(OutReal) >
{
public:
    typedef Rigid3(InReal) In;
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
    enum { adim = InDeriv::total_size - dim };  // size of angular velocity vector

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:   \f$ F = grad p.M = t.grad w.M + A.A0^{-1}.grad(p*-w.t0).M + (p0.grad w + w.I - grad p*).M \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + Omega x.A.A0^{-1}.grad(p*-w.t0).M\f$
    */

    static const bool constant=false;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa0;      ///< =   A0^{-1}.(grad p*- t0.grad w).M   =  dF/dA
    OutCoord PFa;      ///< =    A.PFa0
    OutCoord C;   ///< =  (p0.grad w + w.I - grad p*).M      =  constant term

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dp, const Hessian& /*ddp*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dp[i][0];
        Ft=F0.transposed()*dw;
        Mat<dim,dim,Real> gradps; for (unsigned int j = 0; j < dim; ++j) for (unsigned int k = 0; k < dim; ++k) gradps(j,k)=dp[k][j+1];
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        PFa0.getF()=AOinv*(gradps* F0 - covMN(InPos.getCenter(),Ft) );
        Mat<dim,dim,Real> wI; for (unsigned int j = 0; j < dim; ++j) wI(j,j)=p[0];
        C.getF()=covMN(SPos,Ft) + (wI-gradps)*F0 ;

        rotMat A0; InPos.getOrientation().toMatrix(A0);
        PFa.getF()= A0 * PFa0.getF();
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        rotMat A; data.getOrientation().toMatrix(A);
        PFa.getF()= A * PFa0.getF();  // = update of J according to current transform

        result.getF() +=  covMN(data.getCenter(),Ft) + PFa.getF() + C.getF();;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        const cpMatrix W=crossProductMatrix(getAngular(data));
        result.getF() += covMN(getLinear(data),Ft) + W * PFa.getF();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        getLinear(result) += data.getF() * Ft ;

        for(unsigned int i=0; i<mdim; ++i) getAngular(result) += cross(PFa.getF().col(i),data.getF().col(i));
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];

        for(unsigned int j=0; j<mdim; ++j)
        {
            cpMatrix W=-crossProductMatrix(PFa.getF().col(j));
            for(unsigned int l=0; l<adim; ++l)   for(unsigned int i=0; i<dim; ++i)   J(j+i*mdim,l+dim)+=W(i,l);
        }
        return J;
    }


    // TO DO : implement this !!
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> F321  same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< Rigid3(InReal) , F321(OutReal) > :
    public  BaseJacobianBlock< Rigid3(InReal) , F321(OutReal) >
{
public:
    typedef Rigid3(InReal) In;
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
    enum { adim = InDeriv::total_size - dim };  // size of angular velocity vector

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:   \f$ F = grad p.M = t.grad w.M + A.A0^{-1}.grad(p*-w.t0).M + (p0.grad w + w.I - grad p*).M \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + Omega x.A.A0^{-1}.grad(p*-w.t0).M\f$
    */

    static const bool constant=false;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa0;      ///< =   A0^{-1}.(grad p*- t0.grad w).M   =  dF/dA
    OutCoord PFa;      ///< =    A.PFa0
    OutCoord C;   ///< =  (p0.grad w + w.I - grad p*).M      =  constant term

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dp, const Hessian& /*ddp*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dp[i][0];
        Ft=F0.transposed()*dw;
        Mat<dim,dim,Real> gradps; for (unsigned int j = 0; j < dim; ++j) for (unsigned int k = 0; k < dim; ++k) gradps(j,k)=dp[k][j+1];
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        PFa0.getF()=AOinv*(gradps* F0 - covMN(InPos.getCenter(),Ft) );
        Mat<dim,dim,Real> wI; for (unsigned int j = 0; j < dim; ++j) wI(j,j)=p[0];
        C.getF()=covMN(SPos,Ft) + (wI-gradps)*F0 ;

        rotMat A0; InPos.getOrientation().toMatrix(A0);
        PFa.getF()= A0 * PFa0.getF();
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        rotMat A; data.getOrientation().toMatrix(A);
        PFa.getF()= A * PFa0.getF();  // = update of J according to current transform

        result.getF() +=  covMN(data.getCenter(),Ft) + PFa.getF() + C.getF();;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        const cpMatrix W=crossProductMatrix(getAngular(data));
        result.getF() += covMN(getLinear(data),Ft) + W * PFa.getF();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        getLinear(result) += data.getF() * Ft ;

        for(unsigned int i=0; i<mdim; ++i) getAngular(result) += cross(PFa.getF().col(i),data.getF().col(i));
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];

        for(unsigned int j=0; j<mdim; ++j)
        {
            cpMatrix W=-crossProductMatrix(PFa.getF().col(j));
            for(unsigned int l=0; l<adim; ++l)   for(unsigned int i=0; i<dim; ++i)   J(j+i*mdim,l+dim)+=W(i,l);
        }
        return J;
    }


    // TO DO : implement this !!
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> F311  same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< Rigid3(InReal) , F311(OutReal) > :
    public  BaseJacobianBlock< Rigid3(InReal) , F311(OutReal) >
{
public:
    typedef Rigid3(InReal) In;
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
    enum { adim = InDeriv::total_size - dim };  // size of angular velocity vector

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:   \f$ F = grad p.M = t.grad w.M + A.A0^{-1}.grad(p*-w.t0).M + (p0.grad w + w.I - grad p*).M \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - p* is the mls coordinate
        - w is the mls weight (first value of basis)
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + Omega x.A.A0^{-1}.grad(p*-w.t0).M\f$
    */

    static const bool constant=false;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa0;      ///< =   A0^{-1}.(grad p*- t0.grad w).M   =  dF/dA
    OutCoord PFa;      ///< =    A.PFa0
    OutCoord C;   ///< =  (p0.grad w + w.I - grad p*).M      =  constant term

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dp, const Hessian& /*ddp*/)
    {
        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dp[i][0];
        Ft=F0.transposed()*dw;
        Mat<dim,dim,Real> gradps; for (unsigned int j = 0; j < dim; ++j) for (unsigned int k = 0; k < dim; ++k) gradps(j,k)=dp[k][j+1];
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        PFa0.getF()=AOinv*(gradps* F0 - covMN(InPos.getCenter(),Ft) );
        Mat<dim,dim,Real> wI; for (unsigned int j = 0; j < dim; ++j) wI(j,j)=p[0];
        C.getF()=covMN(SPos,Ft) + (wI-gradps)*F0 ;

        rotMat A0; InPos.getOrientation().toMatrix(A0);
        PFa.getF()= A0 * PFa0.getF();
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        rotMat A; data.getOrientation().toMatrix(A);
        PFa.getF()= A * PFa0.getF();  // = update of J according to current transform

        result.getF() +=  covMN(data.getCenter(),Ft) + PFa.getF() + C.getF();;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        const cpMatrix W=crossProductMatrix(getAngular(data));
        result.getF() += covMN(getLinear(data),Ft) + W * PFa.getF();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        getLinear(result) += data.getF() * Ft ;

        for(unsigned int i=0; i<mdim; ++i) getAngular(result) += cross(PFa.getF().col(i),data.getF().col(i));
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];

        for(unsigned int j=0; j<mdim; ++j)
        {
            cpMatrix W=-crossProductMatrix(PFa.getF().col(j));
            for(unsigned int l=0; l<adim; ++l)   for(unsigned int i=0; i<dim; ++i)   J(j+i*mdim,l+dim)+=W(i,l);
        }
        return J;
    }


    // TO DO : implement this !!
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> F332
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class MLSJacobianBlock< Rigid3(InReal) , F332(OutReal) > :
    public  BaseJacobianBlock< Rigid3(InReal) , F332(OutReal) >
{
public:
    typedef Rigid3(InReal) In;
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
    enum { adim = InDeriv::total_size - dim };  // size of angular velocity vector

    typedef typename MLSInfo< dim, InInfo<In>::order, InReal >::basis Basis;
    typedef Vec<dim,Basis> Gradient;
    typedef Mat<dim,dim,Basis> Hessian;

    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;
    typedef Mat<dim,mdim,Real> mHessian;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

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
        - \f$ d F = dt.grad w.M + Omega x. A.A0^{-1}.grad(p*-w.t0).M\f$
        - \f$ d(grad F)_k = dt.(grad2 w)_k^T.M + Omega x. A.A0^{-1}.[ grad2(p*)_k - t0.grad2(w)_k].M
    */

    static const bool constant=false;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    mHessian dFt;      ///< =   (grad2 w)_k^T.M   =  d (grad F)_k/dt
    OutCoord PFdFa0;      ///< =   A0^{-1}.(grad p*- t0.grad w).M, [A0^{-1}.[ grad2(p*)_k - t0.grad2(w)_k].M]   =  dF/dA , d (grad F)_k/dA
    OutCoord PFdFa;      ///< =   A.PFdFa0
    OutCoord C;   ///< =  (p0.grad w + w.I - grad p*).M , [( I_k.grad w + p0.(grad2 w)_k + (grad w)_k.I - (grad2 p*)_k).M ]       =  constant term


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Basis& p, const Gradient& dp, const Hessian& ddp)
    {
        rotMat A0inv; InPos.getOrientation().inverse().toMatrix(A0inv);
        rotMat A0; InPos.getOrientation().toMatrix(A0);

        SpatialCoord dw; for(unsigned int i=0; i<dim; i++) dw[i]=dp[i][0];
        Ft=F0.transposed()*dw;
        Mat<dim,dim,Real> gradps; for (unsigned int i = 0; i < dim; ++i) for (unsigned int j = 0; j < dim; ++j) gradps(i,j)=dp[j][i+1];
        PFdFa0.getF()=A0inv*(gradps* F0 - covMN(InPos.getCenter(),Ft) );
        PFdFa.getF()= A0 * PFdFa0.getF();
        Mat<dim,dim,Real> wI; for (unsigned int j = 0; j < dim; ++j) wI(j,j)=p[0];
        C.getF()=covMN(SPos,Ft) + (wI-gradps)*F0 ;

        Mat<dim,dim,Real> ddw; for (unsigned int i = 0; i < dim; ++i) for (unsigned int j = 0; j < dim; ++j) ddw(i,j)=ddp(i,j)[0];
        dFt=ddw.transposed()*F0;

        for (unsigned int k = 0; k < dim; ++k)
        {
            for (unsigned int i = 0; i < dim; ++i) for (unsigned int j = 0; j < dim; ++j) gradps(i,j)=ddp(j,k)[i+1];
            PFdFa0.getGradientF(k) = A0inv*(gradps* F0 - covMN(InPos.getCenter(),dFt[k]) );
            PFdFa.getGradientF(k)= A0 * PFdFa0.getGradientF(k);
            for (unsigned int j = 0; j < dim; ++j) wI(j,j)=dw[k];
            SpatialCoord one; one[k]=1;
            C.getGradientF(k)=covMN(SPos,dFt[k]) + covMN(one,Ft) + (wI-gradps)*F0 ;
        }
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        // = update of J according to current transform
        rotMat A; data.getOrientation().toMatrix(A);
        PFdFa.getF()= A * PFdFa0.getF();
        for (unsigned int k = 0; k < dim; ++k)            PFdFa.getGradientF(k)= A * PFdFa0.getGradientF(k);


        result.getF() +=  covMN(data.getCenter(),Ft) + PFdFa.getF()  + C.getF();
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN( data.getCenter(), dFt[k]) + PFdFa.getGradientF(k) + C.getGradientF(k) ;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        const cpMatrix W=crossProductMatrix(getAngular(data));

        result.getF() += covMN(getLinear(data),Ft) + W * PFdFa.getF();
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN(getLinear(data),dFt[k]) + W * PFdFa.getGradientF(k);
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        getLinear(result) += data.getF() * Ft ;
        for (unsigned int k = 0; k < dim; ++k) getLinear(result) += data.getGradientF(k) * dFt[k] ;

        for(unsigned int i=0; i<mdim; ++i)
        {
            getAngular(result) += cross(PFdFa.getF().col(i),data.getF().col(i));
            for (unsigned int k = 0; k < dim; ++k) getAngular(result) += cross(PFdFa.getGradientF(k).col(i),data.getGradientF(k).col(i));
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int j=0; j<mdim; ++j)
        {
            cpMatrix W=-crossProductMatrix(PFdFa.getF().col(j));
            for(unsigned int l=0; l<adim; ++l)   for(unsigned int i=0; i<dim; ++i)   J(j+i*mdim,l+dim)+=W(i,l);
        }
        unsigned int offset=dim*mdim;
        for(unsigned int k=0; k<dim; ++k)
        {
            for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+offset+i*mdim,i)=dFt[k][j];
            for(unsigned int j=0; j<mdim; ++j)
            {
                cpMatrix W=-crossProductMatrix(PFdFa.getGradientF(k).col(j));
                for(unsigned int l=0; l<adim; ++l)   for(unsigned int i=0; i<dim; ++i)   J(j+offset+i*mdim,l+dim)+=W(i,l);
            }
            offset+=dim*mdim;
        }
        return J;
    }

    // TO DO : implement this !!
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};



} // namespace defaulttype
} // namespace sofa



#endif
