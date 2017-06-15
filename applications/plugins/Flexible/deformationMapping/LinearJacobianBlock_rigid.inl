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
#ifndef FLEXIBLE_LinearJacobianBlock_rigid_INL
#define FLEXIBLE_LinearJacobianBlock_rigid_INL

#include "LinearJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "../types/AffineTypes.h"
#include "../types/QuadraticTypes.h"
#include "../types/DeformationGradientTypes.h"
#include <sofa/helper/decompose.h>

namespace sofa
{

namespace defaulttype
{



//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> Vec3
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Rigid3(InReal) , V3(OutReal) > :
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

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part

    /**
    Mapping:   \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0) = w.t + w.A.q0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.

    Jacobian:
        - \f$ dp = w.dt + Omega x w.A.q0 \f$
    Geometric Stiffness:
        - \f$ K = dJ^T/dOmega fc = (fc)x (A.w.q0)x  \f$
      */

    static const bool constant=false;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa0;   ///< =  w.q0    : weighted point in local frame
    OutCoord Pa;   ///< =  Omega x w.q0        : rotated point


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa0= InPos.pointToChild(SPos)*Pt;
        Pa= InPos.rotate(Pa0);  // = (SPos - InPos.getCenter() ) * w[i] ;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        Pa= data.rotate(Pa0);
        result +=  data.getCenter() * Pt + Pa;
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

    KBlock getK(const OutDeriv& childForce, bool stabilization=false)
    {
        // will only work for 3d rigids
        Mat<adim,adim,Real> block = crossProductMatrix( childForce ) * crossProductMatrix( Pa );

        if( stabilization )
        {
            block.symmetrize(); // symmetrization
            helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
        }

        KBlock K;
        for( unsigned i=0; i<adim; ++i )
            for( unsigned j=0; j<adim; ++j )
                K[dim+i][dim+j] = block[i][j];
        return K;
    }

    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const SReal& kfactor )
    {
        getAngular(df) += In::crosscross( childForce, Pa, getAngular(dx) ) * kfactor;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> ExtVec3   same as Vec3 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Rigid3(InReal) , EV3(OutReal) > :
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

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part

    /**
    Mapping:   \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0) = w.t + w.A.q0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.

    Jacobian:
        - \f$ dp = w.dt + Omega x w.A.q0 \f$
    Geometric Stiffness:
        - \f$ K = dJ^T/dOmega fc = (fc)x (A.w.q0)x  \f$
      */

    static const bool constant=false;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa0;   ///< =  w.q0    : weighted point in local frame
    OutCoord Pa;   ///< =  A Pa0        : rotated point


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa0= InPos.pointToChild(SPos)*Pt;
        Pa= InPos.rotate(Pa0);  // = (SPos - InPos.getCenter() ) * w[i] ;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        Pa= data.rotate(Pa0);
        result +=  data.getCenter() * Pt + Pa;
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

    KBlock getK(const OutDeriv& childForce, bool stabilization=false)
    {
        // will only work for 3d rigids
        Mat<adim,adim,Real> block = crossProductMatrix( childForce ) * crossProductMatrix( Pa );

        if( stabilization )
        {
            block.symmetrize(); // symmetrization
            helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
        }

        KBlock K;
        for( unsigned i=0; i<adim; ++i )
            for( unsigned j=0; j<adim; ++j )
                K[dim+i][dim+j] = block[i][j];
        return K;
    }

    void addDForce( InDeriv& df, const InDeriv& dx, const OutDeriv& childForce, const SReal& kfactor )
    {
        getAngular(df) += In::crosscross( childForce, Pa, getAngular(dx) ) * kfactor;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> F331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Rigid3(InReal) , F331(OutReal) > :
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

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0).grad w.M + w.A.A0^{-1}.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + Omega x. A( q0.grad w + w.A0^{-1} ).M\f$
    Geometric Stiffness:
        - \f$ K = dJ^T/dOmega fc = (fc)x (A.( q0.grad w + w.A0^{-1} ).M)x  \f$
    */

    static const bool constant=false;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa0;      ///< =   q0.grad w.M + w.A0^{-1}.M
    OutCoord PFa;      ///< =    A.PFa0

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;

        SpatialCoord vectorInLocalCoordinates = InPos.pointToChild(SPos);  // q0
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        PFa0.getF()=covMN(vectorInLocalCoordinates,Ft) + AOinv * F0 * w;
        rotMat A0; InPos.getOrientation().toMatrix(A0);
        PFa.getF()= A0 * PFa0.getF();
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        rotMat A; data.getOrientation().toMatrix(A);
        PFa.getF()= A * PFa0.getF();  // = update of J according to current transform

        result.getF() +=  covMN(data.getCenter(),Ft) + PFa.getF();
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

        //        getAngular(result)[0] += dot(PFa.getF()[1],data.getF()[2]) - dot(PFa.getF()[2],data.getF()[1]);
        //        getAngular(result)[1] += dot(PFa.getF()[2],data.getF()[0]) - dot(PFa.getF()[0],data.getF()[2]);
        //        getAngular(rresult)[2] += dot(PFa.getF()[0],data.getF()[1]) - dot(PFa.getF()[1],data.getF()[0]);
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


    KBlock getK(const OutDeriv& childForce, bool stabilization=false)
    {
        // will only work for 3d rigids
        KBlock K = KBlock();
        for(unsigned int k=0; k<mdim; ++k)
        {
            Mat<adim,adim,Real> block = crossProductMatrix( childForce.getF().col(k) ) * crossProductMatrix( PFa.getF().col(k) );

            if( stabilization )
            {
                block.symmetrize(); // symmetrization
                helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
            }

            for( unsigned i=0; i<adim; ++i )
                for( unsigned j=0; j<adim; ++j )
                    K[dim+i][dim+j] += block[i][j];
        }
        return K;
    }

    void addDForce( InDeriv& df, const InDeriv& dx,  const OutDeriv& childForce, const SReal& kfactor )
    {
        for(unsigned int i=0; i<mdim; ++i) getAngular(df) += In::crosscross( childForce.getF().col(i), PFa.getF().col(i), getAngular(dx) ) * kfactor;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> F321  same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Rigid3(InReal) , F321(OutReal) > :
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

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0).grad w.M + w.A.A0^{-1}.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + Omega x. A( q0.grad w + w.A0^{-1} ).M\f$
    Geometric Stiffness:
        - \f$ K = dJ^T/dOmega fc = (fc)x (A.( q0.grad w + w.A0^{-1} ).M)x  \f$
    */

    static const bool constant=false;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa0;      ///< =   q0.grad w.M + w.A0^{-1}.M
    OutCoord PFa;      ///< =    A.PFa0

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;

        SpatialCoord vectorInLocalCoordinates = InPos.pointToChild(SPos);  // q0
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        PFa0.getF()=covMN(vectorInLocalCoordinates,Ft) + AOinv * F0 * w;
        rotMat A0; InPos.getOrientation().toMatrix(A0);
        PFa.getF()= A0 * PFa0.getF();
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        rotMat A; data.getOrientation().toMatrix(A);
        PFa.getF()= A * PFa0.getF();  // = update of J according to current transform

        result.getF() +=  covMN(data.getCenter(),Ft) + PFa.getF();
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

    KBlock getK(const OutDeriv& childForce, bool stabilization=false)
    {
        // will only work for 3d rigids
        KBlock K = KBlock();
        for(unsigned int k=0; k<mdim; ++k)
        {
            Mat<adim,adim,Real> block = crossProductMatrix( childForce.getF().col(k) ) * crossProductMatrix( PFa.getF().col(k) );

            if( stabilization )
            {
                block.symmetrize(); // symmetrization
                helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
            }

            for( unsigned i=0; i<adim; ++i )
                for( unsigned j=0; j<adim; ++j )
                    K[dim+i][dim+j] += block[i][j];
        }
        return K;
    }

    void addDForce( InDeriv& df, const InDeriv& dx,  const OutDeriv& childForce, const SReal& kfactor )
    {
        for(unsigned int i=0; i<mdim; ++i) getAngular(df) += In::crosscross( childForce.getF().col(i), PFa.getF().col(i), getAngular(dx) ) * kfactor;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> F311  same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Rigid3(InReal) , F311(OutReal) > :
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

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0).grad w.M + w.A.A0^{-1}.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + Omega x. A( q0.grad w + w.A0^{-1} ).M\f$
    Geometric Stiffness:
        - \f$ K = dJ^T/dOmega fc = (fc)x (A.( q0.grad w + w.A0^{-1} ).M)x  \f$
    */

    static const bool constant=false;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    OutCoord PFa0;      ///< =   q0.grad w.M + w.A0^{-1}.M
    OutCoord PFa;      ///< =    A.PFa0

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;

        SpatialCoord vectorInLocalCoordinates = InPos.pointToChild(SPos);  // q0
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        PFa0.getF()=covMN(vectorInLocalCoordinates,Ft) + AOinv * F0 * w;
        rotMat A0; InPos.getOrientation().toMatrix(A0);
        PFa.getF()= A0 * PFa0.getF();
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        rotMat A; data.getOrientation().toMatrix(A);
        PFa.getF()= A * PFa0.getF();  // = update of J according to current transform

        result.getF() +=  covMN(data.getCenter(),Ft) + PFa.getF();
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

    KBlock getK(const OutDeriv& childForce, bool stabilization=false)
    {
        // will only work for 3d rigids
        KBlock K = KBlock();
        for(unsigned int k=0; k<mdim; ++k)
        {
            Mat<adim,adim,Real> block = crossProductMatrix( childForce.getF().col(k) ) * crossProductMatrix( PFa.getF().col(k) );

            if( stabilization )
            {
                block.symmetrize(); // symmetrization
                helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
            }

            for( unsigned i=0; i<adim; ++i )
                for( unsigned j=0; j<adim; ++j )
                    K[dim+i][dim+j] += block[i][j];
        }
        return K;
    }

    void addDForce( InDeriv& df, const InDeriv& dx,  const OutDeriv& childForce, const SReal& kfactor )
    {
        for(unsigned int i=0; i<mdim; ++i) getAngular(df) += In::crosscross( childForce.getF().col(i), PFa.getF().col(i), getAngular(dx) ) * kfactor;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> F332
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Rigid3(InReal) , F332(OutReal) > :
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

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;
    typedef Mat<dim,mdim,Real> mHessian;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

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
        - \f$ d F = dt.grad w.M + Omega x. A( q0.grad w + w.A0^{-1} ).M\f$
        - \f$ d (grad F)_k = dt.(grad2 w)_k^T.M + Omega x. A [q0.(grad2 w)_k^T + (grad w)_k.A0^{-1} +  A0^{-1}_k.grad w].M \f$
    Geometric Stiffness:
        - \f$ K = dJ^T/dOmega fc = (fc)x (A.( q0.grad w + w.A0^{-1} ).M)x
                  + dJ^T/dOmega fc = (fc)x (A.[q0.(grad2 w)_k^T + (grad w)_k.A0^{-1} +  A0^{-1}_k.grad w].M)x  \f$

    */

    static const bool constant=false;

    mGradient Ft;       ///< =   grad w     =  d F/dt
    mHessian dFt;      ///< =   (grad2 w)_k^T   =  d (grad F)_k/dt
    OutCoord PFdFa0;      ///< =   [q0.grad w + w.A0^{-1}].M, [q0.(grad2 w)_k^T + (grad w)_k.A0^{-1} +  A0^{-1}_k.grad w].M   =  dF/dA , d (grad F)_k/dA
    OutCoord PFdFa;      ///< =   A.PFdFa0

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& ddw)
    {
        Ft=F0.transposed()*dw;
        dFt=ddw.transposed()*F0;

        SpatialCoord vectorInLocalCoordinates = InPos.pointToChild(SPos);  // q0
        rotMat AOinv; InPos.getOrientation().inverse().toMatrix(AOinv);
        PFdFa0.getF()=covMN(vectorInLocalCoordinates,Ft) + AOinv * F0 * w;
        rotMat A0; InPos.getOrientation().toMatrix(A0);
        PFdFa.getF()= A0 * PFdFa0.getF();

        rotMat AOinvT = AOinv.transposed();
        rotMat AOinvM; for (unsigned int k = 0; k < dim; ++k) AOinvM[k]=F0.transposed()*AOinv[k];
        for (unsigned int k = 0; k < dim; ++k)
        {
            PFdFa0.getGradientF(k) = covMN( vectorInLocalCoordinates, dFt[k]) + AOinvM * dw[k] + covMN(AOinvT[k],Ft);
            PFdFa.getGradientF(k)= A0 * PFdFa0.getGradientF(k);
        }
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        // = update of J according to current transform
        rotMat A; data.getOrientation().toMatrix(A);
        PFdFa.getF()= A * PFdFa0.getF();
        for (unsigned int k = 0; k < dim; ++k)            PFdFa.getGradientF(k)= A * PFdFa0.getGradientF(k);


        result.getF() +=  covMN(data.getCenter(),Ft) + PFdFa.getF();
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN( data.getCenter(), dFt[k]) + PFdFa.getGradientF(k);
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

    KBlock getK(const OutDeriv& childForce, bool stabilization=false)
    {
        // will only work for 3d rigids
        KBlock K = KBlock();
        for(unsigned int k=0; k<mdim; ++k)
        {
            Mat<adim,adim,Real> block = crossProductMatrix( childForce.getF().col(k) ) * crossProductMatrix( PFdFa.getF().col(k) );
            for (unsigned int m = 0; m < dim; ++m) block += crossProductMatrix( childForce.getGradientF(m).col(k) ) * crossProductMatrix( PFdFa.getGradientF(m).col(k) );

            if( stabilization )
            {
                block.symmetrize(); // symmetrization
                helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
            }

            for( unsigned i=0; i<adim; ++i )
                for( unsigned j=0; j<adim; ++j )
                    K[dim+i][dim+j] += block[i][j];
        }
        return K;
    }

    void addDForce( InDeriv& df, const InDeriv& dx,  const OutDeriv& childForce, const SReal& kfactor )
    {
        for(unsigned int i=0; i<mdim; ++i)
        {
            getAngular(df) += In::crosscross( childForce.getF().col(i), PFdFa.getF().col(i), getAngular(dx) ) * kfactor;
            for (unsigned int m = 0; m < dim; ++m)         getAngular(df) += In::crosscross( childForce.getGradientF(m).col(i), PFdFa.getGradientF(m).col(i), getAngular(dx) ) * kfactor;
        }
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Rigid3 -> Affine3 = F331 with dw=0
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Rigid3(InReal) , Affine3(OutReal) > :
    public  BaseJacobianBlock< Rigid3(InReal) , Affine3(OutReal) >
{
    public:
    typedef Rigid3(InReal) In;
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
    enum { adim = InDeriv::total_size - dim };  // size of angular velocity vector

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    typedef Mat<dim,adim,Real> cpMatrix; // cross product matrix of angular part
    typedef Mat<dim,dim,Real> rotMat;

    /**
    Mapping:
        - \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0) = w.t + w.A.q0  \f$
        - \f$ F = w.A.A0^{-1}.F0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0,F0 is the position of p,F in the reference configuration.
        - q0 is the local position of p0.
    Jacobian:
        - \f$ dp = w.dt + Omega x w.A.q0\f$
        - \f$ d F = w.Omega x. A.A0^{-1}.F0\f$
    Geometric Stiffness:
        - \f$ K = dJ^T/dOmega fc = (fc)x (A.w.q0)x
                + dJ^T/dOmega fc = (fc)x (x.A.A0^{-1}.F0)x  \f$
    */

    static const bool constant=false;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa0;      ///< =   w.q0      =  dp/dA  , w.A0^{-1}.F0   =  dF/dA  : weighted affine in local frame
    OutCoord Pa;      ///< =    A.Pa0  : rotated affine

    void init( const InCoord& InPos, const OutCoord& OutPos, const SpatialCoord& /*SPos*/, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;

        InCoord inverseInitialTransform = In::inverse(InPos);   // A0^{-1}
        Pa0.getCenter()=inverseInitialTransform.pointToParent(OutPos.getCenter())*w;
        Pa.getCenter() = InPos.rotate(Pa0.getCenter() );

        rotMat AOinv; inverseInitialTransform.getOrientation().toMatrix(AOinv);
        Pa0.getAffine()=AOinv*OutPos.getAffine()*w;
        rotMat A0; InPos.getOrientation().toMatrix(A0);
        Pa.getAffine()=A0 *Pa0.getAffine();
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        // = update of J according to current transform
        Pa.getCenter()= data.rotate(Pa0.getCenter());
        rotMat A; data.getOrientation().toMatrix(A);
        Pa.getAffine()=A *Pa0.getAffine();

        result.getCenter() +=  data.getCenter() * Pt + Pa.getCenter();
        result.getAffine() +=  Pa.getAffine() ;

        for (unsigned int j = 0; j < dim; ++j) result.getAffine()[j][j] -= Pt; // this term cancels the initial identity affine matrix
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getVCenter() += getLinear(data) * Pt + cross(getAngular(data), Pa.getCenter());

        const cpMatrix W=crossProductMatrix(getAngular(data));
        result.getVAffine() += W * Pa.getAffine();
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        getLinear(result) += data.getVCenter() * Pt ;
        getAngular(result) += cross(Pa.getCenter(), data.getVCenter());

        for(unsigned int i=0; i<dim; ++i) getAngular(result) += cross(Pa.getAffine().col(i),data.getVAffine().col(i));
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) J(i,i)=Pt;

        cpMatrix W=-crossProductMatrix(Pa.getCenter());
        for(unsigned int l=0; l<adim; ++l) for (unsigned int i=0; i<dim; ++i) J(i,l+dim)=W(i,l);

        for(unsigned int j=0; j<dim; ++j)
        {
            W=-crossProductMatrix(Pa.getAffine().col(j));
            for(unsigned int l=0; l<adim; ++l)   for(unsigned int i=0; i<dim; ++i)   J(j+(i+1)*dim,l+dim)+=W(i,l);
        }
        return J;
    }


    KBlock getK(const OutDeriv& childForce, bool stabilization=false)
    {
        // will only work for 3d rigids
        KBlock K;
        Mat<adim,adim,Real> block = crossProductMatrix( childForce.getVCenter() ) * crossProductMatrix( Pa.getCenter() );

        if( stabilization )
        {
            block.symmetrize(); // symmetrization
            helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
        }

        for( unsigned i=0; i<adim; ++i )
            for( unsigned j=0; j<adim; ++j )
                K[dim+i][dim+j] = block[i][j];
        for(unsigned int k=0; k<dim; ++k)
        {
            Mat<adim,adim,Real> block = crossProductMatrix( childForce.getVAffine().col(k) ) * crossProductMatrix( Pa.getAffine().col(k) );

            if( stabilization )
            {
                block.symmetrize(); // symmetrization
                helper::Decompose<Real>::NSDProjection( block ); // negative, semi-definite projection
            }

            for( unsigned i=0; i<adim; ++i )
                for( unsigned j=0; j<adim; ++j )
                    K[dim+i][dim+j] += block[i][j];
        }
        return K;
    }

    void addDForce( InDeriv& df, const InDeriv& dx,  const OutDeriv& childForce, const SReal& kfactor )
    {
        getAngular(df) += In::crosscross( childForce.getVCenter(), Pa.getCenter(), getAngular(dx) ) * kfactor;
        for(unsigned int i=0; i<dim; ++i) getAngular(df) += In::crosscross( childForce.getVAffine().col(i), Pa.getAffine().col(i), getAngular(dx) ) * kfactor;
    }
};




} // namespace defaulttype
} // namespace sofa



#endif
