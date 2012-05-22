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
#ifndef FLEXIBLE_LinearJacobianBlock_INL
#define FLEXIBLE_LinearJacobianBlock_INL

#include "../deformationMapping/LinearJacobianBlock.h"
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
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define V3(type) StdVectorTypes<Vec<3,type>,Vec<3,type>,type>
#define EV3(type) ExtVectorTypes<Vec<3,type>,Vec<3,type>,type>
#define F331(type)  DefGradientTypes<3,3,0,type>
#define F321(type)  DefGradientTypes<3,2,0,type>
#define F332(type)  DefGradientTypes<3,3,1,type>
#define Rigid3(type)  StdRigidTypes<3,type>
#define Affine3(type)  StdAffineTypes<3,type>
#define Quadratic3(type)  StdQuadraticTypes<3,type>

//////////////////////////////////////////////////////////////////////////////////
////  helpers
//////////////////////////////////////////////////////////////////////////////////

template<class Real1, class Real2,  int Dim1, int Dim2>
inline Mat<Dim1, Dim2, Real2> covMN(const Vec<Dim1,Real1>& v1, const Vec<Dim2,Real2>& v2)
{
    Mat<Dim1, Dim2, Real2> res;
    for ( unsigned int i = 0; i < Dim1; ++i)
        for ( unsigned int j = 0; j < Dim2; ++j)
        {
            res[i][j] = (Real2)v1[i] * v2[j];
        }
    return res;
}

template<class _Real>
inline Mat<3, 3, _Real> crossProductMatrix(const Vec<3,_Real>& v)
{
    Mat<3, 3, _Real> res;
    res[0][0]=0;
    res[0][1]=-v[2];
    res[0][2]=v[1];
    res[1][0]=v[2];
    res[1][1]=0;
    res[1][2]=-v[0];
    res[2][0]=-v[1];
    res[2][1]=v[0];
    res[2][2]=0;
    return res;
}


//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> Vec3
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< V3(InReal) , V3(OutReal) > :
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

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;

    /**
    Mapping:   \f$ p = w.t + w.(p0-t0)  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.

    Jacobian:    \f$ dp = w.dt \f$
      */

    static const bool constantJ=true;

    OutCoord C;   ///< =  w.(p0-t0)  =  constant term
    Real Pt;      ///< =   w         =  dp/dt


    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        C=(OutPos-InPos)*Pt;
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

    // no geometric striffness (contstant J)
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
};

//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> ExtVec3
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< V3(InReal) , EV3(OutReal) > :
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

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;

    /**
    Mapping:   \f$ p = w.t + w.(p0-t0)  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.

    Jacobian:    \f$ dp = w.dt \f$
      */

    static const bool constantJ=true;

    OutCoord C;   ///< =  w.(p0-t0)  =  constant term
    Real Pt;      ///< =   w         =  dp/dt


    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        C=(OutPos-InPos)*Pt;
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

    // no geometric striffness (contstant J)
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> F331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< V3(InReal) , F331(OutReal) > :
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

    typedef Vec<mdim,Real> Gradient;
    typedef Mat<mdim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;


    /**
    Mapping:
        - \f$ F = grad p = (t+p0-t0).grad w + w.I  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w \f$
      */

    static const bool constantJ=true;

    OutCoord C;       ///< =  (p0-t0).grad w + w.I   =  constant term
    Gradient Ft;  ///< =   grad w     =  d F/dt

    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=dw;
        C.getF()=covMN(OutPos-InPos,Ft);
        for(unsigned int i=0; i<mdim; i++) C.getF()[i][i]+=w; // to do: anisotropy
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

    // no geometric striffness (contstant J)
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> F321
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< V3(InReal) , F321(OutReal) > :
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
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };
//    enum { mdim = Out::spatial_dimensions };

    typedef Vec<mdim,Real> Gradient;
//    typedef Mat<mdim,dim,Real> Hessian;
    typedef Mat<2,3,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;

    /**
    Mapping:
        - \f$ F = grad p = (t+p0-t0).grad w + w.I  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w \f$
      */

    static const bool constantJ=true;

    OutCoord C;       ///< =  w.(p0-t0)   ,  (p0-t0).grad w + w.I   =  constant term
    Gradient Ft;  ///< =   grad w     =  d F/dt

    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=dw;
        C.getF()=covMN(OutPos-InPos,Ft);
        for(unsigned int i=0; i<mdim; i++) C.getF()[i][i]+=w; // to do: anisotropy
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
        MatBlock J;
        for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<mdim; j++) J[j+i*mdim][i]=Ft[j];
        return J;
    }
};


//////////////////////////////////////////////////////////////////////////////////
////  Vec3 -> F332
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< V3(InReal) , F332(OutReal) > :
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

    typedef Vec<mdim,Real> Gradient;
    typedef Mat<mdim,dim,Real> Hessian;
    typedef Mat<dim,mdim,Real> HessianT;
    typedef Vec<dim, Real> SpatialCoord;

    /**
    Mapping:
        - \f$ F = grad p = (t+p0-t0).grad w + w.I  \f$
        - \f$ (grad F)_k = (t+p0-t0).(grad2 w)_k^T + [(grad w)_k.I +  I_k.grad w]  \f$ the second term can be removed because \f$ \sum_i grad w_i =0\f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - grad denotes spatial derivatives
        - _k denotes component/column k
    Jacobian:
        - \f$ d F = dt.grad w \f$
        - \f$ d (grad F)_k = dt.(grad2 w)_k^T \f$
      */

    static const bool constantJ=true;

    OutCoord C;       ///< =  w.(p0-t0)   ,  (p0-t0).grad w + w.I,  (p0-t0).(grad2 w)_k^T + [(grad w)_k.I +  I_k.grad w]   =  constant term
    Real Pt;           ///< =   w     =  dp/dt
    Gradient Ft;  ///< =   grad w     =  d F/dt
    HessianT dFt;  ///< =   (grad2 w)_k^T   =  d (grad F)_k/dt

    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& dw, const Hessian& ddw)
    {
        Ft=dw;
        C.getF()=covMN(OutPos-InPos,Ft);
        for(unsigned int i=0; i<mdim; i++) C.getF()[i][i]+=w; // to do: anisotropy
        dFt=ddw.transposed();
        for (unsigned int k = 0; k < dim; ++k)
        {
            C.getGradientF(k)=covMN(OutPos-InPos,dFt[k]);
            //            for(unsigned int i=0;i<mdim;i++) C.getGradientF(k)[i][i]+=Ft[k]; // to do: anisotropy
            //            for(unsigned int i=0;i<mdim;i++) C.getGradientF(k)[k][i]+=Ft[i]; // to do: anisotropy
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
            for(unsigned int i=0; i<dim; i++) for(unsigned int j=0; j<mdim; i++) J(j+offset+i*mdim,i)=dFt[k][j];
            offset+=mdim*dim;
        }
        return J;
    }

    // no geometric striffness (contstant J)
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
};





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

    /**
    Mapping:   \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0) = w.t + w.A.q0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.

    Jacobian:    \f$ dp = w.dt + w.dA.q0\f$
      */

    static const bool constantJ=true;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa;   ///< =  w.q0      =  dp/dA


    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa=In::inverse(InPos).pointToParent(OutPos)*Pt;
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

    // no geometric striffness (contstant J)
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
};



//////////////////////////////////////////////////////////////////////////////////
////  Affine3 -> ExtVec3
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

    /**
    Mapping:   \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0) = w.t + w.A.q0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.

    Jacobian:    \f$ dp = w.dt + w.dA.q0\f$
    */

    static const bool constantJ=true;

    Real Pt;      ///< =   w         =  dp/dt
    OutCoord Pa;   ///< =  w.q0      =  dp/dA


    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa=In::inverse(InPos).pointToParent(OutPos)*Pt;
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

    // no geometric striffness (contstant J)
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
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

    typedef Vec<mdim,Real> Gradient;
    typedef Mat<mdim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;

    /**
    Mapping:
        - \f$ F = grad p = (t+A.q0).grad w + w.A.A0^{-1}  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w + dA.( q0.grad w + w.A0^{-1} )\f$
    */

    static const bool constantJ=true;

    Gradient Ft;       ///< =   grad w     =  d F/dt
    OutCoord PFa;      ///< =   q0.grad w + w.A0^{-1}   =  dF/dA

    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=dw;
        InCoord inverseInitialTransform = In::inverse(InPos);   // A0^{-1}
        SpatialCoord vectorInLocalCoordinates = inverseInitialTransform.pointToParent(OutPos);  // q0
        PFa.getF()=covMN(vectorInLocalCoordinates,Ft) + inverseInitialTransform.getAffine() * w; // to do: anisotropy
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


    // no geometric striffness (contstant J)
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
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

    typedef Vec<mdim,Real> Gradient;
    typedef Mat<mdim,dim,Real> Hessian;
    typedef Mat<dim,mdim,Real> HessianT;
    typedef Vec<dim, Real> SpatialCoord;

    /**
    Mapping:
        - \f$ F = grad p = (t+A.q0).grad w + w.A.A0^{-1}  \f$
        - \f$ (grad F)_k = (t+A.q0).(grad2 w)_k^T + A.[(grad w)_k.A0^{-1} +  A0^{-1}_k.grad w]  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
        - _k denotes component/column k
    Jacobian:
        - \f$ d F = dt.grad w + dA.( q0.grad w + w.A0^{-1} )\f$
        - \f$ d (grad F)_k = dt.(grad2 w)_k^T + dA.[q0.(grad2 w)_k^T + (grad w)_k.A0^{-1} +  A0^{-1}_k.grad w] \f$
    */

    static const bool constantJ=true;

    Gradient Ft;       ///< =   grad w     =  d F/dt
    HessianT dFt;      ///< =   (grad2 w)_k^T   =  d (grad F)_k/dt
    OutCoord PFdFa;      ///< =   q0.grad w + w.A0^{-1}, [q0.(grad2 w)_k^T + (grad w)_k.A0^{-1} +  A0^{-1}_k.grad w]   =  dF/dA , d (grad F)_k/dA

    void init( const InCoord& InPos, const SpatialCoord& OutPos, const Real& w, const Gradient& dw, const Hessian& ddw)
    {
        Ft=dw;
        dFt=ddw.transposed();

        InCoord inverseInitialTransform = In::inverse(InPos);   // A0^{-1}
        SpatialCoord vectorInLocalCoordinates = inverseInitialTransform.pointToParent(OutPos);  // q0
        PFdFa.getF()=covMN(vectorInLocalCoordinates,Ft) + inverseInitialTransform.getAffine() * w; // to do: anisotropy

        Mat<dim,dim> AOinv = inverseInitialTransform.getAffine();
        Mat<dim,dim> AOinvT = AOinv.transposed();
        for (unsigned int k = 0; k < dim; ++k) PFdFa.getGradientF(k) = covMN( vectorInLocalCoordinates, dFt[k]) + AOinv * dw[k] + covMN(AOinvT[k],dw);
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

    // no geometric striffness (contstant J)
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
};

} // namespace defaulttype
} // namespace sofa



#endif
