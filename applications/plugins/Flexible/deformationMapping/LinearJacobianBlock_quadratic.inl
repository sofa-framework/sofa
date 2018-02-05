/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef FLEXIBLE_LinearJacobianBlock_quadratic_INL
#define FLEXIBLE_LinearJacobianBlock_quadratic_INL

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
////  Quadratic3 -> Vec3
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Quadratic3(InReal) , V3(OutReal) > :
    public  BaseJacobianBlock< Quadratic3(InReal) , V3(OutReal) >
{
public:
    typedef Quadratic3(InReal) In;
    typedef V3(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    enum { dimq = In::num_quadratic_terms };

    enum { dim = Out::spatial_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    /**
    Mapping:   \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0)^* = w.t + w.A.q0^*  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - ^* converts a vector to a 2nd order basis (e.g. (x,y) -> (x,y,x^2,y^2,xy))

    Jacobian:    \f$ dp = w.dt + w.dA.q0^*\f$
      */

    static const bool constant=true;

    Real Pt;      ///< =   w         =  dp/dt
    QuadraticCoord Pa;   ///< =  w.q0^*      =  dp/dA


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa=convertSpatialToQuadraticCoord(In::inverse(InPos).pointToParent(SPos))*Pt;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result +=  data.getCenter() * Pt + data.getQuadratic() * Pa;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += data.getVCenter() * Pt + data.getVQuadratic() * Pa;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data * Pt ;
        for (unsigned int j = 0; j < dim; ++j) result.getVQuadratic()[j] += Pa * data[j];
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
////  Quadratic3 -> ExtVec3   same as Vec3 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Quadratic3(InReal) , EV3(OutReal) > :
    public  BaseJacobianBlock< Quadratic3(InReal) , EV3(OutReal) >
{
public:
    typedef Quadratic3(InReal) In;
    typedef EV3(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    enum { dimq = In::num_quadratic_terms };

    enum { dim = Out::spatial_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    /**
    Mapping:   \f$ p = w.t + w.A.(A0^{-1}.p0-A0^{-1}.t0)^* = w.t + w.A.q0^*  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - ^* converts a vector to a 2nd order basis (e.g. (x,y) -> (x,y,x^2,y^2,xy))

    Jacobian:    \f$ dp = w.dt + w.dA.q0^*\f$
    */

    static const bool constant=true;

    Real Pt;      ///< =   w         =  dp/dt
    QuadraticCoord Pa;   ///< =  w.q0^*      =  dp/dA


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa=convertSpatialToQuadraticCoord(In::inverse(InPos).pointToParent(SPos))*Pt;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result +=  data.getCenter() * Pt + data.getQuadratic() * Pa;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result += data.getVCenter() * Pt + data.getVQuadratic() * Pa;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data * Pt ;
        for (unsigned int j = 0; j < dim; ++j) result.getVQuadratic()[j] += Pa * data[j];
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
////  Quadratic3 -> F331
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Quadratic3(InReal) , F331(OutReal) > :
    public  BaseJacobianBlock< Quadratic3(InReal) , F331(OutReal) >
{
public:
    typedef Quadratic3(InReal) In;
    typedef F331(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    enum { dimq = In::num_quadratic_terms };

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0^*).grad w.M + w.A.grad q0^*.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.( q0^*.grad w + w.grad q0^* ).M\f$
    */

    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    Mat<dimq,mdim,Real> PFa;      ///< =   q0^*.grad w.M + w.grad q0^*.M   =  dF/dA

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;
        InCoord inverseInitialTransform = In::inverse(InPos);   // inverse of quadratic transform (warning: only affine part inverted)

        SpatialCoord q0 = inverseInitialTransform.pointToParent(SPos);
        QuadraticCoord vectorInLocalCoordinates = convertSpatialToQuadraticCoord( q0 ); // q0^*

        Mat<dimq,dim,Real> gradQ0 = SpatialToQuadraticCoordGradient (q0) ;  // grad q0^*
        for (unsigned int i=0; i<dim; ++i) for (unsigned int j=0; j<dim; ++j) gradQ0[i][j]=inverseInitialTransform.getAffine()[i][j];

        PFa=covMN(vectorInLocalCoordinates,Ft) + gradQ0 * F0 * w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getQuadratic()*PFa;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data.getVCenter(),Ft) + data.getVQuadratic()*PFa;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getF() * Ft ;

        for (unsigned int j = 0; j < dim; ++j)
        {
            result.getVQuadratic()[j] += PFa * (data.getF()[j]);
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dimq; ++l)    J(j+i*mdim,l+dim+i*dimq)=PFa[l][j];
        return J;
    }


    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Quadratic3 -> F321  same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Quadratic3(InReal) , F321(OutReal) > :
    public  BaseJacobianBlock< Quadratic3(InReal) , F321(OutReal) >
{
public:
    typedef Quadratic3(InReal) In;
    typedef F321(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    enum { dimq = In::num_quadratic_terms };

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0^*).grad w.M + w.A.grad q0^*.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.( q0^*.grad w + w.grad q0^* ).M\f$
    */

    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    Mat<dimq,mdim,Real> PFa;      ///< =   q0^*.grad w.M + w.grad q0^*.M   =  dF/dA

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;
        InCoord inverseInitialTransform = In::inverse(InPos);   // inverse of quadratic transform (warning: only affine part inverted)

        SpatialCoord q0 = inverseInitialTransform.pointToParent(SPos);
        QuadraticCoord vectorInLocalCoordinates = convertSpatialToQuadraticCoord( q0 ); // q0^*

        Mat<dimq,dim,Real> gradQ0 = SpatialToQuadraticCoordGradient (q0) ;  // grad q0^*
        for (unsigned int i=0; i<dim; ++i) for (unsigned int j=0; j<dim; ++j) gradQ0[i][j]=inverseInitialTransform.getAffine()[i][j];

        PFa=covMN(vectorInLocalCoordinates,Ft) + gradQ0 * F0 * w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getQuadratic()*PFa;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data.getVCenter(),Ft) + data.getVQuadratic()*PFa;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getF() * Ft ;

        for (unsigned int j = 0; j < dim; ++j)
        {
            result.getVQuadratic()[j] += PFa * (data.getF()[j]);
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dimq; ++l)    J(j+i*mdim,l+dim+i*dimq)=PFa[l][j];
        return J;
    }


    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};



//////////////////////////////////////////////////////////////////////////////////
////  Quadratic3 -> F311  same as F331 -> Factorize using partial instanciation ?
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Quadratic3(InReal) , F311(OutReal) > :
    public  BaseJacobianBlock< Quadratic3(InReal) , F311(OutReal) >
{
public:
    typedef Quadratic3(InReal) In;
    typedef F311(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    enum { dimq = In::num_quadratic_terms };

    enum { dim = Out::spatial_dimensions };
    enum { mdim = Out::material_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,mdim,Real> MaterialToSpatial;

    typedef Vec<mdim,Real> mGradient;

    /**
    Mapping:
        - \f$ F = grad p.M = (t+A.q0^*).grad w.M + w.A.grad q0^*.M  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
    Jacobian:
        - \f$ d F = dt.grad w.M + dA.( q0^*.grad w + w.grad q0^* ).M\f$
    */

    static const bool constant=true;

    mGradient Ft;       ///< =   grad w.M     =  d F/dt
    Mat<dimq,mdim,Real> PFa;      ///< =   q0^*.grad w.M + w.grad q0^*.M   =  dF/dA

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        Ft=F0.transposed()*dw;
        InCoord inverseInitialTransform = In::inverse(InPos);   // inverse of quadratic transform (warning: only affine part inverted)

        SpatialCoord q0 = inverseInitialTransform.pointToParent(SPos);
        QuadraticCoord vectorInLocalCoordinates = convertSpatialToQuadraticCoord( q0 ); // q0^*

        Mat<dimq,dim,Real> gradQ0 = SpatialToQuadraticCoordGradient (q0) ;  // grad q0^*
        for (unsigned int i=0; i<dim; ++i) for (unsigned int j=0; j<dim; ++j) gradQ0[i][j]=inverseInitialTransform.getAffine()[i][j];

        PFa=covMN(vectorInLocalCoordinates,Ft) + gradQ0 * F0 * w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getQuadratic()*PFa;
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data.getVCenter(),Ft) + data.getVQuadratic()*PFa;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getF() * Ft ;

        for (unsigned int j = 0; j < dim; ++j)
        {
            result.getVQuadratic()[j] += PFa * (data.getF()[j]);
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dimq; ++l)    J(j+i*mdim,l+dim+i*dimq)=PFa[l][j];
        return J;
    }


    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};

//////////////////////////////////////////////////////////////////////////////////
////  Quadratic3 -> F332
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Quadratic3(InReal) , F332(OutReal) > :
    public  BaseJacobianBlock< Quadratic3(InReal) , F332(OutReal) >
{
public:
    typedef Quadratic3(InReal) In;
    typedef F332(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    enum { dimq = In::num_quadratic_terms };

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
        - \f$ F = grad p.M = (t+A.q0^*).grad w.M + w.A.grad q0^*.M  \f$
        - \f$ (grad F)_k = [ (t+A.q0^*).(grad2 w)_k^T + A.[(grad w)_k.grad q0^* +  grad q0^*_k.grad w] ].M \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0 is the position of p in the reference configuration.
        - q0 is the local position of p0.
        - grad denotes spatial derivatives
        - _k denotes component/column k
    Jacobian:
        - \f$ d F = [dt.grad w + dA.( q0^*.grad w + w.grad q0^* )].M \f$
        - \f$ d (grad F)_k = [dt.(grad2 w)_k^T + dA.[q0^*.(grad2 w)_k^T + (grad w)_k.grad q0^* +  grad q0^*_k.grad w] ].M \f$
    */

    static const bool constant=true;

    mGradient Ft;       ///< =   grad w     =  d F/dt
    mHessian dFt;      ///< =   (grad2 w)_k^T   =  d (grad F)_k/dt
    Vec<dim+1,Mat<dimq,mdim,Real> > PFdFa;      ///< =   q0.grad w + w.grad q0^*, [q0.(grad2 w)_k^T + (grad w)_k.grad q0^* +  grad q0^*_k.grad w]   =  dF/dA , d (grad F)_k/dA

    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& F0, const Real& w, const Gradient& dw, const Hessian& ddw)
    {
        Ft=F0.transposed()*dw;
        dFt=ddw.transposed()*F0;

        InCoord inverseInitialTransform = In::inverse(InPos);   // inverse of quadratic transform (warning: only affine part inverted)

        SpatialCoord q0 = inverseInitialTransform.pointToParent(SPos);
        QuadraticCoord vectorInLocalCoordinates = convertSpatialToQuadraticCoord( q0 ); // q0^*

        Mat<dimq,dim,Real> gradQ0 = SpatialToQuadraticCoordGradient (q0) ;  // grad q0^*
        for (unsigned int i=0; i<dim; ++i) for (unsigned int j=0; j<dim; ++j) gradQ0[i][j]=inverseInitialTransform.getAffine()[i][j];

        PFdFa[0]=covMN(vectorInLocalCoordinates,Ft) + gradQ0 * F0 * w;

        Mat<dim,dimq,Real> gradQ0T = gradQ0.transposed();
        Mat<dimq,mdim> gradQ0M; for (unsigned int k = 0; k < dimq; ++k) gradQ0M[k]=F0.transposed()*gradQ0[k];
        for (unsigned int k = 0; k < dim; ++k) PFdFa[k+1] = covMN( vectorInLocalCoordinates, dFt[k]) + gradQ0M * dw[k] + covMN(gradQ0T[k],Ft);
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getF() +=  covMN(data.getCenter(),Ft) + data.getQuadratic()*PFdFa[0];
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN( data.getCenter(), dFt[k]) + data.getQuadratic() * PFdFa[k+1];
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getF() += covMN(data.getVCenter(),Ft) + data.getVQuadratic()*PFdFa[0];
        for (unsigned int k = 0; k < dim; ++k) result.getGradientF(k) += covMN(data.getVCenter(),dFt[k]) + data.getVQuadratic() * PFdFa[k+1];
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getF() * Ft ;
        for (unsigned int k = 0; k < dim; ++k) result.getVCenter() += data.getGradientF(k) * dFt[k] ;

        for (unsigned int j = 0; j < dim; ++j)
        {
            result.getVQuadratic()[j] += PFdFa[0] * (data.getF()[j]);
            for (unsigned int k = 0; k < dim; ++k) result.getVQuadratic()[j] += PFdFa[k+1] * (data.getGradientF(k)[j]);
        }
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+i*mdim,i)=Ft[j];
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dimq; ++l)    J(j+i*mdim,l+dim+i*dimq)=PFdFa[0][l][j];
        unsigned int offset=dim*mdim;
        for(unsigned int k=0; k<dim; ++k)
        {
            for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) J(j+offset+i*mdim,i)=dFt[k][j];
            for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<mdim; ++j) for(unsigned int l=0; l<dimq; ++l)    J(j+offset+i*mdim,l+dim+i*dimq)=PFdFa[k+1][l][j];
            offset+=dim*mdim;
        }
        return J;
    }

    // no geometric striffness (constant J)
    KBlock getK(const OutDeriv& /*childForce*/, bool=false) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const SReal& /*kfactor */) {}
};


//////////////////////////////////////////////////////////////////////////////////
////  Quadratic3 -> Affine3 = F331 with dw=0
//////////////////////////////////////////////////////////////////////////////////

template<class InReal,class OutReal>
class LinearJacobianBlock< Quadratic3(InReal) , Affine3(OutReal) > :
    public  BaseJacobianBlock< Quadratic3(InReal) , Affine3(OutReal) >
{
public:
    typedef Quadratic3(InReal) In;
    typedef Affine3(OutReal) Out;

    typedef BaseJacobianBlock<In,Out> Inherit;
    typedef typename Inherit::InCoord InCoord;
    typedef typename Inherit::InDeriv InDeriv;
    typedef typename Inherit::OutCoord OutCoord;
    typedef typename Inherit::OutDeriv OutDeriv;
    typedef typename Inherit::MatBlock MatBlock;
    typedef typename Inherit::KBlock KBlock;
    typedef typename Inherit::Real Real;

    typedef typename In::QuadraticCoord QuadraticCoord; // vec9
    enum { dimq = In::num_quadratic_terms };

    enum { dim = Out::spatial_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;
    typedef Vec<dim, Real> SpatialCoord;
    typedef Mat<dim,dim,Real> MaterialToSpatial;

    /**
    Mapping:
        - \f$ p =  w.t + w.A.q0^*  \f$
        - \f$ F = w.A.grad q0^*.F0  \f$
    where :
        - (A0,t0) are the frame orientation and position (A,t) in the reference configuration,
        - p0,F0 is the position of p,F in the reference configuration.
        - q0 is the local position of p0.
    Jacobian:
        - \f$ dp = w.dt + w.dA.q0\f$
        - \f$ d F = w.dA.grad q0^*.F0\f$
    */

    static const bool constant=true;

    Real Pt;      ///< =   w         =  dp/dt
    QuadraticCoord Pa;   ///< =  w.q0^*      =  dp/dA
    Mat<dimq,dim,Real> PFa;      ///< =   w.grad F0^*.M   =  dF/dA

    void init( const InCoord& InPos, const OutCoord& OutPos, const SpatialCoord& /*SPos*/, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;

        InCoord inverseInitialTransform = In::inverse(InPos);   // inverse of quadratic transform (warning: only affine part inverted)

        SpatialCoord q0 = inverseInitialTransform.pointToParent(OutPos.getCenter());
        QuadraticCoord vectorInLocalCoordinates = convertSpatialToQuadraticCoord( q0 ); // q0^*

        Mat<dimq,dim,Real> gradQ0 = SpatialToQuadraticCoordGradient (q0) ;  // grad q0^*
        for (unsigned int i=0; i<dim; ++i) for (unsigned int j=0; j<dim; ++j) gradQ0[i][j]=inverseInitialTransform.getAffine()[i][j];

        Pa=vectorInLocalCoordinates*Pt;
        PFa=gradQ0 * OutPos.getAffine() * w;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getCenter() +=  data.getCenter() * Pt + data.getQuadratic() * Pa;
        result.getAffine() +=  data.getQuadratic() * PFa ;
        for (unsigned int j = 0; j < dim; ++j) result.getAffine()[j][j] -= Pt; // this term cancels the initial identity Affine matrix
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getVCenter() += data.getVCenter() * Pt + data.getVQuadratic() * Pa;
        result.getVAffine() += data.getVQuadratic() * PFa;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result.getVCenter() += data.getVCenter() * Pt ;
        for (unsigned int j = 0; j < dim; ++j) result.getVQuadratic()[j] += Pa * (data.getVCenter())[j] + PFa * (data.getVAffine()[j]);
    }

    MatBlock getJ()
    {
        MatBlock J = MatBlock();
        for(unsigned int i=0; i<dim; ++i) J(i,i)=Pt;
        for(unsigned int i=0; i<dim; ++i) for(unsigned int j=0; j<dim; ++j)
            {
                J(j,i+(j+1)*dim)=Pa[i];
                for(unsigned int l=0; l<dimq; ++l)   J(j+(i+1)*dim,l+dim+i*dimq)=PFa[l][j];
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
