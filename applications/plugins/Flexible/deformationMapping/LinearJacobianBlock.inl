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

#include "LinearJacobianBlock.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "../frame/AffineTypes.h"
#include "../frame/QuadraticTypes.h"
#include "DeformationGradientTypes.h"

namespace sofa
{

namespace defaulttype
{

//////////////////////////////////////////////////////////////////////////////////
////  macros
//////////////////////////////////////////////////////////////////////////////////
#define V3(type) StdVectorTypes<Vec<3,type>,Vec<3,type>,type>
#define EV3(type) ExtVectorTypes<Vec<3,type>,Vec<3,type>,type>
#define F331(type)  DefGradientTypes<3,3,1,type>
#define F332(type)  DefGradientTypes<3,3,2,type>
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
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;

    /**
    Mapping:   \f$ p = w.t + w.(p0-t0)  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.

    Jacobian:    \f$ dp = w.dt \f$
      */

    OutCoord C;   ///< =  w.(p0-t0)  =  constant term
    Real Pt;      ///< =   w         =  dp/dt

    void init( const InCoord& InPos, const OutCoord& OutPos, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        C=(OutPos-InPos)*w;
        Pt=w;
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
        MatBlock J;
        for(unsigned int i=0; i<dim; i++) J[i][i]=Pt;
        return J;
    }
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
    typedef typename Inherit::Real Real;

    enum { dim = Out::spatial_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;

    /**
    Mapping:   \f$ p = w.t + w.(p0-t0)  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.

    Jacobian:    \f$ dp = w.dt \f$
      */

    OutCoord C;   ///< =  w.(p0-t0)  =  constant term
    Real Pt;      ///< =   w         =  dp/dt

    void init( const InCoord& InPos, const OutCoord& OutPos, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        C=(OutPos-InPos)*w;
        Pt=w;
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
        MatBlock J;
        for(unsigned int i=0; i<dim; i++) J[i][i]=Pt;
        return J;
    }
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
    typedef typename Inherit::Real Real;

    enum { dim = Out::material_dimensions };

    typedef Vec<dim,Real> Gradient;
    typedef Mat<dim,dim,Real> Hessian;

    /**
    Mapping:
        - \f$ p = w.t + w.(p0-t0)  \f$
        - \f$ grad p = (t+p0-t0).grad w + w.I  \f$
    where :
        - t0 is t in the reference configuration,
        - p0 is the position of p in the reference configuration.
    Jacobian:
        - \f$ dp = w.dt \f$
        - \f$ d grad p = dt.grad w \f$
      */

    OutCoord C;       ///< =  w.(p0-t0)   ,  (p0-t0).grad w + w.I   =  constant term
    Real Pt;           ///< =   w     =  dp/dt
    Gradient Ft;  ///< =   grad w     =  d grad p/dt

    void init( const InCoord& InPos, const OutCoord& OutPos, const Real& w, const Gradient& dw, const Hessian& /*ddw*/)
    {
        C.getCenter()=(OutPos.getCenter()-InPos)*w;
        C.getMaterialFrame()=covMN(OutPos.getCenter()-InPos,dw); for(unsigned int i=0; i<dim; i++) C.getMaterialFrame()[i][i]+=w;
        Pt=w;
        Ft=dw;
    }

    void addapply( OutCoord& result, const InCoord& data )
    {
        result.getCenter() +=  data * Pt + C.getCenter();
        result.getMaterialFrame() +=  covMN(data,Ft) + C.getMaterialFrame();
    }

    void addmult( OutDeriv& result,const InDeriv& data )
    {
        result.getCenter() += data * Pt ;
        result.getMaterialFrame() += covMN(data,Ft) ;
    }

    void addMultTranspose( InDeriv& result, const OutDeriv& data )
    {
        result += data.getCenter() * Pt ;
        result += data.getMaterialFrame() * Ft ;
    }

    MatBlock getJ()
    {
        MatBlock J;
        for(unsigned int i=0; i<dim; i++) J[i][i]=Pt;
        for(unsigned int i=0; i<dim; i++) J[i+dim][0]=J[i+2*dim][1]=J[i+3*dim][2]=Ft[i];
        return J;
    }
};




} // namespace defaulttype
} // namespace sofa



#endif
