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

    Jacobian:    \f$ dp = w.dt + w.dA.q0\f$
      */

    static const bool constantJ=true;

    Real Pt;      ///< =   w         =  dp/dt
    QuadraticCoord Pa;   ///< =  w.q0      =  dp/dA


    void init( const InCoord& InPos, const OutCoord& /*OutPos*/, const SpatialCoord& SPos, const MaterialToSpatial& /*M*/, const Real& w, const Gradient& /*dw*/, const Hessian& /*ddw*/)
    {
        Pt=w;
        Pa=In::convertToQuadraticCoord(In::inverse(InPos).pointToParent(SPos))*Pt;
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
    KBlock getK(const OutDeriv& /*childForce*/) {return KBlock();}
    void addDForce( InDeriv& /*df*/, const InDeriv& /*dx*/,  const OutDeriv& /*childForce*/, const double& /*kfactor */) {}
};




} // namespace defaulttype
} // namespace sofa



#endif
